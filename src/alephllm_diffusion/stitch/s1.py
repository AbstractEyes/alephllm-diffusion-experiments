"""alephllm_diffusion.stitch.s1 - stage 1 of the stitch instrument: Beatrix's states at the stage-0 rows. Reads s0.pt from the
output folder (settings.OUT_DIR) and her trunk at a stored step through alephllm_diffusion.beatrix.extract (the bare core, the DOC
byte in front, bf16 autocast on the card, LayerNorm without affine per block afterwards); the untrained controls (random:SEED)
differ from her trunk only in their weights (random init, the seed given). Writes s1_<tag>.pt (fp16) to the output folder.
Per block in BLOCKS, for every fit row her state at the row's CLOSING byte (the byte after the token's span) and at its LAST byte;
for every eval caption her state at the phrase's closing byte (the full stop after it, the caption read in full) and at its last
byte; for every phrase token of every eval caption (the matched span, ' and' included) her state at that token's closing and last
byte.
THE MOOD FORM (--mood; the registered fallback caption form of e029, anima-trainer's relayed-phrase picture test): her states at
the phrase tokens of e029's captions (the 16 phrases of the read of record, the registered primary read, on the scenes [::4])
rewritten "..., {phrase} mood.": the bytes up to the phrase's end are the bare caption's, so every phrase token keeps its byte
span and only the last token's closing byte changes ('.' becomes ' '); no tokenizer is called. Output: s1_mood_<tag>.pt (the e029
rows' eval indices, the mood texts, the token rows).
THE MOUNT (the testing plan, step 0; added 2026-10-06 before any read): --mount=<group> mounts the anchors of a trained arm group
(one of the arm refit's groups, e.g. gCA) on her trunk through the library's own route (geolip.alephllm.arm_mount.mount_group,
alephllm >= 0.10.6: the trunk-bound check, the carried members' content hashes, every member live) before any tap; the
extractor's per-block prefill then reads the armed stream (amoe-lora >= 0.2.11 applies each wrapper's adapter in prefill). Tag:
step<N>_<group> (e.g. step245674_gCA). The file records the mount (prog.mounted). Without --mount nothing changes.
THE FRAME (added the same date, for the frame column of the testing plan's prediction PM3(a)): --frame=cap puts the caption arm's
own label "caption: " (its pack rows' line label) in front of every caption, fit and eval alike; every byte position moves by the
label's length, nothing else changes; tag suffix _fcap. The registered bare framing stays the default and the read of record.
Usage: python -m alephllm_diffusion.stitch.s1 <step | random:SEED> [limit] [--mood] [--mount=<group>] [--frame=cap]
  (limit: the first N fit captions and the eval captions of the first two scenes, whole, for stage 2's per-scene centring; a smoke
  test. CUDA_VISIBLE_DEVICES=-1 runs it on the CPU.)
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

from .. import settings  # noqa: E402
from ..beatrix import extract as bx  # noqa: E402

BLOCKS = [8, 12, 16, 18, 20, 22, 24, 28]
OUT_DIR = settings.OUT_DIR
# THE BATCHES (settings.S1_BATCHING): "padded" hands the extractor 1,024 captions at a time and lets it fill right-padded batches
# up to 16,384 bytes (the trunk is causal: a caption's own positions never see the padding); "equal" is the 0.2.0 form (128 captions
# at a time, batches of equal byte length only, ~4 captions a pass). Under bf16 autocast the two round differently (batch shape),
# so a file records its form and the grid refuses to mix forms.
PADDED = settings.S1_BATCHING == "padded"
CHUNK = 1024 if PADDED else 128
BATCHING = ({"form": "padded", "chunk": CHUNK, "max_tokens": 16384, "max_batch": 512} if PADDED else
            {"form": "equal", "chunk": CHUNK, "max_batch": 64})
SCENE_STEP = 4                                                          # e029's scenes: e001's [::4], the contact sheet's rows
FRAMES = {"cap": "caption: "}                                           # the caption arm's own line label (the pack's render_row)


def trunk_for(arg, device, mount=None, frame=None):
    """(tag, model, mount record, frame text): her trunk at a stored step (or an untrained copy), with a refit group mounted when
    asked; the tag names the trunk, the mount and the frame."""
    if arg.startswith("random:"):
        assert mount is None, "a mount needs her trained trunk (the anchors are trunk-bound)"
        seed = int(arg.split(":")[1])
        tag, model = f"random{seed}", bx.build_model(step=None, random_init_seed=seed, device=device)
    else:
        tag, model = f"step{int(arg)}", bx.build_model(step=int(arg), device=device)
    rec = None
    if mount:
        from geolip.alephllm.arm_mount import mount_group
        import amoe.core.adapter as AD
        assert hasattr(AD.BlockWithAdapter, "prefill"), "amoe-lora >= 0.2.11 is needed: prefill must apply the adapter"
        prog = mount_group(model, mount, require_step=int(arg), device=device)
        model.eval()
        rec = dict(prog.mounted)
        tag += f"_{mount}"
        print(f"[stitch s1] {tag}: mounted {rec.get('run')} close {rec.get('close')} ({len(rec.get('members', []))} members, "
              f"every member live) through geolip.alephllm.arm_mount", flush=True)
    assert frame is None or frame in FRAMES, (frame, sorted(FRAMES))
    if frame:
        tag += f"_f{frame}"
    return tag, model, rec, FRAMES.get(frame, "")


def eval_view(s0, limit):
    """The eval captions and the matched-span token rows a run uses (limit: the eval captions of the first two scenes, the token
    rows re-indexed to them; stage 2 calls the same function)."""
    evals, tok = s0["evals"], s0["eval_tok"].clone()
    if not limit:
        return evals, tok
    keep_sc = sorted({e["scene"] for e in evals})[:2]                       # stage 2's smoke selection (two whole scenes)
    keep = [i for i, e in enumerate(evals) if e["scene"] in keep_sc]
    remap = {old: new for new, old in enumerate(keep)}
    m = torch.tensor([int(x) in remap for x in tok[:, 0].tolist()], dtype=torch.bool)
    tok = tok[m]
    tok[:, 0] = torch.tensor([remap[int(x)] for x in tok[:, 0].tolist()], dtype=torch.long)
    return [evals[i] for i in keep], tok


def mood_view(s0, limit):
    """e029's rows in the mood form: (the eval indices in stage 2's view, scene-major over the scenes [::4] in stage 0's phrase
    order, the export's order; the mood texts; the token rows (row in that list, Qwen3 position, closing byte, last byte, the
    view's eval_tok row)). Each mood text is asserted to be the bare caption with '.' replaced by ' mood.'."""
    evals, tok = eval_view(s0, limit)
    rec = [j for j, p in enumerate(s0["phrases"]) if p["cls"] == "FRAGMENT" and p["split"] == "heldout"]
    at = {(e["scene"], e["phrase"]): i for i, e in enumerate(evals)}
    sel = [at[(sc, j)] for sc in range(0, len(s0["subjects"]), SCENE_STEP) for j in rec if (sc, j) in at]
    texts = []
    for i in sel:
        e, p = evals[i], s0["phrases"][evals[i]["phrase"]]["text"]
        assert e["text"].endswith(f", {p}.") and e["text"][e["close_byte"]] == ".", e["text"]
        texts.append(e["text"][:-1] + " mood.")
        assert texts[-1] == s0["prefix"] + f"an illustration of {s0['subjects'][e['scene']]}, {p} mood.", texts[-1]
    by_eval: dict = {}
    for r, (ei, k, close, last, *_rest) in enumerate(tok.tolist()):
        by_eval.setdefault(ei, []).append((r, k, close, last))
    rows = [(n, k, close, last, r) for n, i in enumerate(sel) for r, k, close, last in by_eval[i]]
    return sel, texts, rows


def main_mood(arg: str, limit: int = 0, device: str = "cuda", mount: str | None = None) -> str:
    """Her states at the phrase tokens of e029's mood-form captions (closing and last byte, every block), fp16."""
    import hashlib
    s0_path = os.path.join(OUT_DIR, "s0.pt")
    s0 = torch.load(s0_path, weights_only=False)
    sel, texts, rows = mood_view(s0, limit)
    seed = int(arg.split(":")[1]) if arg.startswith("random:") else None
    tag, model, mount_rec, _ = trunk_for(arg, device, mount=mount)      # the mood form is read in the bare framing only
    tag += f"_limit{limit}" if limit else ""
    d, nb = model.nf.weight.numel(), len(BLOCKS)
    out = {k: torch.zeros(len(rows), nb, d, dtype=torch.float16) for k in ("evaltok_close", "evaltok_last")}
    by_n: dict = {}
    for r, (n, _k, close, last, _vr) in enumerate(rows):
        by_n.setdefault(n, []).append((r, close, last))
    t0 = time.time()
    print(f"[stitch s1] {tag} THE MOOD FORM: {len(texts)} captions, {len(rows)} phrase tokens, blocks {BLOCKS}; batches: "
          f"{BATCHING}", flush=True)
    for c0 in range(0, len(texts), CHUNK):
        ns = list(range(c0, min(c0 + CHUNK, len(texts))))
        ex = bx.extract(model, [texts[n] for n in ns], layers=BLOCKS, taps=("byte",), device=device, amp=(device == "cuda"),
                        pad=PADDED)
        for j, n in enumerate(ns):
            states = [ex["byte"][b][j] for b in BLOCKS]
            assert states[0].shape[0] == len(texts[n].encode("utf-8")), (states[0].shape, len(texts[n]))
            picks = by_n[n]
            idx_r = torch.tensor([p[0] for p in picks])
            for col, pos in (("close", [p[1] for p in picks]), ("last", [p[2] for p in picks])):
                v = torch.stack([F.layer_norm(s[pos].float(), (d,)) for s in states], dim=1)
                assert torch.isfinite(v).all(), (tag, n)
                out[f"evaltok_{col}"][idx_r] = v.half()
    path = os.path.join(OUT_DIR, f"s1_mood_{tag}.pt")
    torch.save({**out, "blocks": BLOCKS, "arg": arg, "seed": seed, "limit": limit, "eval_index": sel, "texts": texts,
                "rows": torch.tensor(rows, dtype=torch.long), "s0_sha256": hashlib.sha256(open(s0_path, "rb").read()).hexdigest(),
                "mount": mount_rec, "batching": BATCHING}, path)
    print(f"[stitch s1] wrote {path} in {time.time() - t0:.0f} s", flush=True)
    return path


def main(arg: str, limit: int = 0, device: str = "cuda", mount: str | None = None, frame: str | None = None) -> str:
    s0 = torch.load(os.path.join(OUT_DIR, "s0.pt"), weights_only=False)
    fit, rows = s0["fit"], s0["rows"]
    evals, eval_tok = eval_view(s0, limit)
    if limit:
        fit = fit[:limit]
        rows = rows[rows[:, 0] < limit]
    seed = int(arg.split(":")[1]) if arg.startswith("random:") else None
    tag, model, mount_rec, pre = trunk_for(arg, device, mount=mount, frame=frame)
    tag += f"_limit{limit}" if limit else ""
    off = len(pre.encode("utf-8"))                                          # the frame moves every byte position by its length
    d = model.nf.weight.numel()
    n_fit, n_eval, nb = len(fit), len(evals), len(BLOCKS)
    out = {k: torch.zeros(n, nb, d, dtype=torch.float16) for k, n in
           (("fit_close", len(rows)), ("fit_last", len(rows)), ("eval_close", n_eval), ("eval_last", n_eval),
            ("evaltok_close", len(eval_tok)), ("evaltok_last", len(eval_tok)))}
    by_caption: dict = {}
    for r, (ci, _t, close, last, _id, _a, _e) in enumerate(rows.tolist()):
        by_caption.setdefault(ci, []).append((r, close + off, last + off))
    by_eval: dict = {}                                                      # the matched span: every phrase token's own bytes
    for r, (ei, _k, close, last, *_rest) in enumerate(eval_tok.tolist()):
        by_eval.setdefault(ei, []).append((r, close + off, last + off))
    t0 = time.time()
    jobs = [("fit", i, pre + f["text"]) for i, f in enumerate(fit)] + [("eval", i, pre + e["text"]) for i, e in enumerate(evals)]
    total = len(jobs)
    print(f"[stitch s1] {tag}: {n_fit} fit captions ({len(rows)} rows) + {n_eval} eval captions = {total} captions, blocks {BLOCKS}"
          + (f"; every caption framed {pre!r} (positions +{off})" if pre else "") + f"; batches: {BATCHING}", flush=True)
    for c0 in range(0, total, CHUNK):
        chunk = jobs[c0:c0 + CHUNK]
        ex = bx.extract(model, [t for _, _, t in chunk], layers=BLOCKS, taps=("byte",), device=device, amp=(device == "cuda"),
                        pad=PADDED)
        for j, (kind, i, text) in enumerate(chunk):
            states = [ex["byte"][b][j] for b in BLOCKS]                   # each (len(text) bytes, d), the DOC position excluded
            assert states[0].shape[0] == len(text.encode("utf-8")), (states[0].shape, len(text))
            sets = ([("fit", by_caption.get(i, []))] if kind == "fit" else
                    [("eval", [(i, evals[i]["close_byte"] + off, evals[i]["last_byte"] + off)]), ("evaltok", by_eval.get(i, []))])
            for name, picks in sets:
                if not picks:
                    continue
                idx_r = torch.tensor([p[0] for p in picks])
                for col, pos in (("close", [p[1] for p in picks]), ("last", [p[2] for p in picks])):
                    v = torch.stack([F.layer_norm(s[pos].float(), (d,)) for s in states], dim=1)   # (picks, blocks, d)
                    assert torch.isfinite(v).all(), (tag, name, i)
                    out[f"{name}_{col}"][idx_r] = v.half()
        done = min(c0 + CHUNK, total)
        el = time.time() - t0
        print(f"[stitch s1] {tag}: {done}/{total} captions, {el:.0f} s spent, about {el / done * (total - done):.0f} s left",
              flush=True)
    path = os.path.join(OUT_DIR, f"s1_{tag}.pt")
    torch.save({**out, "blocks": BLOCKS, "arg": arg, "seed": seed, "limit": limit, "n_eval_tok": len(eval_tok), "mount": mount_rec,
                "frame": pre, "batching": BATCHING}, path)
    print(f"[stitch s1] wrote {path} in {time.time() - t0:.0f} s", flush=True)
    return path


if __name__ == "__main__":
    flags = {x.split("=", 1)[0][2:]: x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--") and "=" in x}
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda"
    if "--mood" in sys.argv:
        assert "frame" not in flags, "the mood form is read in the bare framing only"
        main_mood(a[0], int(a[1]) if len(a) > 1 else 0, device=dev, mount=flags.get("mount"))
    else:
        main(a[0], int(a[1]) if len(a) > 1 else 0, device=dev, mount=flags.get("mount"), frame=flags.get("frame"))
