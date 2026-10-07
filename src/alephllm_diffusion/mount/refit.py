"""alephllm_diffusion.mount.refit - the diffusion reads over the arm refit's group anchors (the arm groups trained on Beatrix's
trunk). The mount itself lives in the library since alephllm 0.10.6: geolip.alephllm.arm_mount (load_trunk, mount_group,
mount_anchors, masked, only, detach_all; the trunk-bound base_model_id check and the carried-member content_hash_v2 check inside
mount_group). This module keeps only what belongs to the diffusion experiments: the caption pack stream, the stored caption read
reproduced in its own fields, the comparison against the stored result, and the full route's reference logits. The names the
notebook was written against are kept as aliases (build_trunk, mount_refit_group). Nothing runs on import; the mount gates
(alephllm_diffusion.mount.gates) drive it.

THE GROUPS (training repo AbstractPhil/alephllm-mini-beatrix-training, mini-beatrix-3/arm_refit/group/anchors/
<run>_close<k>_<member>.safetensors; the last letter of a group's name, <S> = A or B, is its training seed): gRA/gRB close 4 (the
four: stage arms 1-4), gXA/gXB close 8 (the eight stage arms = the package's default stack = M8), gEA/gEB close 8 (from scratch),
gCA/gCB close 9 (the eight of gX<S> held fixed + s9_caption, the caption arm = M9, all nine on). Seed A's gC anchors are also in
the public package (AbstractPhil/mini-beatrix-3, arms/stages-1-8-caption/); seed B only here. Mc (the caption arm with the eight
masked) = only(prog, ["s9_caption"]): in a stack each anchor sees the stream after the earlier ones, so with the eight off the
ninth sees the bare block output; the package's parity gates and the library's test check that identity at 0.0.

THE STORED CAPTION READ (group_gC*.json, close 9): bpb on 8 batches of 2 x 4096 bytes from PackStream("held", 2, VAL_SEED
99_000_001, CTX) over the caption pack (mini-beatrix-3/caption_pack/pack_v2: manifest.json + <source>/<source>-NNNNN.jsonl; the
frame = the pack builder's render_row, which alephllm_diffusion.mount.caption_frame carries), under bf16 autocast with the
compiled arm chain, per-batch SE about .045. caption_read_repro() re-reads the same batches in the same fields (rows /
member_alone / member_in_group) at a chosen precision: "stored" (bf16 autocast on cuda + the compiled chain) or "fp32" (eager).
compare_to_stored() prints the deltas and flags |delta| > .02 (a set tolerance: the two exact gates stay gates, the precision
delta is reported, never failed).

The extractor's trunk (alephllm_diffusion.beatrix.extract.build_model) is the same AlephLM the library's load_trunk builds;
mount_group works on it as is. Block-level prefill taps taken after the mount see the armed stream (amoe-lora >= 0.2.11).
"""
import contextlib
import json
import math

import numpy as np
import torch

from geolip.alephllm.arm_mount import (ANCHOR_REPO as REPO, GROUPS, STEP as TRUNK_STEP, CAPTION_ARM as CAPTION, STAGE_ARMS as EIGHT,  # noqa: F401
                                       anchor_files, check_frozen_members, detach_all, header_metadata as header_meta, load_trunk,
                                       logits, masked, mount_anchors, mount_group, only)

build_trunk = load_trunk                   # the notebook's names
mount_refit_group = mount_group
PACK_DIR = "mini-beatrix-3/caption_pack/pack_v2"
PACK_MIX = {"dff": 0.161, "scf": 0.161, "danbooru": 0.164, "cc12m": 0.406, "coco": 0.108}   # the caption run's --pack-mix
VAL_SEED = 99_000_001
LN2 = math.log(2.0)


def precision_context(mode, device):
    """'stored' = the stored read's autocast (bf16 on cuda; nothing on cpu); 'fp32' = eager fp32."""
    if mode == "stored":
        from geolip.alephllm.train.precision import autocast
        return autocast(str(device))
    assert mode == "fp32", mode
    return contextlib.nullcontext()


def refit_reference_logits(batch, group="gCA", *, step=TRUNK_STEP, device="cpu", local_dir=None, off=(), precision="fp32", trunk=None,
                           **mount_kw):
    """Logits from the full route (the library's load_trunk + mount_group), with `off` members masked. Gate 2 of the mount gates
    compares them against the extractor's trunk + mount_group at 0.0 in the same precision. Returns (logits, prog)."""
    model = trunk if trunk is not None else load_trunk(step, device, local_dir=local_dir)
    prog = mount_group(model, group, local_dir=local_dir, compile_chain=(precision == "stored"), **mount_kw)
    with torch.no_grad(), masked(prog, list(off)), precision_context(precision, device):
        out = logits(model, batch.to(next(model.parameters()).device))
    return out, prog


def stat(d):
    d = np.asarray(d, dtype=np.float64)
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0, "per_batch": [float(v) for v in d]}


def pack_rows(pack_dir, split="held", mix=PACK_MIX):
    """source -> its rows of the split, in shard order (as the refit's training run read them)."""
    import os
    man = json.load(open(os.path.join(pack_dir, "manifest.json"), encoding="utf-8"))
    rows = {}
    for key in mix:
        assert key in man["sources"], f"the pack has no source {key!r} (it has {sorted(man['sources'])})"
        rows[key] = []
        for sh in man["sources"][key]["shards"]:
            with open(os.path.join(pack_dir, sh["file"]), encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("split") == split:
                        rows[key].append(r)
        assert rows[key], f"the pack's {key} has no {split} rows"
    return rows


def pack_render(pack_builder=None):
    """The caption pack's render_row (the frame the caption arm was trained on), from this package."""
    from .caption_frame import render_row
    return render_row


class PackStream:
    """The refit training run's PackStream, the caption pack in the library's stream contract: one rendered row per
    document + newline + DOC; documents drawn by source from the mix with a seeded RNG, each source's rows in a seeded shuffle;
    next_batch() -> (rows, ctx + 1) int64."""

    def __init__(self, rows, render, tok, doc, n_rows, seed, ctx, mix=PACK_MIX):
        self.src, self.render, self.tok, self.doc, self.rows, self.ctx = rows, render, tok, int(doc), n_rows, ctx
        self.keys = [k for k in mix if mix[k] > 0]
        w = np.asarray([mix[k] for k in self.keys], dtype=np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)
        self.perm = {k: np.random.default_rng(seed + 7 * i + 1).permutation(len(self.src[k])) for i, k in enumerate(self.keys)}
        self.pos = {k: 0 for k in self.keys}
        self.buf = np.empty(0, dtype=np.int64)

    def _next_doc(self):
        k = self.keys[int(self.rng.choice(len(self.keys), p=self.p))]
        if self.pos[k] >= len(self.perm[k]):
            self.perm[k] = np.random.default_rng(int(self.rng.integers(1 << 31))).permutation(len(self.src[k]))
            self.pos[k] = 0
        r = self.src[k][int(self.perm[k][self.pos[k]])]
        self.pos[k] += 1
        ids = np.asarray(self.tok.encode(self.render(r) + "\n"), dtype=np.int64)
        return np.append(ids, np.int64(self.doc))

    def next_batch(self):
        need = self.rows * (self.ctx + 1)
        chunks, have = [self.buf], self.buf.size
        while have < need:
            ids = self._next_doc()
            chunks.append(ids)
            have += ids.size
        flat = np.concatenate(chunks)
        take, self.buf = flat[:need], flat[need:]
        return torch.from_numpy(take.reshape(self.rows, self.ctx + 1).copy())


def caption_batches(pack_dir, ctx, *, pack_builder=None, render_row=None, tok=None, nb=8, rows=2, seed=VAL_SEED, split="held"):
    """The stored read's batches: nb x (rows, ctx + 1) from PackStream(split, rows, seed, ctx) over the pack's held rows."""
    from geolip.alephllm.data.special_tokens import DOC
    if tok is None:
        from geolip.alephllm.data.tokenizers import build_tokenizer
        tok = build_tokenizer("byte-trigram")
    render = render_row or pack_render(pack_builder)
    st = PackStream(pack_rows(pack_dir, split), render, tok, DOC, rows, seed, ctx)
    bs = torch.stack([st.next_batch() for _ in range(nb)])
    assert bs.shape[1:] == (rows, ctx + 1), bs.shape
    return bs


@torch.no_grad()
def bpb_batches(model, batches, device, precision="fp32"):
    """bpb per batch, the stored read's loss form: model(x[:, :-1], targets=x[:, 1:]).loss / ln 2."""
    vals = []
    for xb in batches:
        xb = xb.to(device)
        with precision_context(precision, device):
            o = model(xb[:, :-1], targets=xb[:, 1:])
        vals.append(float(o.loss if hasattr(o, "loss") else o[1]) / LN2)
    return np.asarray(vals, dtype=np.float64)


def caption_read_repro(model, prog, batches, *, precision="fp32", key=CAPTION, device=None):
    """The stored caption read's fields on the given batches: rows[key] (off / all_on / all_on_minus_off), member_alone[m][key]
    (alone / alone_minus_off), member_in_group[m][key] (all_on_minus_masked), at the given precision ('stored' also switches the
    compiled chain on, as the stored read had it; 'fp32' = eager). The trunk is left with every member on."""
    device = device or next(model.parameters()).device
    model.eval()
    prog.set_compile(precision == "stored")
    att = list(prog.attached)
    with masked(prog, att):
        off = bpb_batches(model, batches, device, precision)
    on = bpb_batches(model, batches, device, precision)
    rec = {"attached": att, "precision": precision, "batches": int(len(batches)), "ctx": int(batches.shape[-1] - 1), "seed": VAL_SEED,
           "rows": {key: {"off": float(off.mean()), "all_on": float(on.mean()), "all_on_minus_off": stat(on - off)}},
           "member_alone": {}, "member_in_group": {}}
    for m in att:
        with only(prog, [m]):
            alone = bpb_batches(model, batches, device, precision)
        with masked(prog, [m]):
            msk = bpb_batches(model, batches, device, precision)
        rec["member_alone"][m] = {key: {"alone": float(alone.mean()), "alone_minus_off": stat(alone - off)}}
        rec["member_in_group"][m] = {key: {"all_on_minus_masked": stat(on - msk)}}
    return rec


def stored_close(result_path, close=9):
    r = json.load(open(result_path, encoding="utf-8"))
    recs = [c for c in r["closes"] if c.get("close") == close and not c.get("resumed_read")]
    assert recs, f"close {close} is not in {result_path}"
    return recs[-1]


def compare_to_stored(rec, stored, key=CAPTION, bar=0.02):
    """Rows of (reading, stored, reproduced, delta, flagged) for the caption read; flagged = |delta| > bar. Reported, never failed."""
    rows = [("caption rows, arms off", stored["rows"][key]["off"], rec["rows"][key]["off"]),
            ("caption rows, all on", stored["rows"][key]["all_on"], rec["rows"][key]["all_on"])]
    for m in rec["attached"]:
        if m in stored.get("member_alone", {}):
            rows.append((f"{m} alone", stored["member_alone"][m][key]["alone"], rec["member_alone"][m][key]["alone"]))
            rows.append((f"{m} share (all on minus masked)", stored["member_in_group"][m][key]["all_on_minus_masked"]["mean"],
                         rec["member_in_group"][m][key]["all_on_minus_masked"]["mean"]))
    out = []
    for name, s, r in rows:
        d = float(r) - float(s)
        out.append({"reading": name, "stored": round(float(s), 4), "reproduced": round(float(r), 4), "delta": round(d, 4), "flagged": abs(d) > bar})
    return out


def print_comparison(table):
    print(f"| {'reading':<40} | stored | reproduced | delta | flag |")
    print("|---|---|---|---|---|")
    for r in table:
        print(f"| {r['reading']:<40} | {r['stored']:.4f} | {r['reproduced']:.4f} | {r['delta']:+.4f} | {'FLAG' if r['flagged'] else ''} |")
