"""alephllm_diffusion.stitch.export - THE EXPORT FOR e029 (anima-trainer's picture test of a relayed phrase): after the pick,
the picture test's inputs, so the picture run then only renders. For e029's captions (the read of record's 16 phrases on the
eight scenes SUBJECTS[0::4]: stage 0's own captions and ids, nothing tokenized; the read of record = the held-out phrases that
T5 cuts into fragments), the FINAL Qwen3 states (last_hidden_state, every token of the caption, fp32) of each arm:
  ceiling    the caption, unpatched
  filler     the filler caption, unpatched (its own Qwen3 ids, the same token count; e029 pairs it with the caption's T5 ids)
  relay      the pick of record's cell (Beatrix's trunk, block, byte convention, depth, fit), patched at every phrase token
  companion  the companion pick's cell, when it differs from the pick of record
  control    the untrained trunk (the config's first control, random0 on the record) through the pick of record's block,
             convention, depth and fit
  mount_<g>  with STITCH_EXPORT_MOUNTS, per mounted trunk: its relay at the pick of record's cell, and mount_<g>_own at the
             mount's own pick when that is another cell
The pick of record is the cell alephllm_diffusion.stitch.pick chooses by its registered rule; the companion pick, the cell its
paired tie band chooses.
Each relay is stage 2's own arithmetic, recomputed with stage 2's own functions and batches: the targets (Qwen3 over the fit
captions, 64 per batch), the closed-form ridge with its penalty by cross-validation, the patch vectors at every phrase token of
every eval caption, the patched run one scene per batch; then THE CHECKS against the grid, before any file is written: the
refit's penalty equals the cell's and its cross-validated R2 is within 1e-9; the record set's two final-state reads (final_cos,
final_rel: the patched run against the ceiling at the phrase tokens, over every scene) are within the bar of the cell's stored
ones (1e-6: the same card and code as the grid, so equal; STITCH_EXPORT_BAR overrides it for a run on another device). A relay
cell outside the grid (the control in the last-byte convention: the grid ran the controls in the closing-byte convention only)
is computed by the same arithmetic and labelled unverified. A failed check writes nothing and exits 1.
PRECISION (stated for the picture runner's identity check): the export is fp32, the instrument's precision (Qwen3's weights are
stored in bf16, so fp32 is their exact upcast); the picture runner encodes in bf16 under autocast from the same bf16 weights and
pads to 512 tokens (causal attention: the padding does not reach the caption's own positions). Its override casts these states to
its own dtype at the adapter's input, one rounding. Its identity check decides before any relay image: (i) the override fed the
runner's own encoder output reproduces the runner's picture bit for bit (the path); (ii) the unpatched export matches the
runner's own fp32 forward of the same caption within 1e-4 relative (the same encoding, two cards); (iii) the picture difference
between the runner's bf16 encoding and the export is measured and quoted. Every e029 arm, the ceiling included, renders from this
export: no comparison crosses the two precisions.
THE CONTROL OUTSIDE THE GRID: when the pick of record is in the last-byte convention, its control cell is not in the grid; the
session run computes it first in a selection of its own (stage 2, STITCH_EXTRA_RELAYS, run tag '_control_outside';
`--control-cell` prints the key and the relay to run), and the export verifies the control against that run, labels it outside
the registered grid and prints its adapter-level reads. Without that run the control stays unverified.
THE MOOD FORM (on by default; STITCH_EXPORT_MOOD=0 skips it): after the bare file is written, the same arms for e029's captions in
its registered fallback form PREFIX + 'an illustration of {scene}, {phrase} mood.' (the bare caption's text with '.' replaced by
' mood.'). The Qwen3 ids come from the bundled tokenizer, asserted to reproduce stage 0's ids (Qwen3 and T5, the caption and its
filler) on the bare captions and to keep every token up to the phrase's last; each relay applies the same refitted map (no grid
exists for this form) to her states: the bare file's at every byte the two forms share (her model is causal), stage 1's mood file
(s1_mood_<trunk>.pt) at the one byte they do not share, the last phrase token's closing byte (the space) in the closing-byte
convention; so a mood relay differs from the bare one only where the form does: under a last-byte cell the mood relay's inputs
equal the bare relay's and only the caption around the phrase differs; under a closing-byte cell exactly one state per phrase
differs, the last token's; the header says which. Its checks: (a) wherever the two forms share every input (up to the phrase's
last token; before it for a closing-byte relay), the mood states equal the bare file's within STITCH_EXPORT_MOOD_TOL (1e-4 per
token, relative); (b) her mood-form features equal the bare ones at the shared bytes (cosine >= STITCH_EXPORT_MOOD_FEAT, 0.999),
the last token's closing byte reported; (c) each patched arm equals the ceiling before the first phrase token. A failure or an
error exits 8 with the bare file standing and nothing written for the mood form.
Reads (from the output folder, settings.OUT_DIR, unless named): s0.pt; s1_<trunk>.pt for each relay's trunk (s1_mood_<trunk>.pt
for the mood form); the grid's stitch_<name>_full.json; the pick file pick_<name>.json from the report folder
(settings.REPORT_DIR); the models through stage 2's loader (Qwen3 from settings.LLM_PATH; Anima's adapter, unused here, from
settings.DIT_PATH); the tokenizers bundled with the diffusion-pipe fork (settings.DP); with STITCH_EXPORT_MOUNTS, the mounts grid
and its pick file per mount.
Output: <OUT_DIR>/e029_export_<name>.safetensors (states.<arm> [N, Lmax, 1024] fp32, zeros past each caption; qwen_ids,
filler_qwen_ids [N, Lmax]; lengths [N]; the header as metadata 'header') and the header as JSON beside the grid's summary
(<REPORT_DIR>/e029_export_<name>.json; a smoke run's into <OUT_DIR>); STITCH_RUN_TAG is appended to both names; the mood form's
pair with the suffix '_mood'.
Usage: python -m alephllm_diffusion.stitch.export <config name> [--control-cell]   (record; a smoke config for a test;
CUDA_VISIBLE_DEVICES=-1 runs it on the CPU; STITCH_EXPORT_PICK = another pick file, for a test)"""
import hashlib
import json
import os
import platform
import sys
import time

import torch

from . import s2  # noqa: E402  (stage 2's functions, roots, configs and budgets)

SCENE_STEP = 4                                      # e029's scenes: e001's [::4] (anima-trainer RELAY_SCENE_STEP), the sheet rows
BAR = float(os.environ.get("STITCH_EXPORT_BAR", "1e-6"))
BAR_R2 = float(os.environ.get("STITCH_EXPORT_BAR_R2", "1e-9"))  # both overridable for a run on another device (a test)
MOOD_TOL = float(os.environ.get("STITCH_EXPORT_MOOD_TOL", "1e-4"))      # (a) and (c): per token, relative
MOOD_FEAT = float(os.environ.get("STITCH_EXPORT_MOOD_FEAT", "0.999"))   # (b): her features at the same bytes, cosine
PRECISION = __doc__.split("PRECISION (stated for the picture runner's identity check): ")[1].split("\nTHE CONTROL")[0].replace("\n", " ")
MOOD_FORM = ("exported by the same run beside this file (e029_export_<name>_mood, its own header) when its checks pass: e029's "
             "registered fallback form PREFIX + 'an illustration of {scene}, {phrase} mood.' changes the byte after the phrase ('.' "
             "becomes ' '), so her state at the last phrase token's closing byte differs (stage 1's mood files), and every relay applies "
             "the same refitted map with no grid to check it against; a failed mood pass leaves this file standing (exit 8); "
             "STITCH_EXPORT_MOOD=0 skips it")
TOKENIZER_FILES = ("configs/qwen3_06b/tokenizer.json", "configs/qwen3_06b/tokenizer_config.json", "configs/qwen3_06b/vocab.json",
                   "configs/qwen3_06b/merges.txt", "configs/t5_old/spiece.model", "configs/t5_old/tokenizer.json")


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def peak_note():
    """The process's peak committed memory and working set (Windows) or peak resident set (Linux), for the record."""
    import psutil
    m = psutil.Process().memory_info()
    if hasattr(m, "peak_pagefile"):
        return f"; peak committed {m.peak_pagefile / 2 ** 30:.2f} GB, peak working set {m.peak_wset / 2 ** 30:.2f} GB"
    import resource
    return f"; peak resident {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f} GB"


def parse(key):
    """'record|step245674|b16|close|k12' -> (fit, trunk, block, convention, depth label)."""
    fit, trunk, blk, conv, dep = key.split("|")
    return fit, trunk, int(blk[1:]), conv, dep[1:]


def paths(name):
    """The grid's full JSON and the pick file for a config (a smoke config's both in OUT_DIR)."""
    smoke = name.startswith("smoke")
    full_path = os.path.join(s2.OUT_DIR, f"stitch_{name}.json" if smoke else f"stitch_{name}_full.json")
    pick_path = os.environ.get("STITCH_EXPORT_PICK") or os.path.join(s2.OUT_DIR if smoke else s2.REPORT_DIR, f"pick_{name}.json")
    return full_path, pick_path


def control_cell(name):
    """For the session run: prints 'CONTROL_CELL <key> <relay>' when the grid does not hold the pick of record's control cell
    (the grid ran the controls in the closing-byte convention only), else nothing."""
    cfg = s2.CONFIGS[name]
    full_path, pick_path = paths(name)
    fit_name, _trunk, blk, conv, dlab = parse(json.load(open(pick_path, encoding="utf-8"))["pick"]["key"])
    relay = f"{cfg['controls'][0]}|b{blk}|{conv}"
    key = f"{fit_name}|{relay}|k{dlab}"
    if key not in json.load(open(full_path, encoding="utf-8"))["span"]["cells"]:
        print(f"CONTROL_CELL {key} {relay}", flush=True)


def main(name, device):
    t0 = time.time()
    cfg = s2.CONFIGS[name]
    smoke = name.startswith("smoke")
    assert cfg.get("span"), f"{name}: the export reads a matched-span grid"

    def say(msg):
        print(f"[export] {msg} | {time.time() - t0:.0f} s spent{s2.mem_note()}", flush=True)

    frac = float(os.environ.get("STITCH_MEM_FRACTION", "0.57"))
    card_cap = frac * torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if device == "cuda" else 0.0
    hard = s2.hard_memory_limit(max(s2.BUDGET["ram_hard_gb"], s2.LOAD_COMMIT_GB) + card_cap)    # stage 2's ceiling (Windows)
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(frac)
    full_path, pick_path = paths(name)
    grid = json.load(open(full_path, encoding="utf-8"))
    pk = json.load(open(pick_path, encoding="utf-8"))
    cells = grid["span"]["cells"]

    # the arms (the pick tool's own choices; the control at the pick of record only)
    pick_key = pk["pick"]["key"]
    comp = (pk.get("companion") or {}).get("pick")
    fit_name, trunk, blk, conv, dlab = parse(pick_key)
    assert fit_name == "record" and trunk in cfg["her"], pick_key
    relay_arms = {"relay": pick_key}
    if comp is not None and comp["key"] != pick_key:
        relay_arms["companion"] = comp["key"]
    relay_arms["control"] = f"{fit_name}|{cfg['controls'][0]}|b{blk}|{conv}|k{dlab}"
    # THE MOUNT ARMS (registered 2026-10-06, before any read): STITCH_EXPORT_MOUNTS = her mounted trunks (e.g. step245674_gCA).
    # Per mount: 'mount_<group>' = its relay at the pick of record's own cell (the same fit, block, convention and depth: for
    # PM3(b), the registered picture comparison of a mount arm with the relay arm) and 'mount_<group>_own' = its relay at its own
    # pick (pick_mounts_<trunk>.json) when that is another cell. Each is refitted and checked against the mounts grid's cells
    # exactly as the record's arms are checked against the record grid.
    mount_tags = [t for t in os.environ.get("STITCH_EXPORT_MOUNTS", "").split(",") if t]
    mrun = os.environ.get("STITCH_EXPORT_MOUNTS_RUN", "mounts")          # the mounts grid's config (a smoke chain names its own)
    mounts_info = {}
    if mount_tags:
        msmoke = mrun.startswith("smoke")
        mgrid_path = os.path.join(s2.OUT_DIR, f"stitch_{mrun}.json" if msmoke else f"stitch_{mrun}_full.json")
        mgrid = json.load(open(mgrid_path, encoding="utf-8"))
        assert not (set(mgrid["span"]["cells"]) & set(cells)), "the mounts grid and the record grid share cell keys"
        cells = {**cells, **mgrid["span"]["cells"]}
        mounts_info = {"grid": os.path.basename(mgrid_path), "grid_sha256": sha256(mgrid_path),
                       "grid_partial": mgrid.get("partial"), "picks": {}}
        for t in mount_tags:
            g = t.split("_")[1]                                          # step245674_gCA[_limit80] -> gCA
            mp_path = os.path.join(s2.OUT_DIR if msmoke else s2.REPORT_DIR, f"pick_{mrun}_{t}.json")
            own = json.load(open(mp_path, encoding="utf-8"))["pick"]["key"]
            at_pick = "|".join((fit_name, t, f"b{blk}", conv, f"k{dlab}"))
            relay_arms[f"mount_{g}"] = at_pick
            if own != at_pick:
                relay_arms[f"mount_{g}_own"] = own
            mounts_info["picks"][t] = {"own_pick": own, "at_record_pick": at_pick, "pick_file_sha256": sha256(mp_path)}
    dmap = {str(d): d for d in cfg["depths"]}
    say(f"{name}: the grid {os.path.basename(full_path)}{' (PARTIAL: ' + str(grid['partial']) + ')' if grid.get('partial') else ''}"
        f"; the pick {pk['pick']['key']}; the companion "
        f"{'not computable' if comp is None else comp['key'] + (' (the same cell)' if comp['key'] == pick_key else '')}; the relay "
        f"arms {relay_arms}; the bar {BAR:g} on the final-state reads ({device}; hard ceiling "
        f"{'%.1f GB' % hard if hard else 'not set'})")

    # stage 0: the captions; e029's rows
    s0 = torch.load(os.path.join(s2.OUT_DIR, "s0.pt"), weights_only=False)
    fit, rows = s0["fit"], s0["rows"]
    lim = cfg.get("limit", 0)
    evals, eval_tok = s2.eval_view(s0, lim)
    if lim:
        fit = fit[:lim]
        rows = rows[rows[:, 0] < lim]
    E = len(evals)
    subjects, phrases = s0["subjects"], s0["phrases"]
    rec_ph = [j for j, p in enumerate(phrases) if p["cls"] == "FRAGMENT" and p["split"] == "heldout"]
    assert len(rec_ph) == 16, len(rec_ph)
    scenes_e029 = list(range(0, len(subjects), SCENE_STEP))
    at = {(e["scene"], e["phrase"]): i for i, e in enumerate(evals)}
    sel = [at[(sc, j)] for sc in scenes_e029 for j in rec_ph if (sc, j) in at]
    assert smoke or len(sel) == len(scenes_e029) * len(rec_ph) == 128, len(sel)
    rec = torch.tensor([e["cls"] == "FRAGMENT" and e["held_phrase"] and e["read"] for e in evals])     # stage 2's read of record
    assert all(bool(rec[i]) for i in sel)

    # Qwen3 in fp32 (stage 2's loader; the adapter is not used here)
    qwen, ad = s2.load_models(device)
    del ad
    say(f"Qwen3 loaded in fp32; {len(rows)} fit rows over {len(fit)} fit captions, {E} eval captions, e029's {len(sel)} captions "
        f"({len(rec_ph)} phrases x scenes {[sc for sc in scenes_e029 if any(evals[i]['scene'] == sc for i in sel)]})")

    # the targets at the relays' depths: stage 2's batches (64 fit captions in index order), so each refit sees the grid's numbers
    need = sorted({parse(k)[4] for k in relay_arms.values()}, key=lambda x: [str(d) for d in s2.DEPTHS].index(x))
    assert all(x in dmap for x in need), (need, cfg["depths"])
    by_cap: dict = {}
    for r, (ci, t, *_rest) in enumerate(rows.tolist()):
        by_cap.setdefault(ci, []).append((r, t))
    caps = sorted(by_cap)
    Y = {x: torch.zeros(len(rows), 1024) for x in need}
    t1 = time.time()
    for c0 in range(0, len(caps), 64):
        grp = caps[c0:c0 + 64]
        _, g = s2.qwen_run(qwen, [fit[c]["qwen_ids"] for c in grp], device, collect_all=[dmap[x] for x in need])
        gi = torch.tensor([gi for gi, c in enumerate(grp) for _ in by_cap[c]])
        tt = torch.tensor([t for c in grp for _, t in by_cap[c]])
        rr = torch.tensor([r for c in grp for r, _ in by_cap[c]])
        for x in need:
            Y[x][rr] = g[dmap[x]][gi, tt]
        del g
        if (c0 // 64) % 20 == 0:
            done = min(c0 + 64, len(caps))
            say(f"targets: {done}/{len(caps)} fit captions, about {(time.time() - t1) / done * (len(caps) - done):.0f} s left")
    folds = torch.tensor([fit[c]["fold"] for c in rows[:, 0].tolist()])
    coco_rows = torch.tensor([fit[c]["source"] == "coco" for c in rows[:, 0].tolist()])
    say(f"targets at depths {need} collected")

    # the span runs: stage 2's run_span, Qwen3 only (one scene per batch: all its eval captions in stage 0's order)
    scene_of = [e["scene"] for e in evals]
    by_scene = {sc: [i for i, s in enumerate(scene_of) if s == sc] for sc in sorted(set(scene_of))}
    tok_of: dict = {}
    for r, (ei, k, *_rest) in enumerate(eval_tok.tolist()):
        tok_of.setdefault(ei, []).append((r, k))
    assert all([k for _, k in tok_of[i]] == evals[i]["phrase_toks"] for i in range(E)), "the span rows do not match stage 0"
    keep = set(sel)

    def span_run(vecs_at=None, q_ids="qwen_ids"):
        """Per eval caption the final states at its phrase tokens (the reads' input), and for e029's captions the whole caption's
        final states [L, D] (fp32, CPU). vecs_at = None (unpatched) or (depth, [NT, D] vectors, one per phrase-token row)."""
        final, whole = [None] * E, {}
        for idx in by_scene.values():
            q_list = [evals[i][q_ids] for i in idx]
            prow = torch.tensor([j for j, i in enumerate(idx) for _ in tok_of[i]])
            ppos = torch.tensor([k for i in idx for _, k in tok_of[i]])
            prr = torch.tensor([r for i in idx for r, _ in tok_of[i]])
            patch = None if vecs_at is None else (vecs_at[0], prow, ppos, vecs_at[1][prr])
            out, _ = s2.qwen_run(qwen, q_list, device, patch=patch)
            for j, i in enumerate(idx):
                final[i] = out[j, evals[i]["phrase_toks"]].cpu()
                if i in keep:
                    whole[i] = out[j, :len(q_list[j])].cpu()
            del out
        return final, whole

    def final_reads(fin, ceil):
        """reads_span's final_cos and final_rel on the read of record (its arithmetic: per caption, then the fp32 mean)."""
        fc = torch.tensor([float(s2.cosine(a, b).mean()) for a, b in zip(fin, ceil)])
        fr = torch.tensor([float(((a - b).norm(dim=-1) / b.norm(dim=-1)).mean()) for a, b in zip(fin, ceil)])
        return float(fc[rec].mean()), float(fr[rec].mean())

    ceil_fin, whole = span_run()
    states = {"ceiling": whole}
    _, states["filler"] = span_run(q_ids="filler_qwen_ids")
    say("the ceiling and the filler runs done (unpatched)")

    arms, failed, s1_files, maps = {}, [], {}, {}
    outside_path = os.path.join(s2.OUT_DIR, f"stitch_{name}_control_outside" + (".json" if smoke else "_full.json"))
    for arm, key in relay_arms.items():
        fit_n, tr, b, cv, dl = parse(key)
        p1 = os.path.join(s2.OUT_DIR, f"s1_{tr}.pt")
        s1 = torch.load(p1, weights_only=False)
        s1_files[tr] = p1
        assert s1["fit_close"].shape[0] == len(rows) and s1["evaltok_close"].shape[0] == len(eval_tok), (tr, "stage 1 mismatch")
        bi = s1["blocks"].index(b)
        Xf, Xt = s1[f"fit_{cv}"][:, bi].clone(), s1[f"evaltok_{cv}"][:, bi].clone()
        del s1
        mask = torch.ones(len(rows), dtype=torch.bool) if fit_n == "record" else coco_rows
        W, mx, my, alpha, r2 = s2.ridge_cv(Xf[mask].float(), [Y[dl][mask]], folds[mask], device)[0]
        pt = ((Xt.float().to(device, torch.float64) - mx) @ W + my).float().cpu()
        maps[arm] = {"W": W, "mx": mx, "my": my, "depth": dl, "trunk": tr, "block": b, "convention": cv, "Xt_bare": Xt}
        del Xf
        fin, states[arm] = span_run((dmap[dl], pt))
        fc, fr = final_reads(fin, ceil_fin)
        a = {"key": key, "trunk": tr, "block": b, "convention": cv, "depth": dl, "fit": fit_n, "alpha": alpha, "cv_r2": r2,
             "final_cos_record": fc, "final_rel_record": fr}
        c, src = cells.get(key), "the grid"
        if c is None and os.path.exists(outside_path):         # the control's own selection run outside the registered grid
            oc = json.load(open(outside_path, encoding="utf-8"))
            if key in oc.get("span", {}).get("cells", {}) and "|".join(key.split("|")[1:4]) in oc.get("outside_grid", []):
                c, src = oc["span"]["cells"][key], f"its own run OUTSIDE THE REGISTERED GRID ({os.path.basename(outside_path)})"
                a["outside_registered_grid"] = True
                a["adapter_reads"] = {k: c["record"].get(k) for k in (
                    "primary_centred", "primary_centred_ci", "n_phrases", "share", "gap", "gap_ci", "word_signal", "word_signal_ci",
                    "mood_signal", "mood_signal_ci", "mood_cos", "mood_cos_ci", "mood_cos_ceiling", "mood_size", "mood_size_ci",
                    "transfer_axis_cos", "transfer_axis_size", "hit", "leak", "final_cos", "final_rel")}
        if c is None:
            a["grid"] = ("not in the grid (the grid ran the controls in the closing-byte convention only) and no run outside it: "
                         "the same arithmetic, unverified")
        else:
            g = c["record"]
            dif = {"alpha_equal": alpha == c["alpha"], "cv_r2": abs(r2 - c["cv_r2"]), "final_cos": abs(fc - g["final_cos"]),
                   "final_rel": abs(fr - g["final_rel"])}
            ok = dif["alpha_equal"] and dif["cv_r2"] <= BAR_R2 and dif["final_cos"] <= BAR and dif["final_rel"] <= BAR
            a.update({"grid": ("verified" if ok else "FAILED") + ("" if src == "the grid" else f" against {src}"),
                      "grid_alpha": c["alpha"], "grid_cv_r2": c["cv_r2"], "grid_final_cos_record": g["final_cos"],
                      "grid_final_rel_record": g["final_rel"], "differences": dif})
            if not ok:
                failed.append(arm)
        arms[arm] = a
        say(f"{arm} {key}: penalty {alpha:g}, cv R2 {r2:.6f}; the record's final states: cos {fc:.7f}, rel {fr:.7f} -> "
            + (a["grid"] if c is None else f"{a['grid']} (penalty {c['alpha']:g}, cv R2 {c['cv_r2']:.6f}; cos "
               f"{g['final_cos']:.7f}, rel {g['final_rel']:.7f}; differences R2 {dif['cv_r2']:.1e}, cos {dif['final_cos']:.1e}, "
               f"rel {dif['final_rel']:.1e})"))
        if a.get("outside_registered_grid"):
            r_ = a["adapter_reads"]

            def ci(x):
                return "[n/a]" if x is None else f"[{x[0]:+.3f}, {x[1]:+.3f}]"

            def v(k, f="+.3f"):
                return "n/a" if r_.get(k) is None else format(r_[k], f)
            say(f"THE CONTROL CELL OUTSIDE THE REGISTERED GRID ({key}), its adapter-level reads on the read of record (stage 2's own "
                f"functions, its own run): centred primary {v('primary_centred')} {ci(r_['primary_centred_ci'])} over "
                f"{r_['n_phrases']} phrases, gap {v('gap')} {ci(r_['gap_ci'])}, this word beyond its mood {v('word_signal')} "
                f"{ci(r_['word_signal_ci'])}, mood cosine {v('mood_cos')} {ci(r_['mood_cos_ci'])} (ceiling "
                f"{v('mood_cos_ceiling')}), mood size {v('mood_size', '.2f')} {ci(r_['mood_size_ci'])}")
    if failed:
        print(f"[export] THE CHECK FAILED for {failed}: the recomputed relay is not the grid's; nothing written (exit 1)", flush=True)
        sys.exit(1)

    # the file: every arm on e029's rows, zeros past each caption (the runner's own convention for its padding)
    lens = torch.tensor([len(evals[i]["qwen_ids"]) for i in sel])
    assert all(len(evals[i]["filler_qwen_ids"]) == len(evals[i]["qwen_ids"]) for i in sel)
    N, Lmax = len(sel), int(lens.max())
    ids = torch.zeros(N, Lmax, dtype=torch.long)
    fids = torch.zeros(N, Lmax, dtype=torch.long)
    tensors = {"lengths": lens, "qwen_ids": ids, "filler_qwen_ids": fids}
    for n, i in enumerate(sel):
        ids[n, :lens[n]] = torch.tensor(evals[i]["qwen_ids"])
        fids[n, :lens[n]] = torch.tensor(evals[i]["filler_qwen_ids"])
    for arm, wh in states.items():
        x = torch.zeros(N, Lmax, 1024)
        for n, i in enumerate(sel):
            assert wh[i].shape == (int(lens[n]), 1024) and bool(torch.isfinite(wh[i]).all()), (arm, i)
            x[n, :lens[n]] = wh[i]
        tensors[f"states.{arm}"] = x
    prefix = s0["prefix"]
    caps_h = []
    for n, i in enumerate(sel):
        e = evals[i]
        caps_h.append({"row": n, "text": e["text"], "scene_index": e["scene"], "scene": subjects[e["scene"]],
                       "phrase_index": e["phrase"], "phrase": phrases[e["phrase"]]["text"], "mood": phrases[e["phrase"]]["class"],
                       "filler": e["filler"], "filler_text": prefix + f"an illustration of {subjects[e['scene']]}, {e['filler']}.",
                       "length": int(lens[n]), "phrase_toks": e["phrase_toks"], "t5_ids": e["t5_ids"]})
    reg = s2.registration()
    s0_sha = sha256(os.path.join(s2.OUT_DIR, "s0.pt"))
    import transformers
    header = {
        "what": __doc__.split("\nEach relay")[0],
        "name": name, "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "grid": os.path.basename(full_path),
        "grid_partial": grid.get("partial"), "pick": pk["pick"]["key"], "pick_label": pk.get("label"),
        "companion": comp["key"] if comp is not None else None,
        "companion_label": (pk.get("companion") or {}).get("label"), "precision": PRECISION,
        "form": "bare: PREFIX + 'an illustration of {scene}, {phrase}.' (the grid's own captions and ids)", "mood_form": MOOD_FORM,
        "arms": {"ceiling": {"unpatched": "the caption"}, "filler": {"unpatched": "the filler caption (same Qwen3 token count)"},
                 **arms},
        "mounts": mounts_info or None,
        "bars": {"final_reads": BAR, "cv_r2": BAR_R2},
        "scenes": [{"index": sc, "text": subjects[sc]} for sc in scenes_e029],
        "phrases": [{"index": j, "text": phrases[j]["text"], "mood": phrases[j]["class"]} for j in rec_ph],
        "captions": caps_h,
        "provenance": {"s0_sha256": s0_sha,
                       "s1_sha256": {t: sha256(p) for t, p in s1_files.items()}, "grid_sha256": sha256(full_path),
                       "pick_sha256": sha256(pick_path), "qwen3_file": os.path.basename(s2.LLM_PATH),
                       "qwen3_sha256": sha256(s2.LLM_PATH), "export_tool_sha256": sha256(os.path.abspath(__file__)),
                       "registration_frozen": reg.get("frozen"), "torch": torch.__version__,
                       "transformers": transformers.__version__, "python": platform.python_version(),
                       "device": torch.cuda.get_device_name(0) if device == "cuda" else f"cpu ({platform.processor()})"}}
    dest = s2.OUT_DIR if smoke else s2.REPORT_DIR
    os.makedirs(dest, exist_ok=True)
    st_path = os.path.join(s2.OUT_DIR, f"e029_export_{name}{s2.RUN_TAG}.safetensors")
    js_path = os.path.join(dest, f"e029_export_{name}{s2.RUN_TAG}.json")
    from safetensors.torch import load_file, save_file
    save_file({k: v.contiguous() for k, v in tensors.items()}, st_path, metadata={"header": json.dumps(header)})
    back = load_file(st_path)                                  # read back: every tensor equal to what was written
    assert set(back) == set(tensors) and all(torch.equal(back[k], tensors[k]) for k in tensors)
    with open(js_path, "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)
    say(f"WROTE {st_path} ({os.path.getsize(st_path) / 1e6:.1f} MB: {N} captions x {Lmax} positions, arms {list(states)}; read back "
        f"equal) and {js_path}; the checks: " + "; ".join(f"{a}: {v['grid']}" for a, v in arms.items()) + peak_note())

    def mood_pass():
        """THE MOOD FORM: the same arms on e029's captions rewritten '..., {phrase} mood.'; no grid."""
        from transformers import AutoTokenizer, T5TokenizerFast
        cwd = os.getcwd()
        os.chdir(s2.DP)                                        # the bundled tokenizers, stage 0's own construction
        try:
            qtok = AutoTokenizer.from_pretrained("configs/qwen3_06b", local_files_only=True)
            ttok = T5TokenizerFast(vocab_file="configs/t5_old/spiece.model", tokenizer_file="configs/t5_old/tokenizer.json")
        finally:
            os.chdir(cwd)

        def q(text):
            return qtok(text, add_special_tokens=False)["input_ids"]

        def t5(text):
            x = ttok(text)["input_ids"]
            assert x[-1] == ttok.eos_token_id, text
            return x

        # the captions: the tokenizers reproduce stage 0 on bare captions; the mood form keeps every token up to the phrase's last
        texts_m, ftexts_m, ids_m, fids_m, t5_m = [], [], [], [], []
        for i in sel:
            e = evals[i]
            sc, ph = subjects[e["scene"]], phrases[e["phrase"]]["text"]
            ftext = prefix + f"an illustration of {sc}, {e['filler']}."
            assert q(e["text"]) == e["qwen_ids"] and q(ftext) == e["filler_qwen_ids"], ("Qwen3's ids differ from stage 0's", e["text"])
            assert t5(e["text"]) == e["t5_ids"] and t5(ftext) == e["filler_t5_ids"], ("T5's ids differ from stage 0's", e["text"])
            mt, fmt = prefix + f"an illustration of {sc}, {ph} mood.", prefix + f"an illustration of {sc}, {e['filler']} mood."
            assert mt == e["text"][:-1] + " mood." and fmt == ftext[:-1] + " mood.", mt
            mi, fmi, tl = q(mt), q(fmt), e["phrase_toks"][-1]
            assert mi[:tl + 1] == e["qwen_ids"][:tl + 1] and fmi[:tl + 1] == e["filler_qwen_ids"][:tl + 1], ("the phrase moved", mt)
            assert len(fmi) == len(mi) and fmi[tl + 1:] == mi[tl + 1:], ("the filler's mood form differs after the phrase", fmt)
            texts_m.append(mt)
            ftexts_m.append(fmt)
            ids_m.append(mi)
            fids_m.append(fmi)
            t5_m.append(t5(mt))
        say(f"THE MOOD FORM: the bundled tokenizers reproduce stage 0's ids on all {N} bare captions and their fillers (Qwen3 and "
            f"T5); every mood caption keeps its tokens up to the phrase's last, its filler the same count; e.g. {texts_m[0]!r}")

        # her states: stage 1's mood files, one per trunk, on exactly these captions and phrase tokens
        m1s, mrows = {}, None
        for mp in maps.values():
            tr = mp["trunk"]
            if tr in m1s:
                continue
            p = os.path.join(s2.OUT_DIR, f"s1_mood_{tr}.pt")
            m1 = torch.load(p, weights_only=False)
            assert list(m1["eval_index"]) == sel and list(m1["texts"]) == texts_m, (p, "not e029's mood captions")
            assert m1["s0_sha256"] == s0_sha, (p, "made from another stage 0")
            mrows = m1["rows"] if mrows is None else mrows
            assert torch.equal(m1["rows"], mrows), (p, "the token rows differ between the trunks' files")
            m1s[tr] = (p, m1)
        rows_of: dict = {}
        for r, n in enumerate(mrows[:, 0].tolist()):
            rows_of.setdefault(n, []).append(r)
        for n, i in enumerate(sel):
            rs, vrs = rows_of[n], [r for r, _ in tok_of[i]]
            assert mrows[rs, 1].tolist() == evals[i]["phrase_toks"] and mrows[rs, 4].tolist() == vrs, (n, "not the phrase tokens")
            assert mrows[rs, 2].tolist() == eval_tok[vrs, 2].tolist() and mrows[rs, 3].tolist() == eval_tok[vrs, 3].tolist(), n
            assert texts_m[n][int(mrows[rs[-1], 2])] == " ", (n, "the last token's closing byte is not the space")
        last_tok = torch.zeros(len(mrows), dtype=torch.bool)
        last_tok[[rs[-1] for rs in rows_of.values()]] = True

        # (b) her features at the same bytes, then each arm's patch vectors through its own refitted map: the bare file's features
        # wherever the two forms share the byte (her model is causal, the same bytes give the same state, which (b) checks on the
        # mood files), the mood file's at the last phrase token's closing byte (the space) in the closing-byte convention; so a
        # mood relay differs from the bare one only where the form does
        checks, failed_m, pts = {}, [], {}
        for arm, mp in maps.items():
            p, m1 = m1s[mp["trunk"]]
            bi = m1["blocks"].index(mp["block"])
            xm = m1[f"evaltok_{mp['convention']}"][:, bi].float()
            xb = mp["Xt_bare"][mrows[:, 4]].float()
            cs = torch.nn.functional.cosine_similarity(xm, xb, dim=-1)
            same = ~last_tok if mp["convention"] == "close" else torch.ones_like(last_tok)
            ck = {"features_min_cos_same_bytes": float(cs[same].min()), "features_n_same_bytes": int(same.sum())}
            if mp["convention"] == "close":
                lc = cs[last_tok]
                ck["features_last_token_closing_byte_cos"] = {"min": float(lc.min()), "median": float(lc.median()),
                                                              "max": float(lc.max()), "n": int(last_tok.sum())}
            if ck["features_min_cos_same_bytes"] < MOOD_FEAT:
                failed_m.append(f"{arm}: her features (b)")
            checks[arm] = ck
            x_use = torch.where(same[:, None], xb, xm)
            pts[arm] = ((x_use.to(device, torch.float64) - mp["mx"]) @ mp["W"] + mp["my"]).float().cpu()

        # the forwards: one scene per batch (e029's captions of that scene, in the file's order)
        by_scene_m: dict = {}
        for n, i in enumerate(sel):
            by_scene_m.setdefault(evals[i]["scene"], []).append(n)

        def mood_run(id_lists, vec=None):
            whole_m = [None] * N
            for ns in by_scene_m.values():
                q_list = [id_lists[n] for n in ns]
                patch = None
                if vec is not None:
                    rr = torch.tensor([r for n in ns for r in rows_of[n]])
                    prow = torch.tensor([j for j, n in enumerate(ns) for _ in rows_of[n]])
                    patch = (vec[0], prow, mrows[rr, 1], vec[1][rr])
                out, _ = s2.qwen_run(qwen, q_list, device, patch=patch)
                for j, n in enumerate(ns):
                    whole_m[n] = out[j, :len(q_list[j])].cpu()
                del out
            return whole_m

        sm = {"ceiling": mood_run(ids_m), "filler": mood_run(fids_m)}
        for arm, mp in maps.items():
            sm[arm] = mood_run(ids_m, (dmap[mp["depth"]], pts[arm]))
        say(f"THE MOOD FORM: the forwards done (arms {list(sm)})")

        # (a) the mood states against the bare file wherever the two forms share every input; (c) the patch leaves the positions
        # before the phrase alone; and the final-state reads at the phrase tokens (no grid: reads, not checks)
        def rel(a, b):
            return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp(min=1e-12)

        for arm, wh in sm.items():
            conv = maps[arm]["convention"] if arm in maps else None
            ck = checks.setdefault(arm, {})
            worst, pre, pre_bit, fc_m, fr_m, fc_b, fr_b = 0.0, 0.0, True, [], [], [], []
            for n, i in enumerate(sel):
                tk = evals[i]["phrase_toks"]
                upto = tk[-1] if conv == "close" else tk[-1] + 1
                worst = max(worst, float(rel(wh[n][:upto], states[arm][i][:upto]).max()))
                if arm in maps:
                    a_, b_ = wh[n][:tk[0]], sm["ceiling"][n][:tk[0]]
                    pre = max(pre, float(rel(a_, b_).max()) if tk[0] else 0.0)
                    pre_bit &= bool(torch.equal(a_, b_))
                    fc_m.append(float(s2.cosine(wh[n][tk], sm["ceiling"][n][tk]).mean()))
                    fr_m.append(float(rel(wh[n][tk], sm["ceiling"][n][tk]).mean()))
                    fc_b.append(float(s2.cosine(states[arm][i][tk], states["ceiling"][i][tk]).mean()))
                    fr_b.append(float(rel(states[arm][i][tk], states["ceiling"][i][tk]).mean()))
            ck["shared_inputs_max_rel_vs_bare"] = worst
            if worst > MOOD_TOL:
                failed_m.append(f"{arm}: the shared positions (a) {worst:.1e}")
            line = f"{arm}: (a) the shared positions against the bare file, largest {worst:.1e}"
            if arm in maps:
                ck.update({"before_phrase_max_rel_vs_ceiling": pre, "before_phrase_bit_equal": pre_bit,
                           "final_cos_e029_mood": sum(fc_m) / N, "final_rel_e029_mood": sum(fr_m) / N,
                           "final_cos_e029_bare": sum(fc_b) / N, "final_rel_e029_bare": sum(fr_b) / N})
                if pre > MOOD_TOL:
                    failed_m.append(f"{arm}: before the phrase (c) {pre:.1e}")
                line += (f"; (b) her features at the same bytes, smallest cosine {ck['features_min_cos_same_bytes']:.6f} over "
                         f"{ck['features_n_same_bytes']} tokens" + (
                             f", the last token's closing byte (the space, by design) cosine {ck['features_last_token_closing_byte_cos']['min']:.3f}"
                             f"..{ck['features_last_token_closing_byte_cos']['max']:.3f}" if conv == "close" else "")
                         + f"; (c) before the phrase against the ceiling {'bit-equal' if pre_bit else f'largest {pre:.1e}'}"
                         f"; the final states at the phrase tokens against the ceiling over e029's {N} captions: cos "
                         f"{ck['final_cos_e029_mood']:.4f} / rel {ck['final_rel_e029_mood']:.4f} (bare form {ck['final_cos_e029_bare']:.4f}"
                         f" / {ck['final_rel_e029_bare']:.4f})")
            say("THE MOOD FORM " + line)
        if failed_m:
            raise RuntimeError(f"THE MOOD FORM'S CHECKS FAILED: {failed_m}")

        # the file: as the bare one, with the mood captions' ids and states (written under temporary names, renamed at the end)
        lens_m = torch.tensor([len(x) for x in ids_m])
        Lm = int(lens_m.max())
        tm = {"lengths": lens_m, "qwen_ids": torch.zeros(N, Lm, dtype=torch.long), "filler_qwen_ids": torch.zeros(N, Lm, dtype=torch.long)}
        for n in range(N):
            tm["qwen_ids"][n, :lens_m[n]] = torch.tensor(ids_m[n])
            tm["filler_qwen_ids"][n, :lens_m[n]] = torch.tensor(fids_m[n])
        for arm, wh in sm.items():
            x = torch.zeros(N, Lm, 1024)
            for n in range(N):
                assert wh[n].shape == (int(lens_m[n]), 1024) and bool(torch.isfinite(wh[n]).all()), (arm, n)
                x[n, :lens_m[n]] = wh[n]
            tm[f"states.{arm}"] = x
        arms_m = {"ceiling": {"unpatched": "the mood caption"},
                  "filler": {"unpatched": "the filler's mood caption (the same Qwen3 token count)"}}
        for arm, a in arms.items():                            # what differs from the bare relay, per arm
            inputs = ("the same as the bare relay's: a last-byte cell, so every relayed state sits at a byte the two forms share; "
                      "only the caption around the phrase differs" if a["convention"] == "last" else
                      "one state per phrase differs from the bare relay's, the last phrase token's (its closing byte is the space "
                      "here, the full stop in the bare form); every other relayed state is the bare relay's")
            arms_m[arm] = {**{k: a[k] for k in ("key", "trunk", "block", "convention", "depth", "fit", "alpha", "cv_r2")},
                           "grid": f"{a['grid']} (the map, in the bare form); the mood form has no grid to check against",
                           "mood_relay_inputs": inputs,
                           **({"outside_registered_grid": True} if a.get("outside_registered_grid") else {})}
        for arm in sm:
            arms_m[arm]["mood_checks"] = checks[arm]
        caps_m = [{**caps_h[n], "text": texts_m[n], "filler_text": ftexts_m[n], "length": int(lens_m[n]), "t5_ids": t5_m[n],
                   "bare_text": caps_h[n]["text"]} for n in range(N)]
        hm = {**header, "what": "THE MOOD FORM (see 'form'; no grid exists for it) of the export: " + header["what"],
              "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
              "form": "mood: PREFIX + 'an illustration of {scene}, {phrase} mood.' (e029's registered fallback form; the bare "
                      "caption's text with '.' replaced by ' mood.'; the Qwen3 and T5 ids from the bundled tokenizers, checked "
                      "against stage 0's on the bare captions)",
              "mood_form": None, "arms": arms_m, "captions": caps_m,
              "bars": {**header["bars"], "mood_shared_positions": MOOD_TOL, "mood_before_phrase": MOOD_TOL,
                       "mood_features_cos": MOOD_FEAT},
              "provenance": {**header["provenance"], "bare_export_sha256": sha256(st_path),
                             "s1_mood_sha256": {t: sha256(p) for t, (p, _) in m1s.items()},
                             "tokenizers_sha256": {f: sha256(os.path.join(s2.DP, f)) for f in TOKENIZER_FILES}}}
        st_m = os.path.join(s2.OUT_DIR, f"e029_export_{name}{s2.RUN_TAG}_mood.safetensors")
        js_m = os.path.join(dest, f"e029_export_{name}{s2.RUN_TAG}_mood.json")
        save_file({k: v.contiguous() for k, v in tm.items()}, st_m + ".tmp", metadata={"header": json.dumps(hm)})
        back_m = load_file(st_m + ".tmp")
        assert set(back_m) == set(tm) and all(torch.equal(back_m[k], tm[k]) for k in tm)
        with open(js_m + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(hm, fh, indent=1)
        os.replace(st_m + ".tmp", st_m)
        os.replace(js_m + ".tmp", js_m)
        say(f"THE MOOD FORM: WROTE {st_m} ({os.path.getsize(st_m) / 1e6:.1f} MB: {N} captions x {Lm} positions, arms {list(sm)}; "
            f"read back equal) and {js_m}" + peak_note())

    if os.environ.get("STITCH_EXPORT_MOOD", "1") != "1":
        say("THE MOOD FORM: not exported (STITCH_EXPORT_MOOD=0)")
        return
    try:
        mood_pass()
    except Exception:  # noqa: BLE001 - the bare file stands whatever the mood pass does
        import traceback
        traceback.print_exc()
        print("[export] THE MOOD FORM FAILED (above): the bare export stands; nothing written for the mood form (exit 8)", flush=True)
        sys.exit(8)


if __name__ == "__main__":
    if "--control-cell" in sys.argv:
        control_cell(sys.argv[1])
    else:
        main(sys.argv[1], device="cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda")
