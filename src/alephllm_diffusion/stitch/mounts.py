"""alephllm_diffusion.stitch.mounts - THE MOUNT CONTRAST (registered 2026-10-06, before any build or read): does mounting the
nine arms (the eight stage arms + the caption arm, stage 1 --mount) change how much of a caption's mood her (Beatrix's) relay
carries into Qwen3's space? CPU only; reads the per-caption files of stage-2 runs on the same captions.

PM3(a), the registered prediction read here: at the record run's pick of record (M0's chosen cell, M0 = her trunk with nothing
mounted; and, labelled, at its companion when that differs), the per-caption MOOD SIZE of each mount minus M0's, on the read of
record (the held-out fragment-split phrases), with the 95% phrase-cluster bootstrap interval (2,000 draws, seed 0: stage 2's
cluster_ci); PM3(a) holds for a mount when the interval excludes zero above it, and is read on BOTH arm seeds (gCA, gCB). The
per-caption mood size is the pick tool's own arithmetic: the relay's signed projection on the ceiling's leave-one-out mood axis
over the mean of the ceiling's own projection on the captions with a mood (the ceiling is Qwen3's own state: the same in every
run, asserted). Beside it, never deciding it: the centred primary's paired difference at the same cell; each mount's own pick of
record (pick_mounts_<trunk>.json) and M0's size there; the paired difference over every cell of the grid (how many cells move,
either way).
THE FRAME COLUMN (part of PM3(a)'s read, no direction predicted): the same contrast with every caption framed by the caption
arm's own label ('caption: ', stage 1 --frame=cap): each framed mount minus M0 framed, at the same cells, from the 'frames'
selection run; printed beside the bare framing, which stays the read of record (the stricter test: the arm saw mood words only
inside its photo rows' scene field).
Reads pick_<record name>.json (and pick_<mounts name>_<trunk>.json per mount, when present) from the report folder
(settings.REPORT_DIR; a smoke run's from the output folder) and the three runs' stitch_<name>_percaption.pt from the output
folder (settings.OUT_DIR).
Usage: python -m alephllm_diffusion.stitch.mounts [record name] [mounts name] [frames name]
         (default: record mounts frames; a missing frames run is reported, not failed)
       python -m alephllm_diffusion.stitch.mounts --frame-keys [record name] [frames name]
         (prints the frame column's cell keys for stage 2's STITCH_ONLY_CELLS, nothing else)
Writes stitch_<mounts name>_contrast.json to the report folder (a smoke run's to the output folder)."""
import json
import os
import sys

import torch

from . import s2  # noqa: E402

FRAME = "_fcap"


def framed(tag):
    """A trunk tag's framed stage-1 tag (the frame goes before a smoke file's _limit suffix, as stage 1 names it)."""
    return tag.replace("_limit", FRAME + "_limit", 1) if "_limit" in tag else tag + FRAME


def report_dir(name):
    """Where a run's picks and reads live (a smoke run's in the output folder, kept out of the record)."""
    return s2.OUT_DIR if name.startswith("smoke") else s2.REPORT_DIR


def load_pc(name):
    p = os.path.join(s2.OUT_DIR, f"stitch_{name}_percaption.pt")
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def key_parts(key):
    fit, trunk, blk, conv, dep = key.split("|")
    return fit, trunk, blk, conv, dep


def swap_trunk(key, trunk):
    fit, _t, blk, conv, dep = key_parts(key)
    return "|".join((fit, trunk, blk, conv, dep))


def mood_size(pc, key):
    """The pick tool's per-caption mood size (alephllm_diffusion.stitch.pick.companion's size()): projection over the ceiling's
    mean, on the record set's captions with a mood. Returns (size per caption, the mask of captions it is defined on)."""
    rec, ceil = pc["span_sets"]["record"], pc["ceiling_mood_proj_span"]
    pr = pc["mood_proj_span"][key]
    hx = rec & ~torch.isnan(pr)
    return pr / float(ceil[hx].mean()), hx


def paired(a, b, mk, phr):
    d = (a - b)[mk]
    lo, hi = s2.cluster_ci(d, phr[mk])
    return {"diff": float(d.mean()), "ci": [lo, hi], "excludes_zero": lo is not None and (lo > 0 or hi < 0),
            "above_zero": lo is not None and lo > 0, "n_captions": int(mk.sum()), "n_phrases": len(set(phr[mk].tolist()))}


def same_captions(a, b, what):
    for k in ("phrase", "scene"):
        assert torch.equal(a[k], b[k]), f"{what}: the runs read different captions ({k})"
    assert torch.equal(a["span_sets"]["record"], b["span_sets"]["record"]), f"{what}: the read of record differs"
    ca, cb = a["ceiling_mood_proj_span"], b["ceiling_mood_proj_span"]
    both = ~torch.isnan(ca) & ~torch.isnan(cb)
    assert torch.equal(torch.isnan(ca), torch.isnan(cb)) and float((ca[both] - cb[both]).abs().max()) <= 1e-6, \
        f"{what}: the ceiling's mood projections differ (Qwen3's own states must not depend on her mount)"


def contrast_at(pc_m, pc_0, trunk_m, trunk_0, key0, phr):
    """The mount's cell minus M0's at M0's cell key (same fit, block, convention, depth)."""
    km = swap_trunk(key0, trunk_m)
    if km not in pc_m["mood_proj_span"] or key0 not in pc_0["mood_proj_span"]:
        return {"cell": km, "missing": True}
    sm, hm = mood_size(pc_m, km)
    s0, h0 = mood_size(pc_0, key0)
    assert torch.equal(hm, h0), f"{km}: the captions with a mood differ from M0's"
    r = paired(sm, s0, hm, phr)
    r.update({"cell": km, "mount_size": float(sm[hm].mean()), "m0_size": float(s0[h0].mean())})
    rec = pc_0["span_sets"]["record"]
    r["primary_centred"] = paired(pc_m["percap_span"][km], pc_0["percap_span"][key0], rec, phr)
    return r


def main(record="record", mounts="mounts", frames="frames"):
    pick0 = json.load(open(os.path.join(report_dir(record), f"pick_{record}.json"), encoding="utf-8"))
    cells0 = {"pick of record": pick0["pick"]["key"]}
    comp = (pick0.get("companion") or {}).get("pick")
    if comp is not None and comp["key"] != pick0["pick"]["key"]:
        cells0["companion (labelled)"] = comp["key"]
    trunk0 = key_parts(cells0["pick of record"])[1]
    pc0, pcm, pcf = load_pc(record), load_pc(mounts), load_pc(frames)
    assert pc0 is not None and pcm is not None, "the record run's and the mounts run's per-caption files are both needed"
    same_captions(pcm, pc0, mounts)
    phr = pc0["phrase"]
    out = {"record": record, "mounts": mounts, "frames": frames if pcf is not None else None, "m0_trunk": trunk0,
           "m0_cells": cells0, "bare": {}, "framed": {}, "own_picks": {}, "every_cell": {}}
    mount_trunks = list(pcm["cfg"]["her"])
    print(f"[mounts] THE MOUNT CONTRAST: {', '.join(mount_trunks)} against {trunk0} (M0), on the read of record "
          f"({int(pc0['span_sets']['record'].sum())} captions, {len(set(phr[pc0['span_sets']['record']].tolist()))} phrases)",
          flush=True)
    for tm in mount_trunks:
        out["bare"][tm] = {lab: contrast_at(pcm, pc0, tm, trunk0, k0, phr) for lab, k0 in cells0.items()}
        for lab, r in out["bare"][tm].items():
            if r.get("missing"):
                print(f"[mounts]   {tm} at M0's {lab}: the cell {r['cell']} is not in the runs", flush=True)
                continue
            ci = r["ci"]
            print(f"[mounts]   PM3(a) {tm} at M0's {lab} {r['cell'].split('|', 2)[2]}: mood size {r['mount_size']:.3f} vs M0 "
                  f"{r['m0_size']:.3f}, paired {r['diff']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                  f"{' -> ABOVE ZERO' if r['above_zero'] else ' -> below zero' if r['excludes_zero'] else ' -> includes zero'}; "
                  f"the centred primary {r['primary_centred']['diff']:+.3f}", flush=True)
        pk = os.path.join(report_dir(mounts), f"pick_{mounts}_{tm}.json")
        if os.path.exists(pk):
            own = json.load(open(pk, encoding="utf-8"))["pick"]
            r = contrast_at(pcm, pc0, tm, trunk0, swap_trunk(own["key"], trunk0), phr)
            out["own_picks"][tm] = {"pick": own["key"], "at_own_pick": r}
            if not r.get("missing"):
                print(f"[mounts]   {tm}'s own pick {own['key'].split('|', 2)[2]}: mood size {r['mount_size']:.3f}, M0 there "
                      f"{r['m0_size']:.3f} (paired {r['diff']:+.3f})", flush=True)
        moved_up = moved_dn = n = 0
        for km in pcm["mood_proj_span"]:
            if key_parts(km)[1] != tm:
                continue
            k0 = swap_trunk(km, trunk0)
            if k0 not in pc0["mood_proj_span"]:
                continue
            r = contrast_at(pcm, pc0, tm, trunk0, k0, phr)
            out["every_cell"].setdefault(tm, {})[km.split("|", 2)[2]] = {k: r[k] for k in ("diff", "ci", "excludes_zero")}
            n += 1
            moved_up += r["above_zero"]
            moved_dn += r["excludes_zero"] and not r["above_zero"]
        print(f"[mounts]   {tm} over every cell both runs hold: {n} cells; the mood size moved up at {moved_up}, down at "
              f"{moved_dn} (the paired interval excludes zero)", flush=True)
    seeds = [tm for tm in mount_trunks if "gC" in tm]
    verdict = {lab: all(out["bare"][tm][lab].get("above_zero") for tm in seeds) if seeds else None for lab in cells0}
    out["pm3a_bare_holds_on_both_seeds"] = verdict
    print(f"[mounts] PM3(a), the bare framing (the read of record): holds on both arm seeds at the pick of record: "
          f"{verdict.get('pick of record')}", flush=True)
    if pcf is None:
        print(f"[mounts] THE FRAME COLUMN: no '{frames}' run on disk: not read", flush=True)
    else:
        same_captions(pcf, pc0, frames)
        f0 = framed(trunk0)
        for tm in mount_trunks:
            out["framed"][framed(tm)] = {lab: contrast_at(pcf, pcf, framed(tm), f0, swap_trunk(k0, f0), phr)
                                         for lab, k0 in cells0.items()}
            for lab, r in out["framed"][framed(tm)].items():
                if r.get("missing"):
                    print(f"[mounts]   framed {tm} at M0's {lab}: not in the frames run", flush=True)
                    continue
                ci = r["ci"]
                print(f"[mounts]   FRAME COLUMN {framed(tm)} minus {f0} at M0's {lab}: {r['mount_size']:.3f} vs "
                      f"{r['m0_size']:.3f}, paired {r['diff']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", flush=True)
    path = os.path.join(report_dir(record), f"stitch_{mounts}_contrast.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"[mounts] wrote {path}", flush=True)
    return out


def frame_keys(record="record", frames="frames"):
    """The frame column's selection for stage 2 (STITCH_ONLY_CELLS): the record's pick (and its companion when it differs) in each
    framed tag of the frames config, comma-joined."""
    pick0 = json.load(open(os.path.join(report_dir(record), f"pick_{record}.json"), encoding="utf-8"))
    keys = [pick0["pick"]["key"]]
    comp = (pick0.get("companion") or {}).get("pick")
    if comp is not None and comp["key"] not in keys:
        keys.append(comp["key"])
    return ",".join(swap_trunk(k, t) for t in s2.CONFIGS[frames]["her"] for k in keys)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--frame-keys"]:                     # for the run script: prints the selection, nothing else
        print(frame_keys(*sys.argv[2:4]))
    else:
        main(*sys.argv[1:4])
