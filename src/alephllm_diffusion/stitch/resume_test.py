"""alephllm_diffusion.stitch.resume_test - THE RESTART TEST and THE TWO-HALVES TEST of stage 2: the smoke grid (36 cells on
stage 1's limit files, made from the first 80 fit captions and the eval captions of two scenes; its numbers mean nothing), one
process after another on the card, never two at once:
  A   uninterrupted;
  B   stopped cleanly after half its cells (STITCH_MAX_CELLS);
  C   resumed from B's cell folder (STITCH_RESUME): B's cells read back, the rest computed;
  D1  the record fit only (STITCH_ONLY_FIT=record), D2 the coco fit only: the two halves of a two-card run, here in turn (the
      two ridge fits: COCO captions with the experiment's training captions, and COCO captions alone);
  D3  the merge: resumed from D1's and D2's folders, computing nothing.
PASS = C's and D3's JSON each equal A's with the run-specific keys left out (the resource log, the cell folder, the resume note),
their per-caption files equal A's tensor for tensor, and every cell line they print equals A's with the timing words stripped
(they print the first runs' lines for the cells they read back). Reads the smoke config's stage-0 and stage-1 files from the
output folder (settings.OUT_DIR). Writes nothing to the record: everything stays under the output folder's resume_test/<stamp>/.
Card time: six smoke runs, a few minutes. Exit 0 = PASS, 1 = FAIL.
Usage: python -m alephllm_diffusion.stitch.resume_test"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch  # noqa: E402

from . import s2  # noqa: E402

T = os.path.join(s2.OUT_DIR, "resume_test", time.strftime("%Y%m%d_%H%M%S"))
RUN_KEYS = ("resources", "cells_dir", "resumed_from")
TIMING = re.compile(r"; \d+ s spent, about \d+ s left$")


def run(tag, extra, run_tag=""):
    folder = os.path.join(T, f"cells_{tag}")
    env = dict(os.environ, HF_TOKEN="", HF_HUB_OFFLINE="1", STITCH_MEM_FRACTION=os.environ.get("STITCH_MEM_FRACTION", "0.4"),
               STITCH_CELLS_DIR=folder, STITCH_RUN_TAG=run_tag, **extra)
    t0 = time.time()
    log = os.path.join(T, f"{tag}.log")
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, "-u", "-m", "alephllm_diffusion.stitch.s2", "smoke"], env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
    cells = [TIMING.sub("", x) for x in lines if re.match(r"\[stitch s2\] cell \d+/\d+ ", x)]
    keep = {}
    for src, name in ((os.path.join(s2.OUT_DIR, f"stitch_smoke{run_tag}.json"), "json"),
                      (os.path.join(s2.OUT_DIR, f"stitch_smoke{run_tag}_percaption.pt"), "pc")):
        dst = os.path.join(T, f"{tag}_{os.path.basename(src)}")
        shutil.copyfile(src, dst)
        keep[name] = dst
    print(f"[resume test] {tag}: exit {rc}, {len(cells)} cell lines, {time.time() - t0:.0f} s", flush=True)
    assert rc == 0, f"{tag} failed: see {log}"
    return cells, folder, keep


def strip(obj):
    return {k: v for k, v in obj.items() if k not in RUN_KEYS}


def equal(u, v):
    """Bit-for-bit equality with NaN in the same places counted as equal: the per-caption mood rows hold NaN for captions
    without a mood, and torch.equal never calls a tensor holding NaN equal, not even to itself."""
    if u.shape != v.shape or u.dtype != v.dtype:
        return False
    if u.is_floating_point():
        return bool(((u == v) | (u.isnan() & v.isnan())).all())
    return torch.equal(u, v)


def same(a, x, x_cells, a_cells, label):
    ja, jx = (strip(json.load(open(p["json"], encoding="utf-8"))) for p in (a, x))
    same_json = json.dumps(ja, sort_keys=True) == json.dumps(jx, sort_keys=True)
    pa, px = (torch.load(p["pc"], weights_only=False) for p in (a, x))
    diff = []
    for k in ("percap", "percap_span", "mood_proj_span", "mood_cos_span"):
        da, dx = pa.get(k, {}), px.get(k, {})
        if set(da) != set(dx):
            diff.append(f"{k}: keys differ")
        diff += [f"{k}[{q}]" for q in da if q in dx and not equal(da[q], dx[q])]
    lines = a_cells == x_cells
    print(f"[resume test] {label}: JSON equals A's: {same_json}; per-caption file equals A's: {not diff}"
          f"{'' if not diff else ' ' + str(diff[:5])}; its {len(x_cells)} cell lines equal A's {len(a_cells)}: {lines}", flush=True)
    return same_json and not diff and lines


def main():
    os.makedirs(T)
    c = s2.CONFIGS["smoke"]
    ncell = len(c["fits"]) * len(c["depths"]) * (len(c["her"]) * len(c["blocks"]) * len(c["convs"])
                                                 + len(c["controls"]) * len(c["blocks"]) + 2)
    half = ncell // 2
    a_cells, _, a = run("A_uninterrupted", {})
    b_cells, b_dir, _ = run("B_stopped", {"STITCH_MAX_CELLS": str(half)}, "_b")
    c_cells, _, cc = run("C_resumed", {"STITCH_RESUME": b_dir})
    ok_c = same(a, cc, c_cells, a_cells, "C (resumed)") and len(b_cells) == half
    d1_cells, d1_dir, _ = run("D1_record_fit", {"STITCH_ONLY_FIT": "record"}, "_d1")
    d2_cells, d2_dir, _ = run("D2_coco_fit", {"STITCH_ONLY_FIT": "coco"}, "_d2")
    d3_cells, _, d3 = run("D3_merge", {"STITCH_RESUME": f"{d1_dir},{d2_dir}"})
    ok_d = same(a, d3, d3_cells, a_cells, "D3 (two halves merged)") and len(d1_cells) + len(d2_cells) == ncell
    print(f"[resume test] {ncell} cells; B stopped after {half} (it printed {len(b_cells)}); the halves printed {len(d1_cells)} + "
          f"{len(d2_cells)}", flush=True)
    print("[resume test] " + ("PASS" if ok_c and ok_d else f"FAIL (restart {ok_c}, two halves {ok_d})"), flush=True)
    sys.exit(0 if ok_c and ok_d else 1)


if __name__ == "__main__":
    main()
