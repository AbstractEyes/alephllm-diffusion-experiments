"""alephllm_diffusion.stitch.batch_gate - THE BATCHING GATE: the batched forms against the earlier ones, on the card, with times.

  s1    stage 1's padded batches against the equal-length form on the same captions (the first N stage-1 captions of stage 0,
        the trunk at a stored step, optionally an arm group mounted): per block, in fp32 the largest relative difference
        (max |padded - equal| / max |equal|; PASS under 1e-5: kernel rounding, where a padding leak would show at the size of
        the states), in bf16 the mean |difference| / mean |value| (reported beside the cross-card bf16 difference, 1.0-1.3%);
        the seconds and passes of each form.
  grid  the grid's nine port-check cells (the rehearsal configuration on its stage-1 files) run twice, one scene a pass and
        K scenes a pass (STITCH_SCENES_PER_PASS): PASS = every cell line equal with the timing words stripped and the run's own
        scene-batch check passed; the seconds per cell of each form.
Writes a JSON report to the report folder; one job on the card, the forms in turn. Exit 0 = PASS, 1 = FAIL.
Usage: python -m alephllm_diffusion.stitch.batch_gate s1 [n=512] [step=245674] [--mount=gCA]
       python -m alephllm_diffusion.stitch.batch_gate grid [k=8]"""
import json
import os
import re
import subprocess
import sys
import time

import torch

from .. import settings
from ..beatrix import extract as bx
from . import s1 as S1

TIMING = re.compile(r"; \d+ s spent, about \d+ s left$")
CELL = re.compile(r"^\[stitch s2\] cell \d+/\d+ ")


def _run_form(model, texts, amp, pad, device):
    """Stage 1's extraction as stage 1 runs it: the equal form in chunks of 128 captions, the padded form in chunks of 1,024."""
    chunk = 1024 if pad else 128
    got = {b: [] for b in S1.BLOCKS}
    for c0 in range(0, len(texts), chunk):
        ex = bx.extract(model, texts[c0:c0 + chunk], S1.BLOCKS, taps=("byte",), device=device, amp=amp, pad=pad)
        for b in S1.BLOCKS:
            got[b] += ex["byte"][b]
    return got


def gate_s1(n=512, step=245674, mount=None, device="cuda"):
    s0 = torch.load(os.path.join(settings.OUT_DIR, "s0.pt"), weights_only=False)
    evals, _ = S1.eval_view(s0, 0)
    texts = ([f["text"] for f in s0["fit"]] + [e["text"] for e in evals])[:n]
    lens = [bx.encode_bytes(t).numel() for t in texts]
    _, model, _, _ = S1.trunk_for(str(step), device, mount=mount)
    rep = {"n": len(texts), "step": step, "mount": mount, "blocks": S1.BLOCKS, "forms": {}}
    ok = True
    for amp in (False, True):                          # one precision at a time: the two forms, compared, then released
        prec, out = ("bf16" if amp else "fp32"), {}
        for pad in (False, True):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out[pad] = _run_form(model, texts, amp, pad, device)
            if device == "cuda":
                torch.cuda.synchronize()
            secs = time.time() - t0
            passes = (sum(len(bx.padded_groups(lens[c0:c0 + 1024])) for c0 in range(0, len(lens), 1024)) if pad else
                      _equal_passes(lens))
            key = f"{prec}_{'padded' if pad else 'equal'}"
            rep["forms"][key] = {"seconds": round(secs, 2), "passes": passes, "captions_per_pass": round(len(texts) / passes, 1)}
            print(f"[batch gate] stage 1, {key}: {secs:.1f} s, {passes} passes ({len(texts) / passes:.1f} captions a pass)",
                  flush=True)
        rows = {}
        for blk in S1.BLOCKS:                          # caption by caption (no concatenated copies)
            dmax = rmax = dsum = rsum = 0.0
            for A, B in zip(out[True][blk], out[False][blk]):
                assert A.shape == B.shape, (blk, A.shape, B.shape)
                d = (A - B).abs()
                dmax, rmax = max(dmax, float(d.max())), max(rmax, float(B.abs().max()))
                dsum, rsum = dsum + float(d.sum()), rsum + float(B.abs().sum())
            rows[blk] = {"max_relative": dmax / rmax, "mean_relative": dsum / rsum}
        del out
        rep[prec] = rows
        worst = max(r["max_relative"] for r in rows.values())
        mean = max(r["mean_relative"] for r in rows.values())
        if prec == "fp32":
            ok = worst < 1e-5
        print(f"[batch gate] stage 1, {prec}: padded vs equal, largest relative difference {worst:.2e}, mean |difference| / "
              f"mean |value| up to {mean:.2e}" + (f" -> {'PASS' if ok else 'FAIL'} (bar 1e-5)" if prec == "fp32" else
                                                  " (reported; the cross-card bf16 difference was 1.0-1.3e-2)"), flush=True)
    sp = rep["forms"]["bf16_equal"]["seconds"] / max(rep["forms"]["bf16_padded"]["seconds"], 1e-9)
    rep["speedup_bf16"] = round(sp, 2)
    print(f"[batch gate] stage 1: the padded form {sp:.1f}x the equal form's speed (bf16, as stage 1 runs)", flush=True)
    rep["pass"] = ok
    return rep


def _equal_passes(lens, chunk=128, max_batch=64):
    n = 0
    for c0 in range(0, len(lens), chunk):
        part = sorted(lens[c0:c0 + chunk])
        i = 0
        while i < len(part):
            j = i
            while j < len(part) and j - i < max_batch and part[j] == part[i]:
                j += 1
            n += 1
            i = j
    return n


def gate_grid(k=8):
    from .port_gate import GATE_KEYS
    stamp = time.strftime("%Y%m%d_%H%M%S")
    lines, secs = {}, {}
    for per in (1, k):
        folder = os.path.join(settings.OUT_DIR, "batch_gate", stamp, f"cells_{per}")
        log = os.path.join(settings.OUT_DIR, "batch_gate", stamp, f"grid_{per}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        env = dict(os.environ, HF_TOKEN="", HF_HUB_OFFLINE="1", STITCH_ONLY_CELLS=",".join(GATE_KEYS), STITCH_CELLS_DIR=folder,
                   STITCH_RUN_TAG=f"_batchgate{per}", STITCH_SCENES_PER_PASS=str(per))
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run([sys.executable, "-u", "-m", "alephllm_diffusion.stitch.s2", "rehearsal"], env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        text = open(log, encoding="utf-8", errors="replace").read().splitlines()
        assert rc == 0, f"the grid with {per} scene(s) a pass failed: see {log}"
        lines[per] = [TIMING.sub("", x) for x in text if CELL.match(x)]
        spent = [int(m[1]) for x in text if CELL.match(x) and (m := re.search(r"; (\d+) s spent", x))]
        secs[per] = (spent[-1] - spent[0]) / max(1, len(spent) - 1) if len(spent) > 1 else float("nan")
        check = [x for x in text if "THE SCENE-BATCH CHECK" in x]
        print(f"[batch gate] the grid, {per} scene(s) a pass: {len(lines[per])} cells, {time.time() - t0:.0f} s in all, "
              f"{secs[per]:.1f} s a cell after the first" + (f"; {check}" if check else ""), flush=True)
    same = lines[1] == lines[k] and len(lines[1]) == len(GATE_KEYS)
    print(f"[batch gate] the grid: the {len(GATE_KEYS)} cell lines {'EQUAL' if same else 'DIFFERENT'} with {k} scenes a pass "
          f"-> {'PASS' if same else 'FAIL'}; {secs[1] / secs[k]:.2f}x the speed per cell", flush=True)
    return {"k": k, "cells": len(GATE_KEYS), "seconds_per_cell": secs, "lines_equal": same, "pass": same,
            "differences": [(a, b) for a, b in zip(lines[1], lines[k]) if a != b][:5]}


if __name__ == "__main__":
    flags = {x.split("=", 1)[0][2:]: x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--") and "=" in x}
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert a and a[0] in ("s1", "grid"), __doc__
    rep = (gate_s1(int(a[1]) if len(a) > 1 else 512, int(a[2]) if len(a) > 2 else 245674, flags.get("mount"))
           if a[0] == "s1" else gate_grid(int(a[1]) if len(a) > 1 else 8))
    os.makedirs(settings.REPORT_DIR, exist_ok=True)
    path = os.path.join(settings.REPORT_DIR, f"batch_gate_{a[0]}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(f"[batch gate] wrote {path}", flush=True)
    sys.exit(0 if rep["pass"] else 1)
