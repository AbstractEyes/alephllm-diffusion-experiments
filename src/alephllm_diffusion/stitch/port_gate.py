"""alephllm_diffusion.stitch.port_gate - THE PORT GATE (its bar as registered): a new card and a fresh environment are a new
implementation of the instrument, compared with the old one at one reference point before use. Reference = the rehearsal grid on
step 230,415, run on an RTX 4090 (stopped at cell 288; its printed cell lines are the only record of it). The run on the new card
recomputes a few of those cells (stage 2 with STITCH_ONLY_CELLS on the rehearsal config, stage 1 on 230,415 from the training
run's own checkpoint file) and this tool compares them, every statistic in its OWN units (as registered: the fp32 repeat floor is
a same-machine number in 1 - cosine units and does not bound a cross-machine comparison of printed statistics):
PRINTED-ONLY REFERENCE (the default). Per number: EXACT; or CONSISTENT = one unit off in the last printed digit with the new run's
raw value within the tolerance of the rounding boundary between the two printed values (TOL 1e-4, absolute for magnitudes up to
1, relative above); anything else FAILS (more than one unit, a count or an [n/a] that differs, a missing cell, a raw value beyond
the tolerance from the boundary). And over all compared numbers THE IMPLIED MEAN SHIFT = (the sum over the flipped numbers of each
one's last-digit unit) / (the count of all compared numbers) must stay within SHIFT 5e-5 (one flip near a boundary says little; a
systematic shift shows as many flips). PASS = no number fails AND the shift is within the bar.
RAW REFERENCE (--raw-ref <the reference card's JSON of the same selected cells>): the largest raw difference over every number
within TOL in the same units; it is printed with its field and cell, and quoted beside the record as the measured agreement
between the two cards.
A FAIL stops the session run before stage 2's record and sends the grid nowhere by itself: the fields and cells that differ (this
tool's lines) are read and the next step is decided by hand. The tool first checks its own field list: the new run's raw values,
formatted as stage 2 prints them, must reproduce its printed line exactly, or that cell FAILS unexamined.
Usage: python -m alephllm_diffusion.stitch.port_gate reference <the reference run's log>  -> <REPORT_DIR>/port_gate_reference.json
       python -m alephllm_diffusion.stitch.port_gate check <the stage-2 log> <its full JSON> [options]
       options: --raw-ref <json> (the raw reference), --tol <x> (TOL), --shift <x> (SHIFT)
The check reads <REPORT_DIR>/port_gate_reference.json beside the new run's log and full JSON, and prints a verdict per cell and
the overall verdict. Exit 0 = PASS, 1 = FAIL."""
import json
import os
import re
import sys

from .. import settings

REPORT_DIR = settings.REPORT_DIR
REF = os.path.join(REPORT_DIR, "port_gate_reference.json")
CELL = re.compile(r"^\[stitch s2\] cell (\d+)/(\d+) (\S+): (.*)$")
TIMING = re.compile(r"; \d+ s spent, about \d+ s left$")
NUM = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])")
FLOOR = re.compile(r"PHRASE'S ANSWER WRITE.*THE FP32 REPEAT FLOOR: 1 - centred cos mean ([0-9.e+-]+)")
TOL, SHIFT = 1e-4, 5e-5                                        # the registered bar
# the 7 reference cells (the first 7 of the rehearsal) and two of her cells (the preview's top cell and its pick)
GATE_KEYS = [f"record|step230415|b8|close|k{d}" for d in (0, 4, 8, 12, 16, 20, 24)] + [
    "record|step230415|b18|close|k4", "record|step230415|b16|close|k16"]


def cells(lines):
    out = {}
    for x in lines:
        m = CELL.match(TIMING.sub("", x.rstrip("\n")))
        if m:
            out[m.group(3)] = m.group(4)
    return out


def raw_values(res, key):
    """The numbers of a cell's printed line (stage 2's span-form cell line), in print order, as (field, raw value, format)."""
    sc, slot = res["span"]["cells"][key], res["cells"][key]
    rr, ce, h = sc["record"], res["span"]["ceiling"]["record"], slot["held_all"]

    def ci(name, c):
        return [] if c[0] is None else [(f"{name}.lo", c[0], "+.3f"), (f"{name}.hi", c[1], "+.3f")]
    return [("primary_centred", rr["primary_centred"], "+.3f"), *ci("primary_centred_ci", rr["primary_centred_ci"]),
            ("n_phrases", rr["n_phrases"], "d"), ("gap", rr["gap"], "+.3f"), *ci("gap_ci", rr["gap_ci"]),
            ("offdiag", rr["offdiag"], "+.3f"), ("offdiag_same", rr["offdiag_same"], "+.3f"),
            ("offdiag_opposite", rr["offdiag_opposite"], "+.3f"), ("word_signal", rr["word_signal"], "+.3f"),
            *ci("word_signal_ci", rr["word_signal_ci"]), ("mood_signal", rr["mood_signal"], "+.3f"),
            *ci("mood_signal_ci", rr["mood_signal_ci"]), ("primary_pooled", rr["primary_pooled"], "+.3f"),
            ("primary_raw", rr["primary_raw"], "+.3f"), ("mood_cos", rr["mood_cos"], "+.3f"), *ci("mood_cos_ci", rr["mood_cos_ci"]),
            ("mood_cos_ceiling", rr["mood_cos_ceiling"], "+.3f"), ("mood_size", rr["mood_size"], ".2f"),
            *ci("mood_size_ci", rr["mood_size_ci"]), ("transfer_axis_cos", rr["transfer_axis_cos"], "+.2f"),
            ("transfer_axis_size", rr["transfer_axis_size"], ".2f"), ("leak", rr["leak"], ".4f"), ("ceiling.leak", ce["leak"], ".4f"),
            ("hit", rr["hit"], ".4f"), ("ceiling.hit", ce["hit"], ".4f"), ("hit_strong", rr["hit_strong"], ".4f"),
            ("ceiling.hit_strong", ce["hit_strong"], ".4f"), ("band_share_token_head", rr["band_share_token_head"], ".2f"),
            ("filler_band_share_token_head", rr["filler_band_share_token_head"], ".2f"),
            ("record_rank_mean", sc["record_rank_mean"], ".2f"), ("record_rank_chance", sc["record_rank_chance"], ".1f"),
            ("slot.held.primary_centred", h["primary_centred"], "+.3f"), *ci("slot.held.primary_centred_ci", h["primary_centred_ci"]),
            ("cv_r2", sc["cv_r2"], ".3f")]


def reference(log):
    lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
    floor = next((float(m.group(1)) for x in lines for m in [FLOOR.search(x)] if m), None)
    ref = {"source": "the 4090's dress rehearsal on step 230,415 (stopped at cell 288; its printed lines)", "log": os.path.basename(log),
           "floor_span_4090": floor, "gate_keys": GATE_KEYS, "lines": cells(lines)}
    missing = [k for k in GATE_KEYS if k not in ref["lines"]]
    assert not missing and floor is not None, (missing, floor)
    with open(REF, "w", encoding="utf-8") as fh:
        json.dump(ref, fh, indent=1)
    print(f"[port gate] reference: {len(ref['lines'])} cell lines, the 4090's span floor {floor:.2e}, the gate's {len(GATE_KEYS)} "
          f"cells present -> {REF}")


def within(d, x, tol):
    """A difference d at value x against the bar: absolute for magnitudes up to 1, relative above."""
    return d <= tol * max(1.0, abs(x))


def check(log, pod_json, raw_ref=None, tol=TOL, shift_bar=SHIFT):
    ref = json.load(open(REF, encoding="utf-8"))
    pod_lines = cells(open(log, encoding="utf-8", errors="replace").read().splitlines())
    res = json.load(open(pod_json, encoding="utf-8"))
    rref = json.load(open(raw_ref, encoding="utf-8")) if raw_ref else None
    print(f"[port gate] the bar: {tol:g} per number in its own units (absolute up to 1, relative above)"
          + (f"; raw values against {raw_ref}" if rref else f"; the implied mean shift at most {shift_bar:g}"), flush=True)
    verdicts, flips, n_cmp, worst = {}, [], 0, (0.0, None)
    for key in ref["gate_keys"]:
        if key not in pod_lines:
            verdicts[key] = ("FAIL", "the run printed no line for this cell")
            continue
        r_line, p_line = ref["lines"][key], pod_lines[key]
        vals = raw_values(res, key)
        p_nums, r_nums = NUM.findall(p_line), NUM.findall(r_line)
        if [format(int(x), "d") if f == "d" else format(x, f) for _, x, f in vals] != p_nums:
            verdicts[key] = ("FAIL", f"the field list does not reproduce the run's own line ({len(vals)} vs {len(p_nums)} numbers)")
            continue
        if NUM.sub("#", r_line) != NUM.sub("#", p_line):
            verdicts[key] = ("FAIL", "the lines differ beyond their numbers (an [n/a] or the layout)")
            continue
        bad, notes = [], []
        rvals = raw_values(rref, key) if rref is not None else None
        for i, ((name, x, f), rs, ps) in enumerate(zip(vals, r_nums, p_nums)):
            n_cmp += 1
            if rvals is not None:                              # raw against raw
                y = rvals[i][1]
                if f == "d":
                    if int(x) != int(y):
                        bad.append(f"{name}: count {y} -> {x}")
                    continue
                d = abs(x - y)
                rel = d / max(1.0, abs(y))
                if rel > worst[0]:
                    worst = (rel, f"{key} {name}: {y!r} (4090) vs {x!r}")
                if not within(d, y, tol):
                    bad.append(f"{name}: raw {y!r} -> {x!r} ({d:.1e})")
                continue
            if rs == ps:
                continue
            if f == "d":
                bad.append(f"{name}: count {rs} -> {ps}")
                continue
            ulp = 10.0 ** -int(f.split(".")[1][0])
            r, p = float(rs), float(ps)
            if abs(p - r) > 1.0001 * ulp:
                bad.append(f"{name}: {rs} -> {ps} (more than one unit)")
                continue
            past = abs(x - (r + p) / 2)
            flips.append((key, name, ulp))
            (notes if within(past, x, tol) else bad).append(f"{name}: {rs} -> {ps} (raw {x!r}, {past:.1e} past the boundary)")
        verdicts[key] = ("FAIL", "; ".join(bad)) if bad else (("CONSISTENT", "; ".join(notes)) if notes else ("EXACT", ""))
    for key, (v, why) in verdicts.items():
        print(f"[port gate] {v:10s} {key}{': ' + why if why else ''}", flush=True)
    counts = {v: sum(x == v for x, _ in verdicts.values()) for v in ("EXACT", "CONSISTENT", "FAIL")}
    ok = counts["FAIL"] == 0
    if rref is not None:
        print(f"[port gate] THE MEASURED AGREEMENT between the two cards: the largest raw difference {worst[0]:.2e} in its own "
              f"units ({worst[1]}) over {n_cmp} numbers", flush=True)
    else:
        shift = sum(u for _, _, u in flips) / max(1, n_cmp)
        by_field = {}
        for _, name, _ in flips:
            by_field[name] = by_field.get(name, 0) + 1
        print(f"[port gate] THE IMPLIED MEAN SHIFT {shift:.2e} ({len(flips)} flipped of {n_cmp} numbers; the bar {shift_bar:g})"
              + (f"; flips by field: {by_field}" if by_field else ""), flush=True)
        ok = ok and shift <= shift_bar
    print(f"[port gate] VERDICT: {'PASS' if ok else 'FAIL'} ({counts['EXACT']} exact, {counts['CONSISTENT']} consistent, "
          f"{counts['FAIL']} failed of {len(verdicts)} cells)" + ("" if ok else ": the run stops before the grid of record; the "
          "lines above name the fields and cells that differ"), flush=True)
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "reference":
        reference(sys.argv[2])
    else:
        a = sys.argv[2:]
        rr = a[a.index("--raw-ref") + 1] if "--raw-ref" in a else None
        tl = float(a[a.index("--tol") + 1]) if "--tol" in a else TOL
        sh = float(a[a.index("--shift") + 1]) if "--shift" in a else SHIFT
        sys.exit(0 if check(a[0], a[1], rr, tl, sh) else 1)
