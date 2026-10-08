"""alephllm_diffusion.triangulate.arm_experiment - the read of record on a surface arm (triangulate.arm_read) as an experiment of the
Anima experiments repo: experiments/<ARM_READ_ID>/ (README, meta.json, the read's ledger as result.json, its tables, the small
run's ledger), written from the read's ledger into a local folder laid out like the repo and published by the Anima trainer's
step for experiments written elsewhere (AnimaRunner.publish_local), as the dual-extraction read (triangulate.experiment) is. The
id comes from the trainer's registry (anima_experiments.ARM_READ_ID).

Usage: python -m alephllm_diffusion.triangulate.arm_experiment <ledger.json> [--small=<the small run's ledger>]
       [--mirror=<folder>] [--code="..."] [--publish=1]
(--publish=1 needs HF_TOKEN with write access to the experiments repo; the rest runs offline.)"""
import json
import os
import shutil
import sys

from geolip_anima_trainer import anima_experiments as ax

from . import arm_read as AR
from .experiment import CITATIONS as E038_CITATIONS

EXPERIMENT_ID = ax.ARM_READ_ID
DATE = "2026-10-08"                                     # registered and run that day (local time; the ledger's clock is UTC)
TITLE = ("Beatrix reading Qwen3's spelling of a caption through her Qwen tokenizer arm, against the Qwen3 states Anima reads "
         "(a read of her states; no pictures)")
KIND = "beatrix_read"
CODE_URL = "https://github.com/AbstractEyes/alephllm-diffusion-experiments"
ARM_URL = "https://huggingface.co/AbstractPhil/beatrix-tokenizers"
FILES = {"README.md": "this page",
         "meta.json": "the decisions, the numbers on this page and the run's record",
         "result.json": "the read's ledger: every comparison on both caption draws, the decisions",
         "tables.md": "the read's own tables",
         "small_run.json": "the small run before the full read (48 captions a draw, three blocks, 16 dimensions; its numbers are "
                           "a rehearsal of the code, not results)"}
_KEEP = ("Schönemann", "Smith, Turban", "Artetxe", "Kornblith", "Minixhofer", "Radford", "Qwen Team", "Bansal")
CITATIONS = [c for c in E038_CITATIONS if c[0].startswith(_KEEP)]
SETS_TEXT = {
    "A0": "her trunk alone, the caption's own bytes (the dual-extraction read's reading)",
    "B0": "her trunk alone, Qwen3's spelling (the dual-extraction read's reading)",
    "A8": "the eight stage arms the tokenizer arm was trained over mounted, the tokenizer arm masked: the caption's own bytes "
          "(what the arm was trained to match)",
    "B8": "the same mount, the tokenizer arm masked: Qwen3's spelling (the gap before the arm)",
    "B8q": "the same mount with the tokenizer arm on: Qwen3's spelling read through the arm (THE ARM'S READING)",
    "A8q": "the same mount with the tokenizer arm on: the caption's own bytes (does the arm leave her own bytes alone?)",
    "UA0": "an untrained trunk of the same shape (random initialisation, seed 0): the caption's own bytes",
    "UB0": "the untrained trunk: Qwen3's spelling (the mood rule's comparator)",
    "UBq": "the untrained trunk with its own tokenizer arm: Qwen3's spelling through it",
}


def _a(R, name, b, target="Q"):
    return R["C1"][f"{name}|{b}"][target]["alignment"]


def _span(xs) -> str:
    return f"{min(xs):.3f}-{max(xs):.3f}"


def findings(L) -> list:
    """The result in plain sentences, every number from the ledger (draw 1; draw 2 named where it decides)."""
    R, R2, m, dec = L["draw1"], L.get("draw2"), L["_meta"], L["decisions"]
    blocks = m["blocks"]
    gap = [R["pairs"][f"gap|{b}"]["alignment"] for b in blocks]
    before = [R["pairs"][f"gap before|{b}"]["alignment"] for b in blocks]
    sil = [R["pairs"][f"silence|{b}"]["alignment"] for b in blocks]
    b3 = dec["B3"]["draw1"]
    out = [f"Through the arm, Qwen3's spelling lines up with her own bytes: her two readings of one caption align {_span(gap)} "
           f"over blocks {blocks[0]}-{blocks[-1]}, against {_span(before)} with the arm masked.",
           f"It sits as close to the Qwen3 states Anima reads as her own bytes in the same mount: at block {AR.B3_BLOCK} "
           f"{b3['B8q']:.3f} against {b3['A8']:.3f} (draw 2: {dec['B3']['draw2']['B8q']:.3f} against "
           f"{dec['B3']['draw2']['A8']:.3f}); without the arm Qwen3's spelling reads {_a(R, 'B8', AR.B3_BLOCK):.3f}, and her trunk "
           f"alone on her own bytes {_a(R, 'A0', AR.B3_BLOCK):.3f} (the dual-extraction read's {AR.E038_A0}).",
           f"The eight stage arms the tokenizer arm rides on cost a little alignment by themselves: her own bytes read "
           + ", ".join(f"{_a(R, 'A0', b):.3f} -> {_a(R, 'A8', b):.3f}" for b in (blocks[0], AR.SLIDER_BLOCK, AR.B3_BLOCK))
           + f" at blocks {blocks[0]}, {AR.SLIDER_BLOCK} and {AR.B3_BLOCK} once they are mounted.",
           f"The arm leaves her own bytes nearly alone: with it on, her reading of her own bytes aligns {_span(sil)} with the "
           "same reading with it masked.",
           f"The triangle closes: going from her own bytes through the arm's reading to Qwen3 loses at most "
           f"{max(abs(v['gap']) for k, v in R['triangles'].items() if not k.startswith('A8->B8->')):.3f} of the direct path, "
           f"where the path through the spelling without the arm lost "
           f"{_span([R['triangles'][f'A8->B8->Q|{b}']['gap'] for b in blocks])}."]
    band = [b for b in AR.SLIDER_BAND if b in blocks]
    out.append("Her mood direction comes through the arm: the mood axis of the arm's reading in Qwen3's frame "
               + ", ".join(f"{R['C6'][f'B8q|{b}']['cos_axis']:+.3f}" for b in band)
               + f" at blocks {', '.join(map(str, band))}, against her plain reading in the same mount "
               + ", ".join(f"{R['C6'][f'A8|{b}']['cos_axis']:+.3f}" for b in band)
               + f" and Qwen3's spelling without the arm "
               + ", ".join(f"{R['C6'][f'B8|{b}']['cos_axis']:+.3f}" for b in band) + ".")
    sb = dec["slider_block"]
    passing = [b for b, rr in sb["rule"].items() if all(r["passes"] for r in rr)]
    out.append(f"The slider's block: {sb['block']} ({sb['why']}); the mood rule passes on both caption draws at blocks "
               f"{', '.join(map(str, passing))}.")
    return out


def summary(L) -> str:
    dec, R = L["decisions"], L["draw1"]
    gap = [R["pairs"][f"gap|{b}"]["alignment"] for b in L["_meta"]["blocks"]]
    return (f"through the arm Qwen3's spelling aligns {_span(gap)} with her own bytes; against Qwen3's states at block "
            f"{AR.B3_BLOCK} {dec['B3']['draw1']['B8q']:.3f} (her own bytes in the same mount {dec['B3']['draw1']['A8']:.3f}): "
            f"B3 {dec['B3']['draw1']['verdict']} / {dec['B3']['draw2']['verdict']}, B4 {dec['B4']['draw1']['verdict']} / "
            f"{dec['B4']['draw2']['verdict']}; the slider's block {dec['slider_block']['block']}")


def recipe(L, code: "str | None" = None) -> dict:
    m, row = L["_meta"], L["_meta"]["registry_row"]
    if code is None:
        code = f"{CODE_URL}, alephllm_diffusion.triangulate.arm_read"
    rows = m["rows"]
    return {
        "model": f"Beatrix ({ax.HUB_TRUNK}); an untrained trunk of the same shape (random initialisation, seed 0) as the "
                 "comparator of the mood rule",
        "the arm": f"{ARM_URL}, {row['file']} (the member {row['member']}, mounted after the eight stage arms of "
                   f"{row['base'][0]} at close {row['base'][1]} it was trained over, by the library's mount, "
                   "geolip.alephllm.arm_mount.mount_surface)",
        "target": "Qwen3 0.6B (Anima's text encoder file), its final hidden state at each token, as stored by the "
                  f"dual-extraction read ({ax.TRIANGULATION_ID}); beside it the input to its layer 16 and its input embedding",
        "rows": f"the dual-extraction read's rows, rebuilt and checked token by token: the first {m['n_caps']} COCO captions of "
                f"each of two draws ({rows['draw1']['captions']:,} and {rows['draw2']['captions']:,} token rows) and the "
                f"{rows['draw1']['mood']} mood phrases in the grid's caption form (one row per phrase: its full stop)",
        "her states": f"the library's reader (geolip.alephllm.train.surface.read_spelled, one caption a row, the arm's own "
                      f"reader arguments): her block output at blocks {', '.join(map(str, m['blocks']))} at the byte closing each "
                      "token, layer-normed without its affine",
        "preparation": f"each reading reduced to its {m['k']} leading directions on the fit captions, whitened; every rotation "
                       "and ridge fit on half the captions and scored on the other half, both ways round (the dual-extraction "
                       "read's gauges)",
        "precision": "fp32 for the passes (autocast off), float64 for the comparisons (TF32 off)",
        "card": "one RTX 4090",
        "code": code,
    }


def _table(R, R2, blocks, names, fmt) -> list:
    out = ["| reading | " + " | ".join(f"block {b}" for b in blocks) + " |", "|---|" + "---|" * len(blocks)]
    for n in names:
        out.append(f"| {n} | " + " | ".join(fmt(R, R2, n, b) for b in blocks) + " |")
    return out


def readme(L, code: "str | None" = None) -> str:
    m, dec, R, R2 = L["_meta"], L["decisions"], L["draw1"], L.get("draw2")
    blocks = m["blocks"]
    names = [n for n in AR.SETS if f"{n}|{blocks[0]}" in R["C1"]]
    rec = recipe(L, code)
    out = [f"# {EXPERIMENT_ID}: {TITLE}", "",
           f"Date: {DATE}. Model: Beatrix, a byte-level language model ({ax.HUB_TRUNK}), with her Qwen tokenizer arm. No picture "
           "is made: this experiment reads her states against the text states Anima reads.", "",
           "## Question",
           f"The dual-extraction read ({ax.TRIANGULATION_ID}) found that her reading of a caption's own bytes sits much closer "
           "to the Qwen3 states Anima's text adapter reads than her reading of Qwen3's spelling of the same tokens, and that "
           "only her own bytes carry her mood direction onto Qwen3's. Her Qwen tokenizer arm, a detachable adapter after each of "
           "her 32 blocks, was then trained so that she reads Qwen3's spelling as she reads her own bytes. With the arm mounted: "
           "does Qwen3's spelling line up with her own bytes? Does it sit as close to Qwen3's states as her own bytes, and does "
           "the triangle (her own bytes to the arm's reading to Qwen3) close? Does the arm leave her reading of her own bytes "
           "alone? Does her mood direction come through it? And which block should a slider read through the arm use?", "",
           "## Design", ""] + [f"- {k}: {v}" for k, v in rec.items()]
    out += ["", "The readings:", "", "| name | reading |", "|---|---|"]
    out += [f"| {n} | {SETS_TEXT[n]} |" for n in names]
    out += ["", "## The rules (fixed before the run)",
            f"- B3: the arm's reading against Qwen3's final states at block {AR.B3_BLOCK} within {AR.B3_MARGIN} of the "
            f"dual-extraction read's reading of her own bytes ({AR.E038_A0}); the triangle's detours through the arm within "
            f"{AR.B3_MARGIN} on every path.",
            f"- B4: the arm's mood axis in Qwen3's frame at blocks {', '.join(map(str, AR.B4_BLOCKS))} at least half of her plain "
            "reading's in the same mount.",
            f"- The slider's block: {AR.SLIDER_BLOCK}, unless the arm's reading fails the dual-extraction read's mood rule there on "
            f"both caption draws (its mood axis above the untrained trunk's by more than .10 and at least 3 of the 4 unseen "
            f"gloomy phrases on Qwen3's gloomy side); then the block of {AR.SLIDER_BAND[0]}-{AR.SLIDER_BAND[-1]} passing on both "
            "draws with the largest alignment to Qwen3.", "",
            "## Result", "", f"In short: {summary(L)}.", ""]
    out += [f"- {s}" for s in findings(L)]
    out += ["", "### Every reading against Qwen3's final states (whitened alignment on held-out captions; draw 2 in brackets)",
            ""]
    out += _table(R, R2, blocks, names, lambda R, R2, n, b: f"{_a(R, n, b):.3f}" + (f" ({_a(R2, n, b):.3f})" if R2 else ""))
    out += ["", "### Her readings against each other (draw 1)", "",
            "| pair | " + " | ".join(f"block {b}" for b in blocks) + " |", "|---|" + "---|" * len(blocks)]
    for p, (x, y) in AR.PAIRS.items():
        if f"{p}|{blocks[0]}" in R["pairs"]:
            out.append(f"| {p} ({x} against {y}) | " + " | ".join(f"{R['pairs'][f'{p}|{b}']['alignment']:.3f}" for b in blocks)
                       + " |")
    out += ["", "### The triangle (draw 1: direct / through the middle reading / the loss)", "",
            "| path | " + " | ".join(f"block {b}" for b in blocks) + " |", "|---|" + "---|" * len(blocks)]
    for t in AR.TRIANGLES:
        if f"{t}|{blocks[0]}" in R["triangles"]:
            out.append(f"| {t.replace('->', ' -> ')} | " + " | ".join(
                f"{R['triangles'][f'{t}|{b}']['direct']:.3f} / {R['triangles'][f'{t}|{b}']['composed']:.3f} / "
                f"{R['triangles'][f'{t}|{b}']['gap']:+.3f}" for b in blocks) + " |")
    out += ["", "### The mood axis in Qwen3's frame (draw 1: its cosine with Qwen3's axis; unseen gloomy phrases on Qwen3's "
                "gloomy side)", ""]
    out += _table(R, R2, blocks, names, lambda R, R2, n, b: f"{R['C6'][f'{n}|{b}']['cos_axis']:+.3f} "
                  f"({R['C6'][f'{n}|{b}']['mapped_down'][0]}/{R['C6'][f'{n}|{b}']['mapped_down'][1]})")
    out += ["", "### The decisions", ""]
    for d, v in dec["B3"].items():
        out.append(f"- B3, {d.replace('draw', 'draw ')}: {v['B8q']:.3f} against {v['e038_A0']} (her plain reading in the same "
                   f"mount {v['A8']:.3f}): **{v['verdict']}**; detours within {AR.B3_MARGIN} on {v['detours_within']} of "
                   f"{v['paths']} paths (the largest {v['largest_detour']:+.3f})")
    for d, v in dec["B4"].items():
        out.append(f"- B4, {d.replace('draw', 'draw ')}: " + ", ".join(
            f"block {b} {x['B8q']:+.3f} against {x['A8']:+.3f}" for b, x in v["blocks"].items()) + f": **{v['verdict']}**")
    sb = dec["slider_block"]
    out.append(f"- **The slider's block: {sb['block']}** ({sb['why']})")
    for b, rr in sb["rule"].items():
        out.append(f"  - block {b}: " + "; ".join(f"draw {i + 1} {'passes' if r['passes'] else 'fails'} (axis gain "
                                                  f"{r['axis_gain']:+.3f}, unseen gloomy {r['unseen_gloomy'][0]}/"
                                                  f"{r['unseen_gloomy'][1]})" for i, r in enumerate(rr)))
    an = m.get("anchor_against_e038") or {}
    out += ["", "### Checks",
            "- Every row's token and position equal the dual-extraction read's (checked row by row on both draws).",
            "- The library's reader against the dual-extraction read's own stored states (her trunk alone, draw 1, every block): "
            "the largest difference " + ", ".join(f"{v:.1e} ({k})" for k, v in an.items()) + ".",
            "- Not read here: the untrained trunk with its own tokenizer arm (published after this read; the slider experiments "
            "use it as their control)." if "UBq" not in names else "- The untrained trunk with its own arm: in the tables."]
    out += ["", "## References (checked against their primary pages)", ""]
    out += [f"- {a}. {t}. {u}" for a, t, u in CITATIONS]
    out += ["", "## Files", "", "| file | what |", "|---|---|"] + [f"| `{k}` | {v} |" for k, v in FILES.items()] + [""]
    return "\n".join(out)


def meta(L, code: "str | None" = None) -> dict:
    m, R = L["_meta"], L["draw1"]
    out = {"id": EXPERIMENT_ID, "title": TITLE, "date": DATE, "kind": KIND, "status": "done", "recipe": recipe(L, code),
           "result": {"decisions": L["decisions"],
                      "alignment": {k: v["Q"]["alignment"] for k, v in R["C1"].items()},
                      "pairs": {k: v["alignment"] for k, v in R["pairs"].items()},
                      "triangles": R["triangles"],
                      "mood_axis": {k: {"cos": v["cos_axis"], "unseen_gloomy": v["mapped_down"]} for k, v in R["C6"].items()},
                      "anchor": m.get("anchor_against_e038")},
           "summary": summary(L)}
    if m.get("seconds") is not None:
        out["seconds"] = int(m["seconds"])
    if m.get("finished_utc"):
        out["finished_utc"] = m["finished_utc"]
    return out


def write(ledger: str, mirror: str, small: "str | None" = None, code: "str | None" = None) -> dict:
    """<mirror>/experiments/<id>/: README.md, meta.json, result.json (the ledger), tables.md and small_run.json."""
    with open(ledger, encoding="utf-8") as fh:
        L = json.load(fh)
    if L["_meta"].get("smoke"):
        raise ValueError(f"{ledger} is the small run's ledger; the experiment is the full read")
    if "decisions" not in L:
        raise ValueError(f"{ledger} holds no decisions (the read did not finish)")
    folder = os.path.join(mirror, "experiments", EXPERIMENT_ID)
    os.makedirs(folder, exist_ok=True)
    mt = meta(L, code)
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(readme(L, code))
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(mt, fh, indent=1, ensure_ascii=False)
    shutil.copyfile(ledger, os.path.join(folder, "result.json"))
    tables = ledger[:-5] + ".md"
    if os.path.exists(tables):
        shutil.copyfile(tables, os.path.join(folder, "tables.md"))
    if small:
        shutil.copyfile(small, os.path.join(folder, "small_run.json"))
    return mt


def publish(mirror: str) -> dict:
    """The folder into the experiments repo in one commit, then the repo README's index (the Anima trainer's own step)."""
    from geolip_anima_trainer.anima_runner import AnimaRunner
    return AnimaRunner().publish_local(EXPERIMENT_ID, mirror)


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    pos = [a for a in args if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in args if a.startswith("--") and "=" in a)
    if len(pos) != 1:
        raise SystemExit(__doc__)
    ledger = pos[0]
    mirror = opt.get("mirror", os.path.join(os.path.dirname(os.path.abspath(ledger)), "experiments_mirror"))
    mt = write(ledger, mirror, opt.get("small"), opt.get("code"))
    print(f"wrote {os.path.join(mirror, 'experiments', EXPERIMENT_ID)}: {mt['summary']}", flush=True)
    if opt.get("publish") == "1":
        publish(mirror)
    return 0


if __name__ == "__main__":
    sys.exit(main())
