"""alephllm_diffusion.nano.page_comparator - the uniform comparator's folder on the Anima experiments repo (README, meta.json, the
ledger, the readers, a contact sheet), written from the run's own ledger (nano.experiment calls it). Nothing runs on import."""
from __future__ import annotations

import os

from .experiment import ANIMA_LICENCE, ANIMA_REFS, CODE_URL, DATE, JUDGE, STEER_REFS, _f, _load, _refs, sheet
from .page_pictures import AMOUNT, READER_SHORT, READER_TEXT, _group

ANIMA_READERS = ("qwen", "beatrix", "nano", "birth")
FORM_TEXT = {"bare": "D_side (the side measured against the bare neutral caption)",
             "neutral_rel": "D_side - D_neutral (the side measured against a caption carrying a neutral phrase)"}
FORM_SHORT = {"bare": "the registered form", "neutral_rel": "the second form (less the neutral push)"}


def page(root: str, readers: str, img_dir: str, out_sheet: str, gain: float = 2.0, smoke: bool = False) -> tuple:
    led = os.path.join(root, "pictures", f"anima_comparator_g{gain:g}{'_smoke' if smoke else ''}.json")
    L = _load(led)
    R = _load(readers)
    M = L["_meta"]
    geo = M["geometry"]
    S = L["sets"]
    r_rows = []
    for form in FORM_TEXT:
        for side in ("up", "down"):
            for s in sorted({v["s"] for k, v in S.items() if k.startswith(f"R|{form}|{side}|")}):
                x = S[f"R|{form}|{side}|s{s:g}"]
                r_rows.append(f"| {FORM_TEXT[form]} | {side} | {s:g} | {_f(x['mean'], 2, True)} +- {_f(x['se'], 2)} | "
                              f"{x['OUTCOME'].lower()} | {_f(x['content_kept'])} | {_f(x['push_size'])} |")
    t_r = ("| direction | side | s | effect on the mood score | verdict | content kept | push size |\n"
           "|---|---|---|---|---|---|---|\n" + "\n".join(r_rows))
    gates = {f: L["reads"].get(f"gate|{f}", {}).get("OUTCOME", "-").lower() for f in FORM_TEXT}
    gf = L["reads"].get("G_form", "neutral_rel")
    g_rows, heldout = [], {}
    for rd in ANIMA_READERS:
        key = f"G{gain:g}|{rd}"
        if key not in L["reads"]:
            continue
        r = L["reads"][key]
        heldout[rd] = r["HELD_OUT"]
        g_rows.append(f"| {READER_SHORT[rd]} | {_group(r, 'heldout_up')} | {_group(r, 'heldout_down')} | "
                      f"{_f(r['groups']['neutral']['mean'], 2, True)} | {r['HELD_OUT'].lower()}; {(r['NEUTRAL'] or '-').lower()} |")
    t_g = "| reader | unseen cheerful | unseen gloomy | neutral | verdicts |\n|---|---|---|---|---|\n" + "\n".join(g_rows)
    ctl = L["reads"].get(f"G{gain:g}|control", {})
    cells = M["cells"]
    seed0 = cells[0][1]
    scenes = [c[0] for c in cells if c[1] == seed0]
    up_p, dn_p = "elated", "dismal"
    cols = [("base", None), ("R", "up"), ("R", "down")] + [(rd, p) for rd in ANIMA_READERS for p in (up_p, dn_p)]
    labels = ["no push", "up direction, s 2", "down direction, s 2"] + [f"{rd} {p}" for rd in ANIMA_READERS for p in (up_p, dn_p)]

    def path_of(si, col):
        a, b = col
        if a == "base":
            return os.path.join(img_dir, f"base_{si}_{seed0}.jpg")
        if a == "R":
            return os.path.join(img_dir, f"R_{gf}_{b}_s2_{si}_{seed0}.jpg")
        i = R["phrases"].index(b)
        return os.path.join(img_dir, f"G{gain:g}_{a}_{R['cls'][i]}_{i}_{si}_{seed0}.jpg")

    sheet(scenes[:8], cols, path_of, out_sheet, labels=labels)
    summary = (f"the directions alone at s = 2: {gates['bare']} ({FORM_SHORT['bare']}), {gates['neutral_rel']} "
               f"({FORM_SHORT['neutral_rel']}); the readers at g = {gain:g} on {FORM_SHORT[gf]}: "
               + "; ".join(f"{READER_SHORT[rd]} {v.lower()}" for rd, v in heldout.items()))
    title = ("The uniform comparator on Anima: two fixed mood directions from Anima's own prepared conditioning, each reader "
             "setting only the amount (Qwen3, Beatrix 3s, a nano body, an untrained start)")
    readers_line = "; ".join(READER_TEXT[rd] for rd in ANIMA_READERS)
    readme = f"""# {{rid}}: {title}

Date: {DATE}. Model: Anima-Base v1.0 (stock), {M['settings']}. No training: every picture is the stock model with a fixed push added to the text adapter's output.

## Question
The mood sliders of this repo (e039-e044) learned a push direction from Anima's own denoising loss; their cheerful side moved, their gloomy side barely did, and the learned gloomy direction pointed only a quarter of the way toward one that works. Here the directions are fixed, taken once from what Anima's text path prepares for the mood words themselves, and every reader sets only the amount from its own reading of a phrase. Do the fixed directions move the pictures, and which readers carry phrases they were never fitted on?

## Design
- the directions: per side, the mean over the 24 training scenes and that side's 30 training phrases of the text adapter's output (its mean over a caption's T5 slots) for "<prefix> an illustration of <scene>, <phrase>." minus the same for the neutral caption "<prefix> an illustration of <scene>." (the evaluated scenes and phrases never enter). Appending any phrase moves that output one shared way (the two directions at cosine {_f(geo.get('bare_cos_up_down'))}), so a second form subtracts the push of the 4 neutral training phrases (cosine {_f(geo.get('neutral_rel_cos_up_down'))}). Sizes: up {_f(geo.get('bare_up'))} / {_f(geo.get('neutral_rel_up'))}, down {_f(geo.get('bare_down'))} / {_f(geo.get('neutral_rel_down'))}; a context token is about {geo['context_token_size']:.2f} long.
- the push: p = g (max(a, 0) D_up + max(-a, 0) D_down), added to the adapter's output at every T5 slot of the neutral caption, on the conditional branch only (as the words themselves act).
- the amount: {AMOUNT}
- stage R, the directions alone: s x D_side at s = 1, 2, 4, both sides, both forms. Stage G, the readers at g = {gain:g}: {readers_line}; on the 10 phrases no reader was fitted on (4 unseen a side, 2 unseen neutral). The form for stage G was fixed before any picture: the second form, unless only the registered one moves both sides at s = 2.
- cells: the 8 held-out scenes of the slider experiments at seeds 101 and 202 (16 cells), the neutral caption; each picture paired with the same cell's unpushed picture.
- judge: {JUDGE}

## Result

**The directions alone** (at s = 2: {FORM_SHORT['bare']} {gates['bare']}; {FORM_SHORT['neutral_rel']} {gates['neutral_rel']}; stage G read {FORM_SHORT[gf]}):

{t_r}

**The readers** (g = {gain:g}):

{t_g}

The control read (the untrained start against the nano reader): {str(ctl.get('OUTCOME', '-')).lower()} (unseen effects {_f(ctl.get('reader_heldout_effect'), 2)} and {_f(ctl.get('birth_heldout_effect'), 2)}).

![the sheet](sheet.jpg)

The sheet: the held-out scenes at seed {seed0} (rows): the unpushed picture, the stage-G directions alone at s = 2, then for each reader the unseen phrases "{up_p}" and "{dn_p}".

## Limits
16 cells; one gain in stage G; one judge model; directions from 24 scenes and the slider experiments' phrases; the amounts are linear readings of one state per phrase.

## Code
{CODE_URL}, `alephllm_diffusion.nano` (mood, comparator), branch feat/nano-battery.

## Licence
{ANIMA_LICENCE}

## References
{_refs(ANIMA_REFS + STEER_REFS)}

## Files

{{files}}
"""
    files = {"README.md": "this page", "meta.json": "the summary and the run's record",
             "result.json": "every set's paired differences, content kept, push size and the reads",
             "readers.json": "each reader's slider value for the 74 phrases", "sheet.jpg": "the contact sheet"}
    meta = {"title": title, "date": DATE, "kind": "nano_comparator", "status": "done",
            "recipe": {"bed": "Anima-Base v1.0, stock", "push": "p = g (max(a,0) D_up + max(-a,0) D_down), conditional branch",
                       "readers": [READER_TEXT[rd] for rd in ANIMA_READERS], "code": f"{CODE_URL} (alephllm_diffusion.nano)"},
            "result": {"gates": gates, "G_form": gf, "readers": heldout, "control": ctl.get("OUTCOME")}, "summary": summary}
    copies = {"result.json": led, "readers.json": readers, "sheet.jpg": out_sheet}
    return readme, meta, copies, files
