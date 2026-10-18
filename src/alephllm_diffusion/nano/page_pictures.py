"""alephllm_diffusion.nano.page_pictures - the picture experiments' folders: the uniform comparator on Sana and on Anima, and the
relay in pictures on Anima (README, meta.json, the ledgers, a contact sheet), written from the runs' own ledgers (nano.experiment
calls them). Nothing runs on import."""
from __future__ import annotations

import os

from .experiment import (ANIMA_LICENCE, ANIMA_REFS, BODY_NAMES, CODE_URL, DATE, JUDGE, NANO_REPO, READ_REFS, SANA_REFS, STEER_REFS,
                         _f, _load, _refs, sheet)

READER_TEXT = {
    "qwen": "Qwen3 0.6B, its final state (Anima's own text reader)",
    "beatrix": "Beatrix 3s (a 376M byte-level model), block 20",
    "nano": "nano-castor-v1.1 -cf, block 3 (the best nano reader by a rule blind to the unseen phrases)",
    "stack": "the four nano bodies' -fr finals side by side, block 3",
    "birth": "nano-castor-v1.1's untrained start, block 3 (the control)",
}
READER_SHORT = {"qwen": "Qwen3", "beatrix": "Beatrix 3s, block 20", "nano": "castor -cf, block 3",
                "stack": "four nano bodies side by side", "birth": "castor's untrained start (control)"}
AMOUNT = ("Each reader's amount for a phrase is its own reading of it: the phrase in Anima's caption form (\"masterpiece, best "
          "quality, score_7, safe, an illustration of a quiet street, <phrase>.\"), the reader's state at the closing full stop, "
          "standardized per feature over the 64 training phrases (30 cheerful, 30 gloomy, 4 neutral), placed on the line from the "
          "gloomy training centre to the cheerful one: a = (f - m) . u / (u . u / 2), u = the cheerful centre minus the gloomy "
          "one, m = their midpoint, so the centres sit at +1 and -1. The reader for the nano bodies was chosen before any picture "
          "by a rule blind to the evaluated unseen phrases: the most training phrases on their own side when each is left out of "
          "the fit (castor -cf at block 3, 48 of 60; Qwen3 60, Beatrix 57, the controls 32-50).")


def _group(r, g):
    x = r["groups"][g]
    return f"{_f(x['mean'], 2, True)} +- {_f(x['se'], 2)} ({x['OUTCOME'].lower()})"


# ------------------------------------------------------------------------------------------------ Sana: the uniform comparator
def sana_page(root: str, readers: str, img_dir: str, out_sheet: str) -> tuple:
    L = _load(os.path.join(root, "pictures", "sana_comparator.json"))
    R = _load(readers)
    M = L["_meta"]
    order = M["readers"]
    rows = []
    for rd in order:
        for g in M["gains"]:
            r = L["reads"][f"{rd}|g{g:g}"]
            rows.append(f"| {READER_SHORT[rd]} | {g:g} | {_group(r, 'trained_up')} | {_group(r, 'trained_down')} | "
                        f"{_group(r, 'heldout_up')} | {_group(r, 'heldout_down')} | {_f(r['groups']['neutral']['mean'], 2, True)} | "
                        f"{r['TRAINED'].lower()}; {r.get('HELD_OUT', '-').lower()}; {r.get('NEUTRAL', '-').lower()} |")
    t = ("| reader | g | trained cheerful | trained gloomy | unseen cheerful | unseen gloomy | neutral | verdicts |\n"
         "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    ev = [p for p in R["phrases"] if p in ("cheerful and upbeat", "joyful and uplifting", "gloomy and downbeat",
                                           "somber and melancholy") or R["split"][R["phrases"].index(p)] == "heldout"]
    t_a = ("| phrase | class | " + " | ".join(READER_SHORT[rd] for rd in order) + " |\n|---|---|" + "---|" * len(order) + "\n"
           + "\n".join(f"| {p} | {R['cls'][R['phrases'].index(p)]}{' (unseen)' if R['split'][R['phrases'].index(p)] == 'heldout' else ''} | "
                       + " | ".join(f"{R['readers'][rd]['a_in'][p]:+.2f}" for rd in order) + " |" for p in ev))
    ctl = L["reads"].get("control|g2", {})
    # the sheet: 6 subjects x (no push, then per reader one unseen cheerful and one unseen gloomy phrase at g = 2)
    up_p, dn_p = "elated", "dismal"
    cols = [("base", None)] + [(rd, p) for rd in order for p in (up_p, dn_p)]
    labels = ["no push"] + [f"{rd} {p}" for rd in order for p in (up_p, dn_p)]

    def path_of(si, col):
        rd, p = col
        if rd == "base":
            return os.path.join(img_dir, f"base_{si}_{M['seed']}.jpg")
        i = R["phrases"].index(p)
        return os.path.join(img_dir, f"{rd}_g2_{R['cls'][i]}_{R['split'][i]}_{i}_{si}_{M['seed']}.jpg")

    sheet(list(range(6)), cols, path_of, out_sheet, labels=labels)
    q2, b2 = L["reads"]["qwen|g2"], L["reads"]["beatrix|g2"]
    summary = (f"with the direction fixed, Qwen3 and Beatrix 3s carry unseen phrases both ways (g = 2: Qwen3 "
               f"{q2['groups']['heldout_up']['mean']:+.2f} / {q2['groups']['heldout_down']['mean']:+.2f}, Beatrix "
               f"{b2['groups']['heldout_up']['mean']:+.2f} / {b2['groups']['heldout_down']['mean']:+.2f}); the nano reader, the "
               f"four nano bodies side by side and the untrained start do not")
    title = ("The uniform comparator on Sana: one fixed mood direction, each reader setting only the amount (Qwen3, Beatrix 3s, "
             "a nano body, the four nano bodies side by side, an untrained start)")
    readme = f"""# {{rid}}: {title}

Date: {DATE}. Model: Sana 600M 512px (stock; diffusers), 20 steps, guidance 4.5, 512 x 512, the model card's dtypes. No training: every picture is the stock model with a push added to the text encoder's output.

## Question
A mood slider has two parts: a direction in the image model's conditioning and an amount read from a text reader. Learned together (the slider experiments of the Anima repo), the direction came out noisy and the gloomy side barely moved. Here the direction is fixed and the same for every reader: the flavor direction this bed already measured (e002 in this repo: a dial, +0.79 of mood score per unit, every subject). Each reader sets only the amount, from its own reading of a phrase. Which readers carry phrases they were never fitted on, both ways?

## Design
- the push: e'(token) = e(token) + g * a * d on every own token of the neutral prompt "a photo of <subject>" (the negative prompt untouched); d = half the mean difference between the text encoder's output for upbeat and downbeat wordings of 32 subjects (two templates each), its size 10.14; g = 2, then g = 1.
- the amount: {AMOUNT}
- readers: """ + "; ".join(READER_TEXT[rd] for rd in order) + f""". The nano bodies: {NANO_REPO}.
- cells: 32 subjects at seed 11; per reader and gain the 14 evaluated phrases of the slider experiments (two training phrases and four unseen phrases a side, two unseen neutral phrases) = 448 pictures; 4,480 pictures and the 32 unpushed ones in 18 minutes on one RTX 4090.
- judge: {JUDGE}
- checks: d's size reproduces this bed's flavor test ({M['direction_norm']:.4f}); the embeddings path reproduces the stock call pixel for pixel (largest difference {M['parity_max_pixel_diff']:.0f}).

## Result

{t}

Each reader's amount a for the evaluated phrases (the training centres at +1 and -1):

{t_a}

The control read (the untrained start against the nano reader it controls): {ctl.get('OUTCOME', '-').lower()}; their unseen effects at g = 2: {_f(ctl.get('reader_heldout_effect'), 2)} and {_f(ctl.get('birth_heldout_effect'), 2)}.

![the sheet](sheet.jpg)

The sheet: six subjects (rows); the unpushed picture, then for each reader the unseen phrases "{up_p}" and "{dn_p}" at g = 2.

## What it means
1. The direction is fixed and shared, so a reader contributes only its amount.
2. Qwen3's and Beatrix's readings put every unseen phrase on its own side; their pushes carry unseen phrases both ways, Beatrix as far as Qwen3 on the gloomy side. The gloomy failure of the learned sliders is a property of the learned direction, not of her reading.
3. The nano readings sit at their controls' level and put neutral phrases far onto the cheerful side; their pushes are wrong-signed or loud on neutral phrases, so no nano reader is a usable mood reader on this rule, alone or side by side.

## Limits
One seed; 32 subjects; one judge (CLIP ViT-L/14); one direction per bed, taken from mood words; the amounts are linear readings of one state per phrase.

## Code
{CODE_URL}, `alephllm_diffusion.nano` (mood, comparator), branch feat/nano-battery.

## References
{_refs(SANA_REFS + STEER_REFS)}

## Files

{{files}}
"""
    files = {"README.md": "this page", "meta.json": "the summary and the run's record",
             "result.json": "every set's paired differences, content kept and the reads", "readers.json":
             "each reader's slider value for the 74 phrases (on the training axis and left out)", "sheet.jpg": "the contact sheet"}
    meta = {"title": title, "date": DATE, "kind": "nano_comparator", "status": "done",
            "recipe": {"bed": "Sana 600M 512px, stock", "push": "e' = e + g a d on the neutral prompt's own tokens",
                       "readers": [READER_TEXT[rd] for rd in order], "code": f"{CODE_URL} (alephllm_diffusion.nano)"},
            "result": {k: {"TRAINED": v["TRAINED"], "HELD_OUT": v.get("HELD_OUT"), "NEUTRAL": v.get("NEUTRAL"),
                           "heldout_effect": v.get("heldout_effect")} for k, v in L["reads"].items() if "groups" in v},
            "summary": summary}
    copies = {"result.json": os.path.join(root, "pictures", "sana_comparator.json"), "readers.json": readers,
              "sheet.jpg": out_sheet}
    return readme, meta, copies, files


# ------------------------------------------------------------------------------------------------ Anima: the relay in pictures
RELAY_TEXT = {"ceiling": "Qwen3, the usual path (its bf16 encoding)",
              "qwen_fp32": "Qwen3's own fp32 states through the supplied-states path (the maps' target)",
              "floor": "the floor: Qwen3's input embedding mapped (a lookup of the token, no context)",
              "beatrix": "Beatrix 3s, block 24 (her closest to Qwen3 in e038)",
              "nano": "nano-castor-v1.1 -cf, block 3",
              "stack": "the four nano bodies' -fr finals side by side, block 3",
              "births": "the four nano bodies' untrained starts side by side, block 3 (the control)"}


def relay_page(root: str, img_dir: str, out_sheet: str, smoke: bool = False) -> tuple:
    L = _load(os.path.join(root, "relay", f"anima_relay{'_smoke' if smoke else ''}.json"))
    F = _load(os.path.join(root, "relay", "relay_fits.json"))
    arms = L["_meta"]["arms"]
    reads = L["reads"]

    def mean(arm, j):
        v = L["arms"][arm][j]
        return sum(v) / len(v)

    rows = []
    for a in arms:
        fit = F.get(a)
        r = reads.get(a)
        rows.append(f"| {RELAY_TEXT[a]} | " + (f"{fit['cv_r2']:.3f} / {fit['eval_r2']:.3f} / {fit['eval_cos']:.3f}" if fit else "-")
                    + f" | {mean(a, 'caption_score'):.2f} | {mean(a, 'kept'):.4f} | "
                    + (f"{_f(r['caption_score']['vs_floor'], 2, True)} +- {_f(r['caption_score']['se'], 2)} | "
                       f"{_f(r['kept']['vs_floor'], 4, True)} +- {_f(r['kept']['se'], 4)} | "
                       f"{_f(r['kept']['gap_closed'], 2) if r['kept']['gap_closed'] is not None else '-'} | {r['OUTCOME'].lower()}"
                       if r else "(the floor) | | |") + " |")
    t = ("| arm | the map: fit R2 / R2 on the unseen captions / cosine to Qwen3's state | caption score | picture kept (cosine to "
         "Qwen3's fp32 picture) | caption score against the floor | kept against the floor | share of the floor-to-Qwen3 gap "
         "closed (kept) | verdict |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    n_cap = len({c[0] for c in L["_meta"]["cells"]})
    seeds = sorted({c[1] for c in L["_meta"]["cells"]})

    def path_of(i, arm):
        return os.path.join(img_dir, f"{arm}_{i}_{seeds[0]}.jpg")

    sheet(list(range(min(5, n_cap))), arms, path_of, out_sheet, labels=arms)
    ident = L["_meta"]["identity"]
    useful = [a for a in arms if reads.get(a, {}).get("OUTCOME") == "USEFUL IN QWEN'S LANE" and a not in ("ceiling", "qwen_fp32")]
    judge_name = {"caption_score": "the caption score", "kept": "picture kept"}
    above = {j: [a for a in arms if a in reads and a != "control" and reads[a][j]["vs_floor"] > 2 * reads[a][j]["se"]]
             for j in judge_name}
    blind = [j for j in judge_name if "qwen_fp32" not in above[j]]             # a judge on which Qwen3 itself does not pass
    readers_above = {j: [a for a in above[j] if a not in ("ceiling", "qwen_fp32")] for j in judge_name}

    short = {"ceiling": "Qwen3's usual path", "qwen_fp32": "Qwen3's own states", "beatrix": "Beatrix 3s", "nano": "castor -cf",
             "stack": "the four nano bodies side by side", "births": "their untrained starts side by side"}

    def said(j, a):
        r = reads[a][j]
        return f"{short[a]} {_f(r['vs_floor'], 4 if j == 'kept' else 2, True)} +- {_f(r['se'], 4 if j == 'kept' else 2)}"

    findings = []
    for j in blind:
        findings.append(f"{judge_name[j]} cannot separate Qwen3's own states from the token lookup ({said(j, 'qwen_fp32')}): every "
                        f"arm reads the caption's own T5 tokens as the adapter's queries, and they carry the caption's content into "
                        f"every picture, so on this judge the rule of both judges can be met by no arm")
    for j in judge_name:
        if j in blind:
            continue
        ra = readers_above[j]
        findings.append(f"on {judge_name[j]}, above the floor: " + (", ".join(said(j, a) for a in ra) if ra else "no reader")
                        + "; at the floor: " + ", ".join(said(j, a) for a in ("nano", "stack", "births") if a in reads and a not in ra))
    gap = {a: reads[a]["kept"]["gap_closed"] for a in ("beatrix", "nano", "stack", "births") if a in reads}
    summary = ("the registered rule (above the token lookup on both judges) is met by no arm" if not useful else
               "useful in Qwen3's lane: " + ", ".join(useful)) + "; " + "; ".join(findings) + (
               f"; share of the floor-to-Qwen3 gap closed on picture kept: " + ", ".join(f"{a} {v:.2f}" for a, v in gap.items()))
    meaning = "\n".join(f"{i + 1}. {s[0].upper()}{s[1:]}." for i, s in enumerate(findings + [
        "a state map that tracks Qwen3's states better (the four nano bodies side by side: R2 on unseen captions "
        f"{F['stack']['eval_r2']:.2f} against the lookup's {F['floor']['eval_r2']:.2f}) did not move the pictures toward Qwen3's; "
        f"Beatrix's map ({F['beatrix']['eval_r2']:.2f}) did, by {gap.get('beatrix', 0):.0%} of the gap: closeness in Qwen3's own "
        "coordinates is necessary, and the share of it that reaches the pictures is small"]))
    title = ("The relay in pictures over whole captions: Qwen3's states replaced by a mapped reading of Beatrix 3s, a nano body, "
             "the four nano bodies side by side, a lookup of the token, and the untrained nano starts")
    readme = f"""# {{rid}}: {title}

Date: {DATE}. Model: Anima-Base v1.0 (stock), {L['_meta']['settings']}. No training of any model: each arm replaces the Qwen3 states Anima's text adapter reads with a linear map of another reader's states.

## Question
Can a small byte-level model stand in for Qwen3 as Anima's text reader? The read of the nano bodies (e046 in this repo) found each body's middle block close to Qwen3's states (about three quarters of Beatrix's best) and the four bodies side by side closer still. Here the question goes to pictures: a whole caption's Qwen3 states are replaced by each reader's mapped states, and the pictures are compared with Qwen3's own and with a floor that knows only each token's identity.

## Design
- the relay: one closed-form ridge per reader (float64, the penalty chosen by cross-validation over two caption folds) from the reader's state at each token's closing byte to Qwen3's final state, fitted on the dual-extraction read's 4,920 caption rows of its first draw; on a new caption every token but the first gets the map of the reader's state (the caption is read with one space after it, so its last token closes too); the first token keeps Qwen3's own state (its attention sink), the same for every arm. The caption's T5 tokens (the adapter's queries) are its own in every arm.
- arms: """ + "; ".join(RELAY_TEXT[a] for a in arms) + f""".
- cells: the first {n_cap} captions of the read's second draw (never in a fit), bare, seeds {', '.join(map(str, seeds))}: {len(L['_meta']['cells'])} cells, {len(arms) * len(L['_meta']['cells'])} pictures.
- judges (CLIP ViT-L/14, pixels only): the caption score, 100 x the cosine between the picture and its caption; picture kept, the cosine between the picture and the same cell's picture from Qwen3's own fp32 states. The unit for the errors is the caption (its seeds averaged).
- the rules (fixed before any picture): USEFUL IN QWEN'S LANE = above the floor on both judges by more than 2 standard errors; the untrained starts side by side must not be; descriptive: the share of the floor-to-Qwen3 gap each arm closes.
- checks: the token ids are the read's; the supplied fp32 states match the runner's own fp32 encoder ({L['_meta']['encoding_check']:.1e} largest per-token relative); the supplied-states path fed the runner's own encoding reproduces its picture pixel for pixel; the bf16 encoding against fp32: {ident['bf16_vs_fp32_pixel_mean']:.2f} mean pixel difference.

## Result

{t}

![the sheet](sheet.jpg)

The sheet: the first five captions at seed {seeds[0]} (rows), one column per arm in the table's order.

## What it means
{meaning}

## Limits
{n_cap} captions, {len(seeds)} seeds; one linear map per reader; the first token kept from Qwen3; one judge model.

## Code
{CODE_URL}, `alephllm_diffusion.nano` (relay), branch feat/nano-battery.

## Licence
{ANIMA_LICENCE}

## References
{_refs(ANIMA_REFS + READ_REFS[:4] + STEER_REFS[-1:])}

## Files

{{files}}
"""
    files = {"README.md": "this page", "meta.json": "the summary and the run's record",
             "result.json": "every picture's two judges, per arm, and the reads", "maps.json": "each reader's map: penalty and fit",
             "sheet.jpg": "the contact sheet"}
    meta = {"title": title, "date": DATE, "kind": "nano_relay", "status": "done",
            "recipe": {"bed": "Anima-Base v1.0, stock", "arms": [RELAY_TEXT[a] for a in arms],
                       "code": f"{CODE_URL} (alephllm_diffusion.nano)"},
            "result": {a: reads[a] for a in reads}, "summary": summary}
    copies = {"result.json": os.path.join(root, "relay", f"anima_relay{'_smoke' if smoke else ''}.json"),
              "maps.json": os.path.join(root, "relay", "relay_fits.json"), "sheet.jpg": out_sheet}
    return readme, meta, copies, files
