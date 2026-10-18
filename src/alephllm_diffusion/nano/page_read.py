"""alephllm_diffusion.nano.page_read - the read's folder (no pictures) for the Anima experiments repo: README, meta.json and the
ledgers, written from the run's own files (nano.experiment calls it). Nothing runs on import."""
from __future__ import annotations

import os

from .experiment import (BODY_NAMES, BODY_TEXT, CODE_URL, DATE, MODELS_LINE, NANO_REPO, READ_REFS, VERSION_TEXT, _f, _load,
                         _refs)


def _signal(n: str) -> str:
    """'A-trunk-3' -> 'block 3' ('A-trunk-6' -> 'the final norm'); 'A-hub-2r' -> 'the rotary hub at block 2'."""
    if n.startswith("A-trunk-"):
        b = n[len("A-trunk-"):]
        return "the final norm" if b == "6" else f"block {b}"
    h = n[len("A-hub-"):]
    kind = {"r": "the rotary hub", "p": "the plain hub"}.get(h[-1], "the hub")
    return f"{kind} at block {h.rstrip('rp')}"


def page(root: str, e038: str) -> tuple:
    """(README with {rid} and {files} left to fill, meta, {file in the folder: source}, {file: what}) from the run's folder (root:
    compare/, sets/manifest.json) and e038's ledger (the references: Beatrix 3s, her untrained copy, Qwen3's own layers)."""
    B = _load(os.path.join(root, "compare", "nano_battery.json"))
    MOOD = _load(os.path.join(root, "compare", "nano_mood.json"))["readings"]
    MAN = _load(os.path.join(root, "sets", "manifest.json"))
    S4 = _load(os.path.join(root, "compare", "s4_last_block.json"))
    E = _load(e038)["draw1"]
    C1 = B["draw1"]["C1"]
    marks = B["marks"]
    bodies = list(BODY_NAMES)

    def q(k):
        return C1[k]["Q"]["alignment"]

    rows = []
    for b in bodies:
        bm = marks["bodies"][b]
        e = bm["end"]["signals"][bm["end"]["best"]]
        vers = [bm[v]["signals"][bm[v]["best"]]["alignment"] for v in ("end", "fr", "ct", "cf")]
        rows.append(f"| {BODY_NAMES[b]} | {_signal(bm['end']['best'])} | {_f(e['alignment'])} / {_f(e['alignment_draw2'])} | "
                    f"{_f(min(vers))}-{_f(max(vers))} | {_f(e['context'])} | {_f(e['ridge_r2'])} | {_f(e['birth'])} | "
                    f"{_f(e['margin'], sign=True)} | {e['S2'].lower()} |")
    t_bodies = ("| body | best signal | closeness at the end of the diet, draw 1 / 2 | the four trained versions there | context | "
                "ridge R2 | its untrained start | margin over it | verdict |\n|---|---|---|---|---|---|---|---|---|\n"
                + "\n".join(rows))
    ref = [("Beatrix 3s, block 8", E["C1"]["trained|A-trunk-8"]["Q"]),
           ("Beatrix 3s, block 24 (her closest)", E["C1"]["trained|A-trunk-24"]["Q"]),
           ("her untrained copy, block 24", E["C1"]["untrained|A-trunk-24"]["Q"]),
           ("Qwen3's input embedding (a lookup of the token)", E["floors"]["Q0->Q"]),
           ("Qwen3's layer 16", E["floors"]["Q16->Q"])]
    t_ref = ("| reference (the dual-extraction read, e038) | closeness | context | ridge R2 |\n|---|---|---|---|\n"
             + "\n".join(f"| {n} | {_f(r['alignment'])} | {_f(r['context_alignment'])} | {_f(r['ridge_r2'])} |" for n, r in ref))
    moved, held = [], 0
    for b in bodies:
        for v in ("fr", "ct", "cf"):
            sig = marks["bodies"][b][v]["signals"]
            held += sig[marks["bodies"][b][v]["best"]].get("vs_end", {}).get("S3") == "HELD"
            moved += [(b, v, n, r["vs_end"]["delta"]) for n, r in sig.items() if "vs_end" in r and r["vs_end"]["S3"] != "HELD"]
    t_moved = ("| body | version | signal | change against the end of the diet |\n|---|---|---|---|\n"
               + "\n".join(f"| {BODY_NAMES[b]} | -{v} | {_signal(n)} | {_f(d, sign=True)} |" for b, v, n, d in moved))
    hub_rows = []
    for b in bodies[1:]:
        for v in ("end", "fr", "ct", "cf"):
            u4 = marks["bodies"][b][v]["U4"]
            hub_rows.append(f"| {BODY_NAMES[b]} | {v} | {_signal(u4['hub'][0])} {_f(u4['hub'][1])} | {_signal(u4['trunk'][0])} "
                            f"{_f(u4['trunk'][1])} | {u4['verdict'].lower().replace('trunk', 'stream')} |")
    t_hubs = "| body | version | best hub | best stream block | verdict |\n|---|---|---|---|---|\n" + "\n".join(hub_rows)
    st = marks["stacks"]
    st_rows = []
    for name, x in st.items():
        best = max(x, key=lambda t: x[t]["stack"])
        r = x[best]
        label = ({"end": "the four bodies, end of diet", "fr": "the four bodies, -fr", "ct": "the four bodies, -ct",
                  "cf": "the four bodies, -cf", "birth": "the four untrained starts"}[name.split("|")[1]] if name.startswith("stack4|") else
                 f"{BODY_NAMES[name.split('|')[1]]}'s four trained versions")
        adds = sum(1 for t in x if x[t]["U3"] == "THE STACK ADDS")
        st_rows.append(f"| {label} | {_signal(best)} | {_f(r['stack'])} | {_f(r['stack_context'])} | {_f(r['best_member'][1])} | "
                       f"{_f(r['gain'], sign=True)} | {_f(r['bar'])} | {adds} of {len(x)} |")
    t_stacks = ("| stack | best block | closeness | context | its best member | gain | bar | blocks where it adds |\n"
                "|---|---|---|---|---|---|---|---|\n" + "\n".join(st_rows))

    def mrow(label, key):
        m = MOOD[key]
        neu = list(m["neutral"].values())[:2]
        return (f"| {label} | {m['loo_train']} / 60 | {_f(m['heldout_sep'], 2, True)} | {m['heldout_sides']} / 8 | "
                + " / ".join(f"{x:+.2f}" for x in neu) + " |")

    def tap(key):
        return key.split("|")[-1].replace("trunk-", "block ").replace("hub-", "hub ")

    best_tr = {b: max((k for k in MOOD if k.split("|")[0] == b and k.split("|")[1] in ("end", "fr", "ct", "cf")),
                      key=lambda k: (MOOD[k]["loo_train"], MOOD[k]["heldout_sep"])) for b in bodies}
    best_birth = {b: max((k for k in MOOD if k.split("|")[0] == b and k.split("|")[1] == "birth"),
                         key=lambda k: (MOOD[k]["loo_train"], MOOD[k]["heldout_sep"])) for b in bodies}
    t_mood = ("| reader | training phrases on their side (each left out of the fit) | unseen separation | unseen phrases on their "
              "side | two neutral training phrases |\n|---|---|---|---|---|\n"
              + "\n".join([mrow("Qwen3, final state", "qwen3|q-28post"), mrow("Qwen3, input embedding (the lookup)", "qwen3|q-0"),
                           mrow("Beatrix 3s, block 20", "beatrix-trained|trunk-20"),
                           mrow("her untrained copy, block 20", "beatrix-untrained|trunk-20")]
                          + [mrow(f"{BODY_NAMES[b]}, its best ({best_tr[b].split('|')[1]}, {tap(best_tr[b])})", best_tr[b])
                             for b in bodies]
                          + [mrow(f"{BODY_NAMES[b]} untrained, its best ({tap(best_birth[b])})", best_birth[b]) for b in bodies[:3]]
                          + [mrow("the four bodies side by side (-fr, block 3)", "stack4|fr|trunk-3")]))
    t_size = ("| body | last block's output size, median over captions: untrained / end of diet / continuations | its largest "
              "direction's share after normalizing: untrained / trained |\n|---|---|---|\n"
              + "\n".join(f"| {BODY_NAMES[b]} | {S4[f'{b}|birth']['size_median']:,.0f} / {S4[f'{b}|end']['size_median']:,.0f} / "
                          f"{min(S4[f'{b}|{v}']['size_median'] for v in ('fr', 'ct', 'cf')):,.0f}-"
                          f"{max(S4[f'{b}|{v}']['size_median'] for v in ('fr', 'ct', 'cf')):,.0f} | "
                          f"{S4[f'{b}|birth']['top_pc_share_block5_normed']:.2f} / "
                          f"{min(S4[f'{b}|{v}']['top_pc_share_block5_normed'] for v in ('end', 'fr', 'ct', 'cf')):.2f}-"
                          f"{max(S4[f'{b}|{v}']['top_pc_share_block5_normed'] for v in ('end', 'fr', 'ct', 'cf')):.2f} |"
                          for b in bodies))
    par = max(max(d["parity"].values()) if d["parity"] else 0.0 for e in MAN.values() for d in e["draws"].values())
    cz = max(v for e in MAN.values() for k, v in e.get("causality", {}).items() if k.startswith("stream"))
    ruler = B["ruler"]["alignment"]
    best_stack = max((x[t]["stack"], n, t) for n, x in st.items() if n.startswith("stack4|") for t in x)
    ends = [q(f"{b}|end|" + marks["bodies"][b]["end"]["best"]) for b in bodies]
    summary = (f"every body learned, closest at its middle block ({min(ends):.2f}-{max(ends):.2f} against Beatrix 3s's .56 and the "
               f"untrained starts' .27-.29); the continuations held; every hub below its stream; the four bodies side by side "
               f"{best_stack[0]:.2f}, adding at every block; no nano reader places mood phrases on its own axis better than its "
               f"controls")
    title = ("The four nano bodies and their continuations against the Qwen3 states Anima reads: closeness, stacks, hubs and each "
             "reader's own mood axis (a read of states; no pictures)")
    readme = f"""# {{rid}}: {title}

Date: {DATE}. Models: {MODELS_LINE} No picture is made: this experiment reads the bodies' states against the text states Anima reads, with the dual-extraction read's own rows and comparisons (e038 in this repo).

## Question
Anima's text adapter reads Qwen3's hidden states, one per Qwen token. Four small byte-level models were trained in one day on the same diet; they differ in their attention: castor has {BODY_TEXT['castor']}; pollux-canon has {BODY_TEXT['pollux-canon']}; pollux-dual has {BODY_TEXT['pollux-dual']}; pollux-dual-canon has {BODY_TEXT['pollux-dual-canon']}. Each was then continued three ways: on text written in frames, on plain text (the control), and on text that carries its own earlier content forward. Read per Qwen token: how close do their states sit to the states Anima reads, at which block, against Beatrix 3s (a 376M byte-level model, e038) and against a lookup of the token alone? Did the continuations move the caption states? Do the hubs read closer than the stream? Do the four bodies side by side hold more than the best one? And can each reader place mood phrases on its own cheerful-to-gloomy axis, the amount a mood slider needs?

## Design
- readers: each body at {VERSION_TEXT['end']}, {VERSION_TEXT['fr']}, {VERSION_TEXT['ct']} and {VERSION_TEXT['cf']}; and {VERSION_TEXT['birth']}. Weights: {NANO_REPO}, `<craft>/checkpoints/step_<step>.safetensors`, loaded strictly in fp32.
- rows: the dual-extraction read's own: the first 512 COCO captions of each of two draws (4,920 and 5,086 token rows: every Qwen token but a caption's first and last), and the 74 mood phrases of the earlier slider experiments in the grid's caption form (one row per phrase, at its closing full stop).
- signals: per Qwen token, at the byte after it (one pass over the caption, the document byte in front): every block's output, layer-normed without its affine; the final norm's output; and each hub's blackboard (the pollux bodies' blocks 0, 2 and 4; a dual hub's rotary and plain hubs apart; per book the positive minus the negative slot state, each slot unit length; a rotary hub's state turned back to the position as its own query reads it; its exact Gram coordinates, the leading 1,024 kept).
- stacks: the four bodies side by side at one version and block (4 x 256 = 1,024 numbers, Qwen's width); one body's four trained versions side by side; a dual hub's two hubs together.
- comparisons (the read's own code, unchanged): alignment with Qwen's final state (each side reduced to its 128 leading directions and whitened, rotated onto the other on half the captions, scored on the other half, both ways round; 1 = the same geometry up to a rotation, 0 = unrelated; called closeness below), context alignment (the same after removing each token's own mean: what a token carries beyond its identity), ridge R2, debiased CKA; the mood directions mapped into Qwen's frame by the caption rotation.
- the mood axis in the reader's own frame: each signal's state at a phrase's closing stop, standardized per feature over the 64 training phrases, placed on the line from the gloomy training centre (-1) to the cheerful one (+1); every training phrase read left out of its own fit (in-sample, any reader puts a training phrase near its centre, the axis being built through it).
- precision: fp32 passes, float64 comparisons (TF32 off). Card: one RTX 4090 (the 40 sets in 6 minutes; the comparisons on the CPU).

## Checks
- every checkpoint loads strictly; the untrained start is the trainer's own birth state;
- a byte's state read from the whole caption equals a pass over the caption up to that byte (largest difference {cz:.1e} over every block and the final norm);
- every hub's output recomputed from its blackboard equals the hub's own output (largest relative difference {par:.1e});
- the rows are the read's own; Beatrix's stored set re-scored by this code reproduces e038's closeness at block 24 to the digit ({ruler[0]:.6f} against {ruler[1]:.6f}).

## The rules (fixed before the run)
The bar for every verdict is the difference between the two caption draws, never below .01. LEARNED: a trained reader beats its own untrained start at the same signal by more than the bar. A continuation MOVED the caption states when its closeness differs from the end of the diet's by more than the larger bar. A stack ADDS when it beats its best member at the same block by more than the bar. A hub reads closer than the stream when it beats the stream at its block by more than the bar.

## Result

**Every body learned, and reads closest at its middle block.**

{t_bodies}

{t_ref}

The bodies reach Beatrix's block-8 level, about three quarters of her best. On this whitened closeness they sit just under the token lookup (.42), but with context the lookup cannot have (it has none by construction).

**The continuations did not move the caption states where they read best** ({held} of 12 held within the bar at the best block). Where anything moved, it was a late block, upward, by about the bar:

{t_moved}

**Every hub reads below the stream at its block**, and pollux-canon's first hub below its own untrained start:

{t_hubs}

**Side by side, the four bodies hold more than the best one, at every block; their untrained starts side by side hold nothing more.**

{t_stacks}

**No nano reader places mood phrases on its own axis better than its controls.** (Training phrases on their side, of 60, each left out; the unseen separation = the mean value of the 4 unseen cheerful phrases minus that of the 4 unseen gloomy ones, the training centres at +1 and -1.)

{t_mood}

Mapped into Qwen's frame by the caption rotation instead, castor's end-of-diet and plain-control states carry both sides at blocks 2-3 (3 of 4 unseen phrases a side on Qwen's own axis; the mood axis at cosine .27-.38 with Qwen's, its untrained start .01), and pollux-canon -fr at block 3: the mood is there along a direction borrowed from Qwen, not along the body's own axis.

**The last block's output grows a thousandfold and one direction takes about a third of it** (reported, not ruled on):

{t_size}

## What it means
1. Each body learned caption information that lines up with Qwen's, most at its middle block; the last blocks are dominated by one large direction (a thousandfold growth of the output's size, about a third of its variance on one direction) and read worse.
2. The continuations taught a skill on top of the bodies without moving what their middle blocks hold about a caption.
3. The hubs keep a memory the stream reads better than the hubs' own states do: on this measure the stream is the signal to use.
4. The four bodies were trained apart and differ in their attention; side by side they hold more of Qwen's caption geometry than any one of them (+.04-.05 at every block), above the token lookup and about a third of the way from the best body to Beatrix's best; their untrained starts side by side add nothing, so the extra is learned, complementary information.
5. A slider needs a reader's own mood axis; the nano bodies have none beyond their controls' level (Qwen 60 of 60, Beatrix 57, the nano bodies at most 48 against controls at 32-50), while the same phrases carried into Qwen's frame keep part of their mood: what they know of a mood lies along directions the caption rotation finds, not along one they can be read on alone.

## Limits
One training seed per body; one checkpoint per version; 512 captions a draw; linear comparisons only (a rotation, a ridge); the stacks are side-by-side concatenations, not trained combinations.

## Code
{CODE_URL}, `alephllm_diffusion.nano` (bodies, extract, compare, mood), branch feat/nano-battery.

## References
{_refs(READ_REFS)}

## Files

{{files}}
"""
    files = {"README.md": "this page", "meta.json": "the summary, the numbers on this page and the run's record",
             "result.json": "the comparisons' ledger: every signal of every reader and stack on both caption draws, the marks",
             "mood.json": "every reader's slider value of the 74 mood phrases at every signal (training phrases left out)",
             "extraction.json": "the extraction's record: per reader the checkpoint, the hub parity, the prefix check, time and memory",
             "last_block.json": "the last block's output size and its largest direction's share, per reader"}
    meta = {"title": title, "date": DATE, "kind": "nano_read", "status": "done",
            "recipe": {"models": MODELS_LINE, "rows": "the dual-extraction read's (e038)",
                       "code": f"{CODE_URL} (alephllm_diffusion.nano)"},
            "result": {"best": {b: [marks["bodies"][b]["end"]["best"], q(f"{b}|end|" + marks["bodies"][b]["end"]["best"])]
                                for b in bodies},
                       "stack4_best": [best_stack[1], best_stack[2], best_stack[0]],
                       "mood_left_out": {k: MOOD[k]["loo_train"] for k in ("qwen3|q-28post", "beatrix-trained|trunk-20",
                                                                            "stack4|fr|trunk-3")}},
            "summary": summary}
    copies = {"result.json": os.path.join(root, "compare", "nano_battery.json"),
              "mood.json": os.path.join(root, "compare", "nano_mood.json"),
              "extraction.json": os.path.join(root, "sets", "manifest.json"),
              "last_block.json": os.path.join(root, "compare", "s4_last_block.json")}
    return readme, meta, copies, files
