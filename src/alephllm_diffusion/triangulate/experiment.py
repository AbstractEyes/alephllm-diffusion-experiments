"""alephllm_diffusion.triangulate.experiment - the dual-extraction read as an experiment of the Anima experiments repo. Its
folder, experiments/<TRIANGULATION_ID>/ (README, meta.json, the read's ledger as result.json, its tables, the small run's ledger),
is written from the read's ledger into a local folder laid out like the repo, then published by the Anima trainer's step for
experiments written elsewhere (AnimaRunner.publish_local: the folder in one commit, then the repo README's index rebuilt from every
folder's meta.json). The id comes from the trainer's registry (anima_experiments.TRIANGULATION_ID), which keeps it from every
other experiment.

Usage: python -m alephllm_diffusion.triangulate.experiment <ledger.json> [--small=<the small run's ledger>] [--mirror=<folder>]
       [--code="..."] [--publish=1]
(--publish=1 needs HF_TOKEN with write access to the experiments repo; the rest runs offline.)"""
import json
import os
import shutil
import sys

from geolip_anima_trainer import anima_experiments as ax

EXPERIMENT_ID = ax.TRIANGULATION_ID
DATE = "2026-10-07"                                     # registered and run that day (local time; the ledger's clock is UTC)
TITLE =("Beatrix's trunk and hub, reading a caption's own bytes and Qwen3's spelling of its tokens, against the Qwen3 states "
         "Anima reads (a read of her states; no pictures)")
KIND = "beatrix_read"
CODE_URL = "https://github.com/AbstractEyes/alephllm-diffusion-experiments"
FILES = {"README.md": "this page",
         "meta.json": "the decisions, the numbers on this page and the run's record",
         "result.json": "the read's ledger: every comparison on both caption draws, the decisions",
         "tables.md": "the read's own tables",
         "small_run.json": "the small run before the full read (48 captions a draw, two blocks, 16 dimensions; its numbers are a "
                           "rehearsal of the code, not results)"}
CITATIONS = [
    ("Schönemann 1966", "A generalized solution of the orthogonal Procrustes problem. Psychometrika 31(1):1-10",
     "https://doi.org/10.1007/BF02289451"),
    ("Mikolov, Le, Sutskever 2013", "Exploiting Similarities among Languages for Machine Translation",
     "https://arxiv.org/abs/1309.4168"),
    ("Smith, Turban, Hamblin, Hammerla 2017", "Offline bilingual word vectors, orthogonal transformations and the inverted "
     "softmax. ICLR 2017", "https://arxiv.org/abs/1702.03859"),
    ("Artetxe, Labaka, Agirre 2018", "Generalizing and Improving Bilingual Word Embedding Mappings with a Multi-Step Framework "
     "of Linear Transformations. AAAI-18", "https://doi.org/10.1609/aaai.v32i1.11992"),
    ("Conneau, Lample, Ranzato, Denoyer, Jégou 2018", "Word Translation Without Parallel Data. ICLR 2018",
     "https://arxiv.org/abs/1710.04087"),
    ("Raghu, Gilmer, Yosinski, Sohl-Dickstein 2017", "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning "
     "Dynamics and Interpretability. NeurIPS 2017", "https://arxiv.org/abs/1706.05806"),
    ("Kornblith, Norouzi, Lee, Hinton 2019", "Similarity of Neural Network Representations Revisited. ICML 2019",
     "https://arxiv.org/abs/1905.00414"),
    ("Maiorca, Moschella, Norelli, Fumero, Locatello, Rodolà 2023", "Latent Space Translation via Semantic Alignment. "
     "NeurIPS 2023", "https://arxiv.org/abs/2311.00664"),
    ("Moschella et al. 2023", "Relative representations enable zero-shot latent space communication. ICLR 2023",
     "https://arxiv.org/abs/2209.15430"),
    ("Huh, Cheung, Wang, Isola 2024", "Position: The Platonic Representation Hypothesis. ICML 2024",
     "https://arxiv.org/abs/2405.07987"),
    ("Minixhofer, Ponti, Vulić 2024", "Zero-Shot Tokenizer Transfer. NeurIPS 2024", "https://arxiv.org/abs/2405.07883"),
    ("Ethayarajh 2019", "How Contextual are Contextualized Word Representations? EMNLP-IJCNLP 2019",
     "https://arxiv.org/abs/1909.00512"),
    ("Timkey, van Schijndel 2021", "All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure "
     "Representational Quality. EMNLP 2021", "https://arxiv.org/abs/2109.04404"),
    ("Sun, Chen, Kolter, Liu 2024", "Massive Activations in Large Language Models. COLM 2024",
     "https://arxiv.org/abs/2402.17762"),
    ("Xiao, Tian, Chen, Han, Lewis 2024", "Efficient Streaming Language Models with Attention Sinks. ICLR 2024",
     "https://arxiv.org/abs/2309.17453"),
    ("Radford et al. 2019", "Language Models are Unsupervised Multitask Learners (byte-level BPE); the byte-to-character map "
     "is in the released code, src/encoder.py", "https://github.com/openai/gpt-2"),
    ("Bansal, Nakkiran, Barak 2021", "Revisiting Model Stitching to Compare Neural Representations. NeurIPS 2021",
     "https://arxiv.org/abs/2106.07682"),
    ("Phan, Amos, Gat, Havasi, Muckley, Ullrich 2025", "Exact Byte-Level Probabilities from Tokenized Language Models for "
     "FIM-Tasks and Model Ensembles. ICLR 2025", "https://arxiv.org/abs/2410.09303"),
    ("Minixhofer, Vulić, Ponti 2025", "Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching. NeurIPS 2025",
     "https://arxiv.org/abs/2503.20083"),
    ("Qwen Team 2025", "Qwen3 Technical Report", "https://arxiv.org/abs/2505.09388"),
]


def words(name: str) -> str:
    """'A-hub-22' -> "the hub's blackboard at block 22, reading the caption's own bytes"."""
    form, kind, block = name.split("-")
    what = "her stream (the block's output)" if kind == "trunk" else "the hub's blackboard"
    how = "the caption's own bytes" if form == "A" else "Qwen's spelling of the tokens"
    return f"{what} at block {block}, reading {how}"


def _q(C1, trunk, name, col, target="Q"):
    v = C1.get(f"{trunk}|{name}", {}).get(target, {}).get(col)
    return "-" if v is None else f"{v:.3f}"


def _order(name: str):
    """'A-trunk-8' sorts before 'A-trunk-12': form, kind, block number."""
    form, kind, block = name.split("-")
    return form, kind, int(block)


def second_draw(L) -> dict:
    """The decisions again with the second caption draw as the read (its untrained rows taken from draw 1, the only draw that
    holds them) and the first draw as the bar: how many verdicts repeat, the pick, the readings carrying the gloomy side, and
    the largest draw-to-draw gaps."""
    from .read import decide
    d1, d2, D = L["draw1"], L["draw2"], L["decisions"]
    R2 = dict(d2)
    for c in ("C1", "C6"):
        R2[c] = {**d2[c], **{k: v for k, v in d1[c].items() if k.startswith("untrained|")}}
    dd = decide(R2, d1)
    out = {x: [sum(dd[x][k]["verdict"] == D[x][k]["verdict"] for k in D[x] if k in dd[x]), len(D[x])] for x in ("D1", "D2", "D3")}
    out["D4_open"] = [sum(v["verdict"] == "OPEN" for v in dd["D4"].values()), len(dd["D4"])]
    out["pick"] = dd["PICK"].get("signal")
    out["gloomy"] = sorted((n for n, v in dd["D6"].items() if v["down"]["carries"]), key=_order)
    ax_gap = [abs(d1["C6"][k]["cos_axis"] - d2["C6"][k]["cos_axis"]) for k in d2["C6"]]
    out["axis_gap"] = [sum(ax_gap) / len(ax_gap), max(ax_gap)]
    out["alignment_gap"] = max(abs(d1["C1"][k]["Q"]["alignment"] - d2["C1"][k]["Q"]["alignment"]) for k in d2["C1"])
    return out


def findings(L) -> list:
    """The result in plain sentences, every number from the ledger (draw 1; the second draw is the bar)."""
    d, C1, C2, fl = L["decisions"], L["draw1"]["C1"], L["draw1"]["C2"], L["draw1"]["floors"]
    trunk = sorted((k.split("|")[1] for k in C1 if k.startswith("trained|A-trunk-")), key=_order)

    def al(n, target="Q"):
        return C1[f"trained|{n}"][target]["alignment"]

    def blk(n):
        return n.split("-")[2]

    out = []
    d1 = [v["verdict"] for v in d["D1"].values()]
    out.append(f"Her own bytes, not Qwen's spelling: {d1.count('HER OWN BYTES CLOSER')} of the {len(d1)} extraction verdicts "
               f"(stream and blackboard, whitened and not, plain and context) say her own bytes sit closer to Qwen's states.")
    d2 = [v["verdict"] for v in d["D2"].values()]
    out.append(f"Her stream, not her hub: {d2.count('TRUNK CLOSER')} of the {len(d2)} signal verdicts say the stream sits closer.")
    best = max(trunk, key=al)
    out.append(f"Depth: on her own bytes the stream's alignment with Qwen's final state rises from {al(trunk[0]):.3f} at block "
               f"{blk(trunk[0])} to {al(best):.3f} at block {blk(best)}; Qwen's own layer 16 reaches "
               f"{fl['Q16->Q']['alignment']:.3f} against its final state, and the token's identity alone (Qwen's input "
               f"embedding) {fl['Q0->Q']['alignment']:.3f}.")
    b0, b16 = max(trunk, key=lambda n: al(n, "Q0")), max(trunk, key=lambda n: al(n, "Q16"))
    out.append(f"Her depths line up with Qwen's: her block {blk(b0)} matches Qwen's input embedding best ({al(b0, 'Q0'):.3f}), "
               f"her block {blk(b16)} Qwen's layer 16 ({al(b16, 'Q16'):.3f}), her block {blk(best)} Qwen's final state.")
    c2 = sorted((k.split("|")[1] for k in C2 if k.startswith("trained|trunk-")), key=lambda n: int(n.split("-")[1]))
    if c2:
        un = [C2[f"untrained|{n}"]["alignment"] for n in c2 if f"untrained|{n}" in C2]
        out.append(f"Her two readings of one caption part with depth: her own bytes against Qwen's spelling align "
                   f"{C2['trained|' + c2[0]]['alignment']:.3f} at block {c2[0].split('-')[1]} and "
                   f"{C2['trained|' + c2[-1]]['alignment']:.3f} at block {c2[-1].split('-')[1]}, where the untrained copy reads "
                   f"the two as nearly one text ({min(un):.3f}-{max(un):.3f}).")
    gaps = [v["gap"] for v in d["D4"].values()]
    out.append(f"The triangle stays open on {sum(v['verdict'] == 'OPEN' for v in d['D4'].values())} of {len(gaps)} paths: "
               f"going through the other byte form loses {min(gaps):.3f}-{max(gaps):.3f} of the direct alignment.")
    for k, v in d["D3"].items():
        base, top = v.get("solo", v.get("best_pair")), v.get("pair", v.get("four"))
        out.append(f"The pair ({k.split('|')[1]}): ridge R2 {base:.3f} -> {top:.3f}: {v['verdict'].lower()}.")
    carriers = sorted((n for n, v in d["D6"].items() if v["down"]["carries"]), key=_order)
    lost = sorted((n for n, v in d["D6"].items() if n.startswith("A-trunk-") and not v["down"]["carries"]), key=_order)
    if carriers:
        axes = [d["D6"][n]["axis"]["cos"] for n in carriers]
        out.append(f"The gloomy side: {len(carriers)} readings carry it into Qwen's frame (their mood axis on Qwen's at "
                   f"{min(axes):+.3f} to {max(axes):+.3f}; " + ", ".join(f"`{n}`" for n in carriers) + "); "
                   + ("the stream readings on her own bytes that do not: " + ", ".join(f"`{n}`" for n in lost) + "; "
                      if lost else "")
                   + f"Qwen's spelling carries it in {sum(n.startswith('B-') for n in carriers)}.")
    else:
        out.append("The gloomy side: no reading carries it into Qwen's frame.")
    c7 = L["draw1"]["C7"]
    if "trained|A" in c7:
        t, u = c7["trained|A"], c7.get("untrained|A", {})
        out.append(f"Her final block's size follows Qwen's residual size only weakly (rank correlation "
                   f"{t['size_vs_qwen_size']:+.3f}; untrained {u.get('size_vs_qwen_size', float('nan')):+.3f}) and does not "
                   f"tell which rows align ({t.get('size_vs_row_alignment', float('nan')):+.3f}).")
    return out


def summary(L) -> str:
    d = L["decisions"]
    a_t, a_h = d["D1"]["trunk|alignment"], d["D1"]["hub|alignment"]
    pick = d["PICK"]
    carried = [n for n, v in d["D6"].items() if v["down"]["carries"]]
    head = (f"her own bytes against Qwen's spelling: trunk {a_t['verdict']} ({a_t['A'][1]:.2f} against {a_t['B'][1]:.2f}), "
            f"hub {a_h['verdict']} ({a_h['A'][1]:.2f} against {a_h['B'][1]:.2f}); trunk against hub on her own bytes: "
            f"{d['D2']['A|alignment']['verdict']}")
    tail = (f"; the gloomy side carried into Qwen's frame by {len(carried)} of {len(d['D6'])} readings; the pick: "
            + (words(pick["signal"]) if pick.get("signal") else pick.get("verdict", "none")))
    return head + tail


def recipe(L, code: "str | None" = None) -> dict:
    m = L["_meta"]
    if code is None:
        code = f"{CODE_URL}, alephllm_diffusion.triangulate" + (f" at commit {m['commit'][:7]}" if m.get("commit") else "")
    rows = m["rows"]
    return {
        "model": f"Beatrix ({ax.HUB_TRUNK}); an untrained trunk of the same shape (random initialisation, seed 0) as the "
                 "control that must fail",
        "target": "Qwen3 0.6B (Anima's text encoder file), its final hidden state at each token: what Anima's text adapter "
                  "reads; beside it the input to its layer 16 and its input embedding (the token's identity alone)",
        "rows": f"the first {m['n_caps']} COCO captions of each of two draws ({rows['draw1']['captions']:,} and "
                f"{rows['draw2']['captions']:,} token rows: every Qwen token but a caption's first and last), and the "
                f"{rows['draw1']['mood']} mood phrases of the earlier slider experiments in the grid's caption form (one row "
                "per phrase, its last token)",
        "her signals": f"per Qwen token, at the byte after it: the stream at blocks {', '.join(map(str, m['stream_blocks']))} "
                       f"(layer-normed without its affine) and the hub's blackboard at blocks "
                       f"{', '.join(map(str, m['hub_blocks']))} (4 x 64 slots of 1,024 numbers; its exact Gram coordinates, "
                       f"the leading {m['r_hub']:,} kept), both from one pass over the text up to that byte",
        "preparation": f"each signal reduced to its {m['k']} leading directions on the fit captions, whitened (and, beside it, "
                       "unwhitened); every rotation and ridge fit on half the captions and scored on the other half, both "
                       "ways round",
        "precision": "fp32 for the passes, float64 for the comparisons (TF32 off)",
        "card": "one RTX 4090",
        "code": code,
    }


def _example_table() -> list:
    return ["| | what she reads | bytes | the token ' taco' closes at |", "|---|---|---|---|",
            "| A, the caption's own bytes | `a taco on a plate.` | 18 | the space before ' on' |",
            "| B, Qwen's spelling | `aĠtacoĠonĠaĠplate.` | 22 | the first byte of 'Ġon' |",
            "| Qwen | its 6 token ids: a, Ġtaco, Ġon, Ġa, Ġplate, . | - | its own state at 'Ġtaco' |"]


def readme(L, code: "str | None" = None) -> str:
    m, d = L["_meta"], L["decisions"]
    C1 = L["draw1"]["C1"]
    rec = recipe(L, code)
    out = [f"# {EXPERIMENT_ID}: {TITLE}", "",
           f"Date: {DATE}. Model: Beatrix, a byte-level language model "
           f"({ax.HUB_TRUNK}). No picture is made: this experiment reads her states against the text states Anima reads.", "",
           "## Question",
           "Anima's text adapter reads Qwen3's hidden states, one per Qwen token. Beatrix reads bytes, and carries two signals at "
           "every block: her stream (the block's output) and her hub's blackboard (the block's memory of the text so far). "
           "Read per Qwen token, which of her signals sits closest to the states Anima reads: when she reads the caption's own "
           "bytes (A) or Qwen's own spelling of its tokens (B)? Does the second signal or the second form add anything? Do the "
           "three close a triangle (A to B to Qwen against A to Qwen)? And do her mood directions, the cheerful and the gloomy "
           "side, land on Qwen's? The gloomy side is the question the slider experiments left open (e031-e037: the cheerful "
           "side moved, the gloomy side did not).", "",
           "## The two byte forms", "",
           "Qwen3's vocabulary stores every byte as a printable stand-in character (the GPT-2 byte map): a space is 'Ġ' (two "
           "bytes in UTF-8), so ' taco' is the one token 'Ġtaco', six bytes against five. For ASCII text the spelling differs "
           "from the caption's bytes only at the spaces; the two forms are tied by a byte-exact round trip.", ""]
    out += _example_table()
    out += ["", "## Design", ""] + [f"- {k}: {v}" for k, v in rec.items()]
    out += ["", "The comparisons:", "", "| | comparison | what it answers |", "|---|---|---|",
            "| C1 | each of her signals against Qwen's final states: alignment (the cosine of the angle between her rotated rows "
            "and Qwen's on unseen captions: 1 the same geometry up to a rotation, 0 unrelated), context alignment (the same "
            "after removing each token's own mean: the variation beyond the token's identity), unwhitened twins, ridge R2, "
            "debiased CKA | which reading sits closest to what Anima reads |",
            "| C2, C3 | A against B; stream against blackboard | how far apart her forms and her signals are |",
            "| C4 | ridge to Qwen from stream and blackboard together, then all four | whether the second signal adds |",
            "| C5 | A to B to Qwen (each leg fit on its own captions) against A to Qwen | whether the triangle closes |",
            "| C6 | her mood directions mapped into Qwen's frame by the rotation fit on the captions: the mood axis (cheerful "
            "minus gloomy), the part both sides share, each side; the unseen phrases read from her mapped centre along "
            "Qwen's axis | whether her gloomy direction lands on Qwen's |",
            "| C7 | the size of her last block's output against Qwen's residual size and against each row's alignment | "
            "whether her confidence follows Qwen's |"]
    out += ["", "## The rules (fixed before the run; one amendment before the full read)",
            "The bar for every verdict is the difference between the two caption draws on her trained trunk, never below .01. "
            "D1, the extraction: per signal kind, the best block under Qwen's spelling against under her own bytes. D2, the "
            "signal: per byte form, the best blackboard against the best stream. D3: the pair adds when its ridge R2 beats the "
            "better single signal by more than the bar. D4: the triangle closes when the composed path is within the bar of "
            "the direct one. D5: a reading whose alignment beats its untrained copy by no more than the bar is not learned. D6: "
            "a reading carries a side into Qwen's frame when its mapped mood axis beats the untrained copy's cosine by more "
            "than .10 and at least 3 of the 4 unseen phrases of that side land on that side. THE PICK: among the learned "
            "readings that carry the gloomy side, the largest margin over the untrained copy in context alignment. The "
            "whitened numbers decide; the unwhitened verdicts are shown beside them and decide nothing.",
            "", "The amendment (before the full read, after the code's tests and a small run): the triangle's two legs were "
            "first fit on the same captions, which closes any triangle by construction (now each leg has its own captions); "
            "the unseen phrases were first read from Qwen's centre, which a rotation cannot carry (now from her own mapped "
            "centre); Qwen's two sides share most of their direction from neutral, so the side rule moved to the mood axis. "
            "Three cross-checks were added from the literature (the unwhitened twins, CKA, outlier counts).", ""]
    # ---- result
    s2 = second_draw(L)
    first = set(n for n, v in d["D6"].items() if v["down"]["carries"])
    extra, gone = [n for n in s2["gloomy"] if n not in first], sorted(first - set(s2["gloomy"]), key=_order)
    out += ["## Result", "",
            f"In short (caption draw 1). The second draw, read the same way, repeats the extraction verdicts {s2['D1'][0]} of "
            f"{s2['D1'][1]}, the signal verdicts {s2['D2'][0]} of {s2['D2'][1]}, the pair verdicts {s2['D3'][0]} of "
            f"{s2['D3'][1]}, leaves the triangle open on {s2['D4_open'][0]} of {s2['D4_open'][1]} paths and makes the same "
            f"pick ({'yes' if s2['pick'] == d['PICK'].get('signal') else 'no: ' + str(s2['pick'])}); its alignments differ by "
            f"at most {s2['alignment_gap']:.3f}. The direction reads move more between draws (the mood axis cosine by "
            f"{s2['axis_gap'][0]:.2f} on average, {s2['axis_gap'][1]:.2f} at most)"
            + (f"; on the second draw the gloomy side is also carried by " + ", ".join(f"`{n}`" for n in extra) if extra else "")
            + (f"; and lost by " + ", ".join(f"`{n}`" for n in gone) if gone else "") + ".", ""]
    out += [f"- {s}" for s in findings(L)]
    out += ["", "The verdicts:", ""]
    out += [f"- **The extraction (D1)**:{k.replace('|', ', ').replace('_', ' ')}: her own bytes {v['A'][1]:.3f} "
            f"({v['A'][0].split('|')[1]}), Qwen's spelling {v['B'][1]:.3f} ({v['B'][0].split('|')[1]}), bar {v['bar']:.3f}: "
            f"**{v['verdict']}**" for k, v in d["D1"].items()]
    out += [f"- **The signal (D2)**: {k.replace('|', ', ').replace('_', ' ')}: stream {v['trunk'][1]:.3f} "
            f"({v['trunk'][0].split('|')[1]}), blackboard {v['hub'][1]:.3f} ({v['hub'][0].split('|')[1]}), bar "
            f"{v['bar']:.3f}: **{v['verdict']}**" for k, v in d["D2"].items()]
    for k, v in d["D3"].items():
        base, top = v.get("solo", v.get("best_pair")), v.get("pair", v.get("four"))
        out.append(f"- **The pair (D3)**, {k.split('|')[1]}: ridge R2 {base:.3f} -> {top:.3f} (gain {v['gain']:+.3f}, bar "
                   f"{v['bar']:.3f}): **{v['verdict']}**")
    pick = d["PICK"]
    out.append("- **THE PICK**: " + (f"{words(pick['signal'])} (`{pick['signal']}`; alignment {pick['alignment']:.3f}, "
                                      f"context margin over the untrained copy {pick['context_margin']:+.3f})"
                                      if pick.get("signal") else f"{pick.get('verdict')}"))
    out += ["", "### C1: her readings against Qwen's final states (held-out captions; draw 1)", "",
            "| reading | alignment | untrained | context alignment | untrained | unwhitened | ridge R2 | CKA | vs layer 16 | "
            "vs input |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in sorted((k.split("|", 1)[1] for k in C1 if k.startswith("trained|")), key=_order):
        out.append(f"| `{n}` | {_q(C1, 'trained', n, 'alignment')} | {_q(C1, 'untrained', n, 'alignment')} | "
                   f"{_q(C1, 'trained', n, 'context_alignment')} | {_q(C1, 'untrained', n, 'context_alignment')} | "
                   f"{_q(C1, 'trained', n, 'alignment_unwhitened')} | {_q(C1, 'trained', n, 'ridge_r2')} | "
                   f"{_q(C1, 'trained', n, 'cka_debiased')} | {_q(C1, 'trained', n, 'alignment', 'Q16')} | "
                   f"{_q(C1, 'trained', n, 'alignment', 'Q0')} |")
    fl = L["draw1"]["floors"]
    out += ["", f"Qwen's own floors: its input embedding (the token's identity alone) against its final state: alignment "
                f"{fl['Q0->Q']['alignment']:.3f} (context {fl['Q0->Q']['context_alignment']:.3f}); its layer 16 against its "
                f"final state: {fl['Q16->Q']['alignment']:.3f} (context {fl['Q16->Q']['context_alignment']:.3f}).", ""]
    C2 = L["draw1"]["C2"]
    out += ["### C2: her own bytes against Qwen's spelling, read by her and by the untrained copy (held-out alignment)", "",
            "| signal | hers | context | untrained |", "|---|---|---|---|"]
    for k in sorted((k.split("|", 1)[1] for k in C2 if k.startswith("trained|")),
                    key=lambda s: (s.split("-")[0], int(s.split("-")[1]))):
        u = C2.get(f"untrained|{k}", {})
        out.append(f"| {k.replace('-', ' at block ')} | {C2['trained|' + k]['alignment']:.3f} | "
                   f"{C2['trained|' + k]['context_alignment']:.3f} | {u.get('alignment', float('nan')):.3f} |")
    out += ["", "### C6: the mood directions in Qwen's frame (untrained in brackets)", "",
            "| reading | learned | mood axis cos | shared part cos | cheerful unseen | gloomy unseen | carries cheerful / gloomy |",
            "|---|---|---|---|---|---|---|"]
    for n, v in sorted(d["D6"].items(), key=lambda kv: _order(kv[0])):
        out.append(f"| `{n}` | {d['D5'][n]['verdict']} | {v['axis']['cos']:+.3f} ({v['axis']['cos_untrained']:+.3f}) | "
                   f"{v['shared']['cos']:+.3f} ({v['shared']['cos_untrained']:+.3f}) | {v['up']['held_out'][0]}/"
                   f"{v['up']['held_out'][1]} | {v['down']['held_out'][0]}/{v['down']['held_out'][1]} | "
                   f"{'yes' if v['up']['carries'] else 'no'} / {'yes' if v['down']['carries'] else 'no'} |")
    q6 = next(iter(L["draw1"]["C6"].values()))
    out += ["", f"Qwen's own two sides share a cosine of {q6['q_sides_cos']:+.3f} (each measured from the neutral phrases); "
                f"its own unseen phrases on its own axis: cheerful {q6['q_own_up'][0]}/{q6['q_own_up'][1]}, gloomy "
                f"{q6['q_own_down'][0]}/{q6['q_own_down'][1]}.", ""]
    out += ["### C5: the triangle (her trained trunk; held-out alignment)", "", "| path | direct | composed | gap | verdict |",
            "|---|---|---|---|---|"]
    for k, v in d["D4"].items():
        out.append(f"| {k.split('|', 1)[1].replace('|', ', ')} | {v['direct']:.3f} | {v['composed']:.3f} | {v['gap']:+.3f} | "
                   f"{v['verdict']} |")
    out += ["", "### C7: her final block's size (descriptive; rank correlations over the caption rows)", "",
            "| trunk, byte form | against Qwen's residual size | against the row's alignment (her best stream reading) |",
            "|---|---|---|"]
    for k, v in L["draw1"]["C7"].items():
        r = v.get("size_vs_row_alignment")
        out.append(f"| {k.replace('|', ', ')} | {v['size_vs_qwen_size']:+.3f} | "
                   + ("-" if r is None else f"{r:+.3f} (`{v.get('signal')}`)") + " |")
    an = m.get("anchor") or {}
    n_out = sum((m.get("outliers") or {}).values())
    out += ["", "### Checks",
            f"- The pass over each token's prefix reproduces the full caption's state at that byte to {an.get('max_abs_diff', 0):.1e} "
            f"(block {an.get('block')}, {an.get('rows')} rows).",
            "- Rows more than 10 times the median size: " + ("none in any set." if n_out == 0 else f"{n_out} (result.json)."),
            "- The blackboard's kept Gram coordinates hold "
            + ", ".join(f"{min(v.values()):.3f}-{max(v.values()):.3f}" for v in list((m.get("shares") or {}).values())[:1])
            + " of its variance (the first set; every set in result.json)."]
    out += ["", "## References (checked against their primary pages)", ""]
    out += [f"- {a}. {t}. {u}" for a, t, u in CITATIONS]
    out += ["", "## Files", "", "| file | what |", "|---|---|"] + [f"| `{k}` | {v} |" for k, v in FILES.items()] + [""]
    return "\n".join(out)


def meta(L, code: "str | None" = None) -> dict:
    m, d = L["_meta"], L["decisions"]
    C1 = L["draw1"]["C1"]
    keep = ("alignment", "context_alignment", "alignment_unwhitened", "context_alignment_unwhitened", "ridge_r2", "cka_debiased")
    out = {"id": EXPERIMENT_ID, "title": TITLE, "date": DATE, "kind": KIND,
           "status": "done", "recipe": recipe(L, code),
           "result": {"decisions": d,
                      "c1": {k: {c: v["Q"].get(c) for c in keep} for k, v in C1.items()},
                      "floors": L["draw1"]["floors"], "anchor": m.get("anchor"), "outliers": m.get("outliers"),
                      "second_draw": second_draw(L)},
           "summary": summary(L)}
    if m.get("seconds") is not None:
        out["seconds"] = int(m["seconds"])
    if m.get("finished_utc"):
        out["finished_utc"] = m["finished_utc"]
    return out


def write(ledger: str, mirror: str, small: "str | None" = None, code: "str | None" = None) -> dict:
    """<mirror>/experiments/<id>/: README.md, meta.json, result.json (the ledger), tables.md and small_run.json; returns the
    meta."""
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
