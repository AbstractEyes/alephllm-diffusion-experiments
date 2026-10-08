"""alephllm_diffusion.hubs.experiment - the hub read as an experiment of the Anima experiments repo. Its folder,
experiments/<HUB_READ_ID>/ (README, meta.json, the read's ledger as result.json, its tables, the small run's ledger), is written
from the read's ledger into a local folder laid out like the repo, then published by the Anima trainer's step for experiments
written elsewhere (AnimaRunner.publish_local: the folder in one commit, then the repo README's index rebuilt from every folder's
meta.json). The id comes from the trainer's registry (anima_experiments.HUB_READ_ID), which keeps it from every other
experiment; the hub slider arms it chose name it in their own READMEs.

Usage: python -m alephllm_diffusion.hubs.experiment <ledger.json> [--small=<the small run's ledger>] [--mirror=<folder>]
       [--seconds=N] [--finished="YYYY-MM-DD HH:MM"] [--code="..."] [--publish=1]
(--publish=1 needs HF_TOKEN with write access to the experiments repo; the rest runs offline.)"""
import json
import os
import shutil
import statistics
import sys

from geolip_anima_trainer import anima_experiments as ax

EXPERIMENT_ID = ax.HUB_READ_ID
TITLE = "Beatrix's hubs against her stream as features for the sliders (a read of her states; no pictures)"
KIND = "beatrix_read"
CODE_URL = "https://github.com/AbstractEyes/alephllm-diffusion-experiments"
READINGS = ("close/stream/18", "close/hub/22", "close/both/18")   # the three readings the hub sliders (e031-e037) use
CONDITIONS = (("M0", "bare"), ("M9_A", "nine arms A"), ("M9_B", "nine arms B"), ("M8_A", "eight arms A"),
              ("M8_B", "eight arms B"), ("Mc_A", "caption arm A"), ("Mc_B", "caption arm B"), ("U0", "untrained"),
              ("A212", "step 212,000"))
CONDITION_TEXT = {
    "M0": "her final checkpoint, bare",
    "M0_draw2": "the same on a second draw of 2,048 captions (part A only: the noise bar)",
    "M9_A / M9_B": "her nine trained arms mounted, all live (eight stage arms and the caption arm); two training seeds of the "
                   "arm group",
    "M8_A / M8_B": "the eight stage arms, the caption arm masked",
    "Mc_A / Mc_B": "the caption arm alone",
    "U0": "an untrained trunk of the same shape (random initialisation, seed 0): a control that must fail",
    "A212": "her checkpoint at step 212,000, the trunk of the earlier slider experiments (part B only)",
}
FILES = {"README.md": "this page",
         "meta.json": "the decision, the numbers on this page and the run's record",
         "result.json": "the read's ledger: every condition's part A and part B, the floor, the rulers, the decision",
         "tables.md": "every table of the read",
         "small_run.json": "the small run before the full read (256 captions, fewer blocks, one arm seed; its numbers are a "
                           "rehearsal, except the check against the stored features)"}


def words(key: str) -> str:
    """A reading's name in words: form/kind/block."""
    form, kind, block = key.split("/")
    where = {"last": "after the phrase alone", "close": "at the phrase's closing full stop",
             "frame": "inside a caption"}[form]
    what = {"stream": "her stream", "hub": "the hub's blackboard", "both": "the blackboard and the stream together"}[kind]
    at = "blocks 16, 18, 21 and 24 together" if block == "rec" else f"block {block}"
    return f"{what} {where}, {at}"


def _cell(r) -> str:
    return "-" if r is None else (f"{r['separation']:+.2f}; {r['loo_signs'][0]}; {r['heldout_signs'][0]}; "
                                  f"{r['balance']:.2f}")


def _blocks(L):
    return [b for b in L["_meta"]["sweep"] if f"bb@L{b}" in L["A"]["M0"] and f"pool@L{b}" in L["A"]["M0"]]


def _a(L, cond, fam, b, ref, field):
    try:
        return L["A"][cond][f"{fam}@L{b}"][ref][field]
    except KeyError:
        return None


def noise_bar(L) -> dict:
    """{gauge: (mean, largest)} of |draw 1 - draw 2| over every tap and reference both draws hold."""
    A1, A2 = L["A"]["M0"], L["A"].get("M0_draw2", {})
    out = {}
    for field in ("overlap10", "spearman", "cka_deb", "r2_rep2ref"):
        gaps = [abs(A1[k][r][field] - A2[k][r][field]) for k in A1 if k in A2 for r in A1[k]
                if r != "pr" and field in A1[k][r] and field in A2[k].get(r, {})]
        if gaps:
            out[field] = (statistics.mean(gaps), max(gaps))
    return out


def part_a_facts(L) -> dict:
    """The bare trunk's blackboard against its pooled stream: per-gauge block leads, the best block, the ridge ranges."""
    bl = _blocks(L)
    lead = {g: sum(_a(L, "M0", "bb", b, ref, f) > _a(L, "M0", "pool", b, ref, f) for b in bl)
            for g, ref, f in (("clip", "clip_b32", "overlap10"), ("ridge", "adapter_pool", "r2_rep2ref"),
                              ("cka", "adapter_pool", "cka_deb"))}
    best = max(bl, key=lambda b: _a(L, "M0", "bb", b, "clip_b32", "overlap10"))
    r2 = {fam: [_a(L, "M0", fam, b, "adapter_pool", "r2_rep2ref") for b in bl] for fam in ("bb", "pool")}
    return {"blocks": bl, "lead": lead, "best_block": best,
            "best": {fam: (_a(L, "M0", fam, best, "clip_b32", "overlap10"), _a(L, "M0", fam, best, "clip_b32", "spearman"))
                     for fam in ("bb", "pool", "last")},
            "r2_range": {fam: (min(v), max(v)) for fam, v in r2.items() if v and None not in v}}


def summary(L) -> str:
    d, f = L["decision"], part_a_facts(L)
    if d.get("chosen") is None:
        head = f"{d['verdict']}: no reading met the slider read's bar"
    else:
        c, h, s = d["chosen"], d.get("best_hub"), d.get("best_stream")
        other = h if c["kind"] == "stream" else s
        head = f"{d['verdict']}: {words(c['key'])} (separation {c['sep']:.2f}"
        if other is not None:
            head += (f"; the best {'hub' if other['kind'] == 'hub' else 'stream'} reading, {words(other['key'])}, "
                     f"{other['sep']:.2f}, balance {other['bal']:.2f} against {c['bal']:.2f}")
        head += ")"
    bb, pool = f["best"]["bb"], f["best"]["pool"]
    tail = (f"; the blackboards lead on caption geometry (CLIP-B/32 overlap {bb[0]:.2f} against the pooled stream's "
            f"{pool[0]:.2f} at block {f['best_block']}")
    if {"bb", "pool"} <= set(f["r2_range"]):
        rb, rp = f["r2_range"]["bb"], f["r2_range"]["pool"]
        tail += f"; ridge to Anima's caption context R2 {rb[0]:.2f}-{rb[1]:.2f} against {rp[0]:.2f}-{rp[1]:.2f}"
    return head + tail + ")"


def recipe(L, code: "str | None" = None) -> dict:
    m = L["_meta"]
    if code is None:
        code = f"{CODE_URL}, alephllm_diffusion.hubs.read" + (f" at commit {m['commit'][:7]}" if m.get("commit") else "")
    return {
        "model": f"Beatrix ({ax.HUB_TRUNK}); her step 212,000 checkpoint (the earlier sliders' trunk); an untrained trunk "
                 "of the same shape (random initialisation, seed 0)",
        "arms": "her nine trained arms (eight stage arms and the caption arm; two training seeds of the arm group): all nine "
                "live, the eight alone, the caption arm alone",
        "part A": f"{m['n_captions']:,} COCO captions (a second draw of {m['n_captions']:,} for the noise bar); per block "
                  f"{', '.join(map(str, m['sweep']))}: the hub's blackboard (4 x 64 slots of 1,024 numbers, each slot unit "
                  "length), the stream pooled over the caption, the stream at its last byte",
        "part A gauges": "against CLIP ViT-B/32 (the earlier conditioning probes' ruler): the mean share of each caption's 10 "
                         "nearest captions (by cosine) that the ruler also puts among its 10 nearest, and the Spearman "
                         "correlation of the cosine similarities over 100,000 caption pairs; against Anima's text adapter "
                         "output pooled over the caption (the context a slider pushes), T5-XXL and Qwen3 0.6B pooled: "
                         "debiased CKA and a ridge's R2 predicting the reference",
        "part B": "the earlier slider experiments' 74 phrases (per mood 6 phrases and 24 words for training, 4 phrases held "
                  "out; 4 neutral phrases for training, 2 held out), read alone, closed by a full stop and inside a caption; "
                  "per block, each block's output layer-normed without its affine, standardized per feature over 64 "
                  "reference rows that hold no word of a held-out phrase",
        "part B instruments": "the slider's axis from the gloomy to the cheerful training phrases' centres (at -1 and +1); "
                              "leave-one-out signs of the 60 mood training phrases; the 8 held-out mood phrases' signs; the "
                              "separation (the held-out cheerful phrases' mean slider value minus the gloomy ones'); the "
                              "balance |log(sd_up / sd_down)| of the two moods' spreads on the axis (0 is even)",
        "precision": "fp32 throughout (TF32 off)",
        "card": "one RTX 4090",
        "code": code,
    }


def readme(L, code: "str | None" = None) -> str:
    m, d, B, f, nb = L["_meta"], L["decision"], L["B"], part_a_facts(L), noise_bar(L)
    rec = recipe(L, code)
    conds = [(k, lab) for k, lab in CONDITIONS if k in B]
    out = [f"# {EXPERIMENT_ID}: {TITLE}", "",
           f"Date: {m['made'][:10]}. Model: Beatrix, a byte-level language model ({ax.HUB_TRUNK}). No picture is made: this "
           "experiment reads her states, and its decision chose the features of the hub sliders (listed at the end).", "",
           "## Question",
           "Each block of Beatrix carries a hub whose blackboard holds the block's memory of the text so far (4 x 64 slots of "
           "1,024 numbers: 262,144). Is the blackboard a better feature for steering Anima than her stream, the block's "
           "output that the earlier slider experiments read (e018, e019, e023-e025)? Two measurements: how closely each "
           "matches the geometry of captions (the method of the earlier conditioning probes, and Anima's own caption "
           "context), and how well each places mood phrases on a slider's axis (the instruments of the earlier slider "
           "experiments).", "",
           "## Design", "", "| part | what is read | the numbers |", "|---|---|---|",
           f"| A, caption geometry | {rec['part A']} | {rec['part A gauges']} |",
           f"| B, the slider read | {rec['part B']} | {rec['part B instruments']} |", "",
           "The conditions (their keys in result.json):", "", "| key | the trunk |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in CONDITION_TEXT.items()]
    out += ["", "## Recipe"] + [f"- {k}: {v}" for k, v in rec.items() if not k.startswith("part")]
    out += ["", "## The rule (fixed before the run)",
            "Among the bare trunk's readings of a phrase alone or closed by a full stop, at every block and at the earlier "
            "sliders' four blocks together (16, 18, 21, 24), the hub's blackboard and the stream: a reading qualifies with at "
            "least 7 of the 8 held-out mood phrases on their side and at least 50 of the 60 mood training phrases on their side "
            "when each is left out of the fit. The slider's feature is the qualifying reading with the largest separation; "
            "within .05 of it, the more even spreads, then the closer match to Anima's caption context (debiased CKA). The "
            "hub is called better when the chosen reading is a blackboard and the best stream reading trails it by more than "
            ".05 in separation, or is within .05 and less even by more than .10; the stream in the mirror case; otherwise "
            "tied (the sliders would run both). The blackboard and the stream together is read beside them, never a "
            "candidate.", "",
            "**The expectation registered before the read**: the hub's blackboards would be the better features.", "",
            "**Limits fixed in advance**: one draw of captions for the decision (a second only for the noise bar); one "
            "untrained trunk; no picture (the sliders' pictures are their own experiments).", ""]
    # ---- the result
    out += ["## Result"]
    if d.get("chosen") is None:
        out += [f"**{d['verdict']}**: no reading met the bar.", ""]
    else:
        c, h, s = d["chosen"], d.get("best_hub"), d.get("best_stream")
        out += [f"**{d['verdict']}**: the slider's feature is {words(c['key'])} (`{c['key']}`; separation {c['sep']:.2f}, "
                f"balance {c['bal']:.2f})."]
        for o in (h, s):
            if o is not None and o["key"] != c["key"]:
                out[-1] += (f" The best {'hub' if o['kind'] == 'hub' else 'stream'} reading is {words(o['key'])} "
                            f"(`{o['key']}`; separation {o['sep']:.2f}, balance {o['bal']:.2f}).")
        out += [""]
    out += ["### The slider read: the three readings the sliders use, under every condition",
            "Each cell: separation; leave-one-out signs (of 60); held-out signs (of 8); balance.", "",
            "| reading | " + " | ".join(lab for _, lab in conds) + " |", "|---|" + "---|" * len(conds)]
    for k in READINGS:
        out.append(f"| {words(k)} (`{k}`) | " + " | ".join(_cell(B[c].get(k)) for c, _ in conds) + " |")
    held = [p for p in B["M0"][READINGS[0]]["heldout_a"]]
    mood = {p: c for (c, s), ps in _phrase_table().items() if s == "heldout" for p in ps}
    out += ["", "The held-out phrases' slider values on the bare trunk (the training centres at +1 cheerful and -1 gloomy):",
            "", "| phrase | mood | " + " | ".join(f"`{k}`" for k in READINGS) + " |", "|---|---|" + "---|" * len(READINGS)]
    for p in held:
        out.append(f"| {p} | {mood.get(p, '?')} | "
                   + " | ".join(f"{B['M0'][k]['heldout_a'][p]:+.2f}" if p in B["M0"][k].get("heldout_a", {}) else "-"
                                for k in READINGS) + " |")
    means = {k: {c: statistics.mean(B["M0"][k]["heldout_a"][p] for p in held if mood.get(p) == c) for c in ("up", "down")}
             for k in READINGS if "heldout_a" in B["M0"].get(k, {})}
    if means:
        out += ["", "The unseen phrases' mean slider value, cheerful / gloomy: " + "; ".join(
            f"`{k}` {v['up']:+.2f} / {v['down']:+.2f}" for k, v in means.items()) + "."]
    el = sorted((x for x in d.get("candidates", []) if x["eligible"]), key=lambda x: -x["sep"])
    out += ["", f"The qualifying readings ({len(el)} of {len(d.get('candidates', []))}), by separation (the first ten):", "",
            "| reading | separation | balance | CKA with Anima's caption context |", "|---|---|---|---|"]
    out += [f"| `{x['key']}` | {x['sep']:.2f} | {x['bal']:.2f} | {x['cka']:.3f} |" for x in el[:10]]
    out += ["", "### Caption geometry: the blackboard against the stream (bare trunk)",
            "Against CLIP ViT-B/32: top-10 overlap / Spearman. Against Anima's caption context: debiased CKA (ridge R2).", "",
            "| block | blackboard, CLIP-B/32 | stream pooled, CLIP-B/32 | stream last byte, CLIP-B/32 | blackboard, Anima's "
            "context | stream pooled, Anima's context |", "|---|---|---|---|---|---|"]

    def v1(fam, b):
        o, sp = _a(L, "M0", fam, b, "clip_b32", "overlap10"), _a(L, "M0", fam, b, "clip_b32", "spearman")
        return "-" if o is None else f"{o:.3f} / {sp:+.3f}"

    def ctx(fam, b, cond="M0"):
        k, r2 = _a(L, cond, fam, b, "adapter_pool", "cka_deb"), _a(L, cond, fam, b, "adapter_pool", "r2_rep2ref")
        return "-" if k is None else f"{k:.3f}" + ("" if r2 is None else f" ({r2:+.3f})")
    for b in f["blocks"]:
        out.append(f"| {b} | {v1('bb', b)} | {v1('pool', b)} | {v1('last', b)} | {ctx('bb', b)} | {ctx('pool', b)} |")
    bk = f["best_block"]
    out += ["", f"Of the {len(f['blocks'])} blocks the blackboard leads the pooled stream against CLIP-B/32 at "
            f"{f['lead']['clip']}, in ridge R2 to Anima's caption context at {f['lead']['ridge']} and in CKA with it at "
            f"{f['lead']['cka']}."]
    if nb:
        out[-1] += (" The noise bar (draw 1 against draw 2, every tap and reference): " + "; ".join(
            f"{n} {nb[g][0]:.3f} mean, {nb[g][1]:.3f} at most" for g, n in (("overlap10", "overlap"), ("spearman", "Spearman"),
                                                                            ("cka_deb", "CKA"), ("r2_rep2ref", "ridge"))
            if g in nb) + ".")
        if "M0_draw2" in L["A"] and f"bb@L{bk}" in L["A"]["M0_draw2"]:
            out[-1] += (f" The second draw repeats the lead at block {bk}: blackboard "
                        f"{_a(L, 'M0_draw2', 'bb', bk, 'clip_b32', 'overlap10'):.3f}, pooled stream "
                        f"{_a(L, 'M0_draw2', 'pool', bk, 'clip_b32', 'overlap10'):.3f}.")
    arms = [c for c in ("M9_A", "M9_B") if f"bb@L{bk}" in L["A"].get(c, {})]
    if arms:
        out[-1] += (f" Her nine arms leave the blackboard's match to Anima's caption context at block {bk} where it was: "
                    f"{ctx('bb', bk).split(' ')[0]} bare, " + " and ".join(ctx('bb', bk, c).split(' ')[0] for c in arms)
                    + f" with the nine arms (arm seed{'s' if len(arms) > 1 else ''} "
                    + " and ".join(c[-1] for c in arms) + ").")
    if f"bb@L{bk}" in L["A"].get("U0", {}):
        out[-1] += (f" The untrained trunk at block {bk}: blackboard {_a(L, 'U0', 'bb', bk, 'clip_b32', 'overlap10'):.3f}, "
                    f"pooled stream {_a(L, 'U0', 'pool', bk, 'clip_b32', 'overlap10'):.3f} against CLIP-B/32.")
    # ---- the anchor
    out += ["", "### The check against the earlier sliders' features"]
    trunk = {"trained": "her step 212,000 trunk", "random": "the untrained trunk"}
    if "anchor" in L:
        out += ["The read's features of the earlier sliders' trunk and of the untrained trunk reproduce the stored features "
                "file of those experiments to " + " and ".join(f"{v['max_rel']:.1e} ({trunk.get(k, k)})"
                                                                for k, v in L["anchor"].items())
                + " of the largest stored value."]
    elif "anchor_smoke" in L:
        out += ["The small run (the same code, trunks and phrases) reproduced the stored features file of the earlier sliders "
                "to " + " and ".join(f"{v['max_rel']:.1e} ({trunk.get(k, k)})" for k, v in L["anchor_smoke"].items())
                + " of the largest stored value. The full run could not open that file at its last step (an offline cache "
                  "fault, since fixed), after every condition was saved."]
    else:
        out += ["Not read."]
    # ---- what it set
    from .read import FEATURES_REPO
    out += ["", "## What it set", "The hub sliders, each reading its own features file in the data repo "
            f"([{FEATURES_REPO}](https://huggingface.co/datasets/{FEATURES_REPO})):", "",
            "| arm | reading | trunk |", "|---|---|---|"]
    for a in hub_arms():
        trunk = {"trained": "hers", "arms9": "hers with her nine arms mounted", "random": "untrained (control)"}[a.source]
        out.append(f"| {a.id} | {words(_reading_of(a))} | {trunk} |")
    out += ["", "## Files", "", "| file | what |", "|---|---|"] + [f"| `{k}` | {v} |" for k, v in FILES.items()] + [""]
    return "\n".join(out)


def _phrase_table() -> dict:
    from .read import PHRASES
    return PHRASES


def hub_arms() -> list:
    """The hub sliders: the arms on the readings this read chose and measured (later arms on other readings name their own
    read)."""
    files = {ax.HUB_FEATURES[r] for r in READINGS}
    return [a for a in ax.CONNECTOR_ARMS if a.features in files]


def _reading_of(arm) -> str:
    return next(k for k, v in ax.HUB_FEATURES.items() if v == arm.features)


def meta(L, code: "str | None" = None, seconds: "int | None" = None, finished_utc: "str | None" = None) -> dict:
    m, d, B = L["_meta"], L["decision"], L["B"]
    f, nb = part_a_facts(L), noise_bar(L)
    readings = {k: {c: {x: B[c][k][x] for x in ("separation", "balance", "loo_signs", "heldout_signs")}
                    for c, _ in CONDITIONS if c in B and k in B[c]} for k in READINGS}
    part_a = {str(b): {fam: {"clip_b32": [_a(L, "M0", fam, b, "clip_b32", "overlap10"),
                                          _a(L, "M0", fam, b, "clip_b32", "spearman")],
                             "adapter_pool": [_a(L, "M0", fam, b, "adapter_pool", "cka_deb"),
                                              _a(L, "M0", fam, b, "adapter_pool", "r2_rep2ref")]}
                       for fam in ("bb", "pool", "last")} for b in f["blocks"]}
    out = {"id": EXPERIMENT_ID, "title": TITLE, "date": m["made"][:10], "kind": KIND, "status": "done",
           "recipe": recipe(L, code),
           "result": {"decision": {k: d.get(k) for k in ("verdict", "chosen", "best_hub", "best_stream")},
                      "readings": readings, "part_a_bare": part_a,
                      "noise_bar": {k: list(v) for k, v in nb.items()},
                      "block_leads": f["lead"], "anchor": L.get("anchor") or L.get("anchor_smoke"),
                      "anchor_from": "the full run" if "anchor" in L else ("the small run" if "anchor_smoke" in L else None),
                      "arms": [a.id for a in hub_arms()]},
           "summary": summary(L)}
    if seconds is not None or m.get("seconds") is not None:
        out["seconds"] = int(seconds if seconds is not None else m["seconds"])
    if finished_utc or m.get("finished_utc"):
        out["finished_utc"] = finished_utc or m["finished_utc"]
    return out


def write(ledger: str, mirror: str, small: "str | None" = None, code: "str | None" = None, seconds: "int | None" = None,
          finished_utc: "str | None" = None) -> dict:
    """<mirror>/experiments/<id>/: README.md, meta.json, result.json (the ledger), tables.md (the read's tables beside the
    ledger, written fresh when absent) and small_run.json; returns the meta."""
    with open(ledger, encoding="utf-8") as fh:
        L = json.load(fh)
    if L["_meta"].get("smoke"):
        raise ValueError(f"{ledger} is the small run's ledger; the experiment is the full read")
    if "decision" not in L:
        raise ValueError(f"{ledger} holds no decision (the read did not finish)")
    folder = os.path.join(mirror, "experiments", EXPERIMENT_ID)
    os.makedirs(folder, exist_ok=True)
    mt = meta(L, code, seconds, finished_utc)
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(readme(L, code))
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(mt, fh, indent=1, ensure_ascii=False)
    shutil.copyfile(ledger, os.path.join(folder, "result.json"))
    tables = ledger[:-5] + ".md"
    if os.path.exists(tables):
        shutil.copyfile(tables, os.path.join(folder, "tables.md"))
    else:
        from .read import report
        with open(os.path.join(folder, "tables.md"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(report(L))
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
    mt = write(ledger, mirror, opt.get("small"), opt.get("code"),
               int(opt["seconds"]) if "seconds" in opt else None, opt.get("finished"))
    print(f"wrote {os.path.join(mirror, 'experiments', EXPERIMENT_ID)}: {mt['summary']}", flush=True)
    if opt.get("publish") == "1":
        publish(mirror)
    return 0


if __name__ == "__main__":
    sys.exit(main())
