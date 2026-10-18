"""alephllm_diffusion.nano.page_relay_trained - the trained relay's folder on the Anima experiments repo (README, meta.json, the
training and picture ledgers, every connector's weights as safetensors, a contact sheet), written from the runs' own ledgers
(nano.experiment calls it). Nothing runs on import."""
from __future__ import annotations

import os

from .experiment import ANIMA_LICENCE, ANIMA_REFS, CODE_URL, DATE, READ_REFS, _f, _load, _refs, sheet

TEXT = {"qwen_fp32": "Qwen3's own fp32 states (the reference every picture is compared with)",
        "floor": "the floor: Qwen3's input embedding (a lookup of the token, no context)",
        "stack": "the four nano v1.1 bodies' -fr finals side by side, block 3",
        "nano": "nano-castor-v1.1 -cf, block 3",
        "births": "the four nano bodies' untrained starts side by side, block 3 (the control)",
        "beatrix": "Beatrix 3s (a 376M byte-level model), block 24 (the reference reader)",
        "q16": "Qwen3's own layer-16 state (the positive control: a reader that holds Qwen3's own information)"}
SHORT = {"floor": "the token lookup", "stack": "the four nano bodies side by side", "nano": "castor -cf",
         "births": "the untrained starts side by side", "beatrix": "Beatrix 3s", "q16": "Qwen3's layer 16"}
KIND = {"ridge": "the ridge alone", "s0": "trained, seed 0", "s1": "trained, seed 1"}
EXTRA_REFS = [("Hu, Wang, Fang, Fu, Cheng, Yu 2024", "ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment",
               "https://arxiv.org/abs/2403.05135")]


def arm_text(arm: str) -> str:
    if arm == "qwen_fp32":
        return TEXT[arm]
    r, k = arm.split("|")
    return f"{TEXT[r]}: {KIND[k]}"


def to_safetensors(src_dir: str, out_dir: str) -> dict:
    """every connector (a state dict) and ridge (W, mx, my + its penalty and fit) as safetensors; {published name: what}."""
    import torch
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    names = {}
    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith(".pt"):
            continue
        obj = torch.load(os.path.join(src_dir, fn), weights_only=False, map_location="cpu")
        base = fn[:-3]
        if base.endswith("_ridge"):
            t = {k: obj[k].contiguous() for k in ("W", "mx", "my")}
            meta = {"alpha": repr(obj["alpha"]), "cv_r2": repr(obj["cv_r2"])}
            what = "the closed-form ridge (q_hat = (x - mx) W + my; float64), the connector's birth"
        else:
            t = {k: v.contiguous() for k, v in obj.items()}
            meta = {}
            what = "a trained connector (lin = the ridge-born linear part, up / down = the nonlinear part)"
        save_file(t, os.path.join(out_dir, base + ".safetensors"), metadata=meta)
        names[f"connectors/{base}.safetensors"] = what
    return names


def page(root: str, img_dir: str, out_sheet: str, smoke: bool = False) -> tuple:
    sfx = "_smoke" if smoke else ""
    T = _load(os.path.join(root, "relay_train", f"rt_train{sfx}.json"))
    P = _load(os.path.join(root, "relay_train", f"anima_relay_trained{sfx}.json"))
    E = _load(os.path.join(root, "relay", "anima_relay.json"))
    arms, reads, M = P["_meta"]["arms"], P["reads"], T["_meta"]

    def kept(arm, src=P):
        v = src["arms"][arm]["kept"]
        return sum(v) / len(v)

    # the state side
    s_rows = []
    for k, r in T["readers"].items():
        s0 = [x for x in r["runs"] if x["seed"] == 0]
        b0 = max(s0, key=lambda x: x["best_val_r2"])
        s1 = [x for x in r["runs"] if x["seed"] == 1][0]
        s_rows.append(f"| {TEXT[k]} | {r['width']:,} | {r['ridge']['val_context_r2']:.3f} / {r['ridge']['val_state_r2']:.3f} | "
                      + " / ".join(f"{x['best_val_r2']:.3f}" for x in s0) + f" | {r['lr']:g} | {b0['best_val_r2']:.3f} / "
                      f"{s1['best_val_r2']:.3f} | {b0['val_state_r2']:.3f} / {s1['val_state_r2']:.3f} |")
    t_s = ("| reader | width | the ridge: context R2 / Qwen3-state R2 | the rate search at seed 0 (1e-5 / 3e-5 / 1e-4): best "
           "context R2 | rate kept | trained, seeds 0 / 1: context R2 | trained, seeds 0 / 1: Qwen3-state R2 |\n"
           "|---|---|---|---|---|---|---|\n" + "\n".join(s_rows))
    # the pictures
    p_rows = []
    for a in arms:
        r = reads.get(a)
        p_rows.append(f"| {arm_text(a)} | {kept(a):.4f} | " + (
            f"{_f(r['kept']['diff'], 4, True)} +- {_f(r['kept']['se'], 4)} ({r['kept']['captions_above']} of {r['kept']['n']}) | "
            f"{_f(r['gap_closed'], 2) if r['gap_closed'] is not None else '-'} | {_f(r['caption_score']['diff'], 2, True)} +- "
            f"{_f(r['caption_score']['se'], 2)} | {r['OUTCOME'].lower()} |" if r else
            ("(the comparison) | - | - | (the trained floor) |" if a == "floor|s0" else "(the reference) | - | - | - |")))
    t_p = ("| arm | picture kept | against the trained floor (captions above) | share of the floor-to-Qwen3 gap closed | caption "
           "score against the trained floor | verdict |\n|---|---|---|---|---|---|\n" + "\n".join(p_rows))
    t_t = "\n".join(f"- {SHORT[n]}: trained against its ridge {_f(reads[f'training|{n}']['diff'], 4, True)} +- "
                    f"{_f(reads[f'training|{n}']['se'], 4)} ({reads[f'training|{n}']['OUTCOME'].lower()})" for n in ("floor", "stack"))
    e_arms = [a for a in ("floor", "beatrix", "nano", "stack", "births") if a in E["arms"]]
    t_e = "\n".join(f"- {SHORT[a]}: the earlier relay's linear map {kept(a, E):.4f}; here " + ", ".join(
        f"{kept(a + '|' + k):.4f} ({w})" for k, w in (("ridge", "the ridge on this corpus"), ("s0", "trained"))
        if a + "|" + k in P["arms"]) for a in e_arms)
    # what it means
    useful = [a for a in arms if reads.get(a, {}).get("OUTCOME") == "USEFUL WITH TRAINING" and not a.startswith("floor")]
    meaning = []
    if reads["TR3"].startswith("THE POSITIVE CONTROL FAILS"):
        meaning.append("Qwen3's own layer 16, trained the same way, does not rise above the trained floor: on this instrument no "
                       "reader can pass, and every verdict below is inconclusive")
    else:
        meaning.append(f"Qwen3's own layer 16, trained the same way, rises above the trained floor "
                       f"({_f(reads['q16|s0']['kept']['diff'], 4, True)} +- {_f(reads['q16|s0']['kept']['se'], 4)}): the "
                       f"instrument can pass")
    meaning.append("above the trained floor: " + (", ".join(
        f"{SHORT[a.split('|')[0]]} {_f(reads[a]['kept']['diff'], 4, True)} +- {_f(reads[a]['kept']['se'], 4)} "
        f"({_f(reads[a]['gap_closed'], 2)} of the gap)" for a in useful) if useful else "no reader but the positive control")
                   + "; at it: " + ", ".join(SHORT[a.split('|')[0]] for a in arms if a in reads and a not in useful and
                                             a.split('|')[0] in ("stack", "nano", "births", "beatrix") and a.endswith("s0")))
    meaning.append(f"the untrained starts side by side: {reads['TR2'].lower()}")
    meaning.append("training's own effect over the ridge on the same corpus: " + "; ".join(
        f"{SHORT[n]}: {reads[f'training|{n}']['OUTCOME'].lower()}" for n in ("floor", "stack")))
    v0 = [x for x in T["readers"]["floor"]["runs"] if x["seed"] == 0 and x["lr"] == T["readers"]["floor"]["lr"]][0]
    v1 = [x for x in T["readers"]["floor"]["runs"] if x["seed"] == 1][0]
    meaning.append(f"the trained lookup's two seeds, alike in context R2 ({v0['best_val_r2']:.3f} / {v1['best_val_r2']:.3f}), differ "
                   f"by {abs(kept('floor|s0') - kept('floor|s1')):.3f} in picture kept: the sampler magnifies small differences in "
                   f"the states, and that spread is the bar every reader had to clear ({reads['bar']:.4f})")
    meaning_md = "\n".join(f"{i + 1}. {m[0].upper()}{m[1:]}." for i, m in enumerate(meaning))
    summary = ("trained connectors through Anima's frozen text adapter: " + ("; ".join(
        f"{SHORT[a.split('|')[0]]} above the trained floor ({_f(reads[a]['gap_closed'], 2)} of the gap)" for a in useful)
        if useful else "no reader above the trained floor") + f"; {reads['TR3'].lower()}; {reads['TR2'].lower()}")
    # the sheet: the first five captions at the first seed
    seeds = sorted({c[1] for c in P["_meta"]["cells"]})
    n_cap = len({c[0] for c in P["_meta"]["cells"]})
    cols = [a for a in ("qwen_fp32", "floor|s0", "stack|ridge", "stack|s0", "nano|s0", "births|s0", "beatrix|s0", "q16|s0") if a in arms]

    def path_of(i, arm):
        return os.path.join(img_dir, f"{arm.replace('|', '_')}_{i}_{seeds[0]}.jpg")

    sheet(list(range(min(5, n_cap))), cols, path_of, out_sheet, labels=[c.replace("|", " ") for c in cols])
    title = ("The trained relay: a small connector per reader, taught through Anima's frozen text adapter to stand in for "
             "Qwen3 (a token lookup, the nano bodies, Beatrix 3s, Qwen3's own layer 16)")
    readme = f"""# {{rid}}: {title}

Date: {DATE}. Model: Anima-Base v1.0 (stock), {P['_meta']['settings']}. Only the connectors train (a few million weights each); Anima, its text adapter, Qwen3 and every reader stay frozen.

## Question
The relay in pictures (e047 in this repo) replaced Qwen3's states with one linear map of another reader's states: the nano bodies' maps tracked Qwen3's states better than a lookup of the token, yet their pictures stayed at the lookup's level, and Beatrix's map moved the pictures a fifth of the way toward Qwen3's own. Does a small trained connector, taught through Anima's own text adapter to reproduce the context Qwen3's states give, carry what the linear map could not?

## Design
- the connector, per reader: q_hat(t) = W x(t) + b + V2 gelu(V1 n(x(t)) + c1) + c2 for every token but the first (n = LayerNorm without its affine, hidden width 1,024), born at the closed-form ridge to Qwen3's final state (W, b) with V2 and c2 at zero, so it starts as the linear relay exactly; the first token keeps Qwen3's own state.
- the loss: L = sum over the captions' T5 slots of |A(q_hat) - A(q)|^2 / sum of |A(q) - m|^2, A = Anima's text adapter (frozen, float32), q = Qwen3's own float32 states, m = the training targets' mean: the context the diffusion model reads, from the connector's states against Qwen3's. Adam without weight decay, float32, batches of 64 captions, {M['steps']:,} steps, the validation score every {M['eval_every']} steps and the best step kept; the learning rate per reader from {{1e-5, 3e-5, 1e-4}} by the validation score at seed 0, then seed 1 at that rate.
- captions (COCO): training, 2,048 ({M['n']['train']:,} used); validation, 512 others ({M['n']['val']:,} used); pictures, the earlier relay's {n_cap} captions (in no fit and no validation), bare, seeds {', '.join(map(str, seeds))}: {len(P['_meta']['cells'])} cells.
- readers: """ + "; ".join(TEXT[k] for k in T["readers"]) + f""".
- judge (CLIP ViT-L/14, pixels only): picture kept, the cosine between a picture and the same cell's picture from Qwen3's own float32 states; the caption score (100 x the picture-caption cosine) is shown and not ruled (the earlier relay found it cannot separate Qwen3 from the lookup). The unit for the errors is the caption (its seeds averaged).
- the rules (fixed before any training): USEFUL WITH TRAINING = above the trained lookup (seed 0) on picture kept by more than 2 standard errors and by more than the connector-seed bar (the larger of the lookup's and the four bodies' seed-0 / seed-1 differences; here {reads['bar']:.4f}); the untrained starts side by side must not be; Qwen3's layer 16 must be (else nothing can pass); training's own effect = the trained connector against its ridge, by the same rule.
- checks: every reader's states reproduce the earlier read's stored rows and the earlier relay's; the Qwen3 and T5 token ids are the pipeline's own; the stored float32 states match the runner's float32 encoder ({P['_meta']['encoding_check']:.1e} largest per-token relative); Qwen3's float32 pictures against the earlier relay's saved ones: {P['_meta'].get('e047_reference_mean_pixel_difference', float('nan')):.2f} mean pixel difference (jpeg).

## Result

**The state side** (validation captions; context R2 = 1 - the loss's ratio over every T5 slot; Qwen3-state R2 of the connector's states):

{t_s}

**The pictures:**

{t_p}

Training's own effect (the same cells):
{t_t}

Against the earlier relay (e047, a linear map fitted on 512 captions, the same cells and judge):
{t_e}

![the sheet](sheet.jpg)

The sheet: the first five captions at seed {seeds[0]} (rows); columns: Qwen3's own states, the trained lookup, the four nano bodies (ridge, trained), castor -cf, the untrained starts, Beatrix 3s, Qwen3's layer 16 (trained).

## What it means
{meaning_md}

## Limits
{n_cap} captions, {len(seeds)} seeds; one connector form and size; seed 1 for two readers only; one judge model; COCO captions only.

## Code
{CODE_URL}, `alephllm_diffusion.nano` (relay, relay_train), branch feat/nano-battery.

## Licence
{ANIMA_LICENCE}

## References
{_refs(ANIMA_REFS + EXTRA_REFS + READ_REFS[3:4])}

## Files

{{files}}
"""
    files = {"README.md": "this page", "meta.json": "the summary and the run's record",
             "result.json": "every picture's judges, per arm, and the reads", "training.json": "every connector's training record",
             "sheet.jpg": "the contact sheet"}
    meta = {"title": title, "date": DATE, "kind": "nano_relay_trained", "status": "done",
            "recipe": {"bed": "Anima-Base v1.0, stock (frozen)", "connector": "ridge-born, a weight-zero-born nonlinear part, hidden 1,024",
                       "loss": "the adapter's output against Qwen3's, squared error over T5 slots, normalized",
                       "optimizer": "Adam (no weight decay), float32", "code": f"{CODE_URL} (alephllm_diffusion.nano)"},
            "result": {k: v for k, v in reads.items()}, "summary": summary}
    copies = {"result.json": os.path.join(root, "relay_train", f"anima_relay_trained{sfx}.json"),
              "training.json": os.path.join(root, "relay_train", f"rt_train{sfx}.json"), "sheet.jpg": out_sheet}
    return readme, meta, copies, files
