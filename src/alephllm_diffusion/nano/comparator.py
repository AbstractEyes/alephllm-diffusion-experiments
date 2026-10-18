"""alephllm_diffusion.nano.comparator - THE UNIFORM COMPARATOR in pictures (plans/2026-10-08_slider_formula_map.md s.4 (A)-(C) in the
research record; registered for the nano readers as mark P of plans/2026-10-17_nano_diffusion_battery_plan.md, amendment 1):
fixed mood directions taken once from the image model's own prepared conditioning; every reader sets only the amount, from its own
slider reading a of the phrase (nano.mood; a training phrase read on the axis it helped build, as a deployed slider reads it).

  Anima (two directions after the text adapter, on the conditional branch, as the words themselves act):
    p = g * ( max(a, 0) * D_up + max(-a, 0) * D_down )
    D_side = mean over training scenes s and that side's training phrases w of
             [ mean_j c_j(caption of s with w) - mean_j c_j(neutral caption of s) ]      c_j: the adapter's output at T5 slot j
  Sana (the Sana flavor test's one direction d on the text encoder's own tokens; its size 10.14):
    e'(token) = e(token) + g * a * d

Every picture is paired with the same cell's picture without a push; the reads are the record's connector rule on those
differences (geolip_anima_trainer.anima_experiments.connector_reads). Nothing runs on import; torch is the only dependency."""
from __future__ import annotations

import torch

GAINS = (2.0, 1.0)                       # g = 2 first (amendment 1: the g = 1 gloomy response is about two of its errors)
RESPONSE_STEPS = (1.0, 2.0, 4.0)         # the shared directions alone, s * D_side on the neutral caption
EVAL_TRAIN = {"up": ("cheerful and upbeat", "joyful and uplifting"),       # the record's evaluated training phrases
              "down": ("gloomy and downbeat", "somber and melancholy")}


def slot_mean(ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, L, D] -> [B, D]: the mean over each caption's own slots (mask 1), float32."""
    m = mask.float().unsqueeze(-1)
    return (ctx.float() * m).sum(1) / m.sum(1).clamp_min(1)


@torch.no_grad()
def anima_directions(pipe, scenes: list, phrases: dict, template: str, neutral: str, batch: int = 32) -> dict:
    """{side: D_side} (float32 [1024] on the CPU) for every side of phrases ({'up': [...], 'down': [...], 'neutral': [...]}) from
    AnimaPipe's encode + adapt: the training phrases only and the training scenes only (the evaluated cells never enter)."""
    def means(prompts):
        out = []
        for i in range(0, len(prompts), batch):
            conds = pipe.encode(prompts[i:i + batch])
            out.append(slot_mean(pipe.adapt(conds), conds[3]).cpu())
        return torch.cat(out)

    neu = means([neutral.format(s=s) for s in scenes])
    out = {}
    for side, ws in phrases.items():
        M = means([template.format(s=s, p=w) for s in scenes for w in ws]).view(len(scenes), len(ws), -1)
        out[side] = (M - neu[:, None]).mean((0, 1))
    return out


def neutral_relative(D: dict) -> dict:
    """the formula map's second form: each mood direction less the neutral phrases' (what appending any phrase does in common)."""
    return {side: D[side] - D["neutral"] for side in ("up", "down")}


def anima_push(a: float, g: float, D: dict) -> torch.Tensor:
    return g * (max(a, 0.0) * D["up"] + max(-a, 0.0) * D["down"])


def eval_sets(cls: list, split: list, phrases: list, train: bool = True) -> dict:
    """{key: (class, split, phrase)} in render order: per class the two evaluated training phrases (train=True) and every held-out
    phrase; the neutral class its held-out phrases (the record's connector_eval_sets over the mood rows)."""
    out = {}
    for c in ("up", "down", "neutral"):
        if train:
            for t in EVAL_TRAIN.get(c, ()):
                i = next(i for i in range(len(phrases)) if phrases[i] == t and cls[i] == c and split[i] == "train")
                out[f"{c}/train/{phrases[i]}"] = (c, "train", phrases[i])
        for i in range(len(phrases)):
            if cls[i] == c and split[i] == "heldout":
                out[f"{c}/heldout/{phrases[i]}"] = (c, "heldout", phrases[i])
    return out
