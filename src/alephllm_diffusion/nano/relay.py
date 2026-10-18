"""alephllm_diffusion.nano.relay - a reader's states in Qwen3's lane: the relay in pictures carried from a phrase's span to a whole
caption. One closed-form ridge per reader (stitch.s2.ridge_cv: fp64, the penalty chosen by cross-validation over caption folds)
from the reader's tap on the dual-extraction read's caption rows to Qwen3's final state (what Anima's text adapter reads). On a
new caption every token but the first gets the map of the reader's state at the token's closing byte (the caption is read with
one space after it, so its last token closes too); the first token keeps Qwen3's own state (its attention sink, an outlier the
rows never held), the same for every arm.
    q_hat(t) = (x(t) - mu_x) W + mu_y          t = 1 .. L-1
Nothing runs on import."""
from __future__ import annotations

import torch

SPACE = b" "


def caption_folds(groups: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """a fold index per row (0 / 1), whole captions to a fold (the read's own split, triangulate.align.folds)."""
    from ..triangulate import align as AL
    g = torch.as_tensor(groups)
    out = torch.full((len(g),), -1, dtype=torch.long)
    for f, (_, held) in enumerate(AL.folds(g, seed=seed)):
        out[held] = f
    assert bool((out >= 0).all()), "a row in no fold"
    return out


def fit(X: torch.Tensor, Y: torch.Tensor, groups: torch.Tensor, device: str = "cpu") -> dict:
    """the reader's map: {'W' [p, D], 'mx' [p], 'my' [D], 'alpha', 'cv_r2'} (fp64 on the CPU)."""
    from ..stitch import s2
    W, mx, my, alpha, r2 = s2.ridge_cv(X, [Y], caption_folds(groups), device)[0]
    return {"W": W.cpu(), "mx": mx.cpu(), "my": my.cpu(), "alpha": float(alpha), "cv_r2": float(r2)}


def apply(m: dict, X: torch.Tensor) -> torch.Tensor:
    return ((X.double() - m["mx"]) @ m["W"] + m["my"]).float()


def closing_positions(a_end: list) -> list:
    """positions in [DOC] + caption + ' ' of the closing bytes of tokens 1 .. L-1 (the byte after each token's expansion)."""
    return [1 + e for e in a_end[1:]]


def prefixes(raw: bytes, a_end: list) -> list:
    """per token 1 .. L-1, the caption + ' ' through the token's closing byte (the read's form A rows, one prefix a token)."""
    s = raw + SPACE
    return [s[: e + 1] for e in a_end[1:]]


def states(first: torch.Tensor, mapped: torch.Tensor) -> torch.Tensor:
    """[L, D]: Qwen3's own first-token state, then the mapped tokens."""
    return torch.cat([first.float()[None], mapped.float()], 0)
