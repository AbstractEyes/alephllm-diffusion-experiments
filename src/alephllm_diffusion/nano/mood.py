"""alephllm_diffusion.nano.mood - a reader's slider reading of the 74 mood phrases: the reader term of the uniform comparator
(plans/2026-10-08_slider_formula_map.md s.2 and s.4 in the research record; plans/2026-10-17_nano_diffusion_battery_plan.md mark P).

Per tap, a phrase's state at the full stop closing its caption (the dual-extraction read's mood rows: the 74 phrases in the grid
caption form), normalized as the tap is stored, standardized per feature over the 64 training phrases (30 cheerful, 30 gloomy, 4
neutral), placed on the axis from the gloomy training centre to the cheerful one:
    a = (f - m) . u / (u . u / 2)      u = mean(cheerful training) - mean(gloomy training),  m = their midpoint
so the training centres sit at +1 and -1. A TRAINING phrase is read LEFT OUT: its own row leaves the standardization and the axis
(in-sample, any reader puts a training phrase near its centre: the axis is built through it); the held-out and neutral phrases are
read on the full training axis. The uniform comparator's push is then p = g (max(a, 0) D_up + max(-a, 0) D_down) on fixed
directions shared by every reader. Nothing runs on import."""
from __future__ import annotations

import torch

EPS = 1e-6


def _fit(X: torch.Tensor, cls: list, rows: list) -> tuple:
    """(mean, std, midpoint, axis) from the training rows `rows` of X (float64): standardization stats over those rows, then the
    cheerful and gloomy centres in the standardized frame."""
    T = X[rows]
    mu, sd = T.mean(0), T.std(0).clamp_min(EPS)
    Z = (T - mu) / sd
    up = [i for i, r in enumerate(rows) if cls[r] == "up"]
    dn = [i for i, r in enumerate(rows) if cls[r] == "down"]
    cu, cd = Z[up].mean(0), Z[dn].mean(0)
    return mu, sd, (cu + cd) / 2, cu - cd


def _a(x: torch.Tensor, fit: tuple) -> torch.Tensor:
    mu, sd, m, u = fit
    return ((x - mu) / sd - m) @ u / (u @ u / 2)


def readings(F: torch.Tensor, cls: list, split: list) -> torch.Tensor:
    """a for every row of F [74, d] (the mood rows in their stored order): training rows left out, the rest on the full axis."""
    X = F.double()
    train = [i for i, s in enumerate(split) if s == "train"]
    full = _fit(X, cls, train)
    a = torch.empty(len(X), dtype=torch.float64)
    for i in range(len(X)):
        if split[i] == "train":
            a[i] = _a(X[i], _fit(X, cls, [r for r in train if r != i]))
        else:
            a[i] = _a(X[i], full)
    return a


def in_sample(F: torch.Tensor, cls: list, split: list) -> torch.Tensor:
    """a for every row on the full training axis (the record's dead-zone check reads the training phrases this way)."""
    X = F.double()
    return _a(X, _fit(X, cls, [i for i, s in enumerate(split) if s == "train"]))


def summary(a: torch.Tensor, a_in: torch.Tensor, cls: list, split: list, phrases: list) -> dict:
    """The reading's numbers: left-out training accuracy (of 60; the selection statistic, blind to the held-out phrases), the
    held-out separation (mean a of the unseen cheerful minus the unseen gloomy; the record's hub-read statistic) and sides, the
    record's dead-zone check (in-sample gloomy training phrases below 0 / below -0.25; unseen gloomy below -0.25), the neutral
    phrases' values, and every evaluated phrase's a."""
    idx = lambda c, s: [i for i in range(len(cls)) if cls[i] == c and split[i] == s]   # noqa: E731
    tu, td, hu, hd = idx("up", "train"), idx("down", "train"), idx("up", "heldout"), idx("down", "heldout")
    neu = [i for i in range(len(cls)) if cls[i] == "neutral"]
    loo = int((a[tu] > 0).sum() + (a[td] < 0).sum())
    return {"loo_train": loo, "loo_up": int((a[tu] > 0).sum()), "loo_down": int((a[td] < 0).sum()),
            "heldout_sep": float(a[hu].mean() - a[hd].mean()),
            "heldout_sides": int((a[hu] > 0).sum() + (a[hd] < 0).sum()),
            "dead_zone": [int((a_in[td] < 0).sum()), int((a_in[td] < -0.25).sum()), int((a[hd] < -0.25).sum())],
            "neutral": {phrases[i]: float(a[i]) for i in neu},
            "a": {phrases[i]: float(a[i]) for i in range(len(cls))}}
