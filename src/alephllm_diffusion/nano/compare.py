"""alephllm_diffusion.nano.compare - the nano readers under the dual-extraction read's own gauges, and the registered marks
(plans/2026-10-17_nano_diffusion_battery_plan.md in the research record, registered before any read).

Every gauge is triangulate.read.gauges, unchanged: per signal, against the Qwen3 states Anima reads (Q, its final hidden state; Q16 and
Q0 descriptive): the whitened alignment (K 128), its unwhitened twin, the context alignment (each token's mean removed), ridge R^2 and
the debiased CKA; the mood directions mapped into Q's frame (C6); fit on half the captions, scored on the other half. The readers'
sets come from nano.extract (one file a reader and a draw); the stacks are built here from them.

THE MARKS (the plan's section 3; the read's rules carried to the nano readers):
  bar        per signal, max(BAR_FLOOR, abs(draw 1 - draw 2)) of the trained reader's alignment
  S2         LEARNED: the alignment beats the body's own birth at the same signal by more than the bar
  S3         a continuation moved the caption states: its alignment against the diet's end at the same signal, beyond the larger bar
  U2         a side carries: the mapped mood axis beats the birth's cosine by DIR_MARGIN and at least 3 of the 4 held-out phrases of
             that side land on that side
  U3         a stack beats its best member at the same tap by more than the stack's bar
  U4         a hub beats the stream at its block by more than the larger bar (and the dual hubs' two hubs against each other)
Nothing runs on import."""
from __future__ import annotations

import os

import torch

from ..triangulate import align as AL
from ..triangulate import read as RD

K = 128


def D_of(rw: dict) -> dict:
    """the read's row description (draw_rows' fields the gauges use) from nano.extract.rows."""
    return {"mood": rw["mood"], "group": rw["group"], "ids": rw["ids"], "cls": rw["cls"], "split": rw["split"]}


def load(folder: str, reader: str, draw: str) -> dict:
    return torch.load(os.path.join(folder, f"{reader}_{draw}.pt"), weights_only=False)


def stacked(members: list, taps=None) -> dict:
    """the members' stream taps side by side (each tap present in every member), as a set the gauges read."""
    taps = taps if taps is not None else sorted(set.intersection(*(set(m["stream"]) for m in members)))
    return {"stream": {b: torch.cat([m["stream"][b] for m in members], dim=1) for b in taps}, "hub": {},
            "size": members[0]["size"]}


def hub_pair(S: dict, block: int) -> dict:
    """a dual hub's two hubs side by side at one block (their Gram coordinates concatenated)."""
    return {"stream": {}, "hub": {f"{block}rp": torch.cat([S["hub"][f"{block}r"], S["hub"][f"{block}p"]], dim=1)},
            "size": S["size"]}


def gauges(sets: dict, Q: dict, D: dict, k: int = K) -> dict:
    """the read's gauges over {reader: set} (form 'A': her own bytes), plus C3 for every hub against the stream at its block."""
    res = RD.gauges({(r, "A"): S for r, S in sets.items()}, Q, D, k)
    cap = ~D["mood"]
    g, ids = D["group"][cap], D["ids"][cap]
    for r, S in sets.items():
        for name, X in S["hub"].items():
            b = int("".join(ch for ch in name if ch.isdigit()))
            if b in S["stream"]:
                res["C3"][f"{r}|A-{b}|{name}"] = AL.compare(S["stream"][b][cap], X[cap], g, k, ids, ridge=False)
    return res


def bar_of(C1, C1b, key):
    return RD._bar(C1, C1b, (key, "Q", "alignment"))


def body_marks(R1: dict, R2: dict, body: str, versions=("end", "fr", "ct", "cf")) -> dict:
    """S2 / S3 / U2 / U4 and the best signals for one body's readers (the keys '<body>|<version>|A-...')."""
    C1, C1b, C6 = R1["C1"], R2["C1"], R1["C6"]
    out = {}
    for v in versions:
        mine = sorted(key for key in C1 if key.startswith(f"{body}|{v}|"))
        rows = {}
        for key in mine:
            name = key.split("|", 2)[2]
            u = f"{body}|birth|{name}"
            a1 = C1[key]["Q"]["alignment"]
            bar = bar_of(C1, C1b, key)
            row = {"alignment": a1, "alignment_draw2": C1b.get(key, {}).get("Q", {}).get("alignment"), "bar": bar,
                   "context": C1[key]["Q"]["context_alignment"], "ridge_r2": C1[key]["Q"].get("ridge_r2"),
                   "cka": C1[key]["Q"].get("cka_debiased")}
            if u in C1:
                row["birth"] = C1[u]["Q"]["alignment"]
                row["birth_context"] = C1[u]["Q"]["context_alignment"]
                row["margin"] = a1 - row["birth"]
                row["S2"] = "LEARNED" if row["margin"] > bar else "NOT LEARNED"
                t6, u6 = C6[key], C6[u]
                gain = t6["cos_axis"] - u6["cos_axis"]
                row["mood"] = {"cos_axis": t6["cos_axis"], "cos_axis_birth": u6["cos_axis"], "gain": gain,
                               **{side: {"held_out": list(t6[f"mapped_{side}"]),
                                         "carries": bool(gain > RD.DIR_MARGIN and t6[f"mapped_{side}"][0] >= 3)}
                                  for side in ("up", "down")}}
            e = f"{body}|end|{name}"
            if v != "end" and e in C1:
                be = max(bar, bar_of(C1, C1b, e))
                dlt = a1 - C1[e]["Q"]["alignment"]
                row["vs_end"] = {"delta": dlt, "bar": be, "S3": "MOVED UP" if dlt > be else ("MOVED DOWN" if dlt < -be else "HELD")}
            rows[name] = row
        if not rows:
            continue
        best = max(rows, key=lambda n: rows[n]["alignment"])
        best_ctx = max(rows, key=lambda n: rows[n]["context"])
        trunk = [n for n in rows if "-trunk-" in n]
        hubs = [n for n in rows if "-hub-" in n]
        u4 = None
        if trunk and hubs:
            bt, bh = max(trunk, key=lambda n: rows[n]["alignment"]), max(hubs, key=lambda n: rows[n]["alignment"])
            bar = max(rows[bt]["bar"], rows[bh]["bar"])
            u4 = {"trunk": [bt, rows[bt]["alignment"]], "hub": [bh, rows[bh]["alignment"]], "bar": bar,
                  "verdict": RD._verdict(rows[bh]["alignment"], rows[bt]["alignment"], bar, "HUB CLOSER", "TRUNK CLOSER")}
        out[v] = {"signals": rows, "best": best, "best_context": best_ctx, "U4": u4}
    return out


def stack_marks(R1: dict, R2: dict, stack: str, members: dict) -> dict:
    """U3 for one stack: {tap name: the stack's alignment, its best member's, the bar, the verdict}. members = {member reader:
    the C1 key prefix} as written by the gauges ('<reader>|A-trunk-<b>')."""
    C1, C1b = R1["C1"], R2["C1"]
    out = {}
    for key in sorted(k for k in C1 if k.startswith(f"{stack}|")):
        name = key.split("|")[-1]
        mk = [f"{m}|{name}" for m in members if f"{m}|{name}" in C1]
        if not mk:
            continue
        top = max(mk, key=lambda m: C1[m]["Q"]["alignment"])
        bar = max(bar_of(C1, C1b, key), bar_of(C1, C1b, top))
        a, b = C1[key]["Q"]["alignment"], C1[top]["Q"]["alignment"]
        out[name] = {"stack": a, "stack_context": C1[key]["Q"]["context_alignment"], "best_member": [top, b],
                     "best_member_context": C1[top]["Q"]["context_alignment"], "gain": a - b, "bar": bar,
                     "U3": "THE STACK ADDS" if a - b > bar else "NOTHING ADDED"}
    return out
