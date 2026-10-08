"""alephllm_diffusion.triangulate.arm_read - THE READ OF RECORD ON A SURFACE ARM (registered 2026-10-08 before any read; image-free;
no training). A surface arm (geolip.alephllm.arm_mount.mount_surface: a detachable adapter after each of her blocks) makes Beatrix
read a tokenizer's spelling of a text the way she reads its own bytes. This read measures the published arm the way the
dual-extraction read (read.py, experiment e038) measured her bare trunk: one row per Qwen token, against the Qwen3 states Anima's
adapter reads, with e038's captions, mood rows, gauges and stored Qwen rows.
ROWS (e038's, rebuilt identically): the first N_CAPS captions of each COCO draw (every token but a caption's first and last) and
the 74 mood rows in the grid's caption form (the token closing at the full stop). Her states come from the library's own reader
(geolip.alephllm.train.surface.read_spelled, fp32, one document a row) at every stream block of e038:
  A0 / B0     her trunk bare: her own bytes / Qwen's spelling (the same reading as e038's, through the library's reader)
  A8 / B8     the arm's group mounted (the frozen partners it was trained over), the arm masked: the arm's target / the gap before
  B8q / A8q   the arm on: THE ARM'S READING of Qwen's spelling / her own bytes with the arm on (silence)
  UA0 / UB0   the untrained copy (random init, seed 0), bare: the comparator of the mood rule
  UBq         the untrained copy with its own arm (the registry's '<surface>-untrained' row), when published (--untrained-arm=1)
THE GAUGES (triangulate.align): every set against Q (Qwen's final state), Q16 and Q0 (alignment, context alignment, the
unwhitened twins, the ridge to Q), the mood directions in Q's frame; the gap (A8 <-> B8q; before the arm A8 <-> B8 and A0 <-> B0);
the silence (A8 <-> A8q); the triangle (A8 -> B8q -> Q and B8q -> A8 -> Q; before the arm A8 -> B8 -> Q).
THE DECISIONS (registered in plans/2026-10-08_qwen_arm_read_and_session6.md, part A):
  B3   B8q against Q at block 24 within B3_MARGIN of e038's A0 (A8 beside); the detour (direct - composed) within B3_MARGIN on
       every path
  B4   the mood axis of B8q at blocks 12-20 at least half of A8's (its cosine with Q's axis in Q's frame)
  THE SLIDER'S BLOCK = SLIDER_BLOCK unless B8q fails e038's mood rule there on both draws (the axis cosine above UB0's by more
       than read.DIR_MARGIN and at least 3 of the 4 unseen gloomy phrases on Qwen's gloomy side); then the block of 12-20
       passing on both draws with the largest alignment to Q
Writes arm_read_<surface>_step245674.json / .md and every set's rows (resumable) to <settings.HOME>/triangulation; a file named
STOP there ends the run after the current set.
Usage: python -m alephllm_diffusion.triangulate.arm_read [--surface=qwen3] [--untrained-arm=1] [--smoke=1]"""
import json
import os
import sys
import time

import torch

from .. import __version__, settings
from ..beatrix import extract as bx
from ..paths import git_commit, repo_root
from . import align as AL
from . import read as RD

SETS = {   # name -> (trunk, mounted, surface, the arm on)
    "A0": ("trained", None, "A", None), "B0": ("trained", None, "B", None),
    "A8": ("trained", "group", "A", False), "B8": ("trained", "group", "B", False),
    "B8q": ("trained", "group", "B", True), "A8q": ("trained", "group", "A", True),
    "UA0": ("untrained", None, "A", None), "UB0": ("untrained", None, "B", None),
    "UBq": ("untrained", "arm", "B", True),
}
PAIRS = {"gap": ("A8", "B8q"), "gap before": ("A8", "B8"), "gap bare": ("A0", "B0"), "silence": ("A8", "A8q")}
TRIANGLES = {"A8->B8q->Q": ("A8", "B8q"), "B8q->A8->Q": ("B8q", "A8"), "A8->B8->Q": ("A8", "B8")}
E038_A0 = 0.562                  # e038: her own bytes (bare) against Q at block 24, the reference of B3
B3_MARGIN, B3_BLOCK, B4_BLOCKS = 0.05, 24, (12, 14, 16, 18, 20)
SLIDER_BLOCK, SLIDER_BAND = 20, (12, 14, 16, 18, 20)
OUT = RD.OUT
T0 = time.time()


def say(*a):
    print(f"[arm read {time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------------------------------- the rows
def row_index(table, n_caps: int, n_texts: int) -> torch.Tensor:
    """the reader's site rows that are e038's rows, in e038's order: every site of the captions (text index < n_caps), then the
    LAST site of each mood text (its token closing at the full stop). table = the reader's (text index, token position, id)."""
    t = torch.as_tensor(table)
    cap = torch.nonzero(t[:, 0] < n_caps).flatten()
    last = []
    for j in range(n_caps, n_texts):
        hit = torch.nonzero(t[:, 0] == j).flatten()
        assert len(hit), f"text {j} gave no site"
        last.append(int(hit[-1]))
    return torch.cat([cap, torch.tensor(last, dtype=torch.long)])


def check_rows(table, D) -> None:
    """the selected sites must be e038's rows exactly: the same (text, token position) and the same token id, in the same order."""
    t = torch.as_tensor(table)
    want = torch.tensor([[i, p] for i, p in D["rows"]], dtype=torch.long)
    assert t.shape[0] == want.shape[0], ("row count", t.shape[0], want.shape[0])
    assert torch.equal(t[:, :2], want), "the reader's sites are not e038's rows"
    assert torch.equal(t[:, 2], D["ids"]), "the reader's token ids are not e038's"


@torch.no_grad()
def read_set(model, tok, texts, D, surface, blocks, kw):
    """{block: (n_rows, d) fp32} on e038's rows: the library's reader on every text, its sites reduced to e038's rows."""
    from geolip.alephllm.train.surface import read_spelled
    r = read_spelled(model, tok, texts, blocks, surface=surface, amp=False, **kw)
    idx = row_index(r["sites"], len(texts) - int(D["mood"].sum()), len(texts))        # one row per mood text, after the captions
    check_rows(r["sites"][idx], D)
    for b in blocks:
        assert torch.isfinite(r["states"][b][idx]).all(), ("non-finite state", surface, b)
    return {b: r["states"][b][idx].contiguous() for b in blocks}


# ---------------------------------------------------------------------------------------------------- the gauges
def gauges(sets, Q, D, k, blocks):
    """one draw: C1 every set and block against Q / Q16 / Q0; C6 the directions; the pairs and the triangles."""
    cap, mood = ~D["mood"], D["mood"]
    g, ids = D["group"][cap], D["ids"][cap]
    Qc = {RD.Q_NAMES[q]: Q[q][cap] for q in RD.Q_NAMES}
    res = {"C1": {}, "C6": {}, "pairs": {}, "triangles": {}}
    for name, S in sets.items():
        for b in blocks:
            key = f"{name}|{b}"
            res["C1"][key] = {qn: AL.compare(S[b][cap], Qc[qn], g, k, ids, ridge=(qn == "Q")) for qn in ("Q", "Q16", "Q0")}
            res["C6"][key] = AL.directions(S[b][cap], Qc["Q"], S[b][mood], Q["28post"][mood], D["cls"], D["split"], k)
    for pname, (x, y) in PAIRS.items():
        if x in sets and y in sets:
            for b in blocks:
                res["pairs"][f"{pname}|{b}"] = AL.compare(sets[x][b][cap], sets[y][b][cap], g, k, ids, ridge=False)
    for tname, (x, y) in TRIANGLES.items():
        if x in sets and y in sets:
            for b in blocks:
                res["triangles"][f"{tname}|{b}"] = AL.triangle(sets[x][b][cap], sets[y][b][cap], Qc["Q"], g, k)
    return res


# ---------------------------------------------------------------------------------------------------- the decisions
def mood_rule(C6, name, comparator, b):
    """e038's rule for one set and block: (passes, axis gain over the comparator, unseen gloomy phrases right of total)."""
    t, u = C6[f"{name}|{b}"], C6[f"{comparator}|{b}"]
    gain = t["cos_axis"] - u["cos_axis"]
    right, total = t["mapped_down"]
    return bool(gain > RD.DIR_MARGIN and right >= 3), gain, [right, total]


def decide(R1, R2, blocks):
    """the registered decisions from both draws (R1, R2); B3 and B4 read on draw 1 with draw 2 beside."""
    out = {"B3": {}, "B4": {}, "slider_block": {}}
    for d, R in (("draw1", R1), ("draw2", R2)):
        if R is None:
            continue
        armed = R["C1"][f"B8q|{B3_BLOCK}"]["Q"]["alignment"]
        target = R["C1"].get(f"A8|{B3_BLOCK}", {}).get("Q", {}).get("alignment")
        detours = {k: v["gap"] for k, v in R["triangles"].items() if not k.startswith("A8->B8->")}
        out["B3"][d] = {"B8q": armed, "A8": target, "e038_A0": E038_A0,
                        "verdict": "MET" if armed >= E038_A0 - B3_MARGIN else "NOT MET",
                        "detours_within": sum(abs(v) <= B3_MARGIN for v in detours.values()), "paths": len(detours),
                        "largest_detour": max(detours.values(), key=abs) if detours else None}
        per = {}
        for b in B4_BLOCKS:
            if b in blocks:
                q, a = R["C6"][f"B8q|{b}"]["cos_axis"], R["C6"][f"A8|{b}"]["cos_axis"]
                per[b] = {"B8q": q, "A8": a, "met": bool(q >= a / 2)}
        out["B4"][d] = {"blocks": per, "verdict": "MET" if per and all(v["met"] for v in per.values()) else "NOT MET"}
    rules = {b: [mood_rule(R["C6"], "B8q", "UB0", b) for R in (R1, R2) if R is not None] for b in SLIDER_BAND if b in blocks}
    if SLIDER_BLOCK in rules and not all(not r[0] for r in rules[SLIDER_BLOCK]):
        pick, why = SLIDER_BLOCK, "the registered block passes the mood rule on at least one draw"
    else:
        passing = [b for b, rr in rules.items() if rr and all(r[0] for r in rr)]
        pick = max(passing, key=lambda b: R1["C1"][f"B8q|{b}"]["Q"]["alignment"]) if passing else None
        why = ("the registered block fails on both draws; the band's best passing block" if passing else
               "no block of the band passes on both draws: the slider waits for Phil")
    out["slider_block"] = {"block": pick, "why": why,
                           "rule": {b: [{"passes": r[0], "axis_gain": r[1], "unseen_gloomy": r[2]} for r in rr]
                                    for b, rr in rules.items()}}
    return out


# ---------------------------------------------------------------------------------------------------- the run
def _kw(surface_name):
    from geolip.alephllm.arm_mount import SURFACE_ARMS, reader_kwargs
    return reader_kwargs(SURFACE_ARMS[surface_name])


def main(surface_name="qwen3", untrained_arm=False, smoke=False):
    from transformers import AutoTokenizer
    from geolip.alephllm.arm_mount import SURFACE_ARMS, masked, mount_surface
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    n_caps = 48 if smoke else RD.N_CAPS
    blocks = [16, 20, 24] if smoke else RD.STREAM_BLOCKS
    k = 16 if smoke else RD.K
    tag = f"_{surface_name}" + ("_smoke" if smoke else "")
    os.makedirs(OUT, exist_ok=True)
    ledger_path = os.path.join(OUT, f"arm_read{tag}_step{RD.STEP}.json")
    row = SURFACE_ARMS[surface_name]
    member = row["member"]
    kw = _kw(surface_name)
    tok = AutoTokenizer.from_pretrained(os.path.join(settings.DP, "configs", "qwen3_06b"), local_files_only=True)
    caps, mood = RD.texts(n_caps)
    D = {d: RD.draw_rows(tok, caps[d], mood) for d in caps}
    texts = {d: caps[d] + mood for d in caps}
    qpath = os.path.join(OUT, f"sets_qwen{'_smoke' if smoke else ''}.pt")
    if not os.path.exists(qpath):
        raise SystemExit(f"e038's Qwen rows are not at {qpath} (run triangulate.read first)")
    Q = torch.load(qpath)
    for d in D:
        assert Q[d]["28post"].shape[0] == len(D[d]["rows"]), ("e038's Qwen rows do not match these rows", d)
    L = {"_meta": {"surface": surface_name, "registry_row": row, "step": RD.STEP, "k": k, "n_caps": n_caps, "blocks": blocks,
                   "smoke": smoke, "package": __version__, "commit": git_commit(repo_root()),
                   "rows": {d: {"captions": int((~D[d]["mood"]).sum()), "mood": int(D[d]["mood"].sum())} for d in D},
                   "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}}
    names = [n for n in SETS if n != "UBq" or untrained_arm]
    path = {(n, d): os.path.join(OUT, f"arm_sets{tag}_{n}_{d}.pt") for n in names for d in D}
    sets = {d: {} for d in D}
    t_sets, done = time.time(), 0
    for trunk in ("trained", "untrained"):
        todo = [n for n in names if SETS[n][0] == trunk]
        if all(os.path.exists(path[(n, d)]) for n in todo for d in D):
            for n in todo:
                for d in D:
                    sets[d][n] = torch.load(path[(n, d)])
            continue
        model = bx.build_model(step=RD.STEP if trunk == "trained" else None,
                               random_init_seed=None if trunk == "trained" else 0, device=dev)
        prog = None
        for n in todo:
            _, mounted, surface, arm_on = SETS[n]
            if mounted and prog is None:                   # the untrained copy's arm records no step of her trunk
                prog = (mount_surface(model, surface_name, device=dev) if trunk == "trained" else
                        mount_surface(model, f"{surface_name}-untrained", device=dev, require_step=None))
                say(f"mounted {sorted(prog.attached)} on the {trunk} trunk")
            for d in D:
                if os.path.exists(os.path.join(OUT, "STOP")):
                    say("STOP file found: the run ends here (the sets read so far are kept)")
                    return None
                if os.path.exists(path[(n, d)]):
                    sets[d][n] = torch.load(path[(n, d)])
                    continue
                if mounted and not arm_on:
                    with masked(prog, [member]):
                        S = read_set(model, tok, texts[d], D[d], surface, blocks, kw)
                else:
                    S = read_set(model, tok, texts[d], D[d], surface, blocks, kw)
                torch.save(S, path[(n, d)])
                sets[d][n] = S
                done += 1
                left = (time.time() - t_sets) / done * sum(not os.path.exists(p) for p in path.values())
                say(f"set {n} ({trunk}, {mounted or 'bare'}, surface {surface}, arm {'on' if arm_on else 'off'}) {d}: "
                    f"{len(D[d]['rows'])} rows; {time.time() - t_sets:.0f} s on sets, about {left:.0f} s left")
        del model, prog
        if dev == "cuda":
            torch.cuda.empty_cache()
    # the anchor: the library's reader against e038's own rows (her bare trunk, both surfaces, every block, draw 1)
    anchor = {}
    for n, form in (("A0", "A"), ("B0", "B")):
        p = os.path.join(OUT, f"sets_trained_{form}_draw1{'_smoke' if smoke else ''}.pt")
        if os.path.exists(p):
            old = torch.load(p)["stream"]
            anchor[n] = max(float((sets["draw1"][n][b] - old[b]).abs().max()) for b in blocks if b in old)
    L["_meta"]["anchor_against_e038"] = anchor
    say(f"anchor (the largest difference from e038's rows): {anchor}")
    R = {}
    for d in D:
        R[d] = gauges(sets[d], Q[d], D[d], k, blocks)
        L[d] = R[d]
        json.dump(L, open(ledger_path, "w", encoding="utf-8"), indent=1)
        say(f"{d}: every comparison done")
    L["decisions"] = decide(R["draw1"], R.get("draw2"), blocks)
    L["_meta"]["seconds"] = round(time.time() - T0, 1)
    L["_meta"]["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    json.dump(L, open(ledger_path, "w", encoding="utf-8"), indent=1)
    md = report(L)
    open(ledger_path[:-5] + ".md", "w", encoding="utf-8").write(md)
    say(f"wrote {ledger_path} and the tables")
    print(md, flush=True)
    return L


def report(L):
    """the tables, plain."""
    m, dec, R1, R2 = L["_meta"], L["decisions"], L["draw1"], L.get("draw2")
    blocks = m["blocks"]
    out = [f"# The read of record on the surface arm '{m['surface']}' (her step {m['step']}; k = {m['k']})", "",
           f"Rows: {m['rows']}. The anchor (the library's reader against e038's own rows, her bare trunk): "
           f"{m.get('anchor_against_e038')}.", "",
           "## Every set against the Qwen states Anima reads (whitened alignment, held-out captions; draw 1, draw 2 in brackets)",
           "", "| set | " + " | ".join(f"block {b}" for b in blocks) + " |", "|---|" + "---|" * len(blocks)]
    names = sorted({k.split("|")[0] for k in R1["C1"]}, key=lambda n: list(SETS).index(n))
    for n in names:
        cells = []
        for b in blocks:
            a = R1["C1"][f"{n}|{b}"]["Q"]["alignment"]
            a2 = R2["C1"][f"{n}|{b}"]["Q"]["alignment"] if R2 else float("nan")
            cells.append(f"{a:.3f} ({a2:.3f})")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    out += ["", "## The pairs (her readings against each other; draw 1)", "",
            "| pair | " + " | ".join(f"block {b}" for b in blocks) + " |", "|---|" + "---|" * len(blocks)]
    for p in PAIRS:
        if f"{p}|{blocks[0]}" in R1["pairs"]:
            out.append(f"| {p} ({' <-> '.join(PAIRS[p])}) | "
                       + " | ".join(f"{R1['pairs'][f'{p}|{b}']['alignment']:.3f}" for b in blocks) + " |")
    out += ["", "## The triangle (draw 1: direct / composed / gap)", "",
            "| path | " + " | ".join(f"block {b}" for b in blocks) + " |", "|---|" + "---|" * len(blocks)]
    for t in TRIANGLES:
        if f"{t}|{blocks[0]}" in R1["triangles"]:
            out.append(f"| {t} | " + " | ".join(
                f"{R1['triangles'][f'{t}|{b}']['direct']:.3f} / {R1['triangles'][f'{t}|{b}']['composed']:.3f} / "
                f"{R1['triangles'][f'{t}|{b}']['gap']:+.3f}" for b in blocks) + " |")
    out += ["", "## The mood axis in Qwen's frame (draw 1: axis cosine; unseen gloomy phrases on Qwen's gloomy side)", "",
            "| set | " + " | ".join(f"block {b}" for b in blocks) + " |", "|---|" + "---|" * len(blocks)]
    for n in names:
        out.append(f"| {n} | " + " | ".join(
            f"{R1['C6'][f'{n}|{b}']['cos_axis']:+.3f} ({R1['C6'][f'{n}|{b}']['mapped_down'][0]}/"
            f"{R1['C6'][f'{n}|{b}']['mapped_down'][1]})" for b in blocks) + " |")
    out += ["", "## The decisions", ""]
    for d, v in dec["B3"].items():
        out.append(f"- B3 ({d}): the arm's reading against Q at block {B3_BLOCK} {v['B8q']:.3f} (e038's own bytes {v['e038_A0']}, "
                   f"the mounted plain reading {v['A8'] if v['A8'] is None else round(v['A8'], 3)}) -> {v['verdict']}; detours "
                   f"within {B3_MARGIN} on {v['detours_within']} of {v['paths']} paths (largest {v['largest_detour']:+.3f})")
    for d, v in dec["B4"].items():
        out.append(f"- B4 ({d}): " + ", ".join(f"block {b} {x['B8q']:+.3f} against {x['A8']:+.3f} "
                                              f"{'ok' if x['met'] else 'NO'}" for b, x in v["blocks"].items())
                   + f" -> {v['verdict']}")
    sb = dec["slider_block"]
    out.append(f"- THE SLIDER'S BLOCK: {sb['block']} ({sb['why']})")
    for b, rr in sb["rule"].items():
        out.append(f"  - block {b}: " + "; ".join(f"{'passes' if r['passes'] else 'fails'} (axis gain {r['axis_gain']:+.3f}, "
                                                    f"unseen gloomy {r['unseen_gloomy'][0]}/{r['unseen_gloomy'][1]})" for r in rr))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    flags = {x.split("=", 1)[0][2:]: x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--") and "=" in x}
    main(surface_name=flags.get("surface", "qwen3"), untrained_arm=flags.get("untrained-arm", "0") == "1",
         smoke=flags.get("smoke", "0") == "1")
    sys.exit(0)
