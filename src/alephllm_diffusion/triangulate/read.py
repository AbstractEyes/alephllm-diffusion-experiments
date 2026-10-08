"""alephllm_diffusion.triangulate.read - THE DUAL-EXTRACTION READ (registered 2026-10-07 before any read; image-free; no training).
Beatrix's two signals, each read on two byte forms of the same caption, against the Qwen3 states Anima's adapter reads, one row
per Qwen token (triangulate.rows):
  A   her reading of the caption's own bytes          trunk = the block output at the token's closing byte (LayerNorm without
                                                       affine); hub = the block's blackboard after the bytes through that byte
  B   her reading of Qwen's spelling of the tokens    the same two signals at the token's closing byte in the spelled text
  Q   Qwen3 0.6B's final hidden state at the token (Anima's adapter source; the target); descriptive beside it Q16 (the residual
      entering Qwen's layer 16) and Q0 (Qwen's input embedding of the token: token identity alone)
Her trunk at step 245,674 and the same architecture untrained (random init, seed 0: the control that must fail); both signals
from one pass over each row's prefix (the blackboard sums the whole text read so far), fp32, prefixes of equal length batched
together (no padding). The blackboard (262,144 wide) enters through its exact Gram coordinates, the leading R_HUB kept (the
retained share is recorded).
ROWS: the first N_CAPS captions of each COCO draw of the hub read (every token but a caption's first, Qwen's attention sink, and
its last, which no byte follows) and the connectors' 74 mood rows in the grid's caption form (one row per phrase, its last token);
mood rows never enter a fit.
THE COMPARISONS (triangulate.align; whitened to K dims; rotations fit on half the captions, scored on the other half):
  C1 every signal of her -> Q (and Q16, Q0): alignment, context alignment (token means removed), both again unwhitened (each
     side at its own variance), ridge R^2, debiased CKA (a cross-check that fits nothing)
  C2 A <-> B per kind and block;  C3 trunk <-> hub per extraction and block
  C4 the pair: ridge to Q from trunk + hub together against the better one alone; from all four
  C5 the triangle: A -> B -> Q composed (each leg fit on its own captions) against A -> Q direct (and B -> A -> Q)
  C6 the directions: her mood directions mapped into Q's frame against Q's own: the mood axis (cheerful minus gloomy), the part
     both sides share (their mean minus neutral) and each side (a class minus neutral); the held-out phrases read from her own
     mapped centre along Q's axis
  C7 her final block's output size against Qwen's residual size before its final norm, and against each row's alignment
THE DECISIONS (registered; amendment 1 before the full read: the axis rule and the centre of the held-out read): the bar =
|draw 1 - draw 2| on her trained trunk, never below BAR_FLOOR. D1 the extraction (B against A, per kind, best block); D2 the signal
(hub against trunk, per extraction); D3 the pair adds; D4 the triangle closes; D5 a signal not beating its untrained copy by the
bar is NOT LEARNED; D6 a signal carries a side into Q's frame when its mapped mood axis beats the untrained copy's cosine by
DIR_MARGIN and at least 3 of the 4 held-out phrases of that side land on that side; THE PICK. The unwhitened columns' verdicts are
reported beside the whitened ones and decide nothing.
Reads coco_caps_2x2048.json from settings.DATA_DIR, her checkpoint (the hub cache), Anima's Qwen3 file (settings.LLM_PATH).
Writes triangulation_step245674.json (after every set) and .md (the tables) to <settings.HOME>/triangulation; the rows of each
set are kept there (sets_*.pt) so a restart resumes; a file named STOP there ends the run after the current set.
Usage: python -m alephllm_diffusion.triangulate.read [--smoke=1]"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from .. import __version__, settings
from ..beatrix import extract as bx
from ..beatrix import gauges as G
from ..hubs import read as H
from ..mount import read as MR
from ..paths import git_commit, repo_root
from . import align as AL
from . import rows as RW

STEP = 245674
STREAM_BLOCKS = [8, 12, 14, 16, 18, 20, 22, 24, 28]
HUB_BLOCKS = [14, 16, 18, 20, 22, 24]
MAG_BLOCK = 31                                  # her last block: its output's size (the big direction's amplitude), read only
Q_DEPTHS = (0, 16, "28pre", "28post")           # Qwen: input embedding, layer 16's input, before and after the final norm
Q_NAMES = {0: "Q0", 16: "Q16", "28post": "Q"}
K = 128                                         # whitened dimensions
R_HUB = 1024                                    # the blackboard's Gram coordinates kept
N_CAPS = 512                                    # captions per draw
BAR_FLOOR, DIR_MARGIN = 0.01, 0.10
OUT = os.path.join(str(settings.HOME), "triangulation")
STOP = os.path.join(OUT, "STOP")
T0 = time.time()


def say(*a):
    print(f"[triangulation {time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------------------------------- the rows
def texts(n_caps):
    from geolip_anima_trainer import anima_experiments as ax
    caps = json.load(open(os.path.join(settings.DATA_DIR, "coco_caps_2x2048.json"), encoding="utf-8"))
    mood = [ax.PREFIX + MR.PREFIX_FRAME.replace("{w}", p) for _, _, p in H.ROWS]
    return {d: caps[d][:n_caps] for d in ("draw1", "draw2")}, mood


def draw_rows(tok, captions, mood):
    """{'caps': [Caption] (the draw's captions, then the mood texts), 'rows': [(caption index, token position)], 'group',
    'ids', 'mood': row mask, 'cls', 'split' (the mood rows' class and split, in H.ROWS order)}"""
    C = [RW.caption(tok, s) for s in captions] + [RW.caption(tok, s) for s in mood]
    rows = [(i, t) for i in range(len(captions)) for t in RW.token_positions(C[i])]
    n_cap_rows = len(rows)
    for j in range(len(mood)):
        c = C[len(captions) + j]
        assert c.raw.endswith(b"."), c.text
        rows.append((len(captions) + j, RW.phrase_end_token(c, len(c.raw) - 1)))
    return {"caps": C, "rows": rows, "group": torch.tensor([i for i, _ in rows]),
            "ids": torch.tensor([C[i].ids[t] for i, t in rows]),
            "mood": torch.arange(len(rows)) >= n_cap_rows,
            "cls": [c for c, _, _ in H.ROWS], "split": [s for _, s, _ in H.ROWS]}


def sequences(D, form):
    """each row's prefix in byte form 'A' (the caption's bytes) or 'B' (Qwen's spelling), through the token's closing byte."""
    k = 0 if form == "A" else 1
    return [RW.prefixes(D["caps"][i], t)[k] for i, t in D["rows"]]


# ---------------------------------------------------------------------------------------------------- her signals
@torch.no_grad()
def centred_gram(X, chunk=512):
    """the centred Gram (n x n, float64) of fp16 rows on their device, the mean and every block in fp32 chunks (no full fp32 copy
    of the rows: 5,000 blackboard rows are 5 GB in fp32)."""
    n = X.shape[0]
    mu = sum(X[i:i + chunk].float().sum(0, keepdim=True) for i in range(0, n, chunk)) / n
    Kc = torch.zeros(n, n, dtype=torch.float64, device=X.device)
    for i in range(0, n, chunk):
        Ai = X[i:i + chunk].float() - mu
        for j in range(i, n, chunk):
            blk = (Ai @ (X[j:j + chunk].float() - mu).T).double()
            Kc[i:i + chunk, j:j + chunk] = blk
            Kc[j:j + chunk, i:i + chunk] = blk.T
    return Kc


@torch.no_grad()
def her_rows(model, seqs, dev, stream_blocks, hub_blocks, per_pass, r=None):
    """her states at the last byte of each byte sequence (the DOC byte in front), fp32, equal lengths batched:
    stream {block: (n, d)} the block output LayerNorm'd without affine; size (n,) the raw output size of block MAG_BLOCK;
    hub {block: (n, r)} the blackboard's Gram coordinates; share {block: retained variance}."""
    r = R_HUB if r is None else r
    doc = bx._doc_id()
    ids = [torch.tensor([doc] + list(s), dtype=torch.long) for s in seqs]
    groups = H.equal_groups([x.numel() for x in ids])
    n, d = len(seqs), model.nf.weight.numel()
    last = min(MAG_BLOCK, len(model.blocks) - 1)
    stream = {b: torch.zeros(n, d) for b in stream_blocks}
    size = torch.zeros(n)
    for grp in groups:
        gi = torch.tensor(grp)
        x = model.embed(torch.stack([ids[i] for i in grp]).to(dev))
        for bi, blk in enumerate(model.blocks):
            x, _ = blk.prefill(x)
            if bi in stream:
                v = x[:, -1].float()
                assert torch.isfinite(v).all(), ("non-finite state", bi, grp[:4])
                stream[bi][gi] = F.layer_norm(v, (d,)).cpu()
            if bi == last:
                size[gi] = x[:, -1].float().norm(dim=-1).cpu()
                break
    hub, share = {}, {}
    hub_blocks = sorted(hub_blocks)
    for c0 in range(0, len(hub_blocks), per_pass):
        part = hub_blocks[c0:c0 + per_pass]
        buf = {}
        for grp in groups:
            gi = torch.tensor(grp, device=dev)
            x = model.embed(torch.stack([ids[i] for i in grp]).to(dev))
            for bi, blk in enumerate(model.blocks):
                x, cache = blk.prefill(x)
                if bi in part:
                    rows = bx._blackboard(cache)
                    assert torch.isfinite(rows).all(), ("non-finite blackboard", bi, grp[:4])
                    if bi not in buf:
                        buf[bi] = torch.empty(n, rows.shape[1], dtype=torch.float16, device=dev)
                    buf[bi][gi] = rows.to(torch.float16)
                if bi == part[-1]:
                    break
        for b in part:
            Kc = centred_gram(buf.pop(b))
            coords, share[b] = AL.gram_coordinates(Kc, r)
            hub[b] = coords.float().cpu()
            del Kc, coords
        if dev == "cuda":
            torch.cuda.empty_cache()
    return {"stream": stream, "size": size, "hub": hub, "share": share}


# ---------------------------------------------------------------------------------------------------- Qwen's states
def load_qwen(dev):
    """Qwen3 0.6B from Anima's text-encoder file, fp32, as the grid's loader builds it (stitch.s2.load_models' first half; the
    adapter is not needed here, so the diffusion model's file is never opened)."""
    import transformers
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    cfg = transformers.Qwen3Config.from_pretrained(os.path.join(settings.DP, "configs", "qwen3_06b"), local_files_only=True)
    cfg.use_cache = False
    with init_empty_weights():
        lm = transformers.Qwen3ForCausalLM(cfg)
    with safe_open(settings.LLM_PATH, "pt") as f:
        for key in f.keys():
            set_module_tensor_to_device(lm, key, device="cpu", dtype=torch.float32, value=f.get_tensor(key))
    return lm.model.to(dev).eval().requires_grad_(False)


@torch.no_grad()
def qwen_rows(qwen, D, dev, bs=64):
    """{depth: (n_rows, 1024)}: Qwen3's residual at each row's token (Anima's tokenization, right-padded batches, fp32)."""
    from ..stitch import s2
    by_cap: dict = {}
    for r, (i, t) in enumerate(D["rows"]):
        by_cap.setdefault(i, []).append((r, t))
    out = {q: torch.zeros(len(D["rows"]), 1024) for q in Q_DEPTHS}
    order = sorted(by_cap)
    for b0 in range(0, len(order), bs):
        part = order[b0:b0 + bs]
        _, got = s2.qwen_run(qwen, [list(D["caps"][i].ids) for i in part], dev, collect_all=list(Q_DEPTHS))
        for j, i in enumerate(part):
            for r, t in by_cap[i]:
                for q in Q_DEPTHS:
                    out[q][r] = got[q][j, t]
    return out


# ---------------------------------------------------------------------------------------------------- the comparisons
def signals(S, form):
    """{name: tensor} for one set: 'A-trunk-18', 'B-hub-22', ..."""
    out = {f"{form}-trunk-{b}": v for b, v in S["stream"].items()}
    out.update({f"{form}-hub-{b}": v for b, v in S["hub"].items()})
    return out


def row_cosines(X, Y, groups, k):
    """per row, the cosine between its rotated X and its Y, each row read from the fold that held it out."""
    out = torch.zeros(X.shape[0], dtype=torch.float64)
    for fit, held in AL.folds(groups):
        wx, wy = AL.Whitener(X[fit], k), AL.Whitener(Y[fit], k)
        P = wx(X[held]) @ AL.rotation(wx(X[fit]), wy(Y[fit]))
        out[held] = F.cosine_similarity(P, wy(Y[held]), dim=1).cpu()
    return out


def gauges(sets, Q, D, k):
    """every comparison of one draw. sets = {(trunk, form): her_rows output}; Q = qwen_rows output."""
    cap = ~D["mood"]
    g, ids = D["group"][cap], D["ids"][cap]
    cls, split = D["cls"], D["split"]
    Qc = {Q_NAMES[q]: Q[q][cap] for q in Q_NAMES}
    res = {"C1": {}, "C2": {}, "C3": {}, "C4": {}, "C5": {}, "C6": {}, "C7": {}, "floors": {}}
    res["floors"]["Q0->Q"] = AL.compare(Qc["Q0"], Qc["Q"], g, k, ids)
    res["floors"]["Q16->Q"] = AL.compare(Qc["Q16"], Qc["Q"], g, k, ids)
    q_prep = G.prep(Qc["Q"])                                            # the debiased-CKA cross-check (no fit involved)
    for (trunk, form), S in sets.items():
        sig = signals(S, form)
        for name, X in sig.items():
            key = f"{trunk}|{name}"
            res["C1"][key] = {qn: AL.compare(X[cap], Qc[qn], g, k, ids, ridge=(qn == "Q")) for qn in ("Q", "Q16", "Q0")}
            res["C1"][key]["Q"]["cka_debiased"] = G.cka_debiased(G.prep(X[cap]), q_prep)
            res["C6"][key] = AL.directions(X[cap], Qc["Q"], X[D["mood"]], Q["28post"][D["mood"]], cls, split, k)
        for b in sorted(set(S["stream"]) & set(S["hub"])):
            res["C3"][f"{trunk}|{form}-{b}"] = AL.compare(S["stream"][b][cap], S["hub"][b][cap], g, k, ids, ridge=False)
        res["C7"][f"{trunk}|{form}"] = {"size_vs_qwen_size": AL.spearman(S["size"][cap], Q["28pre"][cap].norm(dim=1))}
    for trunk in sorted({t for t, _ in sets}):
        if (trunk, "A") not in sets or (trunk, "B") not in sets:
            continue
        a, b = signals(sets[(trunk, "A")], "A"), signals(sets[(trunk, "B")], "B")
        for name in a:
            other = "B" + name[1:]
            res["C2"][f"{trunk}|{name[2:]}"] = AL.compare(a[name][cap], b[other][cap], g, k, ids, ridge=False)
            res["C5"][f"{trunk}|A->B->Q|{name[2:]}"] = AL.triangle(a[name][cap], b[other][cap], Qc["Q"], g, k)
            res["C5"][f"{trunk}|B->A->Q|{name[2:]}"] = AL.triangle(b[other][cap], a[name][cap], Qc["Q"], g, k)
        # THE PAIR: each extraction's best trunk block with its best hub block, then all four
        best = {}
        for form, sig in (("A", a), ("B", b)):
            for kind in ("trunk", "hub"):
                names = [n for n in sig if n.split("-")[1] == kind]
                if names:
                    best[(form, kind)] = max(names, key=lambda n: res["C1"][f"{trunk}|{n}"]["Q"]["alignment"])
        allsig = {**a, **b}
        for form in ("A", "B"):
            if (form, "trunk") in best and (form, "hub") in best:
                t_, h_ = best[(form, "trunk")], best[(form, "hub")]
                solo = max(res["C1"][f"{trunk}|{t_}"]["Q"]["ridge_r2"], res["C1"][f"{trunk}|{h_}"]["Q"]["ridge_r2"])
                pair = AL.pair_ridge([allsig[t_][cap], allsig[h_][cap]], Qc["Q"], g, k)
                res["C4"][f"{trunk}|{form}"] = {"trunk": t_, "hub": h_, "solo": solo, "pair": pair, "gain": pair - solo}
        if len(best) == 4:
            names = [best[x] for x in sorted(best)]
            pairs = [res["C4"][f"{trunk}|{f}"]["pair"] for f in ("A", "B") if f"{trunk}|{f}" in res["C4"]]
            four = AL.pair_ridge([allsig[n][cap] for n in names], Qc["Q"], g, k)
            res["C4"][f"{trunk}|all four"] = {"signals": names, "best_pair": max(pairs), "four": four, "gain": four - max(pairs)}
        # C7: her size against the per-row alignment of her best A signal
        if ("A", "trunk") in best:
            n_ = best[("A", "trunk")]
            rc = row_cosines(allsig[n_][cap], Qc["Q"], g, k)
            res["C7"][f"{trunk}|A"]["size_vs_row_alignment"] = AL.spearman(sets[(trunk, "A")]["size"][cap], rc)
            res["C7"][f"{trunk}|A"]["signal"] = n_
    return res


# ---------------------------------------------------------------------------------------------------- the decisions
def _bar(r1, r2, path):
    def get(r):
        for p in path:
            r = r[p]
        return r
    try:
        return max(BAR_FLOOR, abs(get(r1) - get(r2)))
    except (KeyError, TypeError):
        return BAR_FLOOR


def _verdict(x, y, bar, x_name, y_name):
    return x_name if x - y > bar else (y_name if y - x > bar else "TIED")


def decide(R1, R2):
    """the registered decisions from draw 1's gauges (R1), each bar from draw 2's (R2)."""
    C1, C1b = R1["C1"], R2["C1"]
    out = {"D1": {}, "D2": {}, "D3": {}, "D4": {}, "D5": {}, "D6": {}}

    def best(form, kind, col="alignment"):
        names = [n for n in C1 if n.startswith(f"trained|{form}-{kind}-")]
        n = max(names, key=lambda n: C1[n]["Q"][col])
        return n, C1[n]["Q"][col], _bar(C1, C1b, (n, "Q", col))

    for kind in ("trunk", "hub"):
        for col in ("alignment", "context_alignment", "alignment_unwhitened", "context_alignment_unwhitened"):
            (na, va, ba), (nb, vb, bb) = best("A", kind, col), best("B", kind, col)
            bar = max(ba, bb)
            out["D1"][f"{kind}|{col}"] = {"A": [na, va], "B": [nb, vb], "bar": bar,
                                          "verdict": _verdict(vb, va, bar, "QWEN'S SPELLING CLOSER", "HER OWN BYTES CLOSER")}
    for form in ("A", "B"):
        for col in ("alignment", "alignment_unwhitened"):
            (nt, vt, bt), (nh, vh, bh) = best(form, "trunk", col), best(form, "hub", col)
            bar = max(bt, bh)
            out["D2"][f"{form}|{col}"] = {"trunk": [nt, vt], "hub": [nh, vh], "bar": bar,
                                          "verdict": _verdict(vh, vt, bar, "HUB CLOSER", "TRUNK CLOSER")}
    for key, v in R1["C4"].items():
        if not key.startswith("trained|"):
            continue
        bar = _bar(R1["C4"], R2["C4"], (key, "gain"))
        out["D3"][key] = {**v, "bar": bar, "verdict": "THE SECOND SIGNAL ADDS" if v["gain"] > bar else "NOTHING ADDED"}
    for key, v in R1["C5"].items():
        if not key.startswith("trained|"):
            continue
        bar = _bar(R1["C5"], R2["C5"], (key, "gap"))
        out["D4"][key] = {**v, "bar": bar, "verdict": "CLOSES" if abs(v["gap"]) <= bar else "OPEN"}
    for key in C1:
        if not key.startswith("trained|"):
            continue
        name = key.split("|", 1)[1]
        u = f"untrained|{name}"
        if u not in C1:
            continue
        bar = _bar(C1, C1b, (key, "Q", "alignment"))
        m = C1[key]["Q"]["alignment"] - C1[u]["Q"]["alignment"]
        mc = C1[key]["Q"]["context_alignment"] - C1[u]["Q"]["context_alignment"]
        out["D5"][name] = {"margin": m, "context_margin": mc, "bar": bar, "verdict": "LEARNED" if m > bar else "NOT LEARNED"}
        t6, u6 = R1["C6"][key], R1["C6"][u]
        axis_gain = t6["cos_axis"] - u6["cos_axis"]
        sides = {"axis": {"cos": t6["cos_axis"], "cos_untrained": u6["cos_axis"], "gain": axis_gain},
                 "shared": {"cos": t6["cos_shared"], "cos_untrained": u6["cos_shared"]}}
        for side in ("up", "down"):
            right, total = t6[f"mapped_{side}"]
            sides[side] = {"cos": t6[f"cos_{side}"], "cos_untrained": u6[f"cos_{side}"],
                           "gain": t6[f"cos_{side}"] - u6[f"cos_{side}"], "held_out": [right, total],
                           "carries": bool(axis_gain > DIR_MARGIN and right >= 3)}
        out["D6"][name] = sides
    pool = [n for n, v in out["D6"].items() if v["down"]["carries"] and out["D5"][n]["verdict"] == "LEARNED"]
    if pool:
        top = max(pool, key=lambda n: out["D5"][n]["context_margin"])
        bar = out["D5"][top]["bar"]
        ties = [n for n in pool if out["D5"][top]["context_margin"] - out["D5"][n]["context_margin"] <= bar]
        pick = max(ties, key=lambda n: C1[f"trained|{n}"]["Q"]["alignment"])
        out["PICK"] = {"signal": pick, "context_margin": out["D5"][pick]["context_margin"],
                       "alignment": C1[f"trained|{pick}"]["Q"]["alignment"], "candidates": pool}
    else:
        out["PICK"] = {"signal": None, "verdict": "NO READING CARRIES THE GLOOMY SIDE"}
    return out


# ---------------------------------------------------------------------------------------------------- the run
def _per_pass(dev, n):
    return H._per_pass(dev, n)


def main(smoke=False):
    from transformers import AutoTokenizer
    global K, R_HUB
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda"
    G.dev = dev
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    n_caps = 48 if smoke else N_CAPS
    stream_blocks, hub_blocks = ([16, 18], [18, 22]) if smoke else (STREAM_BLOCKS, HUB_BLOCKS)
    if smoke:
        K, R_HUB = 16, 256
    os.makedirs(OUT, exist_ok=True)
    tag = "_smoke" if smoke else ""
    ledger_path = os.path.join(OUT, f"triangulation_step{STEP}{tag}.json")
    tok = AutoTokenizer.from_pretrained(os.path.join(settings.DP, "configs", "qwen3_06b"), local_files_only=True)
    caps, mood = texts(n_caps)
    D = {d: draw_rows(tok, caps[d], mood) for d in caps}
    L = {"_meta": {"step": STEP, "k": K, "r_hub": R_HUB, "n_caps": n_caps, "stream_blocks": stream_blocks,
                   "hub_blocks": hub_blocks, "smoke": smoke, "package": __version__, "commit": git_commit(repo_root()),
                   "rows": {d: {"captions": int((~D[d]["mood"]).sum()), "mood": int(D[d]["mood"].sum())} for d in D},
                   "spelling_bytes": {d: [sum(len(c.raw) for c in D[d]["caps"]), sum(len(c.spelled) for c in D[d]["caps"])]
                                      for d in D},
                   "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}}
    say(f"rows: {L['_meta']['rows']}; the spelled text is {L['_meta']['spelling_bytes']} bytes (raw, spelled)")

    # Qwen first (one model on the card at a time)
    qpath = os.path.join(OUT, f"sets_qwen{tag}.pt")
    if os.path.exists(qpath):
        Q = torch.load(qpath)
    else:
        qwen = load_qwen(dev)
        Q = {d: qwen_rows(qwen, D[d], dev) for d in D}
        del qwen
        if dev == "cuda":
            torch.cuda.empty_cache()
        torch.save(Q, qpath)
    say("Qwen's states read")

    plan = [("trained", "A", "draw1"), ("trained", "B", "draw1"), ("trained", "A", "draw2"), ("trained", "B", "draw2"),
            ("untrained", "A", "draw1"), ("untrained", "B", "draw1")]
    sets = {}
    t_sets = time.time()
    done_sets = 0
    for trunk in ("trained", "untrained"):
        todo = [p for p in plan if p[0] == trunk]
        paths = {p: os.path.join(OUT, f"sets_{p[0]}_{p[1]}_{p[2]}{tag}.pt") for p in todo}
        missing = [p for p in todo if not os.path.exists(paths[p])]
        model = None
        if missing:
            model = bx.build_model(step=STEP if trunk == "trained" else None, random_init_seed=None if trunk == "trained" else 0,
                                   device=dev)
        for p in todo:
            if os.path.exists(STOP):
                say("STOP file found: the run ends here (the sets read so far are kept)")
                return None
            if os.path.exists(paths[p]):
                sets[p] = torch.load(paths[p])
            else:
                seqs = sequences(D[p[2]], p[1])
                S = her_rows(model, seqs, dev, stream_blocks, hub_blocks, _per_pass(dev, len(seqs)))
                torch.save(S, paths[p])
                sets[p] = S
                done_sets += 1
                el = time.time() - t_sets
                left = el / done_sets * len([q for q in plan if not os.path.exists(os.path.join(OUT, f"sets_{q[0]}_{q[1]}_{q[2]}{tag}.pt"))])
                say(f"set {p}: {len(seqs)} rows; hub shares kept {[round(v, 4) for v in S['share'].values()]}; "
                    f"{el:.0f} s on sets, about {left:.0f} s left")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    # rows far larger than the rest (a first-position or delimiter outlier would dominate the whitening): counted, kept
    def big(v):
        return int((v > 10 * v.median()).sum())
    L["_meta"]["outliers"] = {
        **{f"qwen|{d}": big(Q[d]["28pre"][~D[d]["mood"]].norm(dim=1)) for d in D},
        **{f"her size|{t}|{f}|{d}": big(sets[(t, f, d)]["size"][~D[d]["mood"]]) for (t, f, d) in sets}}
    say(f"outlier rows: {L['_meta']['outliers']}")
    # the anchor: the prefix pass's stream equals the full caption's per-byte state at the closing byte (A, trained, block 18)
    L["_meta"]["anchor"] = anchor(D["draw1"], dev, stream_blocks[-1] if smoke else 18)
    say(f"anchor: {L['_meta']['anchor']}")

    R = {}
    for d in ("draw1", "draw2"):
        S_d = {(t, f): sets[(t, f, dd)] for (t, f, dd) in sets if dd == d}
        R[d] = gauges(S_d, Q[d], D[d], K)
        L[d] = R[d]
        L["_meta"]["shares"] = {f"{t}|{f}|{dd}": sets[(t, f, dd)]["share"] for (t, f, dd) in sets}
        json.dump(L, open(ledger_path, "w", encoding="utf-8"), indent=1)
        say(f"{d}: every comparison done")
    L["decisions"] = decide(R["draw1"], R["draw2"])
    L["_meta"]["seconds"] = round(time.time() - T0, 1)
    L["_meta"]["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    json.dump(L, open(ledger_path, "w", encoding="utf-8"), indent=1)
    md = report(L)
    open(ledger_path[:-5] + ".md", "w", encoding="utf-8").write(md)
    say(f"wrote {ledger_path} and the tables")
    print(md, flush=True)
    return L


@torch.no_grad()
def anchor(D, dev, block, n=16):
    """the largest difference between the prefix pass's stream at a row's last byte and the full caption's per-byte state at
    that byte (her trained trunk, form A, the first n captions' rows): the two passes must agree to fp32 rounding."""
    model = bx.build_model(step=STEP, device=dev)
    rows = [(i, t) for i, t in D["rows"] if i < n]                       # caption rows (the mood texts come after the captions)
    full = bx.extract(model, [D["caps"][i].text for i in range(n)], [block], taps=("byte",), device=dev, amp=False)
    seqs = [RW.prefixes(D["caps"][i], t)[0] for i, t in rows]
    S = her_rows(model, seqs, dev, [block], [], 1)
    d = model.nf.weight.numel()
    worst = 0.0
    for r, (i, t) in enumerate(rows):
        v = F.layer_norm(full["byte"][block][i][D["caps"][i].a_end[t]].float(), (d,))
        worst = max(worst, float((v - S["stream"][block][r]).abs().max()))
    del model
    return {"block": block, "rows": len(rows), "max_abs_diff": worst}


def report(L):
    """the tables, plain."""
    dec = L["decisions"]
    C1 = L["draw1"]["C1"]
    out = [f"# The dual-extraction read (her step {L['_meta']['step']}; k = {L['_meta']['k']})", ""]
    out += ["## Her signals against the Qwen states Anima reads (held-out captions)", "",
            "| signal | alignment | untrained | context alignment | untrained | unwhitened | unwhitened context | ridge R2 | "
            "debiased CKA | vs Qwen layer 16 | vs Qwen input |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    nan = float("nan")
    for key in sorted(k for k in C1 if k.startswith("trained|")):
        name = key.split("|", 1)[1]
        t, u = C1[key], C1.get(f"untrained|{name}")
        out.append(f"| {name} | {t['Q']['alignment']:.3f} | {u['Q']['alignment'] if u else nan:.3f} | "
                   f"{t['Q']['context_alignment']:.3f} | {u['Q']['context_alignment'] if u else nan:.3f} | "
                   f"{t['Q']['alignment_unwhitened']:.3f} | {t['Q']['context_alignment_unwhitened']:.3f} | "
                   f"{t['Q']['ridge_r2']:.3f} | {t['Q'].get('cka_debiased', nan):.3f} | {t['Q16']['alignment']:.3f} | "
                   f"{t['Q0']['alignment']:.3f} |")
    fl = L["draw1"]["floors"]
    out += ["", f"Qwen's own floors: its input embedding (token identity) -> its final state: alignment "
                f"{fl['Q0->Q']['alignment']:.3f}, context {fl['Q0->Q']['context_alignment']:.3f}; its layer 16 -> final: "
                f"{fl['Q16->Q']['alignment']:.3f}, context {fl['Q16->Q']['context_alignment']:.3f}.", ""]
    out += ["## The decisions", ""]
    for kind, v in dec["D1"].items():
        out.append(f"- D1 the extraction ({kind}): A {v['A'][0]} {v['A'][1]:.3f}, B {v['B'][0]} {v['B'][1]:.3f}, bar {v['bar']:.3f} "
                   f"-> {v['verdict']}")
    for key, v in dec["D2"].items():
        out.append(f"- D2 the signal ({key}): trunk {v['trunk'][0]} {v['trunk'][1]:.3f}, hub {v['hub'][0]} {v['hub'][1]:.3f}, "
                   f"bar {v['bar']:.3f} -> {v['verdict']}")
    for key, v in dec["D3"].items():
        base = v.get("solo", v.get("best_pair"))
        top = v.get("pair", v.get("four"))
        out.append(f"- D3 the pair ({key}): {base:.3f} -> {top:.3f} (gain {v['gain']:+.3f}, bar {v['bar']:.3f}) -> {v['verdict']}")
    for key, v in dec["D4"].items():
        out.append(f"- D4 the triangle ({key}): direct {v['direct']:.3f}, composed {v['composed']:.3f}, gap {v['gap']:+.3f} "
                   f"(bar {v['bar']:.3f}) -> {v['verdict']}")
    out += ["", "## The directions (each signal's mood directions mapped into Qwen's frame; untrained in brackets)", "",
            "| signal | learned (margin) | mood axis cos | shared part cos | cheerful side cos | cheerful held-out | "
            "gloomy side cos | gloomy held-out | carries cheerful / gloomy |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, v in sorted(dec["D6"].items()):
        d5 = dec["D5"][name]
        out.append(f"| {name} | {d5['verdict']} ({d5['margin']:+.3f}) | {v['axis']['cos']:+.3f} ({v['axis']['cos_untrained']:+.3f}) | "
                   f"{v['shared']['cos']:+.3f} ({v['shared']['cos_untrained']:+.3f}) | "
                   f"{v['up']['cos']:+.3f} ({v['up']['cos_untrained']:+.3f}) | {v['up']['held_out'][0]}/{v['up']['held_out'][1]} | "
                   f"{v['down']['cos']:+.3f} ({v['down']['cos_untrained']:+.3f}) | {v['down']['held_out'][0]}/{v['down']['held_out'][1]} | "
                   f"{'yes' if v['up']['carries'] else 'no'} / {'yes' if v['down']['carries'] else 'no'} |")
    q6 = next(iter(L["draw1"]["C6"].values()))
    out += ["", f"Qwen's own sides: cos(cheerful side, gloomy side) {q6['q_sides_cos']:+.3f}; its own held-out phrases on its own "
                f"axis: cheerful {q6['q_own_up'][0]}/{q6['q_own_up'][1]}, gloomy {q6['q_own_down'][0]}/{q6['q_own_down'][1]}.", ""]
    if "outliers" in L["_meta"]:
        out += [f"Outlier rows (more than 10 times the median size): {L['_meta']['outliers']}", ""]
    out.append(f"THE PICK: {dec['PICK']}")
    out += ["", "## Her final-layer size (descriptive)", ""]
    for key, v in L["draw1"]["C7"].items():
        out.append(f"- {key}: " + ", ".join(f"{a} {b:+.3f}" if isinstance(b, float) else f"{a} {b}" for a, b in v.items()))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    flags = {x.split("=", 1)[0][2:]: x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--") and "=" in x}
    main(smoke=flags.get("smoke", "0") == "1")
    sys.exit(0)
