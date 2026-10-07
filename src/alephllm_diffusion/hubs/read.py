"""alephllm_diffusion.hubs.read - THE HUB READ (registered 2026-10-07 before any read; image-free): are Beatrix's hub
blackboards better conditioning features than her stream? The earlier conditioning probes' method (their hub reads of the
earlier models) on mini-beatrix-3 at step 245,674, bare and with the refit arm groups mounted.

THE TWO KINDS OF FEATURE
  the BLACKBOARD of a block's hub: per constellation the prefill state (Sp - Sn) after the text, 4 x 64 slots x 1,024, each slot
  L2-normalized, flattened (262,144 numbers; the earlier probes' convention = beatrix.extract's 'bb' tap); a sum over every
  position, so it is read in batches of equal byte length only
  the STREAM: the block's output, pooled over the text's bytes ('pool') or at its last byte ('last')

THE CONDITIONS (one trunk on the card at a time): M0 the bare trunk; per arm seed S in {A, B} (the nine-arm groups gCA, gCB)
  M9_S all nine arms live, M8_S the caption arm masked, Mc_S the caption arm with the eight masked (the mount read's routes;
  every member masked must give the bare trunk exactly, checked at each mount); U0 the same architecture untrained (random init,
  seed 0: a control that must fail); A212 the trunk at step 212,000 (the slider experiments' trunk: the anchor of part B).

PART A, THE CAPTION GEOMETRY (the 2,048 COCO captions of draw 1; the bare trunk also on draw 2 = the noise bar), per condition
  and tap (the blackboard, pooled and last byte at the blocks of SWEEP; pooled and last byte after the final norm), against
  every reference: CLIP-B/32 (the earlier probe's reference), T5-XXL, bert-base, captionbert (the stored rulers), Qwen3 0.6B
  pooled and Anima's adapter output pooled (hubs.refs; the adapter output is where a slider's push is added):
    overlap10 / spearman   the earlier probe's two numbers, verbatim: the mean share of each caption's 10 nearest captions (by
                           cosine) that the reference also puts in its 10 nearest; the Spearman correlation of the cosine
                           similarities over 100,000 random caption pairs (the pairs and the gaussian floor drawn as it drew them)
    cka_deb, cka_deb_noPC1, proc_theta_cv, pr   beatrix.gauges verbatim (the raw residual taps convention-normalized first,
                           as the earlier depth sweep did; only the ridge depends on the scale)
    r2_rep2ref             the forward ridge (the tap predicting the reference) to T5-XXL, Qwen3 and the adapter output, for
                           M0 (both draws), the nine-arm conditions and the untrained copy
PART B, THE SLIDER READ: the connectors' 74-phrase table (6 phrases + 24 words per mood class, 4 neutral phrases for training;
  4 + 4 + 2 held out, no held-out word in training) and the 60 mood-probe words, read three ways: the text alone ('last': the
  stream at its last byte, the blackboard after it), the text closed by a full stop ('close': the stream at the stop, the
  blackboard after it) and the grid's caption frame ('frame', descriptive: the quality prefix + "an illustration of a quiet
  street, {text}.", the stream at the stop, the blackboard of the whole caption). Features per block of BLOCKS_B and the
  connectors' four blocks concatenated (16, 18, 21, 24): each block LayerNorm'd without affine, concatenated, z-scored per
  feature over the reference rows (the 64 training rows), as the connectors' features file was built. Reads (the slider
  experiments' instruments, verbatim arithmetic): the mood axis from the training rows' up and down centres (a = +1 / -1 at
  the centres), the neutral axis; leave-one-out signs of the 60 mood training rows; the 8 held-out mood phrases' signs; the
  10 held-out phrases' nearest centre (cosine); THE HELD-OUT SEPARATION (mean a of the held-out up phrases minus that of the
  held-out down phrases); the class spreads (sd of a per side) and their balance |log(sd_up / sd_down)|; the shared-ending
  confound (cos(upbeat, downbeat) beside cos(happy, sad), a('gloomy and downbeat')); the gloomy training rows on the negative
  side; the '-ful' words; and, descriptive, the mood probe's 10-fold ridge accuracy on the 60 words (a read of what the feature
  holds that does not depend on the slider's centre-difference axis). Beside each block, descriptive: 'both', the hub and the
  stream together (the mean of their two slider values).
THE ANCHOR: at step 212,000 and on the untrained copy the stream's 'last' features must reproduce the connectors' stored
  features file (both trunks) to fp32 rounding; the run records the differences.
THE DECISION (registered before any read; the plan of 2026-10-07 with its amendment): among the M0 features of the 'last' and
  'close' reads (every block and the concatenation, blackboard and stream) with held-out signs >= 7/8 and leave-one-out signs
  >= 50/60, the slider's feature = the largest held-out separation; within .05 of it the more balanced sides (smaller
  |log(sd_up / sd_down)|), then the higher debiased CKA with the adapter output (part A, the same family and block; the mean
  over the four blocks for the concatenation). HUB BETTER: the chosen feature is a blackboard and the best stream feature
  trails it by more than .05 in separation, or is within .05 and less balanced by more than .10; STREAM BETTER: the mirror;
  otherwise TIED (the sliders run both); NONE QUALIFIES when no feature passes (the slider read waits).

Reads coco_caps_2x2048.json, refs_2x2048.pt and hub_refs_2x2048.pt (hubs.refs) from settings.DATA_DIR. Writes
hub_read_step245674.json (saved after every condition) and hub_read_step245674.md (the tables) to <settings.HOME>/hub_read;
prints the time spent and left; a file named STOP in that folder ends the run after the current condition. One job on the
card; fp32 throughout.
Usage: python -m alephllm_diffusion.hubs.read [--smoke=1]   (--smoke=1: 256 captions, two blocks, one arm group)"""
import contextlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from .. import settings
from ..beatrix import extract as bx
from ..beatrix import gauges as G
from ..mount import read as MR

STEP, ANCHOR_STEP = 245674, 212000
SWEEP = [8, 12, 14, 16, 18, 20, 21, 22, 24, 28]               # part A's blocks
BLOCKS_B = [8, 12, 14, 16, 18, 20, 21, 22, 24, 28]            # part B's blocks
RECORD_BLOCKS = (16, 18, 21, 24)                              # the connectors' features
REFS_V1 = "clip_b32"                                          # the earlier probe's reference
RIDGE_REFS = ("t5_xxl", "qwen3_pool", "adapter_pool")
RIDGE_CONDS = {"M0", "M0_draw2", "U0", "M9_A", "M9_B"}
SEP_TIE, BAL_TIE = 0.05, 0.10                                 # the decision's ties (separation in a units; balance in log units)
OUT = os.path.join(str(settings.HOME), "hub_read")
STOP = os.path.join(OUT, "STOP")                              # touch it: the read stops after the condition it is in
FEATURES_REPO = "AbstractPhil/geolip-beatrix-anima-data"
FEATURES_FILE = "beatrix/mood_phrases_mini-beatrix-3_step212000.safetensors"
T0 = time.time()


def say(*a):
    print(f"[hub read {time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------- the phrase table (the connectors' own, verbatim)
PHRASES = {
    ("up", "train"): ["cheerful and upbeat", "joyful and uplifting", "happy and bright", "sunny and cheerful",
                      "lighthearted and warm", "bright and hopeful"],
    ("up", "heldout"): ["merry and glad", "elated", "jubilant and gleeful", "blissful"],
    ("down", "train"): ["gloomy and downbeat", "somber and melancholy", "sad and dark", "bleak and grey",
                        "sorrowful and heavy", "dreary and dull"],
    ("down", "heldout"): ["mournful", "forlorn and desolate", "dismal", "despairing and grim"],
    ("neutral", "train"): ["an ordinary scene", "plain", "everyday", "neutral"],
    ("neutral", "heldout"): ["typical", "unremarkable"],
}
PROBE = MR.UP + MR.DOWN                                       # the mood probe's 60 words (30 upbeat, 30 downbeat)
ROWS = [(c, s, p) for (c, s), ps in PHRASES.items() for p in ps]
_HELD_WORDS = {w for c, s, p in ROWS if s == "heldout" for w in p.replace(" and ", " ").split()}
REF = [w for w in PROBE if w not in _HELD_WORDS] + [p for c, s, p in ROWS if s == "train"]
_TRAIN_TEXT = {p for c, s, p in ROWS}
ROWS += [("up" if PROBE.index(w) < 30 else "down", "train", w) for w in PROBE if w not in _HELD_WORDS and w not in _TRAIN_TEXT]
assert len(ROWS) == 74 and len(REF) == 64
TEXTS = [p for _, _, p in ROWS]
TESTS = ["upbeat", "downbeat", "cheerful and upbeat", "gloomy and downbeat", "gloomy", "gloomy and sad", "sad and downbeat",
         "cheerful and downbeat", "gloomy and upbeat"]
FUL = [(("up" if PROBE.index(w) < 30 else "down"), w) for w in PROBE if w.endswith("ful")]
STRINGS_B = list(dict.fromkeys(TEXTS + TESTS + PROBE))        # every string part B reads, once
FORMS = {"last": "{s}", "close": "{s}.", "frame": "{prefix}" + MR.PREFIX_FRAME.replace("{w}", "{s}")}


# ---------------------------------------------------------------------------- part A: the blackboards' Grams on the card
def equal_groups(lens, max_batch=64):
    """beatrix.extract's batches: indices sorted by length, consecutive captions of equal length up to max_batch rows."""
    order = sorted(range(len(lens)), key=lambda i: lens[i])
    groups, pos = [], 0
    while pos < len(order):
        g = [order[pos]]
        while pos + len(g) < len(order) and len(g) < max_batch and lens[order[pos + len(g)]] == lens[order[pos]]:
            g.append(order[pos + len(g)])
        pos += len(g)
        groups.append(g)
    return groups


def cosine_matrix(X, chunk=256):
    """The earlier probe's cosine matrix (rows L2-normalized, fp32), in chunks, on X's device; the diagonal left as computed."""
    n = X.shape[0]
    nrm = torch.cat([X[i:i + chunk].float().norm(dim=1) for i in range(0, n, chunk)]).clamp_min(1e-12)
    S = torch.empty(n, n, dtype=torch.float32, device=X.device)
    for i in range(0, n, chunk):
        A = X[i:i + chunk].float() / nrm[i:i + chunk, None]
        for j in range(0, n, chunk):
            S[i:i + chunk, j:j + chunk] = A @ (X[j:j + chunk].float() / nrm[j:j + chunk, None]).T
    return S


@torch.no_grad()
def blackboard_grams(model, texts, blocks, dev, per_pass):
    """{block: (K, S)} for the blackboard rows of `texts`: K the centred Gram (n x n float64, beatrix.gauges' fp32-chunked Gram,
    the path its memory-light prep takes), S the cosine matrix (n x n float32). The rows of up to per_pass blocks stay on the
    card in fp16 (the form beatrix.extract returns them in); each pass runs the trunk to its deepest block only. Batches as
    beatrix.extract forms them (equal byte length, no padding). Both matrices returned on the CPU."""
    ids = [bx.encode_bytes(t) for t in texts]
    groups = equal_groups([x.numel() for x in ids])
    out = {}
    blocks = sorted(blocks)
    for c0 in range(0, len(blocks), per_pass):
        part = blocks[c0:c0 + per_pass]
        buf = {}
        for grp in groups:
            x = model.embed(torch.stack([ids[i] for i in grp]).to(dev))
            for bi, blk in enumerate(model.blocks):
                x, cache = blk.prefill(x)
                if bi in part:
                    rows = bx._blackboard(cache)
                    assert torch.isfinite(rows).all(), ("non-finite blackboard", bi, grp[:4])
                    if bi not in buf:
                        buf[bi] = torch.empty(len(texts), rows.shape[1], dtype=torch.float16, device=dev)
                    buf[bi][torch.tensor(grp, device=dev)] = rows.to(torch.float16)
                if bi == part[-1]:
                    break
        for b in part:
            out[b] = (G._gram_chunked(buf[b]).cpu(), cosine_matrix(buf[b]).cpu())
            del buf[b]
    return out


def prepped_from_gram(K, dev):
    """beatrix.gauges' prep for a wide representation from its centred Gram: U * sqrt(eigenvalues) (n x n)."""
    evals, U = torch.linalg.eigh(K.to(dev))
    return U * evals.clamp_min(0).sqrt().unsqueeze(0)


# ---------------------------------------------------------------------------- part A: the earlier probe's numbers (verbatim)
def v1_draws(n):
    """The gaussian floor (n x 512) and the 100,000 caption pairs (i != j), drawn in the earlier probe's order from one
    generator seeded 3."""
    g = torch.Generator().manual_seed(3)
    rand = torch.randn(n, 512, generator=g)
    ii = torch.randint(0, n, (100_000,), generator=g)
    jj = torch.randint(0, n, (100_000,), generator=g)
    keep = ii != jj
    return rand, ii[keep], jj[keep]


def top10(S):
    S = S.clone()
    S.fill_diagonal_(-2)
    return S.topk(10, dim=-1).indices


def overlap10(top_a, top_b):
    """the mean over captions of |A's 10 nearest  and  B's 10 nearest| / 10 (the earlier probe's 'jaccard@10')."""
    return float((top_a[:, :, None] == top_b[:, None, :]).any(-1).sum(-1).double().mean() / 10.0)


def spearman(a, b):
    """the earlier probe's form: ranks by a double argsort, the Pearson correlation of the ranks."""
    ra, rb = a.double().argsort().argsort().double(), b.double().argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


class Refs:
    """The references of one caption draw, prepared once: prepped coordinates, without their top component, top-10 sets,
    the pair similarities."""

    def __init__(self, refs, pairs, dev):
        self.names, self.ii, self.jj = sorted(refs), pairs[0], pairs[1]
        self.p = {k: G.prep(v) for k, v in refs.items()}
        self.p1 = {k: G.drop_pc1(v) for k, v in self.p.items()}
        S = {k: cosine_matrix(v.to(dev)) for k, v in refs.items()}
        self.top = {k: top10(s) for k, s in S.items()}
        self.pair = {k: s[self.ii.to(s.device), self.jj.to(s.device)].cpu() for k, s in S.items()}


def tap_rows(Xp, S, R, ridge):
    """one tap against every reference: {'pr', ref: {overlap10, spearman, cka_deb, cka_deb_noPC1, proc_theta_cv[, r2]}}."""
    s2v = torch.linalg.svdvals(Xp) ** 2
    out = {"pr": float(s2v.sum() ** 2 / (s2v ** 2).sum().clamp_min(1e-300))}
    top = top10(S)
    pair = S[R.ii.to(S.device), R.jj.to(S.device)].cpu()
    Xp1 = G.drop_pc1(Xp)
    for rn in R.names:
        row = {"overlap10": overlap10(top, R.top[rn].to(top.device)), "spearman": spearman(pair, R.pair[rn]),
               "cka_deb": G.cka_debiased(Xp, R.p[rn]), "cka_deb_noPC1": G.cka_debiased(Xp1, R.p1[rn]),
               "proc_theta_cv": G.procrustes_cv(Xp, R.p[rn])}
        if ridge and rn in RIDGE_REFS:
            row["r2_rep2ref"], row["r2_rep2ref_sd"] = G.ridge_r2(Xp, R.p[rn])
        out[rn] = row
    return out


def part_a(model, dev, caps, R, ridge, per_pass, blocks=None, label=""):
    """{tap@L: rows} for one condition on one caption draw."""
    blocks = SWEEP if blocks is None else blocks
    t0 = time.time()
    f = bx.extract(model, caps, sorted(set(blocks) | {32}), taps=("pool", "last"), device=dev, amp=False)
    res = {}
    for tap in ("pool", "last"):
        for l in sorted(set(blocks) | {32}):
            X = f[tap][l]
            Xp = G.prep(MR.convnorm(X) if l <= 31 else X)
            res[f"{tap}@L{l}"] = tap_rows(Xp, cosine_matrix(X.to(dev)), R, ridge)
            del Xp
    del f
    say(f"  {label}: the stream taps read ({time.time() - t0:.0f} s)")
    t1 = time.time()
    grams = blackboard_grams(model, caps, blocks, dev, per_pass)
    for b, (K, S) in grams.items():
        res[f"bb@L{b}"] = tap_rows(prepped_from_gram(K, dev), S.to(dev), R, ridge)
    del grams
    if dev == "cuda":
        torch.cuda.empty_cache()
    say(f"  {label}: the blackboards read ({time.time() - t1:.0f} s, {per_pass} block(s) a pass)")
    return res


# ---------------------------------------------------------------------------- part B: the slider read
def features_b(model, dev, form, prefix, blocks):
    """{('stream' | 'hub', block): (strings, D) float32 LayerNorm'd without affine} for every string of part B in one form."""
    strs = [FORMS[form].format(s=s, prefix=prefix) for s in STRINGS_B]
    ex = bx.extract(model, strs, blocks, taps=("last", "bb"), device=dev, amp=False)
    out = {}
    for b in blocks:
        out[("stream", b)] = F.layer_norm(ex["last"][b], (ex["last"][b].shape[1],))
        hb = ex["bb"][b].float()
        out[("hub", b)] = F.layer_norm(hb, (hb.shape[1],))
    return out


def zscored(L, blocks):
    """the connectors' features: the blocks' LayerNorm'd rows concatenated, z-scored per feature over the reference rows."""
    X = torch.cat([L[b] for b in blocks], dim=1)
    ref = X[[STRINGS_B.index(s) for s in REF]]
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    return ((X - mu) / sd).double()


def axis_fit(Z, idx):
    up = Z[[i for i in idx if ROWS[i][0] == "up"]].mean(0)
    dn = Z[[i for i in idx if ROWS[i][0] == "down"]].mean(0)
    ax = up - dn
    return (up + dn) / 2, ax / (ax @ ax / 2)


def cv_folds(X, y, lam, folds=10, seed=0):
    """mount.read.cv_folds (the mood probe's 10-fold ridge classifier) with every tensor on X's device: the same arithmetic."""
    g = torch.Generator().manual_seed(seed)
    y = y.to(X.device)
    up, dn = torch.where(y.cpu() > 0)[0], torch.where(y.cpu() < 0)[0]
    up, dn = up[torch.randperm(len(up), generator=g)], dn[torch.randperm(len(dn), generator=g)]
    accs = []
    for k in range(folds):
        te = torch.cat([up[k::folds], dn[k::folds]])
        tr = torch.tensor([i for i in range(len(y)) if i not in set(te.tolist())])
        te, tr = te.to(X.device), tr.to(X.device)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        alpha = torch.linalg.solve(A @ A.T + lam * torch.eye(len(tr), dtype=A.dtype, device=X.device), y[tr])
        pred = (B @ A.T) @ alpha
        accs.append(float(((pred > 0) == (y[te] > 0)).double().mean()))
    return accs


def slider_reads(Z, P=None):
    """Z: the z-scored features of STRINGS_B (rows in that order). The slider experiments' reads (verbatim arithmetic)."""
    zr = Z[[STRINGS_B.index(t) for t in TEXTS]]                              # the 74 table rows
    zt = Z[[STRINGS_B.index(t) for t in TESTS]]
    train = [i for i, r in enumerate(ROWS) if r[1] == "train"]
    mood_train = [i for i in train if ROWS[i][0] != "neutral"]
    mid, v = axis_fit(zr, train)
    a = (zr - mid) @ v
    cen = {c: zr[[i for i in train if ROWS[i][0] == c]].mean(0) for c in ("up", "down", "neutral")}
    nax = cen["neutral"] - mid
    nval = (zr - mid) @ nax / (nax @ nax)
    loo, loo_a = 0, []
    for i in mood_train:
        m2, v2 = axis_fit(zr, [j for j in train if j != i])
        loo_a.append(float((zr[i] - m2) @ v2))
        loo += int((loo_a[-1] > 0) == (ROWS[i][0] == "up"))
    held = [i for i, r in enumerate(ROWS) if r[1] == "heldout"]
    held_mood = [i for i in held if ROWS[i][0] != "neutral"]
    signs = sum(int((float(a[i]) > 0) == (ROWS[i][0] == "up")) for i in held_mood)
    near = sum(int(max(cen, key=lambda c: float(F.cosine_similarity(zr[i], cen[c], dim=0))) == ROWS[i][0]) for i in held)
    hu = [float(a[i]) for i in held_mood if ROWS[i][0] == "up"]
    hd = [float(a[i]) for i in held_mood if ROWS[i][0] == "down"]
    su = float(a[[i for i in mood_train if ROWS[i][0] == "up"]].std())
    sd_ = float(a[[i for i in mood_train if ROWS[i][0] == "down"]].std())
    down_train = [float(a[i]) for i in mood_train if ROWS[i][0] == "down"]
    ful = [(c, w, float(a[TEXTS.index(w)])) for c, w in FUL if w in TEXTS]
    cs = lambda x, y: float(F.cosine_similarity(x, y, dim=0))  # noqa: E731
    at = (zt - mid) @ v
    out = {"loo_signs": [loo, len(mood_train)], "heldout_signs": [signs, len(held_mood)], "heldout_nearest": [near, len(held)],
           "separation": sum(hu) / len(hu) - sum(hd) / len(hd), "heldout_a": {ROWS[i][2]: float(a[i]) for i in held},
           "heldout_n": {ROWS[i][2]: float(nval[i]) for i in held}, "sd_up": su, "sd_down": sd_,
           "balance": abs(math.log(max(su, 1e-12) / max(sd_, 1e-12))),
           "neutral_train_a": float(a[[i for i in train if ROWS[i][0] == "neutral"]].mean()),
           "gloomy_negative": [sum(x < 0 for x in down_train), sum(x < -.25 for x in down_train), len(down_train)],
           "ful_right": [sum((x > 0) == (c == "up") for c, _, x in ful), len(ful)],
           "cos_upbeat_downbeat": cs(zr[TEXTS.index("upbeat")], zt[TESTS.index("downbeat")]),
           "cos_happy_sad": cs(zr[TEXTS.index("happy")], zr[TEXTS.index("sad")]),
           "a_gloomy_and_downbeat": float(at[TESTS.index("gloomy and downbeat")]),
           "a_rows": [float(x) for x in a], "a_tests": [float(x) for x in at], "loo_a": loo_a}
    if P is not None:                                                        # the mood probe (descriptive)
        X = P[[STRINGS_B.index(w) for w in PROBE]].double()
        X = F.layer_norm(X, (X.shape[1],))
        best = max(((sum(fl) / len(fl), fl, lam) for lam in MR.LAMBDAS for fl in [cv_folds(X, MR.MOOD_Y, lam)]),
                   key=lambda r: r[0])
        out["probe_acc"], out["probe_lambda"] = best[0], best[2]
        out["probe_fold_se"] = float(torch.tensor(best[1]).std() / len(best[1]) ** 0.5)
    return out


def combined_reads(rh, rs):
    """THE HUB AND THE STREAM TOGETHER (descriptive, added 2026-10-07 13:3x after the small run's numbers; never a candidate of
    the decision): the slider value a = the mean of the hub's and the stream's own slider values (each fit on its own training
    rows, centres at -1 / +1), so each source counts equally whatever its width; the leave-one-out values likewise. The reads
    that need a feature space (nearest centre, cosines, the probe) are not defined here."""
    mean = lambda x, y: [(p + q) / 2 for p, q in zip(x, y)]  # noqa: E731
    a, at, loo_a = mean(rh["a_rows"], rs["a_rows"]), mean(rh["a_tests"], rs["a_tests"]), mean(rh["loo_a"], rs["loo_a"])
    train = [i for i, r in enumerate(ROWS) if r[1] == "train"]
    mood_train = [i for i in train if ROWS[i][0] != "neutral"]
    held_mood = [i for i, r in enumerate(ROWS) if r[1] == "heldout" and r[0] != "neutral"]
    hu = [a[i] for i in held_mood if ROWS[i][0] == "up"]
    hd = [a[i] for i in held_mood if ROWS[i][0] == "down"]
    su = float(torch.tensor([a[i] for i in mood_train if ROWS[i][0] == "up"], dtype=torch.float64).std())
    sd_ = float(torch.tensor([a[i] for i in mood_train if ROWS[i][0] == "down"], dtype=torch.float64).std())
    down_train = [a[i] for i in mood_train if ROWS[i][0] == "down"]
    ful = [(c, a[TEXTS.index(w)]) for c, w in FUL if w in TEXTS]
    return {"loo_signs": [sum(int((v > 0) == (ROWS[i][0] == "up")) for v, i in zip(loo_a, mood_train)), len(mood_train)],
            "heldout_signs": [sum(int((a[i] > 0) == (ROWS[i][0] == "up")) for i in held_mood), len(held_mood)],
            "heldout_nearest": None, "separation": sum(hu) / len(hu) - sum(hd) / len(hd),
            "heldout_a": {ROWS[i][2]: a[i] for i, r in enumerate(ROWS) if r[1] == "heldout"}, "sd_up": su, "sd_down": sd_,
            "balance": abs(math.log(max(su, 1e-12) / max(sd_, 1e-12))),
            "neutral_train_a": sum(a[i] for i in train if ROWS[i][0] == "neutral") / 4,
            "gloomy_negative": [sum(x < 0 for x in down_train), sum(x < -.25 for x in down_train), len(down_train)],
            "ful_right": [sum((x > 0) == (c == "up") for c, x in ful), len(ful)],
            "a_gloomy_and_downbeat": at[TESTS.index("gloomy and downbeat")], "a_rows": a, "a_tests": at, "loo_a": loo_a}


def part_b(model, dev, prefix, blocks=None, forms=("last", "close", "frame"), record=RECORD_BLOCKS):
    """{form/kind/block: reads} for one condition; block 'rec' = the connectors' four blocks concatenated."""
    blocks = BLOCKS_B if blocks is None else blocks
    res, keep = {}, {}
    for form in forms:
        L = features_b(model, dev, form, prefix, blocks)
        for kind in ("stream", "hub"):
            for b in list(blocks) + (["rec"] if all(r in blocks for r in record) else []):
                bl = list(record) if b == "rec" else [b]
                Z = zscored({x: L[(kind, x)] for x in bl}, bl)
                if form == "last" and kind == "stream" and b == "rec":
                    keep["stream_last_rec"] = Z[[STRINGS_B.index(t) for t in TEXTS]].float()
                P = torch.cat([L[(kind, x)] for x in bl], dim=1).to(dev) if form == "last" else None
                res[f"{form}/{kind}/{b}"] = slider_reads(Z.to(dev), P)       # on the card: the hub rows are ~1M wide
                del Z, P
        for b in list(blocks) + (["rec"] if all(r in blocks for r in record) else []):
            res[f"{form}/both/{b}"] = combined_reads(res[f"{form}/hub/{b}"], res[f"{form}/stream/{b}"])
        del L
    return res, keep


# ---------------------------------------------------------------------------- the anchor and the decision
def stored_features_path():
    """The stored slider features file: the hub cache's current snapshot or download, else any cached snapshot that holds it
    (offline, the cache's main pointer can name a newer snapshot without this file)."""
    from huggingface_hub import hf_hub_download, scan_cache_dir
    try:
        return hf_hub_download(FEATURES_REPO, FEATURES_FILE, repo_type="dataset", token=False)
    except Exception:  # noqa: BLE001 - offline or unreachable: look through every cached snapshot
        for repo in scan_cache_dir().repos:
            if repo.repo_id == FEATURES_REPO and repo.repo_type == "dataset":
                for rev in repo.revisions:
                    for f in rev.files:
                        if f.file_name == os.path.basename(FEATURES_FILE) and str(f.file_path).replace("\\", "/").endswith(
                                FEATURES_FILE):
                            return str(f.file_path)
        raise


def anchor(keep212, keep_u0):
    """the stored features file (both trunks) against this run's 'last' stream features of the same trunks."""
    from safetensors import safe_open
    path = stored_features_path()
    with safe_open(path, framework="pt") as f:
        phrases = json.loads(f.metadata()["phrases"])
        stored = {k: f.get_tensor(k).float() for k in ("trained", "random")}
    assert [p["text"] for p in phrases] == TEXTS, "the phrase table differs from the stored file's"
    out = {}
    for name, mine in (("trained", keep212), ("random", keep_u0)):
        if mine is None:
            continue
        d = (mine - stored[name]).abs()
        out[name] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "max_rel": float(d.max() / stored[name].abs().max())}
    return out


def decide(B0, A0):
    """THE DECISION on the bare trunk's part B rows (B0) with part A's CKA to the adapter output (A0) as the last tie-break."""
    def cka_adapter(kind, b):
        fam = "bb" if kind == "hub" else "last"
        bl = list(RECORD_BLOCKS) if b == "rec" else [int(b)]
        vals = [A0[f"{fam}@L{x}"]["adapter_pool"]["cka_deb"] for x in bl if f"{fam}@L{x}" in A0]
        return sum(vals) / len(vals) if vals else float("nan")
    cands = []
    for key, r in B0.items():
        form, kind, b = key.split("/")
        if form not in ("last", "close") or kind not in ("hub", "stream"):       # 'both' is descriptive, never a candidate
            continue
        ok = r["heldout_signs"][0] >= 7 and r["loo_signs"][0] >= 50
        cands.append({"key": key, "kind": kind, "sep": r["separation"], "bal": r["balance"], "cka": cka_adapter(kind, b),
                      "eligible": ok})
    el = [c for c in cands if c["eligible"]]

    def best_of(group):
        if not group:
            return None
        top = max(c["sep"] for c in group)
        near = [c for c in group if c["sep"] >= top - SEP_TIE]
        lo = min(c["bal"] for c in near)
        near = [c for c in near if c["bal"] <= lo]
        return max(near, key=lambda c: (c["cka"] if c["cka"] == c["cka"] else -1.0))
    chosen = best_of(el)
    if chosen is None:
        return {"verdict": "NONE QUALIFIES", "chosen": None, "candidates": cands}
    hub, stream = best_of([c for c in el if c["kind"] == "hub"]), best_of([c for c in el if c["kind"] == "stream"])
    mine, other = (hub, stream) if chosen["kind"] == "hub" else (stream, hub)
    if other is None or mine["sep"] - other["sep"] > SEP_TIE or other["bal"] - mine["bal"] > BAL_TIE:
        verdict = "HUB BETTER" if chosen["kind"] == "hub" else "STREAM BETTER"
    else:
        verdict = "TIED"
    return {"verdict": verdict, "chosen": chosen, "best_hub": hub, "best_stream": stream, "candidates": cands}


# ---------------------------------------------------------------------------- the run
def _per_pass(dev, n, width=4 * 64 * 1024, reserve_gb=3.0):
    """blackboard blocks held on the card a pass: what fits in the job's free share after a reserve for the trunk's work."""
    if dev != "cuda":
        return 2
    free, total = torch.cuda.mem_get_info()
    cap = bx.MEM_FRACTION * total - torch.cuda.memory_reserved()
    room = min(free, cap) - reserve_gb * 2 ** 30
    return max(1, int(room // (n * width * 2)))


def main(smoke=False, groups=("gCA", "gCB")):
    from geolip.alephllm.arm_mount import CAPTION_ARM, detach_all, masked, mount_group, only
    from geolip_anima_trainer import anima_experiments as ax
    import amoe.core.adapter as AD
    assert hasattr(AD.BlockWithAdapter, "prefill"), "amoe-lora >= 0.2.11 is needed (prefill applies the adapter)"
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda"
    G.dev = dev                                                  # the gauges' device (their module default is the card)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    G.selfcheck()
    caps = json.load(open(os.path.join(settings.DATA_DIR, "coco_caps_2x2048.json"), encoding="utf-8"))
    rulers = torch.load(os.path.join(settings.DATA_DIR, "refs_2x2048.pt"))
    extra = torch.load(os.path.join(settings.DATA_DIR, "hub_refs_2x2048.pt"))
    blocks_a, blocks_b, n = SWEEP, BLOCKS_B, 2048
    if smoke:
        blocks_a, blocks_b, n, groups = [14, 16], [14, 16, 18, 21, 24], 256, groups[:1]
    draws = {}
    for d in ("draw1", "draw2"):
        refs = {**{k: v[:n] for k, v in rulers[d].items()}, **{k: v[:n] for k, v in extra[d].items()}}
        rand, ii, jj = v1_draws(n)
        draws[d] = (caps[d][:n], Refs(refs, (ii, jj), dev), rand)
    prefix = ax.PREFIX
    os.makedirs(OUT, exist_ok=True)
    tag = f"step{STEP}{'_smoke' if smoke else ''}"
    path = os.path.join(OUT, f"hub_read_{tag}.json")
    L = {"_meta": {"step": STEP, "anchor_step": ANCHOR_STEP, "sweep": blocks_a, "blocks_b": blocks_b, "n_captions": n,
                   "groups": list(groups), "refs": draws["draw1"][1].names, "ridge_refs": list(RIDGE_REFS),
                   "record_blocks": list(RECORD_BLOCKS), "prefix": prefix, "frame": FORMS["frame"], "smoke": smoke,
                   "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, "A": {}, "B": {}}

    def save():
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(L, fh, indent=1)
    plan = ["A212", "M0", "M0_draw2"] + [f"{c}_{g[-1]}" for g in groups for c in ("M9", "M8", "Mc")] + ["U0"]
    done = []

    def tick(name):
        done.append(name)
        el = time.time() - T0
        left = el / len(done) * (len(plan) - len(done))
        say(f"{name} done: {len(done)}/{len(plan)} conditions, {el:.0f} s spent, about {left:.0f} s left")
        if os.path.exists(STOP):                                 # the stop file: the run ends here with what is saved
            say(f"stop file {STOP} found: stopping after {name} (the ledger holds every condition read so far)")
            raise SystemExit(3)
    # the floor and the ruler-against-ruler rows (no trunk)
    caps1, R1, rand = draws["draw1"]
    L["A"]["floor"] = {"gauss_512": tap_rows(G.prep(rand), cosine_matrix(rand.to(dev)), R1, ridge=False)}
    L["A"]["rulers"] = {}
    for rn in R1.names:
        X = extra["draw1"][rn][:n] if rn in extra["draw1"] else rulers["draw1"][rn][:n]
        L["A"]["rulers"][rn] = tap_rows(G.prep(X), cosine_matrix(X.to(dev)), R1, ridge=False)
    save()
    # A212: the slider experiments' trunk (part B and the anchor)
    model = bx.build_model(ANCHOR_STEP, device=dev)
    L["B"]["A212"], keep212 = part_b(model, dev, prefix, blocks_b)
    del model
    save()
    tick("A212")
    model = bx.build_model(STEP, device=dev)
    pp = _per_pass(dev, n)
    say(f"THE HUB READ at {STEP:,}: {n} captions, blocks {blocks_a}, {len(plan)} conditions; {pp} blackboard block(s) a pass")
    L["A"]["M0"] = part_a(model, dev, caps1, R1, True, pp, blocks_a, "M0")
    L["B"]["M0"], _ = part_b(model, dev, prefix, blocks_b)
    save()
    tick("M0")
    caps2, R2, _ = draws["draw2"]
    L["A"]["M0_draw2"] = part_a(model, dev, caps2, R2, True, pp, blocks_a, "M0 draw 2")
    save()
    tick("M0_draw2")
    probe = caps1[:256]
    bare_probe = bx.extract(model, probe, blocks_a, taps=("pool",), device=dev, amp=False)["pool"]
    for g in groups:
        S = g[-1]
        prog = mount_group(model, g, require_step=STEP, device=dev)
        model.eval()
        with masked(prog, list(prog.attached)):
            tog = bx.extract(model, probe, blocks_a, taps=("pool",), device=dev, amp=False)["pool"]
        toggle = {f"L{l}": float((tog[l] - bare_probe[l]).abs().max()) for l in blocks_a}
        assert all(v == 0.0 for v in toggle.values()), ("every member masked must equal bare exactly", toggle)
        L["_meta"][f"toggle_{g}"] = toggle
        for cname, ctx in ((f"M9_{S}", contextlib.nullcontext()), (f"M8_{S}", masked(prog, [CAPTION_ARM])),
                           (f"Mc_{S}", only(prog, [CAPTION_ARM]))):
            with ctx:
                L["A"][cname] = part_a(model, dev, caps1, R1, cname in RIDGE_CONDS, pp, blocks_a, cname)
                L["B"][cname], _ = part_b(model, dev, prefix, blocks_b)
            save()
            tick(cname)
        detach_all(prog, verify=True)
        model.eval()
        del prog
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    model = bx.build_model(None, device=dev, random_init_seed=0)
    L["A"]["U0"] = part_a(model, dev, caps1, R1, True, pp, blocks_a, "U0")
    L["B"]["U0"], keepu0 = part_b(model, dev, prefix, blocks_b)
    del model
    save()
    tick("U0")
    L["decision"] = decide(L["B"]["M0"], L["A"]["M0"])           # the decision and the tables first: nothing after them can
    save()                                                        # cost the run its ending
    say(f"THE DECISION: {L['decision']['verdict']}; chosen {L['decision']['chosen'] and L['decision']['chosen']['key']}")
    if all(b in blocks_b for b in RECORD_BLOCKS):
        try:
            L["anchor"] = anchor(keep212.get("stream_last_rec"), keepu0.get("stream_last_rec"))
            say(f"THE ANCHOR (the stored slider features against this run's): {L['anchor']}")
        except Exception as e:  # noqa: BLE001 - the anchor is a check beside the read, recorded either way
            L["anchor_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            say(f"THE ANCHOR WAS NOT READ: {L['anchor_error']}")
        save()
    with open(path[:-5] + ".md", "w", encoding="utf-8") as fh:
        fh.write(report(L))
    say(f"wrote {path} and the tables beside it")
    return L


# ---------------------------------------------------------------------------- the tables
def report(L):
    A, B, meta = L["A"], L["B"], L["_meta"]
    blocks = meta["sweep"]
    out = [f"# The hub read at step {meta['step']:,} ({meta['n_captions']} captions; made {meta['made']})", ""]

    def g(cond, key, ref, field):
        try:
            return A[cond][key][ref][field]
        except KeyError:
            return float("nan")
    out += ["## Part A: the earlier probe's numbers against CLIP-B/32, the bare trunk (top-10 overlap / Spearman)", "",
            "| block | blackboard | stream pooled | stream last byte |", "|---|---|---|---|"]
    for b in blocks + [32]:
        cells = []
        for fam in ("bb", "pool", "last"):
            k = f"{fam}@L{b}"
            cells.append("-" if k not in A["M0"] else f"{g('M0', k, REFS_V1, 'overlap10'):.3f} / {g('M0', k, REFS_V1, 'spearman'):+.3f}")
        out.append(f"| {b} | " + " | ".join(cells) + " |")
    fl = A["floor"]["gauss_512"][REFS_V1]
    out += ["", f"Gaussian floor {fl['overlap10']:.3f} / {fl['spearman']:+.3f}. The earlier 2s model (step 52,000): blackboard "
            "block 14 .328 / +.407, block 9 .282 / +.257, block 19 .261 / +.268, pooled .249 / +.219, last token .088 / +.012.", ""]
    for ref, lab in (("adapter_pool", "Anima's adapter output (the context the sliders push)"), ("t5_xxl", "T5-XXL"),
                     ("qwen3_pool", "Qwen3 0.6B")):
        out += [f"## Part A: debiased CKA with {lab}; forward ridge R2 in brackets", "",
                "| block | blackboard bare | stream pooled bare | stream last bare | blackboard nine arms (A / B) | blackboard untrained |",
                "|---|---|---|---|---|---|"]
        for b in blocks + [32]:
            def c(cond, fam):
                k = f"{fam}@L{b}"
                if cond not in A or k not in A[cond]:
                    return "-"
                r2 = g(cond, k, ref, "r2_rep2ref")
                return f"{g(cond, k, ref, 'cka_deb'):.3f}" + ("" if r2 != r2 else f" ({r2:+.3f})")
            out.append(f"| {b} | {c('M0', 'bb')} | {c('M0', 'pool')} | {c('M0', 'last')} | {c('M9_A', 'bb')} / {c('M9_B', 'bb')} | "
                       f"{c('U0', 'bb')} |")
        out.append("")
    if "M0_draw2" in A:
        gaps = [abs(g("M0", k, r, "cka_deb") - g("M0_draw2", k, r, "cka_deb")) for k in A["M0"] if k in A["M0_draw2"]
                for r in A["M0"][k] if r != "pr"]
        gaps = [x for x in gaps if x == x]
        if gaps:
            out += [f"The noise bar (draw 1 against draw 2, every tap and reference): CKA {sum(gaps) / len(gaps):.3f} mean, "
                    f"{max(gaps):.3f} at most.", ""]
    out += ["## Part B: the slider read, the bare trunk (the text alone; the text closed by a full stop)", "",
            "| feature | leave-one-out signs | held-out signs | held-out nearest | separation | sd up / down | balance | "
            "probe accuracy |", "|---|---|---|---|---|---|---|---|"]
    def brow(key, r):
        near = "-" if not r.get("heldout_nearest") else f"{r['heldout_nearest'][0]}/{r['heldout_nearest'][1]}"
        return (f"| {key} | {r['loo_signs'][0]}/{r['loo_signs'][1]} | {r['heldout_signs'][0]}/{r['heldout_signs'][1]} | {near} | "
                f"{r['separation']:+.2f} | {r['sd_up']:.2f} / {r['sd_down']:.2f} | {r['balance']:.2f} | "
                + (f"{r['probe_acc']:.2f}" if "probe_acc" in r else "-") + " |")
    for key, r in B.get("M0", {}).items():
        if key.split("/")[0] != "frame":
            out.append(brow(key, r))
    out += ["", "('both' = the mean of the hub's and the stream's slider values at that block: descriptive, never a candidate.)",
            "", "## Part B: the caption frame (descriptive; the bare trunk)", "",
            "| feature | leave-one-out signs | held-out signs | held-out nearest | separation | sd up / down | balance | "
            "probe accuracy |", "|---|---|---|---|---|---|---|---|"]
    out += [brow(key, r) for key, r in B.get("M0", {}).items() if key.split("/")[0] == "frame"]
    out.append("")
    d = L.get("decision")
    picks = [] if not d else [c["key"] for c in (d.get("chosen"), d.get("best_hub"), d.get("best_stream")) if c]
    picks = list(dict.fromkeys(picks + [p.replace("/hub/", "/both/") for p in picks if "/hub/" in p]))
    if picks:
        conds = [c for c in ("M0", "M9_A", "M9_B", "M8_A", "M8_B", "Mc_A", "Mc_B", "U0", "A212") if c in B]
        out += ["## Part B: the decided features under every condition (separation; leave-one-out signs; held-out signs; "
                "balance)", "", "| feature | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
        for p in picks:
            cells = []
            for c in conds:
                r = B[c].get(p)
                cells.append("-" if r is None else f"{r['separation']:+.2f}; {r['loo_signs'][0]}; {r['heldout_signs'][0]}/8; "
                                                   f"{r['balance']:.2f}")
            out.append(f"| {p} | " + " | ".join(cells) + " |")
        out.append("")
    if d:
        ch = d.get("chosen")
        out += ["## The decision", "", f"**{d['verdict']}**" + (f": the slider's feature = {ch['key']} (separation "
                f"{ch['sep']:+.2f}, balance {ch['bal']:.2f})" if ch else ""), ""]
        for lab in ("best_hub", "best_stream"):
            c = d.get(lab)
            if c:
                out.append(f"- {lab.replace('_', ' ')}: {c['key']} (separation {c['sep']:+.2f}, balance {c['bal']:.2f}, CKA with "
                           f"the adapter output {c['cka']:.3f})")
        out.append("")
    if L.get("anchor"):
        out += ["## The anchor", "", "| trunk | largest difference | relative to the largest stored value |", "|---|---|---|"]
        out += [f"| {k} | {v['max_abs']:.2e} | {v['max_rel']:.2e} |" for k, v in L["anchor"].items()]
        out.append("")
    elif L.get("anchor_error"):
        out += ["## The anchor", "", f"Not read in this run: {L['anchor_error']}", ""]
        if L.get("anchor_smoke"):
            out += ["The small run (same code, same trunks) against the stored slider features:", "",
                    "| trunk | largest difference | relative to the largest stored value |", "|---|---|---|"]
            out += [f"| {k} | {v['max_abs']:.2e} | {v['max_rel']:.2e} |" for k, v in L["anchor_smoke"].items()]
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main(smoke="--smoke=1" in sys.argv)
    sys.exit(0)
