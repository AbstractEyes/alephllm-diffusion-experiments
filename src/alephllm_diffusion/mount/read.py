"""alephllm_diffusion.mount.read - THE MOUNT READ (the testing plan, step 3; registered 2026-10-06 before any build or read;
image-free): do the refit arm groups change what Beatrix's CAPTION states carry? The predictions it reads (the plan's section 5):
  PM1  THE EIGHT ARE QUIET ON CAPTIONS: M8 equals M0 on every gauge here at the grid's blocks (within the noise bar)
  PM2  THE CAPTION ARM MOVES CAPTION STATES: M9 differs from M0 at one or more middle blocks, beyond max(the noise bar, the
       difference between the two arm seeds); no direction predicted
The mounts (the library's geolip.alephllm.arm_mount on the extractor's trunk at step 245,674; each group mounted, read, detached
exact):
  M0 = the bare trunk; per arm seed S in {A, B} (gCA, gCB: the nine-arm group, trained with two seeds): M9_S = gC<S> with all nine
  arms live; M8_S = gC<S> with the caption arm masked (the eight stage arms; = gX<S> close 8, the carried members checked by
  content hash at the mount); Mc_S = the caption arm with the eight masked (labelled MASKED: a masked member is not a solo
  member).
The gauges, one function each, every mount through the same code (the same reads used on the arms during their training; fp32,
the DOC byte, the per-block taps through each wrapper's prefill = the armed stream):
  WRITE  per block of the grid ([8, 12, 16, 18, 20, 22, 24, 28]) the arm's write on the 2,048 COCO captions (draw 1):
         ||mount - bare|| / ||bare|| of the pooled block output (mean over captions), beside ten times the fp32 repeat floor (bare
         twice, the same call): a mount under that bar at every grid block is read as M0 (no grid of its own)
  MOOD   the mood probe (30 upbeat + 30 downbeat single words, a ridge to +-1, 10-fold, best of lambda 1 / 10 / 100, chance .50)
         at two taps per grid block: (a) the probe's own form, the word alone, its last byte (measured earlier: .97-.98);
         (b) the grid's caption frame (PREFIX + 'an illustration of a quiet street, {word}.'), the closing byte (the full stop)
         and the word's last byte; the per-fold accuracies give the fold spread (the probe's noise bar)
  R      CKA (debiased), CKA without the top component, split Procrustes and the participation-ratio effective rank against
         T5-XXL, bert-base and captionbert on the same 2,048 captions (pool at 14 / 24 / 32, the blackboard at 14), the earlier
         conditioning study's rulers (the stored reference file holds no CLIP ruler; a CLIP-L ruler waits for its embeddings); the
         ridge columns for M0 and the M9s only; the noise bar: draw-to-draw CKA .017 mean / .039 max (between the two caption
         draws)
  4B     the functional binding probe of the earlier conditioning study (PROBE-4b, verbatim) at pool 14 / 24 / 32, last 24 / 32,
         blackboard 14 / 24, with paired bootstrap intervals against M0
  P2     attribute-to-noun binding at the last byte (the earlier study's formula)
Reads coco_caps_2x2048.json (the caption draws) and refs_2x2048.pt (the rulers' embeddings of the same captions) from
settings.DATA_DIR, and the quality prefix from stage 0's s0.pt in settings.OUT_DIR when it is there. Writes
mount_read_step245674.json to settings.MOUNT_READ_OUT, saved after every condition; prints the time spent and left.
Usage: python -m alephllm_diffusion.mount.read [gCA,gCB] [--smoke=1]   (--smoke=1: the first 256 captions, file suffix _smoke;
CUDA_VISIBLE_DEVICES=-1 runs it on the CPU)"""
import contextlib
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

from .. import settings  # noqa: E402
from ..beatrix import extract as bx  # noqa: E402
from ..beatrix import gauges as G  # noqa: E402

STEP = 245674
BLOCKS = [8, 12, 16, 18, 20, 22, 24, 28]                     # the grid's blocks
DATA = settings.DATA_DIR
OUT = settings.MOUNT_READ_OUT
PREFIX_FRAME = "an illustration of a quiet street, {w}."     # the grid's caption form, one fixed scene (after the quality prefix)
L_4B = {"pool": [14, 24, 32], "bb": [14, 24], "last": [24, 32]}
L_R = {"pool": [14, 24, 32], "bb": [14]}
L_P2 = [0, 4, 8, 14, 20, 24, 28, 31, 32]
RIDGE_REPS = {("pool", 24), ("bb", 14)}
CKA_BAR = {"mean": 0.017, "max": 0.039}                      # the draw-to-draw CKA difference (the registered noise bar)
T0 = time.time()


def say(*a):
    print(f"[mount read {time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------- the PROBE-4b generator (VERBATIM)
def probe4b():
    rng = random.Random(7)
    OBJ = ["car", "bicycle", "house", "umbrella", "backpack", "chair", "boat",
           "kite", "ball", "cup", "jacket", "door", "flag", "scarf", "vase",
           "lamp", "truck", "bench", "bird", "fish"]
    COL = ["red", "blue", "green", "yellow", "black", "white", "orange",
           "purple"]
    REL = [("on", "under"), ("above", "below"), ("behind", "in front of")]
    NUM = ["two", "three", "four", "five", "six", "seven"]
    quads, flips = [], []
    for _ in range(300):
        o1, o2 = rng.sample(OBJ, 2)
        c1, c2 = rng.sample(COL, 2)
        fam = rng.choice(("att", "rel", "cnt"))
        if fam == "att":
            C0 = f"a {c1} {o1} and a {c2} {o2}"
            C1 = f"a {c2} {o1} and a {c1} {o2}"
            P0 = f"the {o1} that is {c1} beside the {o2} that is {c2}"
            P1 = f"the {o1} that is {c2} beside the {o2} that is {c1}"
            extra = {}
        elif fam == "rel":
            r, _ = rng.choice(REL)
            C0 = f"the {c1} {o1} is {r} the {c2} {o2}"
            C1 = f"the {c2} {o2} is {r} the {c1} {o1}"
            P0 = f"positioned {r} the {c2} {o2}, there is a {c1} {o1}"
            P1 = f"positioned {r} the {c1} {o1}, there is a {c2} {o2}"
            extra = {"P0k": f"there is a {c1} {o1} positioned {r} the {c2} {o2}", "P1k": f"there is a {c2} {o2} positioned {r} the {c1} {o1}"}
        else:
            n1, n2 = rng.sample(NUM, 2)
            C0 = f"{n1} {c1} {o1}s and {n2} {c2} {o2}s"
            C1 = f"{n2} {c1} {o1}s and {n1} {c2} {o2}s"
            P0 = f"a group of {n1} {o1}s in {c1} next to {n2} {c2} {o2}s"
            P1 = f"a group of {n2} {o1}s in {c1} next to {n1} {c2} {o2}s"
            extra = {}
        quads.append(dict(fam=fam, C0=C0, C1=C1, P0=P0, P1=P1, para=f"a photo of {C0}", **extra))
    for _ in range(200):
        o1, o2 = rng.sample(OBJ, 2)
        c2 = rng.choice([c for c in COL if c not in ("red", "blue")])
        flips.append((f"a red {o1} next to a {c2} {o2}", f"a blue {o1} next to a {c2} {o2}"))
    texts = sorted({t for q in quads for k, t in q.items() if k != "fam"} | {t for p in flips for t in p})
    return quads, flips, texts


QUADS, FLIPS, TEXTS4B = probe4b()
T2I = {t: i for i, t in enumerate(TEXTS4B)}


def score4b(V):
    """the binding probe's scoring (verbatim, as in the arms' training reads) + per-quad outcome vectors for the paired
    bootstrap."""
    def sim(a, b):
        return float((V[T2I[a]] * V[T2I[b]]).sum())
    ok, grp, mrg, mv_flip, mv_para = [], [], [], [], []
    fam_ok = {"att": [], "rel": [], "cnt": []}
    relk_ok, relk_grp = [], []
    for q in QUADS:
        s00, s10, s11, s01 = sim(q["C0"], q["P0"]), sim(q["C1"], q["P0"]), sim(q["C1"], q["P1"]), sim(q["C0"], q["P1"])
        o = s00 > s10
        g = (s00 > s10) and (s11 > s01) and (s00 > s01) and (s11 > s10)
        ok.append(o); grp.append(g); mrg.append(s00 - s10); fam_ok[q["fam"]].append(o)  # noqa: E702
        mv_flip.append(float((V[T2I[q["C0"]]] - V[T2I[q["C1"]]]).norm()))
        mv_para.append(float((V[T2I[q["C0"]]] - V[T2I[q["para"]]]).norm()))
        if q["fam"] == "rel":
            k00, k10, k11, k01 = sim(q["C0"], q["P0k"]), sim(q["C1"], q["P0k"]), sim(q["C1"], q["P1k"]), sim(q["C0"], q["P1k"])
            relk_ok.append(k00 > k10)
            relk_grp.append((k00 > k10) and (k11 > k01) and (k00 > k01) and (k11 > k10))
    n = len(QUADS)
    D = torch.stack([V[T2I[f]] - V[T2I[o]] for o, f in FLIPS])
    Dn = F.normalize(D, dim=-1)
    coher = float(((Dn @ Dn.T).sum() - len(D)) / (len(D) * (len(D) - 1)))
    rel_rec, rel_keep = sum(fam_ok["rel"]) / len(fam_ok["rel"]), sum(relk_ok) / len(relk_ok)
    row = dict(contrast_acc=sum(ok) / n, margin=sum(mrg) / n, group=sum(grp) / n, snr=(sum(mv_flip) / n) / max(sum(mv_para) / n, 1e-9),
               coherence=coher, fam_acc={k: sum(v) / len(v) for k, v in fam_ok.items()},
               rel_order_reversed_record=rel_rec, rel_order_kept=rel_keep, rel_balanced=(rel_rec + rel_keep) / 2)
    return row, dict(ok=[int(x) for x in ok], grp=[int(x) for x in grp], relk=[int(x) for x in relk_ok])


def boot_delta(a, b, n_boot=2000, seed=0):
    """paired bootstrap of mean(b) - mean(a) over the same items: (delta, lo95, hi95)."""
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(a), (n_boot, len(a)), generator=g)
    d = (b[idx] - a[idx]).mean(1)
    return float((b - a).mean()), float(d.quantile(.025)), float(d.quantile(.975))


# ---------------------------------------------------------------------------- the mood probe (verbatim arithmetic)
UP = ["happy", "joyful", "cheerful", "glad", "merry", "elated", "jubilant", "blissful", "delighted", "bright", "sunny", "upbeat",
      "uplifting", "hopeful", "lighthearted", "playful", "ecstatic", "content", "thrilled", "gleeful", "buoyant", "radiant", "jolly",
      "festive", "carefree", "euphoric", "exuberant", "optimistic", "warm", "lively"]
DOWN = ["sad", "gloomy", "somber", "melancholy", "mournful", "forlorn", "desolate", "dismal", "grim", "bleak", "dreary", "sorrowful",
        "depressed", "miserable", "despairing", "downcast", "glum", "morose", "heavy", "dark", "grey", "tearful", "lonely", "hopeless",
        "wistful", "doleful", "woeful", "joyless", "cheerless", "bitter"]
MOOD_Y = torch.tensor([1.0] * len(UP) + [-1.0] * len(DOWN), dtype=torch.float64)
LAMBDAS = (1.0, 10.0, 100.0)


def cv_folds(X, y, lam, folds=10, seed=0):
    """The mood probe's 10-fold ridge classifier, returning the per-fold accuracies."""
    g = torch.Generator().manual_seed(seed)
    up, dn = torch.where(y > 0)[0], torch.where(y < 0)[0]
    up, dn = up[torch.randperm(len(up), generator=g)], dn[torch.randperm(len(dn), generator=g)]
    accs = []
    for k in range(folds):
        te = torch.cat([up[k::folds], dn[k::folds]])
        tr = torch.tensor([i for i in range(len(y)) if i not in set(te.tolist())])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        alpha = torch.linalg.solve(A @ A.T + lam * torch.eye(len(tr), dtype=A.dtype), y[tr])
        pred = (B @ A.T) @ alpha
        accs.append(float(((pred > 0) == (y[te] > 0)).double().mean()))
    return accs


def mood_probe(model, dev, prefix):
    """{tap@block: {acc (best lambda), folds, lambda}} for (a) the word alone at its last byte and (b) the caption frame at the
    closing byte and the word's last byte, at the grid's blocks."""
    out = {}
    words = UP + DOWN
    ex = bx.extract(model, words, layers=BLOCKS, taps=("last",), device=dev, amp=False)
    frames = [prefix + PREFIX_FRAME.format(w=w) for w in words]
    exb = bx.extract(model, frames, layers=BLOCKS, taps=("byte",), device=dev, amp=False)
    for b in BLOCKS:
        cols = {"alone_last": ex["last"][b]}
        close, last = [], []
        for i, (w, t) in enumerate(zip(words, frames)):
            st = exb["byte"][b][i]                              # (bytes of the text, d), the DOC byte excluded
            end = len(t.encode("utf-8")) - 1                     # the full stop = the closing byte
            close.append(st[end])
            last.append(st[end - 1])                             # the word's last byte
        cols["frame_closing"], cols["frame_last"] = torch.stack(close), torch.stack(last)
        for tap, X in cols.items():
            X = F.layer_norm(X.double(), (X.shape[1],))
            best = max(((sum(f) / len(f), f, lam) for lam in LAMBDAS for f in [cv_folds(X, MOOD_Y, lam)]), key=lambda r: r[0])
            out[f"{tap}@{b}"] = {"acc": best[0], "folds": best[1], "lambda": best[2],
                                 "fold_se": float(torch.tensor(best[1]).std() / len(best[1]) ** 0.5)}
    return out


# ---------------------------------------------------------------------------- R, P2, write (as in the arms' training reads)
def convnorm(X):
    Xf = X.float()
    return Xf / float((Xf - Xf.mean(0, keepdim=True)).pow(2).sum(1).mean().sqrt().clamp_min(1e-30))


def r_rows(rep, refs, refp, ridge):
    Xp = G.prep(rep)
    s2 = torch.linalg.svdvals(Xp) ** 2
    pr = float(s2.sum() ** 2 / (s2 ** 2).sum().clamp_min(1e-300))
    res = {}
    for rn, R in refs.items():
        Rp = refp[rn]
        row = {"cka_deb": G.cka_debiased(Xp, Rp), "cka_deb_noPC1": G.cka_debiased(G.drop_pc1(Xp), G.drop_pc1(Rp)),
               "proc_theta_cv": G.procrustes_cv(Xp, Rp), "pr_rep": pr}
        if ridge:
            row["r2_rep2ref"], row["r2_rep2ref_sd"] = G.ridge_r2(Xp, Rp)
            row["r2_ref2rep"], row["r2_ref2rep_sd"] = G.ridge_r2(Rp, Xp)
        res[rn] = row
    del Xp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def triples(n, seed, cols):
    objects = ["cube", "sphere", "car", "dog", "chair", "ball"]
    r = random.Random(seed)
    out = []
    for _ in range(n):
        a1, a2, a3 = r.sample(cols, 3)
        b1, b2 = r.sample(objects, 2)
        out.append((f"a {a1} {b1} on a {a2} {b2}", f"a {a3} {b1} on a {a2} {b2}", f"a {a1} {b1} on a {a3} {b2}"))
    return out


P2 = {"6col": triples(60, 0, ["red", "blue", "green", "black", "white", "brown"]),
      "eqcol": triples(60, 0, ["green", "black", "white", "brown"])}
P2_TEXTS = {k: [t for tri in v for t in tri] for k, v in P2.items()}


def bind2(f, l):
    S = f["last"][l].view(-1, 3, f["last"][l].shape[-1])
    d_own, d_oth = (S[:, 0] - S[:, 2]).norm(dim=1), (S[:, 0] - S[:, 1]).norm(dim=1)
    okm = d_oth > 1e-4
    r = d_own[okm] / d_oth[okm]
    return {"n": int(okm.sum()), "bind2_mean": float(r.mean()) if okm.any() else float("nan"),
            "bind2_frac_gt1": float((r > 1).float().mean()) if okm.any() else float("nan")}


# ---------------------------------------------------------------------------- one condition
def read_condition(model, dev, caps, refs, refp, prefix, ridge):
    rec = {"4b": {}, "4b_vec": {}, "R": {}, "P2": {}}
    f4 = bx.extract(model, TEXTS4B, sorted(set(L_4B["pool"]) | set(L_4B["last"])), taps=("pool", "last"), device=dev, amp=False)
    f4["bb"] = bx.extract(model, TEXTS4B, L_4B["bb"], taps=("bb",), device=dev, amp=False)["bb"]
    for tap, ls in L_4B.items():
        for l in ls:
            row, vec = score4b(F.normalize(f4[tap][l].float(), dim=-1))
            rec["4b"][f"{tap}@L{l}"], rec["4b_vec"][f"{tap}@L{l}"] = row, vec
    del f4
    fc = bx.extract(model, caps, sorted(set(L_R["pool"]) | set(BLOCKS)), taps=("pool",), device=dev, amp=False)
    fc["bb"] = bx.extract(model, caps, L_R["bb"], taps=("bb",), device=dev, amp=False)["bb"]
    for tap, ls in L_R.items():
        for l in ls:
            rep = convnorm(fc[tap][l]) if tap == "pool" and l <= 31 else fc[tap][l]
            rec["R"][f"{tap}@L{l}"] = r_rows(rep, refs, refp, ridge and (tap, l) in RIDGE_REPS)
    pooled = {l: fc["pool"][l].clone() for l in BLOCKS}
    del fc
    for k, v in P2_TEXTS.items():
        rec["P2"][k] = {f"L{l}": bind2(bx.extract(model, v, L_P2, taps=("last",), device=dev, amp=False), l) for l in L_P2}
    rec["mood"] = mood_probe(model, dev, prefix)
    return rec, pooled


# ---------------------------------------------------------------------------- the run
def main(groups=("gCA", "gCB"), smoke=False):
    from geolip.alephllm.arm_mount import CAPTION_ARM, detach_all, masked, mount_group, only
    import amoe.core.adapter as AD
    assert hasattr(AD.BlockWithAdapter, "prefill"), "amoe-lora >= 0.2.11 is needed (prefill applies the adapter)"
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda"
    caps = json.load(open(os.path.join(DATA, "coco_caps_2x2048.json"), encoding="utf-8"))["draw1"]
    refs = torch.load(os.path.join(DATA, "refs_2x2048.pt"))["draw1"]
    if smoke:
        caps, refs = caps[:256], {k: v[:256] for k, v in refs.items()}
    refp = {k: G.prep(v) for k, v in refs.items()}
    from geolip.alephllm.data.special_tokens import DOC  # noqa: F401  (the extractor puts the DOC byte in front)
    prefix = _prefix()
    n_cond = 1 + 3 * len(groups)
    ledger = {"_meta": {"step": STEP, "groups": list(groups), "blocks": BLOCKS, "smoke": smoke, "n_captions": len(caps),
                        "prefix": prefix, "frame": PREFIX_FRAME, "cka_bar": CKA_BAR, "rulers": sorted(refs),
                        "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"mount_read_step{STEP}{'_smoke' if smoke else ''}.json")

    def save():
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, indent=1)

    model = bx.build_model(STEP, device=dev)
    say(f"THE MOUNT READ at {STEP:,}: {len(caps)} captions, {len(TEXTS4B)} 4B strings, 60 mood words, blocks {BLOCKS}; "
        f"{n_cond} conditions (M0, then per arm seed M9 / M8 / Mc)")
    done = 0

    def tick(name):
        nonlocal done
        done += 1
        el = time.time() - T0
        say(f"{name} read: {done}/{n_cond} conditions, about {el / done * (n_cond - done):.0f} s left")

    ledger["M0"], bare_pool = read_condition(model, dev, caps, refs, refp, prefix, ridge=True)
    # The repeat floor and the toggle compare calls of the same shape. The extractor batches captions of equal byte length
    # together, so the first 256 rows of the 2,048-caption call ran in other batches than a 256-caption call does, and kernel
    # batching moves their last bits: bare_probe is the bare trunk's own 256-caption call, and the floor repeats it.
    probe = caps[:256]
    bare_probe = bx.extract(model, probe, BLOCKS, taps=("pool",), device=dev, amp=False)["pool"]
    rep_pool = bx.extract(model, probe, BLOCKS, taps=("pool",), device=dev, amp=False)["pool"]
    floor = {f"L{l}": float(((rep_pool[l] - bare_probe[l]).norm(dim=1) / bare_probe[l].norm(dim=1).clamp_min(1e-12)).max())
             for l in BLOCKS}
    ledger["_meta"]["repeat_floor_rel_max"] = floor
    ledger["_meta"]["repeat_floor_form"] = "the bare trunk twice, the same 256-caption call (the toggle uses the same call)"
    save()
    tick("M0 (bare)")
    for g in groups:
        S = g[-1]
        prog = mount_group(model, g, require_step=STEP, device=dev)
        model.eval()
        with masked(prog, list(prog.attached)):
            tog = bx.extract(model, probe, BLOCKS, taps=("pool",), device=dev, amp=False)["pool"]
        toggle = {f"L{l}": float((tog[l] - bare_probe[l]).abs().max()) for l in BLOCKS}
        assert all(v == 0.0 for v in toggle.values()), ("every member masked must equal bare exactly", toggle)
        ledger["_meta"][f"toggle_{g}"] = toggle
        eight = [m for m in prog.attached if m != CAPTION_ARM]
        for cname, ctx, ridge in ((f"M9_{S}", contextlib.nullcontext(), True), (f"M8_{S}", masked(prog, [CAPTION_ARM]), False),
                                  (f"Mc_{S}", only(prog, [CAPTION_ARM]), False)):
            with ctx:
                rec, pooled = read_condition(model, dev, caps, refs, refp, prefix, ridge=ridge)
            rec["write_ratio"] = {f"L{l}": float(((pooled[l] - bare_pool[l]).norm(dim=1) / bare_pool[l].norm(dim=1).clamp_min(1e-12)).mean())
                                  for l in BLOCKS}
            rec["write_over_10x_floor"] = {k: v > 10 * floor[k] for k, v in rec["write_ratio"].items()}
            rec["4b_delta_vs_M0"] = {k: {"acc": boot_delta(ledger["M0"]["4b_vec"][k]["ok"], v["ok"]),
                                         "group": boot_delta(ledger["M0"]["4b_vec"][k]["grp"], v["grp"])}
                                     for k, v in rec["4b_vec"].items()}
            rec["label"] = {"M9": "all nine live", "M8": "the caption arm masked (the eight)",
                            "Mc": "MASKED: the caption arm with the eight masked"}[cname[:2]]
            ledger[cname] = rec
            save()
            tick(cname)
        detach_all(prog, verify=True)
        model.eval()
        del prog
    ledger["_reads"] = reads(ledger, groups)
    save()
    say(f"wrote {path}")
    return ledger


def _prefix():
    """The grid's caption prefix (the picture runner's quality prefix), from stage 0's file when present, else empty (stated)."""
    p = os.path.join(settings.OUT_DIR, "s0.pt")
    if os.path.exists(p):
        return torch.load(p, weights_only=False)["prefix"]
    say(f"no stage-0 file at {p}: the caption frame is read without the quality prefix")
    return ""


def reads(L, groups):
    """PM1 and PM2 as registered, printed in plain tables; decides nothing beyond the registered bars."""
    seeds = [g[-1] for g in groups]
    out = {"PM1": {}, "PM2": {}}

    def cka(rec, key, ruler):
        return rec["R"][key][ruler]["cka_deb"]
    say("PM1 (the eight quiet on captions) and PM2 (the caption arm moves caption states), per gauge:")
    for S in seeds:
        for mount, pm in ((f"M8_{S}", "PM1"), (f"M9_{S}", "PM2")):
            rec = L[mount]
            d_cka = {f"{k}|{r}": cka(rec, k, r) - cka(L["M0"], k, r) for k in rec["R"] for r in rec["R"][k]}
            d_mood = {k: rec["mood"][k]["acc"] - L["M0"]["mood"][k]["acc"] for k in rec["mood"]}
            spread = {k: max(rec["mood"][k]["fold_se"], L["M0"]["mood"][k]["fold_se"]) for k in rec["mood"]}
            out[pm][mount] = {"cka_delta": d_cka, "cka_beyond_bar": {k: abs(v) > CKA_BAR["max"] for k, v in d_cka.items()},
                              "mood_delta": d_mood, "mood_beyond_2se": {k: abs(v) > 2 * spread[k] for k, v in d_mood.items()},
                              "write_ratio": rec["write_ratio"], "write_over_10x_floor": rec["write_over_10x_floor"],
                              "4b_acc_delta": {k: v["acc"] for k, v in rec["4b_delta_vs_M0"].items()}}
            o = out[pm][mount]
            say(f"  {pm} {mount}: CKA changes {sum(o['cka_beyond_bar'].values())}/{len(d_cka)} beyond .039 (largest "
                f"{max(map(abs, d_cka.values())):.3f}); mood probe {sum(o['mood_beyond_2se'].values())}/{len(d_mood)} beyond two fold "
                f"SEs (largest {max(map(abs, d_mood.values())):.3f}); write over 10x the repeat floor at "
                f"{sum(o['write_over_10x_floor'].values())}/{len(BLOCKS)} grid blocks (largest {max(o['write_ratio'].values()):.2e})")
    if len(seeds) == 2:
        a, b = (L[f"M9_{s}"] for s in seeds)
        out["PM2_seed_gap"] = {f"{k}|{r}": abs(cka(a, k, r) - cka(b, k, r)) for k in a["R"] for r in a["R"][k]}
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(tuple(args[0].split(",")) if args else ("gCA", "gCB"), smoke="--smoke=1" in sys.argv)
