"""alephllm_diffusion.beatrix.gauges - the conditioning-gauge functions of the earlier conditioning study, copied VERBATIM from
its geometry probe (prep, hsic1, cka_debiased, drop_pc1, procrustes_cv, ridge_r2: the probe's lines 113-203), so the rows here
are computed by the same code as that study's rows. Additions below the verbatim block: erank, a row builder, and a memory-light
path for wide representations (prep and erank are rebound to it at the end of the module; selfcheck() asserts the equivalence).
The gauges compare two representations of the same n items, (n, d) tensors with rows in the same order, and run on the card
(dev). Use: row(rep, ref) returns one row of every gauge as a dict; nothing runs on import and nothing is written.
"""
import torch
import torch.nn.functional as F

dev = "cuda"


def prep(X):
    X = X.double().to(dev)
    X = X - X.mean(0, keepdim=True)
    if X.shape[1] > X.shape[0]:
        # exact Gram-space reduction: every gauge here is invariant to
        # right-rotation, so U*sqrt(evals) (n x n) is lossless and keeps
        # SVD workspaces off the 32k-dim blackboard rep
        K = X @ X.T
        evals, U = torch.linalg.eigh(K)
        X = U * evals.clamp_min(0).sqrt().unsqueeze(0)
    return X


def hsic1(K, L):
    n = K.shape[0]
    K = K.clone()
    L = L.clone()
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)
    t1 = (K * L).sum()
    t2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * (K.sum(0) @ L.sum(1))
    return (t1 + t2 - t3) / (n * (n - 3))


def cka_debiased(X, Y):
    K, L = X @ X.T, Y @ Y.T
    return float(hsic1(K, L)
                 / torch.sqrt(hsic1(K, K) * hsic1(L, L)).clamp_min(1e-30))


def drop_pc1(X):
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return X - (U[:, :1] * S[:1]) @ Vh[:1]


def procrustes_cv(X, Y):
    n = X.shape[0]
    g = torch.Generator().manual_seed(11)
    perm = torch.randperm(n, generator=g)
    a, b = perm[:n // 2], perm[n // 2:]
    Xa, Ya, Xb, Yb = X[a], Y[a], X[b], Y[b]
    Xa = Xa / Xa.norm()
    Ya = Ya / Ya.norm()
    Xb = Xb / Xb.norm()
    Yb = Yb / Yb.norm()
    U, S, Vh = torch.linalg.svd(Xa.T @ Ya, full_matrices=False)
    T = U @ Vh                                  # fit rotation on half A
    ip = float((Xb @ T * Yb).sum()
               / (Xb.norm() * Yb.norm()).clamp_min(1e-30))
    return float(torch.arccos(torch.tensor(max(-1.0, min(1.0, ip)))))


def ridge_r2(X, Y, folds=10):
    """R^2 of ridge X->Y, dual form, lambda by inner 3-fold CV."""
    n = X.shape[0]
    g = torch.Generator().manual_seed(13)
    idx = torch.randperm(n, generator=g)
    fold_sz = n // folds
    lams = [10.0 ** e for e in range(-2, 6)]
    r2s = []
    for f in range(folds):
        te = idx[f * fold_sz:(f + 1) * fold_sz]
        tr = torch.cat([idx[:f * fold_sz], idx[(f + 1) * fold_sz:]])
        Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
        inner = len(tr) // 3
        best, best_lam = -1e18, lams[0]
        for lam in lams:
            itr, ite = tr[inner:], tr[:inner]
            Xi, Yi, Xv, Yv = X[itr], Y[itr], X[ite], Y[ite]
            Kii = Xi @ Xi.T
            A = torch.linalg.solve(
                Kii + lam * torch.eye(len(itr), dtype=torch.float64,
                                      device=dev), Yi)
            pred = (Xv @ Xi.T) @ A
            ss = 1 - ((pred - Yv) ** 2).sum() \
                / ((Yv - Yv.mean(0)) ** 2).sum().clamp_min(1e-30)
            if float(ss) > best:
                best, best_lam = float(ss), lam
        K = Xtr @ Xtr.T
        A = torch.linalg.solve(
            K + best_lam * torch.eye(len(tr), dtype=torch.float64,
                                     device=dev), Ytr)
        pred = (Xte @ Xtr.T) @ A
        r2s.append(float(
            1 - ((pred - Yte) ** 2).sum()
            / ((Yte - Yte.mean(0)) ** 2).sum().clamp_min(1e-30)))
    t = torch.tensor(r2s)
    return float(t.mean()), float(t.std())


# ---------------------------------------------------------------- additions (not in the study's file)
def erank(X):
    """effective rank (exp of the entropy of the normalized singular values) of the centered rep."""
    Xc = X.double().to(dev)
    Xc = Xc - Xc.mean(0, keepdim=True)
    if Xc.shape[1] > Xc.shape[0]:
        K = Xc @ Xc.T
        s = torch.linalg.eigvalsh(K).clamp_min(0).sqrt()
    else:
        s = torch.linalg.svdvals(Xc)
    p = s / s.sum().clamp_min(1e-30)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def row(rep, ref, rep_prepped=None, ref_prepped=None):
    """one (rep, ref) row exactly as the study's loop builds it, plus erank and the participation ratio of the rep. `rep`, `ref`
    are (n, d) tensors, same row order."""
    Xp = prep(rep) if rep_prepped is None else rep_prepped
    Rp = prep(ref) if ref_prepped is None else ref_prepped
    Rp1 = drop_pc1(Rp)
    out = {
        "cka_deb": cka_debiased(Xp, Rp),
        "cka_deb_noPC1": cka_debiased(drop_pc1(Xp), Rp1),
        "proc_theta_cv": procrustes_cv(Xp, Rp),
    }
    r2f, r2f_sd = ridge_r2(Xp, Rp)
    r2b, r2b_sd = ridge_r2(Rp, Xp)
    # pr_rep = the standard effective rank of these studies: the participation ratio (sum s^2)^2 / sum s^4 of the centred rep,
    # computed from the centred Gram-reduced coordinates Xp (same singular values as the centred rep). erank_rep = the exp-entropy
    # form (NOT the standard statistic; kept for continuity with the first results, reads 10-25x higher).
    s2 = torch.linalg.svdvals(Xp) ** 2
    pr = float(s2.sum() ** 2 / (s2 ** 2).sum().clamp_min(1e-300))
    out.update(r2_rep2ref=r2f, r2_rep2ref_sd=r2f_sd, r2_ref2rep=r2b, r2_ref2rep_sd=r2b_sd, erank_rep=erank(rep), pr_rep=pr)
    return out


# ---------------------------------------------------------------- memory-light path for wide reps (verbatim functions untouched)
# The verbatim prep() copies a wide (n x d) matrix to fp64 on the card: 4.3 GB for a 2048 x 262,144 blackboard, and erank() above
# copies it again. On a shared card that once stacked past the memory cap, so the rows here use this equivalent: the SAME
# Gram-space reduction (centre, K = X X^T, eigh, U*sqrt(evals)) with K accumulated in fp32 chunks from the (fp16/fp32) rows, K and
# eigh in fp64 (n x n only). Equivalence is asserted by selfcheck().
_prep_verbatim = prep
_erank_full = erank


def _gram_chunked(X, chunk=256):
    n = X.shape[0]
    Xg = X.to(dev)
    mu = Xg.float().mean(0, keepdim=True)
    K = torch.zeros(n, n, dtype=torch.float64, device=dev)
    for i in range(0, n, chunk):
        Ai = Xg[i:i + chunk].float() - mu
        for j in range(0, n, chunk):
            K[i:i + chunk, j:j + chunk] = (Ai @ (Xg[j:j + chunk].float() - mu).T).double()
    del Xg
    return K


def prep_light(X):
    if X.shape[1] <= X.shape[0]:
        return _prep_verbatim(X)
    evals, U = torch.linalg.eigh(_gram_chunked(X))
    return U * evals.clamp_min(0).sqrt().unsqueeze(0)


def erank_light(X):
    if X.shape[1] <= X.shape[0]:
        return _erank_full(X)
    s = torch.linalg.eigvalsh(_gram_chunked(X)).clamp_min(0).sqrt()
    p = s / s.sum().clamp_min(1e-30)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def selfcheck():
    """asserts the light path equals the verbatim prep / the full erank on a wide test matrix (prints the numbers)."""
    T = torch.randn(300, 5000, generator=torch.Generator().manual_seed(3)).half()
    a, b = _prep_verbatim(T), prep_light(T)
    ka, kb = a @ a.T, b @ b.T
    rel = float((ka - kb).abs().max() / ka.abs().max())
    e1, e2 = _erank_full(T), erank_light(T)
    assert rel < 1e-5 and abs(e1 - e2) / e1 < 1e-3, (rel, e1, e2)
    print(f"[gauges selfcheck] prep_light vs record prep rel diff {rel:.2e}; erank {e2:.3f} vs {e1:.3f}", flush=True)


prep = prep_light            # row() and the drivers resolve these names at call time
erank = erank_light
