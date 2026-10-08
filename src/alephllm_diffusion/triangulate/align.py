"""alephllm_diffusion.triangulate.align - the arithmetic of the dual-extraction read (pure torch, float64, any device). Every
comparison is between two representations of the same rows (n x d tensors, rows in the same order) and is scored on captions the
fit never saw: the rows are split by caption into two halves and each half in turn is the fit half (folds).
  Whitener          centre on the fit rows, keep their k leading principal directions, scale each to unit variance
  rotation          the orthogonal Procrustes rotation between two whitened fit sets: U V^T from the SVD of X^T Y
  alignment         on held-out rows, the cosine of the angle between X R and Y: 1 = the same geometry up to a rotation, 0 =
                    unrelated (the cosine form of the split-Procrustes angle)
  ridge_r2          held-out R^2 of a ridge map X -> Y, the penalty chosen by an inner split of the fit captions
  compare           alignment and ridge, and the same after removing each token's fit-half mean from both sides (the CONTEXT
                    alignment: the variation beyond the token's identity), as means over the two folds
  triangle          the composed rotation X -> Y -> Z against the direct X -> Z, both scored on held-out rows
  directions        class directions (a class mean minus the neutral mean) of a signal mapped by its rotation into the target's
                    frame, against the target's own; held-out rows read on the target's own class axis
  gram_coordinates  exact coordinates of wide rows from their centred Gram (U sqrt(eigenvalues)), the leading r kept
"""
import torch

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)          # ridge penalties, as multiples of the fit rows' count (unit-variance inputs)


def folds(groups, seed=0):
    """[(fit, held), (held, fit)]: two complementary row masks, half the captions (group ids, shuffled by `seed`) on each side."""
    g = torch.as_tensor(groups)
    caps = torch.unique(g)
    perm = caps[torch.randperm(len(caps), generator=torch.Generator().manual_seed(seed))]
    a = torch.isin(g, perm[: len(caps) // 2])
    return [(a, ~a), (~a, a)]


class Whitener:
    """centre on X's rows, project on their k leading principal directions, scale each to unit variance (scale=False: keep each
    direction at its own variance, the unwhitened form); `share` = the variance those k directions hold. The directions come
    from the d x d scatter matrix when the rows outnumber the columns (the same directions as the SVD, without its n x d factor),
    else from the SVD."""

    def __init__(self, X, k, scale=True, _from=None):
        if _from is not None:                                   # the same directions, the other scaling
            self.mu, self.V, self.s, self.n, self.k, self.share = (_from.mu, _from.V, _from.s, _from.n, _from.k, _from.share)
        else:
            X = X.double()
            self.mu = X.mean(0, keepdim=True)
            Xc = X - self.mu
            self.n, d = Xc.shape
            if self.n > d:
                ev, V = torch.linalg.eigh(Xc.T @ Xc)
                S, V = ev.flip(0).clamp_min(0).sqrt(), V.flip(1)
            else:
                _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
                V = Vh.T
            self.k = 0 if float(S[0]) <= 0 else min(k, int((S > S[0] * 1e-9).sum()))
            self.V, self.s = V[:, :self.k], S[:self.k]
            self.share = float((S[:self.k] ** 2).sum() / (S ** 2).sum().clamp_min(1e-300))
        self.W = self.V / (self.s / max(self.n - 1, 1) ** 0.5) if scale else self.V

    def rescaled(self, scale):
        return Whitener(None, None, scale, _from=self)

    def __call__(self, X):
        return (X.double().to(self.mu.device) - self.mu) @ self.W


def rotation(Xf, Yf):
    U, _, Vh = torch.linalg.svd(Xf.T @ Yf, full_matrices=False)
    return U @ Vh


def alignment(X, Y, R):
    return float((X @ R * Y).sum() / (X.norm() * Y.norm()).clamp_min(1e-30))


def r2(P, Y):
    return float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum().clamp_min(1e-30))


def _ridge(Xf, Yf, lam):
    d = Xf.shape[1]
    return torch.linalg.solve(Xf.T @ Xf + lam * Xf.shape[0] * torch.eye(d, dtype=Xf.dtype, device=Xf.device), Xf.T @ Yf)


def ridge_r2(Xf, Yf, gf, Xh, Yh):
    """(held-out R^2, the penalty): the penalty chosen on an inner caption split of the fit rows."""
    ia, ib = folds(gf, seed=1)[0]
    best = max(LAMBDAS, key=lambda lam: r2(Xf[ib] @ _ridge(Xf[ia], Yf[ia], lam), Yf[ib]))
    return r2(Xh @ _ridge(Xf, Yf, best), Yh), best


def _token_residuals(Xf, idf, Xh, idh, min_count=2):
    """subtract each token id's fit-half mean; rows whose id occurs fewer than min_count times in the fit half are dropped
    (both halves). Returns (Xf', keep_f, Xh', keep_h)."""
    uniq, inv, counts = torch.unique(idf, return_inverse=True, return_counts=True)
    means = torch.zeros(len(uniq), Xf.shape[1], dtype=torch.float64, device=Xf.device).index_add_(0, inv.to(Xf.device),
                                                                                                    Xf.double())
    means = means / counts.to(Xf.device)[:, None]
    ok_ids = uniq[counts >= min_count]
    keep_f, keep_h = torch.isin(idf, ok_ids), torch.isin(idh, ok_ids)
    pos_h = torch.searchsorted(uniq, idh[keep_h])
    return (Xf.double()[keep_f] - means[inv[keep_f].to(Xf.device)], keep_f,
            Xh.double()[keep_h] - means[pos_h.to(Xh.device)], keep_h)


def _aligned(wx, wy, Xf, Yf, Xh, Yh):
    """held-out alignment through the given reductions of each side (fit on Xf / Yf); 0 when a side has no variation."""
    if min(wx.k, wy.k) == 0:
        return 0.0
    return alignment(wx(Xh), wy(Yh), rotation(wx(Xf), wy(Yf)))


def compare(X, Y, groups, k, ids=None, ridge=True):
    """X -> Y on held-out captions, means over the two folds: alignment (whitened sides) and alignment_unwhitened (each side at
    its own variance: whitening both sides and rotating is canonical correlation, which weighs a weak direction as much as a
    strong one), ridge_r2 (if ridge), context_alignment, its unwhitened twin and n_context (if token ids are given), and the
    variance the k directions keep on each side."""
    g = torch.as_tensor(groups)
    acc = {}

    def add(name, v):
        acc.setdefault(name, []).append(v)

    for fit, held in folds(g):
        wx, wy = Whitener(X[fit], k), Whitener(Y[fit], k)
        Xf, Yf, Xh, Yh = wx(X[fit]), wy(Y[fit]), wx(X[held]), wy(Y[held])
        add("alignment", alignment(Xh, Yh, rotation(Xf, Yf)) if min(wx.k, wy.k) else 0.0)
        add("alignment_unwhitened", _aligned(wx.rescaled(False), wy.rescaled(False), X[fit], Y[fit], X[held], Y[held]))
        add("share_x", wx.share)
        add("share_y", wy.share)
        if ridge:
            add("ridge_r2", ridge_r2(Xf, Yf, g[fit], Xh, Yh)[0])
        if ids is not None:
            idt = torch.as_tensor(ids)
            Xfr, kf, Xhr, kh = _token_residuals(X[fit], idt[fit], X[held], idt[held])
            Yfr, _, Yhr, _ = _token_residuals(Y[fit], idt[fit], Y[held], idt[held])
            # a side with no variation beyond the token's identity (Qwen's input embedding) has nothing to align: 0
            flat = (float(Xfr.norm()) <= 1e-9 * float(X[fit].double().norm()) or
                    float(Yfr.norm()) <= 1e-9 * float(Y[fit].double().norm()))
            if flat:
                add("context_alignment", 0.0)
                add("context_alignment_unwhitened", 0.0)
            else:
                cx, cy = Whitener(Xfr, k), Whitener(Yfr, k)
                add("context_alignment", _aligned(cx, cy, Xfr, Yfr, Xhr, Yhr))
                add("context_alignment_unwhitened", _aligned(cx.rescaled(False), cy.rescaled(False), Xfr, Yfr, Xhr, Yhr))
            add("n_context", int(kh.sum()))
    return {name: (sum(v) / len(v)) for name, v in acc.items()}


def pair_ridge(Xs, Y, groups, k):
    """held-out ridge R^2 to Y from several signals side by side (each whitened on its own, then concatenated); fold mean."""
    g = torch.as_tensor(groups)
    out = []
    for fit, held in folds(g):
        ws, wy = [Whitener(X[fit], k) for X in Xs], Whitener(Y[fit], k)
        Xf = torch.cat([w(X[fit]) for w, X in zip(ws, Xs)], 1)
        Xh = torch.cat([w(X[held]) for w, X in zip(ws, Xs)], 1)
        out.append(ridge_r2(Xf, wy(Y[fit]), g[fit], Xh, wy(Y[held]))[0])
    return sum(out) / len(out)


def triangle(X, Y, Z, groups, k):
    """{direct, composed, gap}: held-out alignment of X -> Z by its own rotation and by the rotations X -> Y then Y -> Z; gap =
    direct - composed; fold means. Each leg is fit on its OWN captions (the fit half split by caption into two quarters: X -> Y on
    one, Y -> Z on the other, both ways round; the direct rotation on each quarter alone, the same size as a leg). Fit on the same
    rows the two legs would close by construction through ANY middle space: the polar factor of Y^T X cancels against itself."""
    g = torch.as_tensor(groups)
    d, c = [], []
    for fit, held in folds(g):
        wx, wy, wz = Whitener(X[fit], k), Whitener(Y[fit], k), Whitener(Z[fit], k)
        Xf, Yf, Zf = wx(X[fit]), wy(Y[fit]), wz(Z[fit])
        Xh, Zh = wx(X[held]), wz(Z[held])
        (q1, q2), _ = folds(g[fit], seed=2)
        for a, b in ((q1, q2), (q2, q1)):
            d.append(alignment(Xh, Zh, rotation(Xf[a], Zf[a])))
            c.append(alignment(Xh, Zh, rotation(Xf[a], Yf[a]) @ rotation(Yf[b], Zf[b])))
    d, c = sum(d) / len(d), sum(c) / len(c)
    return {"direct": d, "composed": c, "gap": d - c}


def _cos(a, b):
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def axis_value(M, mid, axis):
    """the slider's centre-difference arithmetic: (M - mid) along `axis`, scaled so a row at mid + axis / 2 reads +1."""
    return (M - mid) @ axis / (axis @ axis / 2).clamp_min(1e-30)


def directions(S_fit, Q_fit, S_rows, Q_rows, cls, split, k):
    """S's class directions in Q's frame against Q's own. The whitening and the rotation are fit on (S_fit, Q_fit) only (the
    caption rows); S_rows / Q_rows are the class rows (cls in {'up', 'down', 'neutral'}, split in {'train', 'heldout'}).
    Returns the cosines of S's mapped directions with Q's: per side (cos_up, cos_down: a class mean minus the neutral mean), the
    MOOD AXIS (cos_axis: the up mean minus the down mean) and the SHARED part (cos_shared: the mean of the two sides minus the
    neutral mean; a side direction = the shared part plus or minus half the axis); Q's and S's own cosine between their two sides;
    and the held-out rows' signs as (right, total): mapped_* = S's mapped rows read from S's OWN mapped centre along Q's axis
    (does her deviation point Qwen's way; S's centre, because the rotation does not carry one set's mean onto the other's),
    q_own_* = Q's rows on Q's axis, s_own_* = S's rows on S's own axis."""
    ws, wq = Whitener(S_fit, k), Whitener(Q_fit, k)
    R = rotation(ws(S_fit), wq(Q_fit))
    Sw, Qw = ws(S_rows), wq(Q_rows)
    Sm = Sw @ R
    idx = {(c, s): [i for i in range(len(cls)) if cls[i] == c and split[i] == s] for c in ("up", "down", "neutral")
           for s in ("train", "heldout")}

    def centres(M):
        return {c: M[idx[(c, "train")]].mean(0) for c in ("up", "down", "neutral")}

    cs, cq, co = centres(Sm), centres(Qw), centres(Sw)

    def shared(c):
        return (c["up"] + c["down"]) / 2 - c["neutral"]

    out = {"cos_up": _cos(cs["up"] - cs["neutral"], cq["up"] - cq["neutral"]),
           "cos_down": _cos(cs["down"] - cs["neutral"], cq["down"] - cq["neutral"]),
           "cos_axis": _cos(cs["up"] - cs["down"], cq["up"] - cq["down"]),
           "cos_shared": _cos(shared(cs), shared(cq)),
           "q_sides_cos": _cos(cq["up"] - cq["neutral"], cq["down"] - cq["neutral"]),
           "s_sides_cos": _cos(co["up"] - co["neutral"], co["down"] - co["neutral"])}
    q_axis = cq["up"] - cq["down"]
    for name, M, mid, axis in (("mapped", Sm, (cs["up"] + cs["down"]) / 2, q_axis),
                               ("q_own", Qw, (cq["up"] + cq["down"]) / 2, q_axis),
                               ("s_own", Sw, (co["up"] + co["down"]) / 2, co["up"] - co["down"])):
        v = axis_value(M, mid, axis)
        for c, sign in (("up", 1), ("down", -1)):
            rows = idx[(c, "heldout")]
            out[f"{name}_{c}"] = (int(sum(float(v[i]) * sign > 0 for i in rows)), len(rows))
        out[f"{name}_values"] = {c: [round(float(v[i]), 4) for i in idx[(c, "heldout")]] for c in ("up", "down", "neutral")}
    return out


def gram_coordinates(K, r):
    """(coords n x r, retained share): exact coordinates of n rows from their centred Gram, U sqrt(eigenvalues), the r leading
    components kept (every rotation-invariant number above sees the full rows up to the dropped tail)."""
    evals, U = torch.linalg.eigh(K.double())
    evals, U = evals.flip(0).clamp_min(0), U.flip(1)
    r = min(r, len(evals))
    return U[:, :r] * evals[:r].sqrt(), float(evals[:r].sum() / evals.sum().clamp_min(1e-300))


def spearman(a, b):
    ra, rb = a.double().argsort().argsort().double(), b.double().argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])
