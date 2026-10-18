"""alephllm_diffusion.nano.extract - a nano body's taps on the rows of the dual-extraction read (e038), image-free, no training.

THE ROWS are e038's, rebuilt from the exported tokenizer file (export/anima_tokenizer_e038_captions.json: every caption of both COCO
draws under Anima's own Qwen3 tokenizer, the same renderer the read used) and the read's 74 mood rows (H.ROWS in the grid caption
form, exported as JSON): per draw, the first 512 captions, one row per Qwen token but a caption's first and last, read at the token's
closing byte (the byte after its bytes), then the mood rows read at their final '.'. The DOC byte goes in front of every text, as the
read put it in front of Beatrix's. Row order, groups (the caption of each row) and token ids are the read's, so the stored Qwen sets
(sets_qwen.pt) line up row for row; the caption-row counts are asserted (4,920 / 5,086).

THE TAPS, one fp32 forward per text (texts of equal length batched, no padding; TF32 off; the state at a byte of a causal model is
the state a prefix pass ending there gives, checked by `causality`):
  stream {b: (n, 256)}   block b's output LayerNorm'd without affine (the read's trunk tap), b = 0..5; b = 6 the final norm's output
                         (what the output layer reads), as it is
  size (n,)              the last block's raw output size (the read's C7)
  hub {name: (n, r)}     each hub's BLACKBOARD at the row's byte: per book (Sp - Sn) over its anchors, each anchor's slot L2-normalized,
                         flattened (the read's tap: K x d per book). A hub with the address rotary keeps its state in a rotating frame;
                         its blackboard is taken as the query at that byte reads it (each anchor pair rotated back by the byte's own
                         angle), so it is comparable across bytes. A dual hub's two hubs are read apart ('<block>r' rotary, '<block>p'
                         plain). The blackboards enter as their exact Gram coordinates (the leading r kept, the share recorded).
  parity {name: float}   for every hub, the largest difference between the hub's own output and the output recomputed from the
                         blackboard's state (the state is exactly what the hub reads, or the read stops)
Nothing runs on import."""
from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F

from ..triangulate import align as AL

N_CAPS = 512
R_HUB = 1024
CAPTION_ROWS = {"draw1": 4920, "draw2": 5086}          # e038's caption-row counts (its log); asserted
PARITY_MAX = 1e-3                                       # relative: the recomputed hub output against the hub's own


# ---------------------------------------------------------------------------------------------------- the rows
def rows(export_path: str, mood_path: str, n_caps: int = N_CAPS) -> dict:
    """{draw: dict(texts [bytes], pos [[positions in the DOC-prefixed text]], group, ids, mood, cls, split, n_cap_rows)}."""
    E = json.load(open(export_path, encoding="utf-8"))
    M = json.load(open(mood_path, encoding="utf-8"))["rows"]
    out = {}
    for d in ("draw1", "draw2"):
        caps = E["draws"][d][:n_caps]
        texts, pos, group, ids = [], [], [], []
        for i, c in enumerate(caps):
            raw = c["text"].encode("utf-8")
            ts = list(range(1, len(c["ids"]) - 1))
            assert all(c["a_end"][t] < len(raw) for t in ts), c["text"]
            texts.append(raw)
            pos.append([1 + c["a_end"][t] for t in ts])
            group += [i] * len(ts)
            ids += [c["ids"][t] for t in ts]
        n_cap_rows = len(group)
        if n_caps == N_CAPS:
            assert n_cap_rows == CAPTION_ROWS[d], (d, n_cap_rows)
        for j, m in enumerate(M):
            raw = m["text"].encode("utf-8")
            assert raw.endswith(b"."), m["text"]
            texts.append(raw)
            pos.append([len(raw)])
            group.append(len(caps) + j)
            ids.append(-1)
        out[d] = dict(texts=texts, pos=pos, group=torch.tensor(group), ids=torch.tensor(ids),
                      mood=torch.arange(len(group)) >= n_cap_rows, cls=[m["cls"] for m in M], split=[m["split"] for m in M],
                      n_cap_rows=n_cap_rows)
    return out


# ---------------------------------------------------------------------------------------------------- the hubs
def hubs(model) -> dict:
    """{name: CausalSplatHUB}: '<block>' for a single hub, '<block>r' / '<block>p' for a dual hub's rotary / plain hub."""
    out = {}
    for bi, blk in enumerate(model.blocks):
        if not blk.is_hub:
            continue
        if getattr(blk.attn, "dual", False):
            out[f"{bi}r"], out[f"{bi}p"] = blk.attn.rot, blk.attn.plain
        else:
            out[f"{bi}"] = blk.attn
    return out


def hub_state(hub, h):
    """the hub's blackboard at every byte of h (B, n, d) and the hub's output recomputed from that state.
    -> (bb (B, n, books x K x d), out (B, n, d))."""
    from geolip.alephllm.model.attention import dtype_floor
    qc, kc = hub._code_cat_qk(h)                                   # (B, n, books x 2K): per book [K pos | K neg]
    v = hub.v(h)
    B, n, W = kc.shape
    K = hub.K
    H, d = W // (2 * K), v.shape[-1]
    kr = hub._rotate_codes(kc) if hub.rotary else kc
    S = torch.cumsum(kr.unsqueeze(-1) * v.unsqueeze(-2), dim=1)    # (B, n, W, d): the numerator's state after each byte
    if hub.rotary:                                                 # the state as the query at that byte reads it
        cos, sin = hub._rope.tables(n, 0, h.device, h.dtype)       # (n, K)
        if hub.n_rot < K:
            keep = torch.zeros(K, device=h.device, dtype=torch.bool)
            keep[:hub.n_rot] = True
            cos = torch.where(keep, cos, torch.ones_like(cos))
            sin = torch.where(keep, sin, torch.zeros_like(sin))
        Sv = S.view(B, n, H, 2, K, d)
        p, q = Sv[:, :, :, 0], Sv[:, :, :, 1]
        c, s = cos[None, :, None, :, None], sin[None, :, None, :, None]
        S = torch.stack([p * c + q * s, q * c - p * s], dim=3).reshape(B, n, W, d)
    z = torch.cumsum(kc, dim=1)                                    # the agreement mass (never rotated)
    num = torch.einsum("bnw,bnwd->bnd", qc, S)
    den = (qc * z).sum(-1, keepdim=True)
    out = hub.o(num / den.clamp_min(dtype_floor(den)))
    Sv = S.view(B, n, H, 2, K, d)
    bb = F.normalize(Sv[:, :, :, 0] - Sv[:, :, :, 1], dim=-1).reshape(B, n, H * K * d)
    return bb, out


@torch.no_grad()
def centred_gram(X, chunk=512):
    """the centred Gram (n x n, float64) of fp16 rows on their device (triangulate.read.centred_gram, copied: that module's imports
    need the read's environment)."""
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


# ---------------------------------------------------------------------------------------------------- the read
class _Taps:
    """forward hooks: every block's output, the final norm's output, every hub's input and output."""

    def __init__(self, model):
        self.got, self.handles = {}, []
        for bi, blk in enumerate(model.blocks):
            self.handles.append(blk.register_forward_hook(self._save(("blk", bi))))
        self.handles.append(model.nf.register_forward_hook(self._save("nf")))
        for name, hub in hubs(model).items():
            self.handles.append(hub.register_forward_pre_hook(self._save_in(("hin", name))))
            self.handles.append(hub.register_forward_hook(self._save(("hout", name))))

    def _save(self, key):
        def f(_m, _i, o):
            self.got[key] = o
        return f

    def _save_in(self, key):
        def f(_m, i):
            self.got[key] = i[0]
        return f

    def close(self):
        for h in self.handles:
            h.remove()


def _groups(lengths, max_batch):
    by = {}
    for i, L in enumerate(lengths):
        by.setdefault(L, []).append(i)
    out = []
    for L in sorted(by):
        idx = by[L]
        out += [idx[k:k + max_batch] for k in range(0, len(idx), max_batch)]
    return out


@torch.no_grad()
def read(model, texts, pos, device="cuda", r=R_HUB, max_batch=64, hub_batch=16, log=print) -> dict:
    """the reader's taps at every row (texts with their read positions, rows in text order). -> dict(stream, size, hub, share,
    parity, n, time_s, peak_gb)."""
    from geolip.alephllm.data.special_tokens import DOC
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    L, d = len(model.blocks), model.cfg.d_model
    starts, n = [], 0
    for p in pos:
        starts.append(n)
        n += len(p)
    H = hubs(model)
    stream = {b: torch.zeros(n, d) for b in range(L + 1)}
    size = torch.zeros(n)
    bb, parity = {}, {name: 0.0 for name in H}
    taps = _Taps(model)
    try:
        for grp in _groups([len(t) + 1 for t in texts], max_batch):
            x = torch.tensor([[int(DOC)] + list(texts[i]) for i in grp], dtype=torch.long, device=device)
            model(x)
            g = taps.got
            for k, i in enumerate(grp):
                rr = torch.arange(starts[i], starts[i] + len(pos[i]))
                pp = torch.tensor(pos[i], device=device)
                for b in range(L):
                    v = g[("blk", b)][k, pp].float()
                    assert torch.isfinite(v).all(), ("non-finite state", b, i)
                    stream[b][rr] = F.layer_norm(v, (d,)).cpu()
                    if b == L - 1:
                        size[rr] = v.norm(dim=-1).cpu()
                stream[L][rr] = g["nf"][k, pp].float().cpu()
            for name, hub in H.items():
                h_all, o_all = g[("hin", name)], g[("hout", name)]
                for k0 in range(0, len(grp), hub_batch):
                    part = grp[k0:k0 + hub_batch]
                    state, out = hub_state(hub, h_all[k0:k0 + hub_batch].float())
                    ref = o_all[k0:k0 + hub_batch].float()
                    rel = float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
                    parity[name] = max(parity[name], rel)
                    assert rel < PARITY_MAX, ("the blackboard's state is not the hub's", name, rel)
                    if name not in bb:
                        bb[name] = torch.empty(n, state.shape[-1], dtype=torch.float16, device=device)
                    for k, i in enumerate(part):
                        rr = torch.arange(starts[i], starts[i] + len(pos[i]), device=device)
                        bb[name][rr] = state[k, torch.tensor(pos[i], device=device)].to(torch.float16)
                    del state, out
            taps.got.clear()
    finally:
        taps.close()
    hub_out, share = {}, {}
    for name in sorted(bb):
        Kc = centred_gram(bb.pop(name))
        coords, share[name] = AL.gram_coordinates(Kc, r)
        hub_out[name] = coords.float().cpu()
        del Kc, coords
        if device == "cuda":
            torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0
    log(f"  read {len(texts)} texts, {n} rows in {time.time() - t0:.1f} s; peak {peak:.2f} GB; hub parity "
        + (" ".join(f"{k} {v:.1e}" for k, v in parity.items()) or "-"))
    return dict(stream=stream, size=size, hub=hub_out, share=share, parity=parity, n=n, time_s=time.time() - t0, peak_gb=peak)


@torch.no_grad()
def causality(model, texts, pos, picks, device="cuda") -> dict:
    """for a few (text, position) picks: the taps of the full text at the position against a prefix pass ending there (its last
    byte). -> {tap: largest absolute difference}."""
    full = read(model, [texts[i] for i, _ in picks], [[pos[i][j]] for i, j in picks], device, r=len(picks), log=lambda *_: None)
    pre_texts = [texts[i][:pos[i][j]] for i, j in picks]                     # the DOC byte is position 0: the prefix ends at pos
    pre = read(model, pre_texts, [[len(t)] for t in pre_texts], device, r=len(picks), log=lambda *_: None)
    out = {f"stream-{b}": float((full["stream"][b] - pre["stream"][b]).abs().max()) for b in full["stream"]}
    out["size"] = float((full["size"] - pre["size"]).abs().max())
    return out
