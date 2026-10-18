"""alephllm_diffusion.nano.relay_train - the trained relay (the record's step-6 form: a small trained map where the ridge stops, the
reader taught to stand in for Qwen3): a reader's states carried into Qwen3's lane by a connector born at the closed-form ridge of
nano.relay, with a nonlinear part born weight-zero, trained through Anima's frozen text adapter on one squared error: the adapter's
output (the context the DiT reads) from the connector's states against its output from Qwen3's own states, over each caption's T5
slots. The first token keeps Qwen3's own state, as in the relay.

    q_hat(t) = W x(t) + b + V2 gelu(V1 n(x(t)) + c1) + c2      t = 1 .. L-1; n = LayerNorm without affine; V2 = 0, c2 = 0 at birth
    loss     = sum_slots |A(q_hat) - A(q)|^2 / sum_slots |A(q) - m|^2     A = the adapter (frozen), m = the training targets' mean

Pure Adam (no decay), fp32; the step with the best validation score is kept. Nothing runs on import."""
from __future__ import annotations

import copy

import torch
from torch import nn


class Connector(nn.Module):
    """Born at the ridge map exactly: the linear part is the ridge (W^T, my - mx W); the nonlinear part's output layer is born
    weight-zero and bias-zero (a zero write, not a closed gate)."""

    def __init__(self, ridge: dict, hidden: int = 1024, seed: int = 0):
        super().__init__()
        W, mx, my = ridge["W"].double(), ridge["mx"].double(), ridge["my"].double()
        p, d = W.shape
        g = torch.Generator().manual_seed(seed)
        self.lin, self.up, self.down = nn.Linear(p, d), nn.Linear(p, hidden), nn.Linear(hidden, d)
        with torch.no_grad():
            self.lin.weight.copy_(W.T.float())
            self.lin.bias.copy_((my - mx @ W).float())
            bound = p ** -0.5
            self.up.weight.copy_(torch.empty(hidden, p).uniform_(-bound, bound, generator=g))
            self.up.bias.zero_()
            self.down.weight.zero_()
            self.down.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = nn.functional.layer_norm(x, x.shape[-1:])
        return self.lin(x) + self.down(nn.functional.gelu(self.up(n)))


def offsets(rows_per_caption: list) -> list:
    """row offsets per caption into a flat row table (len = captions + 1)."""
    out = [0]
    for n in rows_per_caption:
        out.append(out[-1] + int(n))
    return out


def source(conn: nn.Module, data: dict, idx: list) -> tuple:
    """([B, L, D] source states, [B, L] mask) for captions idx: Qwen3's own first-token state, then the connector's output on the
    caption's reader rows (L = the longest caption in the batch; zeros past a caption's end)."""
    offs, X, first = data["offs"], data["X"], data["first"]
    ns = [offs[i + 1] - offs[i] for i in idx]
    Y = conn(torch.cat([X[offs[i]:offs[i + 1]] for i in idx]))
    out = Y.new_zeros(len(idx), max(ns) + 1, Y.shape[-1])
    mask = torch.zeros(len(idx), max(ns) + 1, dtype=torch.long, device=Y.device)
    k = 0
    for b, (i, n) in enumerate(zip(idx, ns)):
        out[b, 0] = first[i]
        out[b, 1:n + 1] = Y[k:k + n]
        mask[b, :n + 1] = 1
        k += n
    return out, mask


def targets(data: dict, idx: list) -> tuple:
    """([B, T] T5 ids, [B, T] T5 mask, [B, T, D] the target context, zeros at padding) for captions idx."""
    t5, tgt = [data["t5"][i] for i in idx], [data["tgt"][i] for i in idx]
    T = max(len(t) for t in t5)
    dev = data["X"].device
    ids = torch.zeros(len(idx), T, dtype=torch.long, device=dev)
    msk = torch.zeros(len(idx), T, dtype=torch.long, device=dev)
    out = torch.zeros(len(idx), T, tgt[0].shape[-1], device=dev)
    for b, (t, y) in enumerate(zip(t5, tgt)):
        ids[b, :len(t)], msk[b, :len(t)], out[b, :len(t)] = t, 1, y
    return ids, msk, out


def context(adapter, src: torch.Tensor, mask: torch.Tensor, ids: torch.Tensor, msk: torch.Tensor) -> torch.Tensor:
    """the adapter's output (what the DiT reads), zeroed at T5 padding as the layer chain does."""
    out = adapter(source_hidden_states=src, target_input_ids=ids, target_attention_mask=msk, source_attention_mask=mask)
    return out * msk[..., None].to(out.dtype)


def batch_sums(conn, adapter, data: dict, idx: list) -> tuple:
    """(sum of squared error, sum of squares about the training targets' mean) over the valid T5 slots of captions idx."""
    src, mask = source(conn, data, idx)
    ids, msk, tgt = targets(data, idx)
    pred = context(adapter, src, mask, ids, msk)
    valid = msk.bool()
    sse = ((pred - tgt)[valid] ** 2).sum()
    sst = ((tgt - data["m"])[valid] ** 2).sum()
    return sse, sst


@torch.no_grad()
def score(conn, adapter, data: dict, idx: list, batch: int = 64) -> float:
    """the context R2 over captions idx: 1 - SSE / SST, every valid slot of every caption pooled."""
    sse = sst = 0.0
    for s in range(0, len(idx), batch):
        a, b = batch_sums(conn, adapter, data, idx[s:s + batch])
        sse, sst = sse + float(a), sst + float(b)
    return 1.0 - sse / sst


@torch.no_grad()
def state_r2(conn, data: dict, idx: list) -> float:
    """descriptive: the R2 of the connector's states against Qwen3's own final states on the reader rows of captions idx (about
    the training rows' mean)."""
    offs = data["offs"]
    rows = torch.cat([torch.arange(offs[i], offs[i + 1]) for i in idx]).to(data["X"].device)
    Q = data["Q"][rows]
    return 1.0 - float(((conn(data["X"][rows]) - Q) ** 2).sum()) / float(((Q - data["Q_mean"]) ** 2).sum())


def train(conn, adapter, data: dict, train_idx: list, val_idx: list, *, steps: int, lr: float, batch: int, seed: int,
          eval_every: int, log=print) -> dict:
    """pure Adam on the connector (the adapter frozen); the validation context R2 every eval_every steps; the best step kept
    (the connector is left at it). Returns the run's record."""
    opt = torch.optim.Adam(conn.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    born = score(conn, adapter, data, val_idx)
    best = {"step": 0, "val_r2": born, "state": copy.deepcopy(conn.state_dict())}
    history, order, run_loss = [{"step": 0, "val_r2": born}], [], []
    for step in range(1, steps + 1):
        if len(order) < batch:
            order = [train_idx[j] for j in torch.randperm(len(train_idx), generator=g).tolist()]
        idx, order = order[:batch], order[batch:]
        sse, sst = batch_sums(conn, adapter, data, idx)
        loss = sse / sst
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss.append(float(loss.detach()))
        if step % eval_every == 0 or step == steps:
            v = score(conn, adapter, data, val_idx)
            history.append({"step": step, "train_loss": sum(run_loss) / len(run_loss), "val_r2": v})
            run_loss = []
            if v > best["val_r2"]:
                best = {"step": step, "val_r2": v, "state": copy.deepcopy(conn.state_dict())}
            log(f"step {step}: train loss {history[-1]['train_loss']:.4f}, validation context R2 {v:.4f} "
                f"(born {born:.4f}, best {best['val_r2']:.4f} at {best['step']})")
    conn.load_state_dict(best["state"])
    return {"born_val_r2": born, "best_val_r2": best["val_r2"], "best_step": best["step"], "steps": steps, "lr": lr,
            "batch": batch, "seed": seed, "history": history}
