"""alephllm_diffusion.beatrix.extract - the feature extractor for the diffusion-capacity tests on mini-beatrix-3 (Beatrix).

One pass of the trunk over a list of caption strings; taps at any subset of the 32 blocks:
  pool   mean over the caption's byte positions of the block output (the DOC position is excluded from the mean, kept as context)
  last   the block output at the caption's final byte
  bb     the BLACKBOARD of the block's hub: per constellation the prefill state (Sp - Sn) (K=64 slots x d=1024), each slot
         L2-normalized, all constellations concatenated and flattened (4*64*1024 = 262,144), NOT renormalized (row norm 16): the
         representation the geometry probe of the earlier conditioning study (PROBE-4a) used on the earlier 2s model (that study's
         binding-probe tap, PROBE-4b, additionally normalizes; the scale matters only to the ridge columns)
  byte   the per-byte states (only when asked: token-state tests)
Layer index: -1 = the embedding output, 0..31 = the block outputs, 32 = after the final LayerNorm (pool / last only).

Captions of equal byte length are batched together (no padding at all, so every row is bit-for-bit what a single-caption run
gives up to kernel batching), because the blackboard sums over every position and a pad would enter it.

Bare core only: no stage arm is attached anywhere in this module. The trunk is loaded as the earlier recall probe loaded it:
tokenless, the DOC byte (the trunk's document-boundary token) in front of every sequence, bf16 autocast over the fp32 weights on
the card. Checkpoints come from AbstractPhil/alephllm-mini-beatrix-training (the hub cache first, else settings.CKPT_DIR).
Writes nothing: extract() returns the taps on the CPU. Use: model = build_model(step); out = extract(model, captions, layers,
taps=("pool", "last")). Importing the module sets the job's card-memory cap (below); nothing else runs on import.
"""
import os
import time

import torch
import torch.nn.functional as F

from .. import settings

REPO = "AbstractPhil/alephllm-mini-beatrix-training"
CRAFT = "mini-beatrix-3"
CKPT_DIR = settings.CKPT_DIR

# THE JOB'S OWN MEMORY CAP: a job caps its own card memory so the other programs on the machine keep theirs. The cap arrives as a
# fraction of the card in the variable read below; a queue runner computes it when a job starts (the runner these experiments
# used: min(19,000, 23,000 - what everything else uses - 1,000) MiB). torch's allocator enforces it, so a job past it fails with
# its own out-of-memory error instead of taking memory from the other programs. Set at import: every job script imports this
# module before touching the card. Without the variable (a run outside such a runner) 0.73 applies.
MEM_FRACTION = float(os.environ.get("BTX3_MEM_FRACTION", "0.73"))
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)


def ckpt_path(step, allow_download=True):
    """The local file for a stored trunk checkpoint: the HF cache first (no bytes moved), then settings.CKPT_DIR (a tokenless
    download)."""
    from huggingface_hub import hf_hub_download
    fn = f"{CRAFT}/checkpoints/step_{step:08d}.safetensors"
    try:
        return hf_hub_download(REPO, fn, token=False, local_files_only=True)
    except Exception:
        pass
    local = os.path.join(CKPT_DIR, fn.replace("/", os.sep))
    if os.path.exists(local) and abs(os.path.getsize(local) - 754555546) < 4096:
        return local
    if not allow_download:
        raise FileNotFoundError(fn)
    os.makedirs(CKPT_DIR, exist_ok=True)
    return hf_hub_download(REPO, fn, local_dir=CKPT_DIR, token=False)


def build_model(step=None, device="cuda", random_init_seed=None, mem_fraction=None):
    """Beatrix's trunk (mini-beatrix-3) at a stored step (step=None + random_init_seed -> a random-init copy of the same
    architecture: a control that must fail)."""
    from safetensors.torch import load_file
    from geolip.alephllm.presets import make_v3_preset
    from geolip.alephllm.model.alephlm import AlephLM
    cfg = make_v3_preset(32, data_scale=4.0, epoch_cap=2.0, rebalance_to="generators", name="mini-beatrix-3").model
    if random_init_seed is not None:
        torch.manual_seed(random_init_seed)
    model = AlephLM(cfg)
    if step is not None:
        res = model.load_state_dict(load_file(ckpt_path(step)), strict=False)
        assert not res.missing_keys and not res.unexpected_keys, (res.missing_keys[:4], res.unexpected_keys[:4])
    if device == "cuda":
        # the job's own cap (MEM_FRACTION above) unless the caller passes one; never a fixed 0.73
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION if mem_fraction is None else mem_fraction)
    return model.to(device).eval()


def _doc_id():
    from geolip.alephllm.data.special_tokens import DOC
    return int(DOC)


def encode_bytes(text, doc=True):
    b = text.encode("utf-8", "replace")
    ids = ([_doc_id()] if doc else []) + list(b)
    return torch.tensor(ids, dtype=torch.long)


def _blackboard(cache):
    """(B, 4*64*1024) fp32 rows of per-slot unit vectors (row norm 16) from one block's prefill cache (fusion=None:
    cache['consts'])."""
    books = cache["consts"] if "consts" in cache else [cache]
    sl = torch.cat([F.normalize((b["Sp"] - b["Sn"]).float(), dim=-1) for b in books], dim=1)   # (B, C*K, d)
    # the geometry probe's convention (PROBE-4a): per-slot unit rows, flattened, NOT renormalized (row norm = sqrt(C*K) = 16).
    # Renormalizing the row (F.normalize, the PROBE-4b tap) gives 1/16 of this scale: every scale-invariant gauge is identical,
    # but the ridge lambda grid sees a 256x smaller Gram (measured: forward R2 moves by up to .015).
    return sl.flatten(1)


@torch.no_grad()
def extract(model, texts, layers, taps=("pool", "last", "bb"), device="cuda", amp=True, max_batch=64, doc=True,
            bb_dtype=torch.float16, progress=0):
    """-> {tap: {layer: tensor(N, dim)}} on the CPU (pool / last fp32, bb fp16), rows in the order of `texts`;
    tap 'byte' -> {layer: [tensor(L_i, d)] per caption} (fp32: fp16 overflows at block 31)."""
    layers = sorted(set(layers))
    need_bb = "bb" in taps
    n = len(texts)
    ids = [encode_bytes(t, doc) for t in texts]
    order = sorted(range(n), key=lambda i: ids[i].numel())
    out = {t: {l: None for l in layers} for t in taps}
    if "byte" in taps:
        out["byte"] = {l: [None] * n for l in layers}
    D = model.nf.weight.numel()
    for t in ("pool", "last"):
        if t in taps:
            for l in layers:
                out[t][l] = torch.zeros(n, D, dtype=torch.float32)
    pos = 0
    t0 = time.time()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (amp and device == "cuda") else torch.autocast("cpu", enabled=False)
    while pos < n:
        L = ids[order[pos]].numel()
        grp = [order[pos]]
        while pos + len(grp) < n and len(grp) < max_batch and ids[order[pos + len(grp)]].numel() == L:
            grp.append(order[pos + len(grp)])
        pos += len(grp)
        x_ids = torch.stack([ids[i] for i in grp]).to(device)
        with ctx:
            x = model.embed(x_ids)
            states = {}
            if -1 in layers:
                states[-1] = (x, None)
            for bi, blk in enumerate(model.blocks):
                x, cache = blk.prefill(x)
                if bi in layers:
                    states[bi] = (x, cache)
            if 32 in layers:
                states[32] = (model.nf(x), None)
        gi = torch.tensor(grp)
        s0 = 1 if doc else 0                                   # skip the DOC position in pooled statistics
        for l, (h, cache) in states.items():
            hf = h.float()
            assert torch.isfinite(hf).all(), ("non-finite hidden state", l, grp[:4])
            if "pool" in taps:
                out["pool"][l][gi] = hf[:, s0:].mean(1).cpu()
            if "last" in taps:
                out["last"][l][gi] = hf[:, -1].cpu()
            if need_bb and 0 <= l <= len(model.blocks) - 1 and cache is not None:
                bbv = _blackboard(cache).to(bb_dtype).cpu()
                if out["bb"][l] is None:
                    out["bb"][l] = torch.zeros(n, bbv.shape[1], dtype=bb_dtype)
                out["bb"][l][gi] = bbv
            if "byte" in taps:
                # fp32, never fp16: block 31's residual stream reaches |x| ~ 2.9e5 (> fp16 max 65,504) and would overflow to inf
                for r, i in enumerate(grp):
                    out["byte"][l][i] = hf[r, s0:].cpu()
        if progress and (pos // len(grp)) % progress == 0:
            print(f"  extract {pos}/{n} ({time.time() - t0:.0f}s)", flush=True)
    return out


@torch.no_grad()
def last_logits_parity(model, texts, device="cuda", doc=True):
    """Parity of the tap loop against the model's own forward: max |logit difference| at the last byte (should be 0 to bf16 noise)."""
    worst = 0.0
    for t in texts:
        ids = encode_bytes(t, doc)[None].to(device)
        ref = model(ids).logits[:, -1].float()
        x = model.embed(ids)
        for blk in model.blocks:
            x, _ = blk.prefill(x)
        mine = model.head(model.nf(x)[:, -1:]).float()[:, -1]
        worst = max(worst, float((ref - mine).abs().max()))
    return worst
