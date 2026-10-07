"""alephllm_diffusion.hubs.refs - the caption references the hub read adds to the stored rulers, on the same two caption draws
(coco_caps_2x2048.json in settings.DATA_DIR; the stored rulers T5-XXL, bert-base and captionbert are refs_2x2048.pt):
  clip_b32      CLIP ViT-B/32's projected text embedding (openai/clip-vit-base-patch32), the earlier conditioning probe's own
                reference, built by its recipe (padding, truncation, batches of 256, L2-normalized); a reference to compare against,
                never a target
  qwen3_pool    Anima's text encoder (Qwen3 0.6B, the file the picture runs load): its final hidden state, mean over the
                caption's tokens (Anima's tokenization: no special tokens), L2-normalized
  adapter_pool  Anima's LLM adapter output for the caption (the Qwen3 states as its source, the caption's T5 ids with the end token
                as its queries): the context the image model reads and where a slider's push is added; mean over the T5
                positions, L2-normalized
Every model in fp32, right-padded batches with masks. Writes hub_refs_2x2048.pt = {"draw1": {name: (2048, d)}, "draw2": {...},
"_meta": {...}} to settings.DATA_DIR. Tokenless: every file must already be on disk (the hub cache, settings.MODELS_DIR).
Usage: python -m alephllm_diffusion.hubs.refs"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from .. import settings

OUT = os.path.join(settings.DATA_DIR, "hub_refs_2x2048.pt")
CLIP_ID = "openai/clip-vit-base-patch32"
T0 = time.time()


def say(*a):
    print(f"[hub refs {time.time() - T0:6.1f}s]", *a, flush=True)


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1)


@torch.no_grad()
def clip_b32(caps, dev):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(CLIP_ID, local_files_only=True)
    clip = CLIPTextModelWithProjection.from_pretrained(CLIP_ID, local_files_only=True).to(dev).eval()
    out = []
    for i in range(0, len(caps), 256):
        b = tok(caps[i:i + 256], padding=True, truncation=True, return_tensors="pt").to(dev)
        out.append(F.normalize(clip(**b).text_embeds.float(), dim=-1).cpu())
    del clip
    return torch.cat(out)


@torch.no_grad()
def anima_text(caps, dev, bs=64):
    """(qwen3_pool, adapter_pool) for the captions, through the grid's own model loader and runners (stitch.s2)."""
    from transformers import AutoTokenizer, T5TokenizerFast
    from ..stitch import s2
    qtok = AutoTokenizer.from_pretrained(os.path.join(settings.DP, "configs", "qwen3_06b"), local_files_only=True)
    t5 = T5TokenizerFast(vocab_file=os.path.join(settings.DP, "configs", "t5_old", "spiece.model"),
                         tokenizer_file=os.path.join(settings.DP, "configs", "t5_old", "tokenizer.json"))
    qwen, ad = s2.load_models(dev)
    qp, ap = [], []
    for i in range(0, len(caps), bs):
        part = caps[i:i + bs]
        qids = [qtok(c, add_special_tokens=False)["input_ids"] for c in part]
        tids = [t5(c)["input_ids"] for c in part]
        assert all(t[-1] == t5.eos_token_id for t in tids)
        h, _ = s2.qwen_run(qwen, qids, dev)                         # [B, L, D] float32, right-padded
        _, qm = s2.pad_batch(qids)
        ti, tm = s2.pad_batch(tids)
        o = s2.adapter_module(ad, h, qm, ti, tm, dev)                 # [B, T, D] float32
        qp.append(F.normalize(masked_mean(h, qm.to(dev)), dim=-1).cpu())
        ap.append(F.normalize(masked_mean(o, tm.to(dev)), dim=-1).cpu())
    del qwen, ad
    return torch.cat(qp), torch.cat(ap)


def main():
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda"
    caps = json.load(open(os.path.join(settings.DATA_DIR, "coco_caps_2x2048.json"), encoding="utf-8"))
    draws = [k for k in ("draw1", "draw2") if k in caps]
    R = {d: {} for d in draws}
    for d in draws:
        R[d]["clip_b32"] = clip_b32(caps[d], dev)
        say(f"{d}: CLIP-B/32 {tuple(R[d]['clip_b32'].shape)}")
    if dev == "cuda":
        torch.cuda.empty_cache()
    for d in draws:
        R[d]["qwen3_pool"], R[d]["adapter_pool"] = anima_text(caps[d], dev)
        say(f"{d}: Qwen3 pooled {tuple(R[d]['qwen3_pool'].shape)}, adapter output pooled {tuple(R[d]['adapter_pool'].shape)}")
    R["_meta"] = {"clip": CLIP_ID, "qwen3": os.path.basename(settings.LLM_PATH), "adapter": os.path.basename(settings.DIT_PATH),
                  "dtype": "float32", "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "n": {d: len(caps[d]) for d in draws}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(R, OUT)
    say(f"wrote {OUT}")
    return R


if __name__ == "__main__":
    main()
    sys.exit(0)
