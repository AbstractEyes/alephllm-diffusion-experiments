"""The public model files the experiment needs, fetched without a token.

- Anima (circlestone-labs/Anima): the Qwen3 0.6B text encoder and the DiT the grid reads, and the VAE the picture runs decode
  with, into settings.MODELS_DIR in the hub's split_files layout.
- The mood judge (openai/clip-vit-large-patch14, a CLIP-L used only to score pictures): its configs, tokenizer files and one
  weight file, into the picture runs' Hugging Face cache (<settings.ANIMA_DATA>/hf_cache), where they look for it.
Beatrix's checkpoints are fetched on first use by the extractor (alephllm_diffusion.beatrix.extract).
"""
from __future__ import annotations

import os
import time

from . import settings

ANIMA = "circlestone-labs/Anima"
ANIMA_FILES = [("split_files/text_encoders/qwen_3_06b_base.safetensors", 1.2),
               ("split_files/diffusion_models/anima-base-v1.0.safetensors", 4.18),
               ("split_files/vae/qwen_image_vae.safetensors", 0.25)]
JUDGE = "openai/clip-vit-large-patch14"
JUDGE_GB = 1.7


def judge_cache() -> str:
    return os.path.join(settings.ANIMA_DATA, "hf_cache", "hub")


def fetch(judge: bool = True, say=print) -> None:
    """Every file not yet on disk; prints each item with the time spent and an estimate of the time left."""
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    t0, done = time.time(), 0.0
    total = sum(gb for _, gb in ANIMA_FILES) + (JUDGE_GB if judge else 0.0)
    for fn, gb in ANIMA_FILES:
        dst = os.path.join(settings.MODELS_DIR, *fn.split("/"))
        fresh = not os.path.exists(dst)
        if fresh:
            hf_hub_download(ANIMA, fn, local_dir=settings.MODELS_DIR, token=False)
        done += gb
        el = time.time() - t0
        say(f"Anima {fn.split('/')[-1]} ({gb} GB): {'downloaded' if fresh else 'on disk'}; {el:.0f} s so far"
            + (f", about {el / done * (total - done):.0f} s left" if fresh and done < total else ""))
    if judge:
        names = list_repo_files(JUDGE, token=False)
        weight = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
        snapshot_download(JUDGE, allow_patterns=[n for n in names if n.endswith((".json", ".txt"))] + [weight],
                          cache_dir=judge_cache(), token=False)
        say(f"the mood judge {JUDGE} ({JUDGE_GB} GB) ready; {time.time() - t0:.0f} s in all")
