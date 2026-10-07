"""Where a run reads and writes.

Every location has a default under one workspace folder and can be overridden by an environment variable; the stage programs
run as separate processes and inherit them. The workspace is ALEPHLLM_DIFFUSION_HOME, else `runs/` in the checkout (an editable
install), else ~/.cache/alephllm_diffusion.

| setting | variable | default |
|---|---|---|
| OUT_DIR | STITCH_OUT_DIR | <home>/out (stage files, grids, exports) |
| REPORT_DIR | STITCH_REPORT_DIR | <home>/reports (summaries, picks, the reference files the grid reads) |
| DATA_DIR | BTX3_DATA_DIR | <home>/data (the caption draws and ruler embeddings) |
| CKPT_DIR | BTX3_CKPT_DIR | <home>/ckpts (Beatrix checkpoints when not in the hub cache) |
| MODELS_DIR | ALEPHLLM_DIFFUSION_MODELS | <home>/models (Anima's files, in the hub's split_files layout) |
| LLM_PATH | STITCH_LLM_PATH | <models>/split_files/text_encoders/qwen_3_06b_base.safetensors |
| DIT_PATH | STITCH_DIT_PATH | <models>/split_files/diffusion_models/anima-base-v1.0.safetensors |
| DP | STITCH_DP | the diffusion-pipe fork (external/diffusion-pipe) |
| MOUNT_DATA_DIR | MOUNT_DATA_DIR | <out>/mount_data (the caption pack and the arm results the mount gates read) |
| MOUNT_READ_OUT | MOUNT_READ_OUT | <home>/mount_read |
| ANIMA_DATA | ALEPHLLM_DIFFUSION_ANIMA_DATA | <home>/anima (the picture runs' pictures, results and Hugging Face cache) |
"""
from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

from .paths import diffusion_pipe_dir, repo_root


def _home() -> Path:
    env = os.environ.get("ALEPHLLM_DIFFUSION_HOME")
    if env:
        return Path(env)
    root = repo_root()
    return root / "runs" if root else Path.home() / ".cache" / "alephllm_diffusion"


def _fork() -> str:
    try:
        return str(diffusion_pipe_dir())
    except FileNotFoundError:
        return ""


HOME = _home()
OUT_DIR = os.environ.get("STITCH_OUT_DIR") or str(HOME / "out")
REPORT_DIR = os.environ.get("STITCH_REPORT_DIR") or str(HOME / "reports")
DATA_DIR = os.environ.get("BTX3_DATA_DIR") or str(HOME / "data")
CKPT_DIR = os.environ.get("BTX3_CKPT_DIR") or str(HOME / "ckpts")
MODELS_DIR = os.environ.get("ALEPHLLM_DIFFUSION_MODELS") or str(HOME / "models")
LLM_PATH = os.environ.get("STITCH_LLM_PATH") or os.path.join(MODELS_DIR, "split_files", "text_encoders",
                                                              "qwen_3_06b_base.safetensors")
DIT_PATH = os.environ.get("STITCH_DIT_PATH") or os.path.join(MODELS_DIR, "split_files", "diffusion_models",
                                                              "anima-base-v1.0.safetensors")
DP = os.environ.get("STITCH_DP") or _fork()
MOUNT_DATA_DIR = os.environ.get("MOUNT_DATA_DIR") or os.path.join(OUT_DIR, "mount_data")
MOUNT_READ_OUT = os.environ.get("MOUNT_READ_OUT") or str(HOME / "mount_read")
ANIMA_DATA = os.environ.get("ALEPHLLM_DIFFUSION_ANIMA_DATA") or str(HOME / "anima")


def registration() -> dict:
    """The grid's registration record (package data): when its definitions were frozen and what changed in the stage programs
    since, each change listed with the reason no registered number moves."""
    with resources.files("alephllm_diffusion").joinpath("data/registration.json").open(encoding="utf-8") as fh:
        return json.load(fh)
