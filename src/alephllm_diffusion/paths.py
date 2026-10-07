"""Where the pieces live: the repository checkout, the diffusion-pipe fork and its ComfyUI submodule.

The fork is a git submodule of this repository (external/diffusion-pipe). ALEPHLLM_DIFFUSION_PIPE points elsewhere when the
package is installed without a checkout (for example from a wheel)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def repo_root() -> Path | None:
    """The checkout this package was installed from (editable installs), or None."""
    root = PACKAGE_DIR.parents[1]
    return root if (root / "pyproject.toml").exists() and (root / "external").is_dir() else None


def diffusion_pipe_dir() -> Path:
    """The diffusion-pipe fork: ALEPHLLM_DIFFUSION_PIPE, else external/diffusion-pipe in the checkout."""
    env = os.environ.get("ALEPHLLM_DIFFUSION_PIPE")
    if env:
        return Path(env)
    root = repo_root()
    if root is None:
        raise FileNotFoundError("no checkout found: set ALEPHLLM_DIFFUSION_PIPE to a diffusion-pipe fork checkout")
    return root / "external" / "diffusion-pipe"


def comfyui_dir() -> Path:
    return diffusion_pipe_dir() / "submodules" / "ComfyUI"


def git_commit(path: Path) -> str | None:
    """The checked-out commit of a git working tree, or None."""
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
