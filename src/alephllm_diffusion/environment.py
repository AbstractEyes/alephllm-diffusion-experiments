"""The environment check: the Python and torch builds, the card, every pin of requirements.txt against what is installed,
the AlephLLM libraries' commits and the diffusion-pipe fork's commit.

    python -m alephllm_diffusion.environment      # prints the report; exit code 1 when anything differs from the pins
"""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

from . import __version__
from .paths import comfyui_dir, diffusion_pipe_dir, git_commit, repo_root

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*==\s*([^\s;]+)")
_GIT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*git\+\S+@([0-9a-f]{7,40})")


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _applies(marker: str) -> bool:
    if not marker:
        return True
    try:
        from packaging.markers import Marker
        return Marker(marker).evaluate()
    except Exception:                                          # noqa: BLE001 (an unreadable marker is checked anyway)
        return True


def read_pins(path: Path) -> dict:
    """{'versions': {name: version}, 'commits': {name: commit}} from a requirements file (markers applied)."""
    versions, commits = {}, {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        spec, _, marker = line.partition(";")
        if not _applies(marker.strip()):
            continue
        spec = spec.strip()
        if m := _GIT.match(spec):
            commits[_canon(m.group(1))] = m.group(2)
        elif m := _PIN.match(spec):
            versions[_canon(m.group(1))] = m.group(3)
    return {"versions": versions, "commits": commits}


def installed_commit(dist: str) -> str | None:
    """The VCS commit pip recorded for a distribution installed from git (direct_url.json), or None."""
    try:
        info = json.loads(md.distribution(dist).read_text("direct_url.json") or "{}")
    except md.PackageNotFoundError:
        return None
    return (info.get("vcs_info") or {}).get("commit_id")


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _submodule_pin(root: Path, rel: str) -> str | None:
    """The submodule commit recorded in HEAD, else the one staged in the index (a checkout before its first commit)."""
    for cmd in (["ls-tree", "HEAD", rel], ["ls-files", "-s", rel]):
        try:
            out = subprocess.run(["git", "-C", str(root), *cmd], capture_output=True, text=True, check=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            continue
        if cmd[0] == "ls-tree" and len(out) >= 3 and out[1] == "commit":
            return out[2]
        if cmd[0] == "ls-files" and len(out) >= 2 and out[0] == "160000":
            return out[1]
    return None


def check(verbose: bool = True) -> dict:
    rep = {"package": __version__, "python": platform.python_version(), "platform": sys.platform, "problems": []}
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["torch_cuda"] = torch.version.cuda
        rep["cuda_works"] = bool(torch.cuda.is_available())
        if rep["cuda_works"]:
            cap = torch.cuda.get_device_capability(0)
            rep["card"] = torch.cuda.get_device_name(0)
            rep["sm"] = f"sm_{cap[0]}{cap[1]}"
            x = torch.ones(1024, device="cuda")
            rep["cuda_math_ok"] = float((x * 2).sum()) == 2048.0          # a kernel really runs on the card
        else:
            rep["problems"].append("torch cannot see a CUDA card")
    except ImportError:
        rep["problems"].append("torch is not installed")

    root = repo_root()
    req = root / "requirements.txt" if root else None
    if req and req.exists():
        pins = read_pins(req)
        diff = []
        for name, want in pins["versions"].items():
            have = _version(name)
            if have != want:
                diff.append(f"{name} pinned {want}, installed {have or 'nothing'}")
        for name, want in pins["commits"].items():
            have = installed_commit(name)
            if not (have or "").startswith(want[:7]):
                diff.append(f"{name} pinned @{want[:7]}, installed @{(have or 'nothing')[:7]}")
        rep["pins_checked"] = len(pins["versions"]) + len(pins["commits"])
        rep["pin_differences"] = diff
        rep["problems"] += diff
    else:
        rep["pins_checked"] = 0
    rep["libraries"] = {d: {"version": _version(d), "commit": installed_commit(d)}
                        for d in ("geolip-alephllm", "amoe-lora", "geolip-anima-trainer")}

    try:
        dp = diffusion_pipe_dir()
        rep["diffusion_pipe"] = {"path": str(dp), "present": (dp / "train.py").exists(), "commit": git_commit(dp),
                                 "pinned": _submodule_pin(root, "external/diffusion-pipe") if root else None,
                                 "comfyui_commit": git_commit(comfyui_dir()) if (comfyui_dir() / "comfy").is_dir() else None}
        d = rep["diffusion_pipe"]
        if not d["present"]:
            rep["problems"].append("the diffusion-pipe fork is missing: git submodule update --init external/diffusion-pipe")
        elif d["pinned"] and d["commit"] != d["pinned"]:
            rep["problems"].append(f"the diffusion-pipe fork is at {str(d['commit'])[:7]}, pinned {d['pinned'][:7]}")
        if d["present"] and not d["comfyui_commit"]:
            rep["problems"].append("ComfyUI is missing: git -C external/diffusion-pipe submodule update --init submodules/ComfyUI")
    except FileNotFoundError as e:
        rep["problems"].append(str(e))

    if verbose:
        _print(rep)
    return rep


def _print(rep: dict) -> None:
    print(f"alephllm-diffusion-experiments {rep['package']}: environment check")
    card = (f"{rep['card']} ({rep['sm']}), CUDA works" if rep.get("cuda_works") else "no CUDA card visible")
    print(f"  Python {rep['python']} | torch {rep.get('torch', 'missing')} (CUDA {rep.get('torch_cuda')}) | {card}")
    libs = " | ".join(f"{k} {v['version']} @{(v['commit'] or '?')[:7]}" for k, v in rep["libraries"].items())
    print(f"  libraries: {libs}")
    d = rep.get("diffusion_pipe") or {}
    if d:
        print(f"  diffusion-pipe fork: {'present' if d['present'] else 'MISSING'} @{str(d.get('commit'))[:7]} "
              f"(pinned @{str(d.get('pinned'))[:7]}); ComfyUI @{str(d.get('comfyui_commit'))[:7]}")
    if rep["pins_checked"]:
        n = len(rep["pin_differences"])
        print(f"  pins: {rep['pins_checked']} checked, " + ("all match" if not n else f"{n} differ"))
    for p in rep["problems"]:
        print(f"  PROBLEM: {p}")
    print("  ready" if not rep["problems"] else f"  {len(rep['problems'])} problem(s)")


if __name__ == "__main__":
    sys.exit(1 if check()["problems"] else 0)
