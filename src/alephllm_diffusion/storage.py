"""The private data store: a Hugging Face repo holding the reference data runs start from and a run's own files, so a later
session on another machine can continue. Data only; the code lives in this package.

    fetch_reference()                 the fixed reference inputs, where the stage programs read them
    save(patterns)                    workspace files matching the patterns -> <repo>/workspace/
    restore(patterns)                 the reverse, for files missing on this machine
    Checkpointer(patterns, every_s)   while a long step runs, new or changed files are saved every every_s seconds

The token comes from HF_TOKEN, else from Colab's Secrets panel; it is never printed.
"""
from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import os
import threading
import time
from importlib import resources

from . import settings

REPO = os.environ.get("ALEPHLLM_DIFFUSION_DATA_REPO", "AbstractPhil/beatrix-diffusion-colab")   # private
REFERENCE = {                                                    # <repo>/reference/<name> -> the local folder that reads it
    "s0.pt": settings.OUT_DIR,                                   # stage 0's captions, token maps and phrase classes
    "s1_step212000_limit80.pt": settings.OUT_DIR,                # the restart test's three small stage-1 inputs
    "s1_random0_limit80.pt": settings.OUT_DIR,
    "s1_random1_limit80.pt": settings.OUT_DIR,
    "logit_scale.json": settings.REPORT_DIR,                     # the reading heads and their real-word logit bands
    "stitch_shakedown.json": settings.REPORT_DIR,                # the shakedown run the grid's memory estimate reads
    "port_gate_reference.json": settings.REPORT_DIR,             # the cell lines of the reference run (RTX 4090) the port check
    #                                                              compares with
    "coco_caps_2x2048.json": settings.DATA_DIR,                  # the mount read's caption draws
    "refs_2x2048.pt": settings.DATA_DIR,                         # their ruler embeddings (T5-XXL, bert-base, captionbert)
}


def token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
        except Exception:                                      # noqa: BLE001 (not on Colab, or no secret)
            tok = None
    if not tok:
        raise RuntimeError("HF_TOKEN is needed for the private data store (Colab: the Secrets panel)")
    return tok


def reference_manifest() -> dict:
    """Each reference file's size and SHA-256 (package data, written when the files were uploaded)."""
    with resources.files("alephllm_diffusion").joinpath("data/reference.json").open(encoding="utf-8") as fh:
        files = json.load(fh)["files"]
    assert set(files) == set(REFERENCE), "the reference manifest and the file list differ"
    return files


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_reference() -> list:
    return [name for name, folder in REFERENCE.items() if not os.path.exists(os.path.join(folder, name))]


def check_reference() -> list:
    """The reference files on disk whose size or SHA-256 differs from the manifest."""
    man = reference_manifest()
    bad = []
    for name, folder in REFERENCE.items():
        p = os.path.join(folder, name)
        if os.path.exists(p) and (os.path.getsize(p) != man[name]["bytes"] or _sha256(p) != man[name]["sha256"]):
            bad.append(name)
    return bad


def fetch_reference(tok=None, say=print) -> None:
    """The missing reference files from the data store, each checked against the manifest's SHA-256."""
    from huggingface_hub import hf_hub_download
    man, missing = reference_manifest(), missing_reference()
    for name in missing:
        folder = REFERENCE[name]
        os.makedirs(folder, exist_ok=True)
        p = hf_hub_download(REPO, f"reference/{name}", token=tok or token(), local_dir=os.path.join(settings.HOME, "_fetch"))
        if _sha256(p) != man[name]["sha256"]:
            raise RuntimeError(f"reference/{name} from {REPO} does not match its SHA-256 in the package")
        os.replace(p, os.path.join(folder, name))
    say(f"the reference data is in place ({len(REFERENCE)} files, {len(missing)} fetched and checked)")


def _commit(rels, tok, message):
    from huggingface_hub import CommitOperationAdd, HfApi
    ops = []
    for r in rels:
        src = os.path.join(settings.HOME, r)
        if r.endswith(".log"):                                 # a log may grow while it uploads: send a snapshot
            with open(src, "rb") as fh:
                src = fh.read()
        ops.append(CommitOperationAdd(path_in_repo=f"workspace/{r}", path_or_fileobj=src))
    HfApi(token=tok).create_commit(repo_id=REPO, operations=ops, commit_message=message)


def _matches(patterns):
    rels = set()
    for g in patterns:
        for p in glob.glob(os.path.join(settings.HOME, g), recursive=True):
            if os.path.isfile(p):
                rels.add(os.path.relpath(p, settings.HOME).replace(os.sep, "/"))
    return sorted(rels)


def save(patterns, message="a session's files", tok=None, say=print) -> int:
    rels = _matches(patterns)
    for i in range(0, len(rels), 200):
        _commit(rels[i:i + 200], tok or token(), message)
    say(f"saved {len(rels)} file(s) to the data store")
    return len(rels)


def restore(patterns, say=print) -> int:
    from huggingface_hub import HfApi, snapshot_download
    tok = token()
    names = [f for f in HfApi(token=tok).list_repo_files(REPO) if f.startswith("workspace/")]
    want = [f for f in names if any(fnmatch.fnmatch(f[len("workspace/"):], p) for p in patterns)
            and not os.path.exists(os.path.join(settings.HOME, f[len("workspace/"):]))]
    if want:
        tmp = os.path.join(settings.HOME, "_fetch")
        snapshot_download(REPO, allow_patterns=want, token=tok, local_dir=tmp, max_workers=16)
        for f in want:
            dst = os.path.join(settings.HOME, f[len("workspace/"):])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(os.path.join(tmp, f), dst)
    say(f"restored {len(want)} file(s) from the data store")
    return len(want)


class Checkpointer:
    """While a long step runs: every `every` seconds, files matching `patterns` that are new or changed since the last round go to
    the data store (a file written in the last minute waits for the next round; a log goes as a snapshot). A failed round never
    stops the run."""

    def __init__(self, patterns, every=1800, tok=None, say=print):
        self.patterns, self.every, self.tok, self.say = patterns, every, tok, say
        self.stop, self.sent = threading.Event(), {}
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=900)

    def _loop(self):
        while not self.stop.wait(self.every):
            try:
                now = time.time()
                new = []
                for r in _matches(self.patterns):
                    m = os.path.getmtime(os.path.join(settings.HOME, r))
                    if self.sent.get(r) != m and (r.endswith(".log") or now - m > 60):
                        new.append((r, m))
                for i in range(0, len(new), 200):
                    part = new[i:i + 200]
                    _commit([r for r, _ in part], self.tok, f"checkpoint: {len(part)} file(s)")
                    self.sent.update(dict(part))
                if new:
                    self.say(f"checkpoint: {len(new)} file(s) saved to the data store")
            except Exception as e:                             # noqa: BLE001
                self.say(f"checkpoint failed ({type(e).__name__}); the next round retries")
