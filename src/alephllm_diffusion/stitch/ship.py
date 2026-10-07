"""alephllm_diffusion.stitch.ship - ship one matched-span stitch run to the data repo (AbstractPhil/geolip-beatrix-anima-data,
beatrix/stitch/): the phrase bank (s0.pt), the run's full JSON and its per-caption reads, and the data card's
beatrix/stitch/ section, in one commit (a run ships as soon as it completes; the full JSON goes to the data repo). Reads
s0.pt, stitch_<run>_full.json and stitch_<run>_percaption.pt from the output folder (settings.OUT_DIR; STITCH_OUT_DIR
overrides it) and the data card (README.md) from the repo.

The repo is public, so the shipped JSON is a scrubbed copy: string values that carry an in-house note (the registration
line; the prior run's local path) are replaced by plain wording, and the upload stops if any in-house word or local path
is left in any shipped file (the .pt files are loaded with weights_only=True and every string in them is checked; they
are shipped as written).

Usage: python -m alephllm_diffusion.stitch.ship rehearsal|record|mounts|frames [--dry]
(mounts, frames: the runs with Beatrix's arm groups mounted, and with the caption arm's label in front of every caption;
her cells only, so no margin rule is required of them)
--dry: everything except the upload (prints the upload plan, the scrubbed keys and the card section)."""
import io
import json
import os
import re
import sys

import torch

from .. import settings

REPO = "AbstractPhil/geolip-beatrix-anima-data"
CACHE = settings.OUT_DIR
# in-house words, Windows paths, and the root folders of a rented cloud machine or Colab (/root/, /workspace/, /content/)
BANNED = re.compile(r"S-1\d|\bPhil\b|docket|canon/|Fable|claude|Opus|research lead|[A-Za-z]:\\"
                    r"|(?<![\w.])/(?:root|workspace|content)/")
PLAIN = {"frozen": "the analysis code and every setting were fixed before this run and not changed during it",
         "prior_source": "an earlier small run of the same instrument on the same checkpoint"}
RUNS = {"rehearsal": "mini-beatrix-3 at step 230,415 (the start of its final training stage): a full-scale rehearsal of the "
                     "grid, run before the final checkpoint existed.",
        "record": "mini-beatrix-3 at step 245,674 (the end of its training): the final read.",
        "mounts": "mini-beatrix-3 at step 245,674 with trained adapter arms mounted (eight stage arms and a caption arm trained "
                  "over them; two training seeds, each its own trunk in the grid): Beatrix's cells only (the first ridge fit, both "
                  "byte positions, every block and depth); the controls and floors are the record run's, which do not depend on "
                  "the arms.",
        "frames": "the bare trunk and the two armed trunks with every caption prefixed by 'caption: ' (the caption arm's own "
                  "training label), read only at the record run's chosen cells.",
        "smoke": "a smoke test (never shipped)."}
SECTION = """## beatrix/stitch/

A read of how far Beatrix's states can stand in for those of Anima's text encoder (Qwen3) inside a caption. Each caption
holds one mood phrase. Beatrix reads the caption byte by byte; her states at the phrase's tokens, mapped into Qwen3's
space by a ridge regression, replace Qwen3's own states at every token of the phrase at one depth of Qwen3, and the
change in the output of Anima's text adapter is measured against two references: the unpatched caption, and the same
caption with the phrase replaced by a neutral filler of the same token count. Two untrained trunks of Beatrix's shape
(seeds 0 and 1) run the same grid as controls.

The grid: 3 trunks (Beatrix and the two controls) x 8 blocks of the trunk (8, 12, 16, 18, 20, 22, 24, 28) x 2 byte
positions per token (the token's last byte, and the byte that closes it) x 2 ridge fits (COCO captions with the
experiment's training captions, and COCO captions alone) x 9 depths of Qwen3 (the residual entering layers 0, 4, 8, 12,
16, 20 and 24, and after the last layer, before and after its final norm). 86 phrases in 32 scenes; the main read is 16
held-out phrases whose words the T5 tokenizer cuts into meaningless pieces (8 cheerful, 8 gloomy).

- `s0.pt`: the phrase bank: the phrases with their classes and the held-out split, the captions, each phrase's neutral
  filler, the token positions and the ridge-fit captions.
- `<run>/stitch_<run>_full.json`: every cell's reads, the precision checks and the margins.
- `<run>/stitch_<run>_percaption.pt`: the per-caption effects behind the intervals (a cluster bootstrap over phrases).

Every `.pt` file loads with `torch.load(path, weights_only=True)`.

Runs:
"""


def scrub(obj, path="", log=None):
    """A copy of obj with in-house string values replaced (PLAIN by key, else a neutral note); log lists what changed."""
    if isinstance(obj, dict):
        return {k: scrub(v, f"{path}/{k}", log) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, f"{path}[{i}]", log) for i, v in enumerate(obj)]
    if isinstance(obj, str) and BANNED.search(obj):
        key = path.rsplit("/", 1)[-1]
        log.append(path)
        return PLAIN.get(key, "(an internal note, removed)")
    return obj


def strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from strings(v)
    elif isinstance(obj, str):
        yield obj


def card(readme: str, runs: list) -> str:
    """The data card with its beatrix/stitch/ section written (replaced when present, appended otherwise)."""
    sec = SECTION + "".join(f"- `{r}/`: {RUNS[r]}\n" for r in runs)
    m = re.search(r"^## beatrix/stitch/\n.*?(?=^## |\Z)", readme, flags=re.S | re.M)
    if m:
        return readme[:m.start()] + sec + ("\n" if readme[m.end():] else "") + readme[m.end():]
    return readme.rstrip("\n") + "\n\n" + sec


def main(run: str, dry: bool):
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    full = os.path.join(CACHE, f"stitch_{run}_full.json" if run != "smoke" else "stitch_smoke.json")
    pc = os.path.join(CACHE, f"stitch_{run}_percaption.pt")
    s0 = os.path.join(CACHE, "s0.pt")
    for p in (full, pc, s0):
        assert os.path.exists(p), f"missing {p}"
    if run == "smoke":
        assert dry, "a smoke run is never shipped"
    result = json.load(open(full, encoding="utf-8"))
    assert result.get("config") == run, (result.get("config"), run)
    assert run not in ("rehearsal", "record") or "margins" in result, \
        "the margin analysis is not in the JSON yet (rerun: python -m alephllm_diffusion.stitch.s2 margins <name>)"   # mounts, frames: her cells only
    log = []
    clean = scrub(result, "", log)
    text = json.dumps(clean, indent=1)
    assert not BANNED.search(text), BANNED.search(text).group(0)
    for p in (pc, s0):
        bad = sorted({s[:80] for s in strings(torch.load(p, weights_only=True)) if BANNED.search(s)})
        assert not bad, f"{os.path.basename(p)} carries in-house strings: {bad[:5]}"
    api = HfApi()
    have = api.list_repo_files(REPO, repo_type="dataset")
    runs = [r for r in ("rehearsal", "record", "mounts", "frames")
            if r == run or any(f.startswith(f"beatrix/stitch/{r}/") for f in have)]
    readme = open(hf_hub_download(REPO, "README.md", repo_type="dataset", force_download=True), encoding="utf-8").read()
    new_card = card(readme, runs)
    assert not BANNED.search(new_card.replace("AbstractPhil/", "")), "the card carries an in-house word"
    ops = {"beatrix/stitch/s0.pt": s0,f"beatrix/stitch/{run}/stitch_{run}_full.json": text.encode("utf-8"),
           f"beatrix/stitch/{run}/stitch_{run}_percaption.pt": pc, "README.md": new_card.encode("utf-8")}
    print(f"[ship] {run}: scrubbed {len(log)} string(s): {log}")
    for k, v in ops.items():
        size = len(v) if isinstance(v, bytes) else os.path.getsize(v)
        print(f"[ship]   {k}  {size / 1e6:.2f} MB")
    print("[ship] the card's section:\n" + new_card[new_card.index("## beatrix/stitch/"):])
    if dry:
        print("[ship] dry run: nothing uploaded")
        return
    info = api.create_commit(REPO, repo_type="dataset", commit_message=f"stitch {run}: the full reads, per-caption effects, "
                             "the phrase bank and the card section",
                             operations=[CommitOperationAdd(k, io.BytesIO(v) if isinstance(v, bytes) else v)
                                         for k, v in ops.items()])
    print(f"[ship] pushed {run} -> {REPO}: {info.commit_url}")


if __name__ == "__main__":
    main(sys.argv[1], "--dry" in sys.argv[2:])
