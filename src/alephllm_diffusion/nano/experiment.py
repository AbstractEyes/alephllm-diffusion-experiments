"""alephllm_diffusion.nano.experiment - the nano battery as experiments of the two experiments repos. Each folder (README, meta.json,
the ledgers, a contact sheet where pictures were made) is written from the run's own ledgers into a local folder laid out like the
repo, then published by the Anima trainer's step for experiments written elsewhere (the folder in one commit, then the repo
README's index rebuilt from every folder's meta.json). The ids come from the trainer's registry, which keeps them from every other
experiment: on the Anima repo NANO_READ_ID (the read, no pictures), NANO_RELAY_ID (the relay in pictures), NANO_COMPARATOR_ID (the
uniform comparator), NANO_RELAY_TRAINED_ID (the trained relay); on the Sana repo its NANO_COMPARATOR_ID.

Usage: python -m alephllm_diffusion.nano.experiment write <read|relay|comparator|sana|relay_trained> --root=<the run's folder>
           --mirror=<folder>
       python -m alephllm_diffusion.nano.experiment publish <read|relay|comparator|sana|relay_trained> --mirror=<folder>
(publish needs HF_TOKEN with write access to the repo; the rest runs offline.)"""
from __future__ import annotations

import json
import os
import shutil
import sys

DATE = "2026-10-17"
CODE_URL = "https://github.com/AbstractEyes/alephllm-diffusion-experiments"
NANO_REPO = "AbstractPhil/alephllm-nano-castor-pollux"
BODY_NAMES = {"castor": "nano-castor-v1.1", "pollux-canon": "nano-pollux-canon-v1.1", "pollux-dual": "nano-pollux-dual-v1.1",
              "pollux-dual-canon": "nano-pollux-dual-canon-v1.1"}
BODY_TEXT = {
    "castor": "rotary softmax attention in all six blocks",
    "pollux-canon": "a hub with rotary addresses in blocks 0, 2 and 4, rotary softmax attention in blocks 1, 3 and 5, Canon layers",
    "pollux-dual": "dual hubs (a rotary hub beside a plain one) in blocks 0, 2 and 4, rotary softmax attention in blocks 1, 3 and 5",
    "pollux-dual-canon": "the dual hubs of pollux-dual, with Canon layers"}
VERSION_TEXT = {"end": "the end of the training diet (step 61,422)", "fr": "the frames continuation (-fr, step 65,237)",
                "ct": "the plain-text control continuation (-ct, step 65,237)", "cf": "the carry continuation (-cf, step 65,237)",
                "birth": "the untrained start: the same architecture and seed, nothing trained (the control)"}
MODELS_LINE = ("Four small byte-level language models from the same training program, the nano v1.1 bodies ({repo}: "
               + ", ".join(f"{n}" for n in BODY_NAMES.values()) + "), 256 wide, six blocks, about 9 million parameters each; "
               "each body at the end of its training diet and in its three continuations, and each body's untrained start "
               "as the control.").format(repo=NANO_REPO)
READ_REFS = [
    ("Schönemann 1966", "A generalized solution of the orthogonal Procrustes problem. Psychometrika 31(1):1-10",
     "https://doi.org/10.1007/BF02289451"),
    ("Kornblith, Norouzi, Lee, Hinton 2019", "Similarity of Neural Network Representations Revisited. ICML 2019",
     "https://arxiv.org/abs/1905.00414"),
    ("Raghu, Gilmer, Yosinski, Sohl-Dickstein 2017", "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning "
     "Dynamics and Interpretability. NeurIPS 2017", "https://arxiv.org/abs/1706.05806"),
    ("Bansal, Nakkiran, Barak 2021", "Revisiting Model Stitching to Compare Neural Representations. NeurIPS 2021",
     "https://arxiv.org/abs/2106.07682"),
    ("Huh, Cheung, Wang, Isola 2024", "Position: The Platonic Representation Hypothesis. ICML 2024",
     "https://arxiv.org/abs/2405.07987"),
    ("Ethayarajh 2019", "How Contextual are Contextualized Word Representations? EMNLP-IJCNLP 2019",
     "https://arxiv.org/abs/1909.00512"),
    ("Sun, Chen, Kolter, Liu 2024", "Massive Activations in Large Language Models. COLM 2024", "https://arxiv.org/abs/2402.17762"),
    ("Xiao, Tian, Chen, Han, Lewis 2024", "Efficient Streaming Language Models with Attention Sinks. ICLR 2024",
     "https://arxiv.org/abs/2309.17453"),
    ("Qwen Team 2025", "Qwen3 Technical Report", "https://arxiv.org/abs/2505.09388"),
]
STEER_REFS = [
    ("Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid 2023", "Steering Language Models With Activation Engineering",
     "https://arxiv.org/abs/2308.10248"),
    ("Yu, Luo, Wang, Zhao 2024", "Uncovering the Text Embedding in Text-to-Image Diffusion Models", "https://arxiv.org/abs/2404.01154"),
    ("Gandikota, Materzynska, Zhou, Torralba, Bau 2023", "Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models",
     "https://arxiv.org/abs/2311.12092"),
    ("Brack, Friedrich, Hintersdorf, Struppek, Schramowski, Kersting 2023", "SEGA: Instructing Text-to-Image Models using Semantic "
     "Guidance. NeurIPS 2023", "https://arxiv.org/abs/2301.12247"),
    ("Radford, Kim, Hallacy et al. 2021", "Learning Transferable Visual Models From Natural Language Supervision",
     "https://arxiv.org/abs/2103.00020"),
]
ANIMA_REFS = [
    ("CircleStone Labs and Comfy Org 2026", "Anima (model card)", "https://huggingface.co/circlestone-labs/Anima"),
    ("NVIDIA et al. 2025", "Cosmos World Foundation Model Platform for Physical AI", "https://arxiv.org/abs/2501.03575"),
    ("Liu, Gong, Liu 2022", "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow",
     "https://arxiv.org/abs/2209.03003"),
]
SANA_REFS = [("Xie, Chen, Chen, Cai, Tang et al. 2024", "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion "
              "Transformers", "https://arxiv.org/abs/2410.10629")]
ANIMA_LICENCE = ("Anima's weights are under the [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/"
                 "Anima/blob/main/LICENSE.md) (Anima is a derivative of NVIDIA Cosmos-Predict2-2B, under the NVIDIA Open Model "
                 "License). Generated images are not restricted by that licence.")
JUDGE = ("The mood judge scores pixels only, with CLIP ViT-L/14: 100 x (the mean cosine to \"a cheerful, upbeat image\", \"a happy, "
         "joyful scene\", \"a bright, uplifting photo\" minus the same for \"a gloomy, downbeat image\", \"a sad, melancholy scene\", "
         "\"a dark, depressing photo\"). Every picture is paired with the same cell's picture without the push; an effect is the "
         "mean of those paired differences. The verdict rule (the earlier slider experiments' rule): MOVES IT = the mean goes the "
         "group's way, at least 75% of the pairs go that way and the mean is beyond 3 standard errors; NO EFFECT = within 2 "
         "standard errors or under 60% of the pairs that way; otherwise MIXED.")


def _f(x, d=3, sign=False):
    return "-" if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _refs(refs) -> str:
    return "\n".join(f"- {a}. {t}. {u}" for a, t, u in refs)


def _files(files: dict) -> str:
    return "| file | what |\n|---|---|\n" + "\n".join(f"| `{k}` | {v} |" for k, v in files.items())


def _scrub(x):
    """a ledger's local file paths replaced (a shipped file names no machine's folders)."""
    import re
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    if isinstance(x, str) and re.match(r"^[A-Za-z]:[\\/]", x):
        return "(a local folder of the run)"
    return x


def _write(folder: str, readme: str, meta: dict, copies: dict) -> dict:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(readme)
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    for dst, src in copies.items():
        if dst.endswith(".json"):
            with open(os.path.join(folder, dst), "w", encoding="utf-8", newline="\n") as fh:
                json.dump(_scrub(_load(src)), fh, indent=1, ensure_ascii=False)
        else:
            shutil.copyfile(src, os.path.join(folder, dst))
    return meta


def check_public(text: str) -> None:
    """the shipped prose carries no in-house words or local paths (the ship lint of this package)."""
    from ..stitch.ship import BANNED
    hit = BANNED.search(text)
    if hit:
        raise ValueError(f"in-house word or local path in public text: {text[max(0, hit.start() - 60):hit.end() + 60]!r}")


def sheet(cells: list, cols: list, path_of, out: str, tile: int = 256, labels=None) -> None:
    """a contact sheet: rows = cells, columns = cols; path_of(cell, col) -> an image file or None."""
    from PIL import Image, ImageDraw
    head = 28 if labels else 0
    canvas = Image.new("RGB", (tile * len(cols), head + tile * len(cells)), "white")
    if labels:
        dr = ImageDraw.Draw(canvas)
        for j, lab in enumerate(labels):
            dr.text((j * tile + 6, 8), str(lab)[:40], fill=(0, 0, 0))
    for i, c in enumerate(cells):
        for j, col in enumerate(cols):
            p = path_of(c, col)
            if p and os.path.exists(p):
                canvas.paste(Image.open(p).convert("RGB").resize((tile, tile)), (j * tile, head + i * tile))
    canvas.save(out, quality=85)


# ------------------------------------------------------------------------------------------------ the folders and the publishing
def ids() -> dict:
    """{which: (the experiment id from the trainer's registry, the repo)}."""
    from geolip_anima_trainer import anima_experiments as ax
    from geolip_anima_trainer import sana_experiments as sx
    return {"read": (ax.NANO_READ_ID, ax.DEFAULT_REPO), "relay": (ax.NANO_RELAY_ID, ax.DEFAULT_REPO),
            "comparator": (ax.NANO_COMPARATOR_ID, ax.DEFAULT_REPO), "sana": (sx.NANO_COMPARATOR_ID, sx.DEFAULT_REPO),
            "relay_trained": (ax.NANO_RELAY_TRAINED_ID, ax.DEFAULT_REPO)}


def write(which: str, root: str, mirror: str, e038: str, readers: str, images: dict) -> dict:
    """<mirror>/experiments/<id>/ for one experiment (which: read | relay | comparator | sana); images = {which: the run's
    picture folder}. Returns the meta."""
    rid, _ = ids()[which]
    folder = os.path.join(mirror, "experiments", rid)
    os.makedirs(folder, exist_ok=True)
    sheet_path = os.path.join(folder, "sheet.jpg")
    if which == "read":
        from .page_read import page
        readme, meta, copies, files = page(root, e038)
    elif which == "sana":
        from .page_pictures import sana_page
        readme, meta, copies, files = sana_page(root, readers, images["sana"], sheet_path)
    elif which == "relay":
        from .page_pictures import relay_page
        readme, meta, copies, files = relay_page(root, images["relay"], sheet_path)
    elif which == "comparator":
        from .page_comparator import page
        readme, meta, copies, files = page(root, readers, images["comparator"], sheet_path)
    elif which == "relay_trained":
        from .page_relay_trained import page, to_safetensors
        readme, meta, copies, files = page(root, images["relay_trained"], sheet_path)
        files.update(to_safetensors(os.path.join(root, "relay_train", "connectors"), os.path.join(folder, "connectors")))
    else:
        raise ValueError(which)
    readme = readme.replace("{rid}", rid).replace("{files}", _files(files))
    check_public(readme)
    check_public(json.dumps(meta, ensure_ascii=False))
    meta = {"id": rid, **meta}
    copies = {k: v for k, v in copies.items() if os.path.abspath(v) != os.path.abspath(os.path.join(folder, k))}
    return _write(folder, readme, meta, copies)


def publish(which: str, mirror: str) -> dict:
    """the folder into its experiments repo in one commit, then the repo README's index (the trainer's own steps)."""
    rid, repo_id = ids()[which]
    if which != "sana":
        from geolip_anima_trainer.anima_runner import AnimaRunner
        return AnimaRunner().publish_local(rid, mirror)
    from pathlib import Path
    from geolip_anima_trainer.cache_factory import get_hf_token
    from geolip_anima_trainer.sana_runner import SanaRunner, _HubRepo, _LocalRepo
    local = _LocalRepo(mirror)
    files = {f: Path(local.root) / f for f in local.files() if f.startswith(f"experiments/{rid}/")}
    meta = local.metas().get(rid)
    if not files or meta is None:
        raise FileNotFoundError(f"{mirror} holds no experiments/{rid}/meta.json")
    runner = SanaRunner()
    repo = _HubRepo(runner.cfg.repo_id, get_hf_token())
    repo.commit(files, f"{rid}: {meta.get('status')} (written offline, {len(files)} files)")
    runner._publish_index(repo, {rid: meta})
    print(f"published {len(files)} files of {rid} to {runner.cfg.repo_id}", flush=True)
    return meta


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    pos = [a for a in args if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in args if a.startswith("--") and "=" in a)
    if len(pos) != 2 or pos[0] not in ("write", "publish"):
        raise SystemExit(__doc__)
    if pos[0] == "write":
        mt = write(pos[1], opt["root"], opt["mirror"], opt.get("e038", ""), opt.get("readers", ""),
                   {k: opt.get(f"images_{k}", "") for k in ("sana", "relay", "comparator", "relay_trained")})
        print(f"wrote {mt['id']}: {mt['summary']}", flush=True)
    else:
        publish(pos[1], opt["mirror"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
