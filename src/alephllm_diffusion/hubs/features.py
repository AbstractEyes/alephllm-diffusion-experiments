"""alephllm_diffusion.hubs.features - the slider experiments' features file for a chosen reading of Beatrix's states: per trunk
condition the connectors' 74 phrases in the connectors' format (each block LayerNorm'd without affine, concatenated, z-scored
per feature over the 64 reference rows; hubs.read's part B makes the same features), the phrase table, the reading and the
reference set in the metadata. The image-model side (anima-trainer's connector arms) reduces the rows to the slider values
itself (its connector_axis, fit on the training rows only).

THE READING, named as the hub read names its features: "<form>/<kind>/<block>"
  form   last (the phrase alone) | close (the phrase closed by a full stop)
  kind   stream (her block output at the last byte) | hub (the block's hub blackboard after the text) |
         both (the hub's and the stream's rows side by side, each divided by the length of its own mood axis on the training
         rows: through the connectors' axis the slider value is then exactly the mean of the two sources' own slider values)
  block  a block index, or rec = the connectors' four blocks (16, 18, 21, 24) concatenated
THE CONDITIONS (the file's tensors): trained = the bare trunk at step 245,674; arms9 = the nine-arm group gCA mounted, all
nine live; random = the same architecture untrained (random init, seed 0). Every condition is z-scored on its own reference
rows.

Writes <settings.OUT_DIR>/mood_phrases_<form>-<kind>-<block>_mini-beatrix-3_step245674.safetensors; upload() puts it in the data
repo under beatrix/. One job on the card; fp32.
Usage: python -m alephllm_diffusion.hubs.features close/stream/18,close/hub/22 [--upload=1]"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from .. import settings
from ..beatrix import extract as bx
from . import read as H

DATA_REPO = "AbstractPhil/geolip-beatrix-anima-data"
CONDITIONS = ("trained", "arms9", "random")


def parse(reading):
    form, kind, block = reading.split("/")
    assert form in ("last", "close") and kind in ("stream", "hub", "both"), reading
    blocks = list(H.RECORD_BLOCKS) if block == "rec" else [int(block)]
    return form, kind, blocks


def _z(L, kind, blocks):
    """the z-scored rows of every part-B string for one kind (H.zscored), float64."""
    return H.zscored({b: L[(kind, b)] for b in blocks}, blocks)


def axis_length(Z):
    """the length of the mood axis on the training rows (the up centre minus the down centre), the connectors' own fit."""
    zr = Z[[H.STRINGS_B.index(t) for t in H.TEXTS]]
    train = [i for i, r in enumerate(H.ROWS) if r[1] == "train"]
    up = zr[[i for i in train if H.ROWS[i][0] == "up"]].mean(0)
    dn = zr[[i for i in train if H.ROWS[i][0] == "down"]].mean(0)
    return float((up - dn).norm())


def rows_for(model, dev, reading, prefix=""):
    """(74, D) float32: the table's rows (H.TEXTS order) of one reading on one trunk condition."""
    form, kind, blocks = parse(reading)
    L = H.features_b(model, dev, form, prefix, blocks)
    if kind == "both":
        zh, zs = _z(L, "hub", blocks), _z(L, "stream", blocks)
        Z = torch.cat([zh / axis_length(zh), zs / axis_length(zs)], dim=1)
    else:
        Z = _z(L, kind, blocks)
    return Z[[H.STRINGS_B.index(t) for t in H.TEXTS]].float().contiguous()


def build(readings, dev=None, groups=("gCA",)):
    """{reading: {condition: (74, D)}} for one reading or a list of them, one trunk on the card at a time (each trunk loaded
    once for every reading)."""
    from geolip.alephllm.arm_mount import detach_all, mount_group
    readings = [readings] if isinstance(readings, str) else list(readings)
    for r in readings:
        parse(r)
    dev = dev or ("cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda")
    out = {r: {} for r in readings}
    model = bx.build_model(H.STEP, device=dev)
    for r in readings:
        out[r]["trained"] = rows_for(model, dev, r)
    prog = mount_group(model, groups[0], require_step=H.STEP, device=dev)
    model.eval()
    for r in readings:
        out[r]["arms9"] = rows_for(model, dev, r)
    detach_all(prog, verify=True)
    del model, prog
    if dev == "cuda":
        torch.cuda.empty_cache()
    model = bx.build_model(None, device=dev, random_init_seed=0)
    for r in readings:
        out[r]["random"] = rows_for(model, dev, r)
    del model
    return out


def reads_of(F_):
    """the connectors' held-out reads on a file's rows (the features file's own 'reads' field, as the earlier file carried)."""
    Z = F_.double()
    held = [i for i, r in enumerate(H.ROWS) if r[1] == "heldout"]
    train = [i for i, r in enumerate(H.ROWS) if r[1] == "train"]
    cent = {c: Z[[i for i in train if H.ROWS[i][0] == c]].mean(0) for c in ("up", "down", "neutral")}
    axis, mid = cent["up"] - cent["down"], (cent["up"] + cent["down"]) / 2
    near = sum(int(max(cent, key=lambda c: float(F.cosine_similarity(Z[i], cent[c], dim=0))) == H.ROWS[i][0]) for i in held)
    signs = [(H.ROWS[i][0], float((Z[i] - mid) @ axis)) for i in held if H.ROWS[i][0] != "neutral"]
    ok = sum((v > 0) == (c == "up") for c, v in signs)
    return {"heldout_nearest": [int(near), len(held)], "heldout_sign": [int(ok), len(signs)]}


def write(reading, feats, out_dir=None):
    from safetensors.torch import save_file
    form, kind, blocks = parse(reading)
    name = f"mood_phrases_{reading.replace('/', '-')}_mini-beatrix-3_step{H.STEP}.safetensors"
    path = os.path.join(out_dir or settings.OUT_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tap = {"stream": "her block output at the {pos}", "hub": "the block's hub blackboard (4 x 64 slots x 1,024, each slot "
           "unit length) after the text", "both": "the hub blackboard and the block output at the {pos} side by side, each "
           "divided by the length of its own mood axis on the training rows"}[kind]
    pos = "phrase's last byte" if form == "last" else "full stop closing the phrase"
    meta = {"phrases": json.dumps([{"class": c, "split": s, "text": p} for c, s, p in H.ROWS]),
            "reading": reading,
            "tap": (tap.format(pos=pos) + f"; the text {'alone' if form == 'last' else 'closed by a full stop'}; blocks "
                    f"{', '.join(map(str, blocks))}; LayerNorm without affine per block; concatenated; z-scored per feature over "
                    "the reference set"),
            "trunk": (f"AbstractPhil/alephllm-mini-beatrix-training mini-beatrix-3 step {H.STEP} (trained: bare; arms9: the "
                      "nine-arm group gCA mounted, all live; random: same shape, untrained, seed 0)"),
            "reference": json.dumps(H.REF), "reads": json.dumps({k: reads_of(v) for k, v in feats.items()}),
            "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    save_file({k: v.contiguous() for k, v in feats.items()}, path, metadata=meta)
    return path


def upload(path, token=None):
    from huggingface_hub import HfApi
    dest = f"beatrix/{os.path.basename(path)}"
    HfApi(token=token or os.environ.get("HF_TOKEN") or None).upload_file(
        path_or_fileobj=path, path_in_repo=dest, repo_id=DATA_REPO, repo_type="dataset",
        commit_message=f"the slider features of {os.path.basename(path)}")
    return f"{DATA_REPO}/{dest}"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert args, __doc__
    t0 = time.time()
    built = build(args[0].split(","))
    for r, feats in built.items():
        p = write(r, feats)
        print(f"[hub features] {r}: wrote {p} ({', '.join(f'{k} {tuple(v.shape)}' for k, v in feats.items())}; reads "
              f"{ {k: reads_of(v) for k, v in feats.items()} })", flush=True)
        if "--upload=1" in sys.argv:
            print(f"[hub features] uploaded to {upload(p)}", flush=True)
    print(f"[hub features] {len(built)} reading(s) in {time.time() - t0:.0f} s", flush=True)
    sys.exit(0)
