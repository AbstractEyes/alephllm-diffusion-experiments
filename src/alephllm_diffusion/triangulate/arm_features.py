"""alephllm_diffusion.triangulate.arm_features - the slider features read through a surface arm (session 6; registered before any
code in plans/2026-10-08_qwen_arm_read_and_session6.md, part B). A surface arm (geolip.alephllm.arm_mount.mount_surface) makes
Beatrix read a tokenizer's spelling of a text as she reads its own bytes; these features give the slider arms her reading of a
mood phrase through it.
THE ROWS: each of the connectors' 74 mood phrases and 64 reference texts (hubs.read's table) in the grid's caption form,
PREFIX + 'an illustration of a quiet street, {phrase}.' (the form of the dual-extraction read's mood rows; a phrase alone gives
the library's reader no site), read by the library's reader (geolip.alephllm.train.surface.read_spelled, fp32, the arm's own
reader arguments) at the text's last site, which closes at its full stop; one block; LayerNorm'd without affine (the reader's);
z-scored per feature over the 64 reference texts read the same way. THE TENSORS:
  qwen         her final trunk with the arm's group and the arm mounted: Qwen3's spelling of the text read through the arm
  plain        the same mount with the arm masked: the text's own bytes (her plain reading in the same mount)
  qwen_random  the untrained copy (random init, seed 0) with its own arm (the registry's '<surface>-untrained' row): Qwen3's
               spelling read through it (the control)
Every site is checked against the dual-extraction read's mood rows (triangulate.read.draw_rows: the same token, the same id).
Writes <settings.OUT_DIR>/mood_phrases_frame-<surface>-arm-<block>_mini-beatrix-3_step245674.safetensors; upload() puts it in the
data repo under beatrix/ (hubs.features.upload). One job on the card; fp32.
Usage: python -m alephllm_diffusion.triangulate.arm_features [--surface=qwen3] [--block=20] [--untrained=1] [--upload=1]"""
import json
import os
import sys
import time

import torch

from .. import settings
from ..beatrix import extract as bx
from ..hubs import features as HF
from ..hubs import read as H
from ..mount import read as MR
from . import arm_read as AR
from . import read as RD

TENSORS = ("qwen", "plain", "qwen_random")
STRINGS = list(dict.fromkeys(H.TEXTS + H.REF))             # every text read once: the 74 phrases, then the reference texts
T0 = time.time()


def say(*a):
    print(f"[arm features {time.time() - T0:7.1f}s]", *a, flush=True)


def reading(surface: str, block: int) -> str:
    return f"frame/{surface}-arm/{block}"


def file_name(surface: str, block: int) -> str:
    return f"mood_phrases_{reading(surface, block).replace('/', '-')}_mini-beatrix-3_step{RD.STEP}.safetensors"


def framed(s: str) -> str:
    from geolip_anima_trainer import anima_experiments as ax
    return ax.PREFIX + MR.PREFIX_FRAME.replace("{w}", s)


def tokenizer():
    """Anima's own Qwen3 tokenizer (the diffusion-pipe fork's configs), as the dual-extraction read used."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(settings.DP, "configs", "qwen3_06b"), local_files_only=True)


@torch.no_grad()
def read_closing(model, tok, block: int, surface: str, kw: dict) -> torch.Tensor:
    """(len(STRINGS), d) fp32: the reader's state at each framed text's last site (its full stop), checked against the
    dual-extraction read's mood rows."""
    from geolip.alephllm.train.surface import read_spelled
    texts = [framed(s) for s in STRINGS]
    r = read_spelled(model, tok, texts, [block], surface=surface, amp=False, **kw)
    idx = AR.row_index(r["sites"], 0, len(texts))
    AR.check_rows(r["sites"][idx], RD.draw_rows(tok, [], texts))
    X = r["states"][block][idx].float()
    assert torch.isfinite(X).all(), ("non-finite state", surface, block)
    return X


def zscored(X: torch.Tensor) -> torch.Tensor:
    """(74, d) float32 in the table's order: z-scored per feature over the reference texts (hubs.read.zscored's rule)."""
    ref = X[[STRINGS.index(s) for s in H.REF]]
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    Z = ((X - mu) / sd).double()
    return Z[[STRINGS.index(t) for t in H.TEXTS]].float().contiguous()


def build(surface: str = "qwen3", block: int = AR.SLIDER_BLOCK, dev=None, untrained: bool = True) -> dict:
    """{tensor: (74, d)} for the three tensors (qwen_random only with untrained=True), one trunk on the card at a time."""
    from geolip.alephllm.arm_mount import SURFACE_ARMS, masked, mount_surface, reader_kwargs
    dev = dev or ("cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok, row = tokenizer(), SURFACE_ARMS[surface]
    kw, out = reader_kwargs(row), {}
    model = bx.build_model(step=RD.STEP, device=dev)
    prog = mount_surface(model, surface, device=dev)
    model.eval()
    say(f"mounted {sorted(prog.attached)} on her trunk at step {RD.STEP}")
    out["qwen"] = zscored(read_closing(model, tok, block, "B", kw))
    with masked(prog, [row["member"]]):
        out["plain"] = zscored(read_closing(model, tok, block, "A", kw))
    say(f"qwen and plain read at block {block}")
    del model, prog
    if dev == "cuda":
        torch.cuda.empty_cache()
    if untrained:
        name = f"{surface}-untrained"
        model = bx.build_model(None, device=dev, random_init_seed=0)
        prog = mount_surface(model, name, device=dev, require_step=None)
        model.eval()
        say(f"mounted {sorted(prog.attached)} ({name}) on the untrained copy (seed 0)")
        out["qwen_random"] = zscored(read_closing(model, tok, block, "B", reader_kwargs(SURFACE_ARMS[name])))
        del model, prog
        if dev == "cuda":
            torch.cuda.empty_cache()
    return out


def write(surface: str, block: int, feats: dict, out_dir=None) -> str:
    from geolip.alephllm.arm_mount import SURFACE_ARMS, SURFACE_REPO
    from safetensors.torch import save_file
    path = os.path.join(out_dir or settings.OUT_DIR, file_name(surface, block))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = SURFACE_ARMS[surface]
    control = ("; qwen_random: same shape, untrained, seed 0, with its own arm "
               f"{SURFACE_REPO}/{SURFACE_ARMS[surface + '-untrained']['file']}" if "qwen_random" in feats else "")
    meta = {"phrases": json.dumps([{"class": c, "split": s, "text": p} for c, s, p in H.ROWS]),
            "reading": reading(surface, block),
            "tap": (f"her block output after block {block} at the full stop closing the text (the reader's last site), the "
                    f"text in the grid's caption form '{framed('{phrase}')}'; LayerNorm without affine; z-scored per feature "
                    "over the reference set read the same way"),
            "trunk": (f"AbstractPhil/alephllm-mini-beatrix-training mini-beatrix-3 step {RD.STEP} (qwen: the surface arm "
                      f"{SURFACE_REPO}/{row['file']} mounted over its group {row['base']}, Qwen3's spelling read through it; "
                      f"plain: the same mount, the arm masked, the text's own bytes{control})"),
            "surface": json.dumps(dict(row, name=surface)),
            "reference": json.dumps(H.REF), "reads": json.dumps({k: HF.reads_of(v) for k, v in feats.items()}),
            "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    save_file({k: v.contiguous() for k, v in feats.items()}, path, metadata=meta)
    return path


def main(surface="qwen3", block=AR.SLIDER_BLOCK, untrained=True, upload=False):
    feats = build(surface, block, untrained=untrained)
    p = write(surface, block, feats)
    say(f"wrote {p} ({', '.join(f'{k} {tuple(v.shape)}' for k, v in feats.items())}; reads "
        f"{ {k: HF.reads_of(v) for k, v in feats.items()} })")
    if upload:
        say(f"uploaded to {HF.upload(p)}")
    return p


if __name__ == "__main__":
    flags = {x.split("=", 1)[0][2:]: x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--") and "=" in x}
    main(surface=flags.get("surface", "qwen3"), block=int(flags.get("block", AR.SLIDER_BLOCK)),
         untrained=flags.get("untrained", "1") == "1", upload=flags.get("upload", "0") == "1")
    sys.exit(0)
