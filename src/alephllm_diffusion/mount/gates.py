"""alephllm_diffusion.mount.gates - THE MOUNT GATES (the testing plan, section 4; registered before any build): the nine-arm mount
(the library's geolip.alephllm.arm_mount, alephllm >= 0.10.6) on the stitch grid's own trunk
(alephllm_diffusion.beatrix.extract.build_model), proved on the card before any read, for both arm seeds (gCA, gCB: the nine-arm
group, the eight stage arms + the caption arm, trained with seeds A and B):
  THE REPEAT FLOOR (reported first): the bare trunk's logits twice on the same batch; an exact gate below means 0.0, and a nonzero
      floor names a nondeterministic kernel on this card (the gates are then failed and the floor printed beside them).
  GATE 1 (exact): every member masked = the bare trunk, and the group detached again = the bare trunk (detach_all verifies its own
      fingerprint too): max abs logit difference 0.0 (fp32, the same batch and device).
  GATE 2 (exact): our mount (the extractor's trunk + mount_group) = the full route (alephllm_diffusion.mount.refit's
      refit_reference_logits: load_trunk + mount_group, a second trunk), logits at 0.0 in the same precision (fp32), all on and
      with the eight stage arms masked (Mc: the caption arm with the eight masked).
  THE TAP CHECK (reported): the extractor's per-block prefill on the armed trunk against the model's own forward at the last byte
      (alephllm_diffusion.beatrix.extract.last_logits_parity); the grid reads through prefill.
  THE REPRODUCTION (reported, flagged at .02 bpb, never failed): the stored caption read on the SAME 8 batches (PackStream("held",
      2, 99_000_001, 4096) over the caption pack, in the frame the caption arm was trained on, from
      alephllm_diffusion.mount.caption_frame) at the stored precision (bf16 autocast + the compiled arm chain, as the stored read
      had it) and in fp32 eager, beside group_gC<S>.json's close 9 (a same-precision reproduction on another card agrees to about
      1e-3, never 0.0).
The batch for the exact gates: the first caption batch (2 x 4096 bytes). Reads the caption pack and the two seeds' results files
from AbstractPhil/alephllm-mini-beatrix-training (tokenless; kept in settings.MOUNT_DATA_DIR, default
<settings.OUT_DIR>/mount_data). Prints the time spent and left; writes mount_gates.json to settings.REPORT_DIR; exits 1 when an
exact gate fails (the reproduction never fails it).
Usage: python -m alephllm_diffusion.mount.gates [gCA,gCB]   (CUDA_VISIBLE_DEVICES=-1 runs it on the CPU)
MOUNT_GATES_SKIP_REPRO=1 runs the exact gates only, on a fixed random-byte batch (seed 0), without the caption pack."""
import json
import os
import platform
import sys
import time

import torch

from .. import settings  # noqa: E402
from ..beatrix import extract as bx  # noqa: E402  (sets the card's memory fraction at import)
from . import caption_frame  # noqa: E402
from . import refit as RM  # noqa: E402

REPO = "AbstractPhil/alephllm-mini-beatrix-training"
RESULTS = "mini-beatrix-3/arm_refit/group/results"
STEP = 245674
OUT_DIR = settings.OUT_DIR
REPORT_DIR = settings.REPORT_DIR
DATA = settings.MOUNT_DATA_DIR
BAR = 0.02
torch.backends.cuda.matmul.allow_tf32 = False           # TF32 off for gauges: full fp32 matmuls
torch.backends.cudnn.allow_tf32 = False
T0 = time.time()


def say(msg):
    print(f"[gates] {msg} | {time.time() - T0:.0f} s spent", flush=True)


@torch.no_grad()
def logits(model, x):
    return RM.logits(model, x)


def maxabs(a, b):
    return float((a - b).abs().max())


def fetch_data():
    """The caption pack's folder and the two seeds' results files (tokenless; kept when already there)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download(REPO, allow_patterns=[f"{RM.PACK_DIR}/*"], local_dir=DATA, token=False)
    res = {g: hf_hub_download(REPO, f"{RESULTS}/group_{g}.json", local_dir=DATA, token=False) for g in ("gCA", "gCB")}
    return os.path.join(DATA, *RM.PACK_DIR.split("/")), res


def main(groups=("gCA", "gCB")):
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" else "cuda"
    import amoe
    import geolip.alephllm as al
    import amoe.core.adapter as AD
    assert hasattr(AD.BlockWithAdapter, "prefill"), "amoe-lora >= 0.2.11 is needed (prefill applies the adapter)"
    out = {"made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "device": torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor(),
           "torch": torch.__version__, "alephllm": getattr(al, "__version__", "?"), "amoe": getattr(amoe, "__version__", "?"),
           "python": platform.python_version(), "step": STEP, "bar_reproduction": BAR, "groups": {}, "failed": []}
    say(f"THE MOUNT GATES on {out['device']}: torch {out['torch']}, alephllm {out['alephllm']}, amoe-lora {out['amoe']}; groups "
        f"{list(groups)}; about 10-20 min on a large card (the reproduction compiles the arm chain once per seed)")
    skip_repro = os.environ.get("MOUNT_GATES_SKIP_REPRO") == "1"
    pack_dir, res_files = fetch_data() if not skip_repro else (None, {})
    batches = None
    if not skip_repro:
        batches = RM.caption_batches(pack_dir, 4096, render_row=caption_frame.render_row)
        say(f"the stored read's batches rebuilt: {tuple(batches.shape)} from {pack_dir} (frame: caption_frame.render_row)")
    else:
        g = torch.Generator().manual_seed(0)
        batches = torch.randint(0, 256, (1, 2, 4097), generator=g)        # the exact gates only: a fixed random-byte batch
    x = batches[0][:, :-1].to(dev)
    for gi, grp in enumerate(groups):
        rec = {}
        model = bx.build_model(STEP, device=dev)
        bare = logits(model, x)
        rec["repeat_floor"] = maxabs(logits(model, x), bare)
        say(f"{grp}: the extractor's trunk @ {STEP:,}; THE REPEAT FLOOR (bare twice) {rec['repeat_floor']:.3g}")
        prog = RM.mount_refit_group(model, grp, require_step=STEP, device=dev)
        rec["mounted"] = {k: v for k, v in prog.mounted.items() if k != "anchors"}
        rec["anchor_hashes"] = {m: a.get("content_hash_v2") for m, a in prog.mounted["anchors"].items()}
        on = logits(model, x)
        rec["all_on_minus_bare_max_abs"] = maxabs(on, bare)
        with RM.masked(prog, list(prog.attached)):
            rec["gate1_all_masked_vs_bare"] = maxabs(logits(model, x), bare)
        with RM.only(prog, [RM.CAPTION]):
            mc = logits(model, x)
        say(f"{grp}: mounted ({len(prog.attached)} members, checked against {prog.mounted.get('checked_against')}); all on moves "
            f"the logits by {rec['all_on_minus_bare_max_abs']:.3g}; GATE 1a every member masked vs bare "
            f"{rec['gate1_all_masked_vs_bare']:.3g}")
        ref_on, ref_prog = RM.refit_reference_logits(x, grp, step=STEP, device=dev, precision="fp32")
        rec["gate2_all_on"] = maxabs(ref_on, on)
        with RM.only(ref_prog, [RM.CAPTION]):
            rec["gate2_mc"] = maxabs(logits(ref_prog.model, x), mc)
        del ref_on, ref_prog
        if dev == "cuda":
            torch.cuda.empty_cache()
        say(f"{grp}: GATE 2 our mount vs the full route: all on {rec['gate2_all_on']:.3g}, the caption arm alone (Mc) "
            f"{rec['gate2_mc']:.3g}")
        texts = ["an illustration of a lighthouse on a cliff, elated.", "a photo of a dog running on the beach at sunset"]
        rec["tap_check_last_byte_max_abs"] = bx.last_logits_parity(model, texts, device=dev)
        say(f"{grp}: THE TAP CHECK (the extractor's prefill taps on the armed trunk vs its forward, last byte) "
            f"{rec['tap_check_last_byte_max_abs']:.3g}")
        if not skip_repro:
            stored = RM.stored_close(res_files[grp], 9)
            for prec in ("stored", "fp32"):
                t1 = time.time()
                rep = RM.caption_read_repro(model, prog, batches, precision=prec, device=dev)
                table = RM.compare_to_stored(rep, stored, bar=BAR)
                rec[f"reproduction_{prec}"] = {"table": table, "seconds": round(time.time() - t1)}
                say(f"{grp}: THE REPRODUCTION at the {prec} precision ({round(time.time() - t1)} s; flagged at |delta| > {BAR}, "
                    f"never failed):")
                RM.print_comparison(table)
            prog.set_compile(False)
        RM.detach_all(prog, verify=True)
        rec["gate1_detached_vs_bare"] = maxabs(logits(model, x), bare)
        say(f"{grp}: GATE 1b detached (detach_all verified its fingerprint) vs bare {rec['gate1_detached_vs_bare']:.3g}")
        exact = {k: rec[k] for k in ("gate1_all_masked_vs_bare", "gate1_detached_vs_bare", "gate2_all_on", "gate2_mc")}
        rec["exact_gates_pass"] = all(v == 0.0 for v in exact.values())
        if not rec["exact_gates_pass"]:
            out["failed"] += [f"{grp}:{k}={v:.3g}" for k, v in exact.items() if v != 0.0]
        out["groups"][grp] = rec
        del model, prog, bare, on, mc
        if dev == "cuda":
            torch.cuda.empty_cache()
        left = (time.time() - T0) / (gi + 1) * (len(groups) - gi - 1)
        say(f"{grp}: the exact gates {'PASS' if rec['exact_gates_pass'] else 'FAIL'}; about {left:.0f} s left")
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "mount_gates.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    verdict = "ALL EXACT GATES PASS" if not out["failed"] else f"EXACT GATES FAILED: {out['failed']}"
    say(f"{verdict}; wrote {path}")
    return 0 if not out["failed"] else 1


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("gCA", "gCB")))
