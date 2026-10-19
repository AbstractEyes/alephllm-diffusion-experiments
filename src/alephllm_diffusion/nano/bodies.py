"""alephllm_diffusion.nano.bodies - the nano crafts and their loader.

A reader is (body, version): the body is one of the four nano v1.1 crafts, the version the diet's end (step 61,422), one of its three
continuations (frames -fr, plain control -ct, carry -cf, each at 65,237), one of the later arms (notes -nt at 65,237; frames then carry
-fr-cf and carry then notes -cf-nt at 69,052; the mixed phase -mx at 67,145; the caption phase -mx-cp and its control -mx-cpx at
72,868, and its second form -mx-cp2 with its control -mx-cpx2 at 72,868; each read once it is final on the Hub), or "birth": the
body's own untrained start (the same
architecture and the same seed, nothing trained: the control that must fail). The body's architecture and seed come from the trainer's
own table (geolip.nano_beatrix.nano_v1.ALL), never restated here, so a body means one thing everywhere.

Weights: AbstractPhil/alephllm-nano-castor-pollux <craft>/checkpoints/step_<step>.safetensors, or for the superseded arms (MOVED: the
frames, control, carry and notes arms and their stacks) AbstractPhil/alephllm-nano-beatrix-training castor-pollux-v1.1/<craft>/... where
they now live, byte for byte (both public; read tokenless), loaded strictly in fp32 into geolip.alephllm's AlephLM built from
geolip.nano_beatrix.bodies.craft_cfg at the crafts' context (4,096)."""
from __future__ import annotations

REPO = "AbstractPhil/alephllm-nano-castor-pollux"
TRAINING_REPO, MOVED_DIR = "AbstractPhil/alephllm-nano-beatrix-training", "castor-pollux-v1.1"
MOVED = frozenset({"fr", "ct", "cf", "nt", "fr-cf", "cf-nt"})      # the superseded arms, kept in the training repo
CTX = 4096
BODIES = {"castor": "nano-castor-v1", "pollux-canon": "nano-pollux-canon-v1", "pollux-dual": "nano-pollux-dual-v1",
          "pollux-dual-canon": "nano-pollux-dual-canon-v1"}
VERSIONS = {"end": ("", 61422), "fr": ("-fr", 65237), "ct": ("-ct", 65237), "cf": ("-cf", 65237),
            # the later arms (not every body has each one): notes from the diet's end, frames then carry, carry then notes,
            # and the mixed phase
            "nt": ("-nt", 65237), "fr-cf": ("-fr-cf", 69052), "cf-nt": ("-cf-nt", 69052), "mx": ("-mx", 67145),
            # the caption phase from the mixed phase's end (captions with their attributes as framed records) and its
            # control (the same phase, every frame taken from another caption)
            "mx-cp": ("-mx-cp", 72868), "mx-cpx": ("-mx-cpx", 72868),
            # the caption phase again with the frame withheld in half the caption documents (the question answered from the
            # caption alone) and its control (the frames and the questions' records taken from another caption)
            "mx-cp2": ("-mx-cp2", 72868), "mx-cpx2": ("-mx-cpx2", 72868)}
ORDER = ("end", "fr", "ct", "cf", "birth")
# Pollux dual's v1.2 run: the same birth and recipe as v1.1, half of every phase's rows framed from the first step; one checkpoint
# at each phase boundary of the diet (version "v12-<step>", body "pollux-dual" only), read like the versions above
V12 = "nano-pollux-dual-v1.2"
V12_STEPS = (1145, 20219, 22890, 25561, 29376, 33954, 38532, 42347, 45399, 50740, 53792, 57607, 61422)
CRAFTS = {f"v12-{s}": (V12, s) for s in V12_STEPS}


def craft(body: str, version: str) -> tuple:
    """(the craft's name on the Hub, its step) for a trained version."""
    if version in CRAFTS:
        if body != "pollux-dual":
            raise KeyError(f"{version} is a pollux-dual craft, not {body}'s")
        return CRAFTS[version]
    suffix, step = VERSIONS[version]
    return f"{BODIES[body]}.1{suffix}", step


def location(body: str, version: str) -> tuple:
    """(the repo, the path in it) of a trained version's checkpoint: the release repo, or the training repo for a moved arm."""
    name, step = craft(body, version)
    rel = f"{name}/checkpoints/step_{step:08d}.safetensors"
    return (TRAINING_REPO, f"{MOVED_DIR}/{rel}") if version in MOVED else (REPO, rel)


def spec(body: str) -> dict:
    """the trainer's own entry for the body's v1.1 craft: arm, seed, code, body text."""
    from geolip.nano_beatrix import nano_v1
    return dict(nano_v1.ALL[f"{BODIES[body]}.1"])


def state(body: str, version: str, token=False) -> dict:
    """the reader's weights (fp32, CPU): the stored checkpoint, or the birth for version 'birth'."""
    from geolip.nano_beatrix import nano_v1
    if version == "birth":
        s = spec(body)
        return nano_v1.birth_state(s["arm"], s["seed"])
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    repo, rel = location(body, version)
    path = hf_hub_download(repo, rel, token=token)
    return {k: v.float().contiguous() for k, v in load_file(path).items()}


def build(body: str, version: str, device: str = "cuda", token=False):
    """the reader as an eval-mode fp32 AlephLM on `device`, its weights loaded strictly (no missing or unexpected key)."""
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.nano_beatrix.bodies import craft_cfg
    model = AlephLM(craft_cfg(spec(body)["arm"], CTX))
    model.load_state_dict(state(body, version, token), strict=True)
    return model.to(device).eval().requires_grad_(False)
