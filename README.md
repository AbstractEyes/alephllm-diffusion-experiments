# alephllm-diffusion-experiments

Diffusion experiments for AlephLLM models. Beatrix (mini-beatrix-3, a byte-level AlephLLM) reads captions; her hidden
states are mapped into the text-conditioning space of the Anima image model, with and without her trained arm groups
mounted, and picture tests read what changes. Her hubs' blackboards (each block's memory of the text) are measured against
her stream as conditioning features, and her readings of a phrase drive learned sliders in Anima's text context.

This repository is the launch point for those experiments: one pinned environment, one package, and the diffusion-pipe
fork at a fixed commit. The model code stays in its own libraries:

| piece | where |
|---|---|
| Beatrix (AlephLLM) and the arm mount | [alephllm](https://github.com/AbstractEyes/alephllm) (`geolip.alephllm`) |
| the arm adapters | [amoe-lora](https://github.com/AbstractEyes/amoe-lora) (`amoe`) |
| Anima rendering and the picture runner | [anima-trainer](https://github.com/AbstractEyes/anima-trainer) (`geolip_anima_trainer`) |
| Anima's model code, with ComfyUI | the [diffusion-pipe fork](https://github.com/AbstractEyes/diffusion-pipe), a submodule at `external/diffusion-pipe` |
| the experiments | this repository (`alephllm_diffusion`) |

## Install

Python 3.12 and an NVIDIA driver new enough for CUDA 12.8 (R570 or newer). The CUDA runtime ships inside the torch wheels,
so no system CUDA toolkit is needed.

```bash
git clone https://github.com/AbstractEyes/alephllm-diffusion-experiments
cd alephllm-diffusion-experiments
git submodule update --init external/diffusion-pipe
git -C external/diffusion-pipe submodule update --init submodules/ComfyUI
python -m venv .venv                     # or let the IDE create it
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip install -r requirements.txt
python -m alephllm_diffusion.environment
```

The fork carries submodules for other models; only ComfyUI is needed here, so a `--recursive` clone is not.

## The pinned environment

`requirements.txt` pins every direct dependency to the set that ran together on an RTX PRO 6000 Blackwell on 2026-10-06.

| package | version |
|---|---|
| torch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0, CUDA 12.8 builds |
| transformers / huggingface_hub / datasets | 4.57.6 / 0.36.2 / 2.21.0 |
| diffusers / accelerate / peft | 0.35.2 / 1.15.0 / 0.21.2 |
| alephllm | 0.10.7, commit 4da43e7 |
| amoe-lora | commit 9e95a0a (the `experimental` branch; `amoe` 0.2.11) |
| anima-trainer | commit 40e7b04 |
| diffusion-pipe fork | commit 84e7fe3, its ComfyUI at 0ba903b |

On Windows the file adds `triton-windows` (for `torch.compile`); DeepSpeed, used for multi-card picture runs, installs on
Linux only. `pyproject.toml` carries compatible ranges for the package itself; `requirements.txt` is the tested set.

## Checking the environment

```bash
python -m alephllm_diffusion.environment   # builds, the card, any package off its pin, the libraries' and the fork's commits
pytest                                     # the same checks as tests
```

The check exits with code 1 when anything differs from the pins.

## Running the experiment

The experiments run in sessions on one card. On Colab, open
[`notebooks/colab_sessions.ipynb`](notebooks/colab_sessions.ipynb)
([open it in Colab](https://colab.research.google.com/github/AbstractEyes/alephllm-diffusion-experiments/blob/v0.3.0/notebooks/colab_sessions.ipynb)):
cell 1 installs this package, cell 2 runs a session, and cell 3 shows what is done. Anywhere else, after the install above:

```python
from alephllm_diffusion import sessions
sessions.run(1)        # then 2, then 3; 4 on its own; running a session again skips the steps already done
sessions.status()
```

| session | steps | about (RTX PRO 6000) |
|---|---|---|
| 1 | the mount checks, the mount read, the port check, the restart test, stage 1 of the grid, the picture test's stage A | 2.5-3 h |
| 2 | the grid of record, the pick, the mount grids and their contrast, the export, publishing the grids | 4-7 h |
| 3 | the picture test's stage B | 1-1.5 h |
| 4 | the hub sliders: Beatrix's slider features at three readings (her stream at a phrase's closing stop, her hub's blackboard, both), then seven slider arms trained by Anima's own objective and judged on held-out scenes | 1.5-2.5 h |

Each step runs as its own process and keeps its full output in `runs/logs/`. The fixed reference inputs (stage 0's captions,
the reference run's cell lines, the caption draws and their ruler embeddings) come from a private data repository and are
checked against the SHA-256 recorded in `alephllm_diffusion/data/reference.json`; the session files go to the same repository,
so a session can continue on another machine. `sessions.run(1, smoke=True, store=False)` rehearses the runner on 80 captions in
a scratch workspace (`ALEPHLLM_DIFFUSION_HOME`), without the pictures or any upload.

## What is inside

| module | what it does |
|---|---|
| `alephllm_diffusion.beatrix.extract` | Beatrix's hidden states for a list of captions, at any of her 32 blocks |
| `alephllm_diffusion.beatrix.gauges` | the conditioning gauges (debiased CKA, Procrustes, ridge fits, effective rank) |
| `alephllm_diffusion.mount.gates` | the exact checks of the arm mount, run before any read |
| `alephllm_diffusion.mount.read` | what the mounted arm groups change in Beatrix's caption states |
| `alephllm_diffusion.stitch.s0` ... `s2` | the grid: caption token maps, Beatrix's states, and the cells that map them into Anima's text space |
| `alephllm_diffusion.stitch.pick`, `mounts`, `export` | the pick, the mount contrast, and the export the picture runs read |
| `alephllm_diffusion.stitch.port_gate`, `resume_test` | the port check against the reference run and the restart test |
| `alephllm_diffusion.stitch.ship` | publishes a grid to the public data repository (scrubbed of local details) |
| `alephllm_diffusion.hubs.refs` | the caption references the hub read adds (CLIP-B/32, Qwen3 0.6B pooled, Anima's adapter output pooled) |
| `alephllm_diffusion.hubs.read` | the hub read: the blackboards against the stream (caption geometry and the slider instruments), with its decision |
| `alephllm_diffusion.hubs.features` | the slider arms' features files for a chosen reading (bare, the nine arms, untrained) |
| `alephllm_diffusion.hubs.experiment` | the hub read as experiment e030 of the Anima experiments repo: its page and files from the read's ledger, published with the Anima trainer's own step |
| `alephllm_diffusion.sessions` | the session runner |
| `alephllm_diffusion.settings`, `storage`, `models` | where files live, the private data store, the public model downloads |

## Licences

The code is Apache-2.0. Model weights carry their own licences: Anima's weights are non-commercial (see its model card), and
Beatrix's follow the mini-beatrix-3 model card.
