# alephllm-diffusion-experiments

Diffusion experiments for AlephLLM models. Beatrix (mini-beatrix-3, a byte-level AlephLLM) reads captions; her hidden
states are mapped into the text-conditioning space of the Anima image model, with and without her trained arm groups
mounted, and picture tests read what changes.

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
| anima-trainer | commit 98954d8 |
| diffusion-pipe fork | commit 84e7fe3, its ComfyUI at 0ba903b |

On Windows the file adds `triton-windows` (for `torch.compile`); DeepSpeed, used for multi-card picture runs, installs on
Linux only. `pyproject.toml` carries compatible ranges for the package itself; `requirements.txt` is the tested set.

## Checking the environment

```bash
python -m alephllm_diffusion.environment   # builds, the card, any package off its pin, the libraries' and the fork's commits
pytest                                     # the same checks as tests
```

The check exits with code 1 when anything differs from the pins.

## Status

The experiment code moves in next: the grid that maps Beatrix's states into Anima's conditioning space, the arm-mount checks
and reads, and the export the picture runs consume. A short Colab notebook follows, installing this package and running one
session per call.

## Licences

The code is Apache-2.0. Model weights carry their own licences: Anima's weights are non-commercial (see its model card), and
Beatrix's follow the mini-beatrix-3 model card.
