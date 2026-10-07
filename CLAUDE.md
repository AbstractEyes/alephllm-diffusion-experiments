# Working in this repository

This is the launch point for diffusion experiments with AlephLLM models. The libraries it builds on (alephllm, amoe-lora,
anima-trainer, the diffusion-pipe fork) stay where they are; experiment code lives here, in `src/alephllm_diffusion`.

- The code is public: comments explain the code, never internal notes, session logs or private record references.
- `requirements.txt` is the tested environment. Change a pin only after the environment check and the tests pass with the
  new version; libraries are pinned by commit and move together with a note in the commit message.
- The diffusion-pipe fork is a submodule at `external/diffusion-pipe`; initialise only its `submodules/ComfyUI`.
- Fixes ship as commits and version bumps. Nothing is patched at run time, and no code is shipped as an archive or through
  the Hugging Face hub; the hub holds data and results only.
- A large, structured piece of code that would otherwise be bundled, vendored or imported ad hoc gets a proper home: a
  module here, or its own repository.
- Before a commit: `python -m alephllm_diffusion.environment` and `pytest`.
