"""alephllm_diffusion: diffusion experiments for AlephLLM models.

Beatrix (mini-beatrix-3, a byte-level AlephLLM) reads captions; her hidden states are mapped into the text-conditioning
space of the Anima image model, with and without her trained arm groups mounted, and picture tests read what changes.

The AlephLLM model and its arm mount come from `geolip.alephllm`; the adapters from `amoe`; picture rendering from
`geolip_anima_trainer` and the diffusion-pipe fork carried in `external/diffusion-pipe`.

    python -m alephllm_diffusion.environment      # what is installed, the card, the fork's commit
"""

__version__ = "0.3.1"
