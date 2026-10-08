"""alephllm_diffusion.triangulate - THE DUAL-EXTRACTION READ (registered 2026-10-07 before any read; image-free; no training):
Beatrix's two signals (the trunk's block output and the hub's blackboard), each read on two byte forms of the same caption (A: the
caption's own bytes; B: Qwen3's spelling of its tokens), against the Qwen3 states Anima's adapter reads, one row per Qwen token.
  rows    the token rows: where each byte form closes each Qwen token (pure Python over the tokenizer)
  align   the arithmetic: whitening, rotations fit on one half of the captions and scored on the other, ridge, the triangle, the
          class directions (pure torch)
  read    the run on the card: her prefix passes, Qwen's states, every comparison, the registered decisions
"""
