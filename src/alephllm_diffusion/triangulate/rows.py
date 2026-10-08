"""alephllm_diffusion.triangulate.rows - the token rows of the dual-extraction read. A caption goes through Anima's own Qwen3
tokenizer call (no special tokens; Qwen3 adds no BOS) and, per Qwen token, the read records where each of Beatrix's two byte forms
closes it:
  A, the caption's own bytes   the UTF-8 bytes of the caption. A token's bytes are its exact byte expansion (the tokenizer's
                               byte-to-character map read backwards); the token closes at the byte AFTER its expansion (the
                               relay's convention: her state there has read the whole token).
  B, Qwen's spelling           the tokens' vocabulary strings in order, as UTF-8. A byte-level BPE stores every byte as a printable
                               stand-in character (GPT-2's map): a space is 'Ġ' (C4 A0 in UTF-8), a newline 'Ċ', every byte at or
                               above 0x80 a two-byte stand-in. For ASCII text the spelling differs from the raw bytes only at the
                               spaces: ' taco' is the one token 'Ġtaco', six bytes against five. The token closes at the first byte
                               of the next token's spelling.
The two forms are tied by a round trip: the spelling read back through the map must give the caption's bytes exactly.
THE RENDERER is the library's (geolip.alephllm.train.surface.spell, the GPT-2 byte-level convention): the instrument, the surface
arms it reads and their training share one spelling. It replaced this module's own copy (identical byte for byte on both caption
draws, 4,096 captions; tests/test_triangulate.py keeps the old copy as the reference). A text without an exact spelling raises
ValueError. Rows leave out the caption's first token (Qwen's first position is its attention sink, an outlier state) and its last
(no byte follows it in A). Nothing runs on import."""
from geolip.alephllm.train.surface import Spelled as Caption  # noqa: F401  (the row type: text, ids, raw, spelled, a_end, b_end)
from geolip.alephllm.train.surface import char_to_byte, spell  # noqa: F401


def caption(tok, text: str) -> Caption:
    """the caption's two byte forms and each token's closing byte on both, by the library's renderer."""
    return spell(tok, text, convention="gpt2")


def token_positions(c: Caption) -> list:
    """the token positions a caption contributes rows for: every token but the first and the last."""
    return list(range(1, len(c.ids) - 1))


def prefixes(c: Caption, t: int) -> tuple:
    """(A prefix, B prefix): each byte form through token t's closing byte."""
    assert 0 <= t < len(c.ids) - 1, (t, len(c.ids))
    return c.raw[: c.a_end[t] + 1], c.spelled[: c.b_end[t] + 1]


def phrase_end_token(c: Caption, phrase_end_byte: int) -> int:
    """the token whose expansion ends exactly at a phrase's end (the byte after its last character); asserted to exist, so a
    phrase's last token is never shared with the text after it."""
    hits = [t for t, e in enumerate(c.a_end) if e == phrase_end_byte]
    assert len(hits) == 1, (c.text, phrase_end_byte, c.a_end)
    return hits[0]
