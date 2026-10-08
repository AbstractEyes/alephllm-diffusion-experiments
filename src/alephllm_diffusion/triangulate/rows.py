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
The two forms are tied by a round trip: the spelling read back through the map must give the caption's bytes exactly (asserted).
Rows leave out the caption's first token (Qwen's first position is its attention sink, an outlier state) and its last (no byte
follows it in A). Nothing runs on import."""
from dataclasses import dataclass
from itertools import accumulate


def char_to_byte() -> dict:
    """GPT-2's byte-level map read backwards: stand-in character -> byte (the printable bytes stand for themselves; the other 68
    take the characters from U+0100 on, in byte order)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}


_C2B = char_to_byte()


@dataclass(frozen=True)
class Caption:
    text: str
    ids: tuple            # Qwen3 token ids (Anima's call: no special tokens)
    raw: bytes            # A: what she reads
    spelled: bytes        # B: what she reads
    a_end: tuple          # per token: the byte after its expansion in `raw` (its closing byte; len(raw) for the last)
    b_end: tuple          # per token: the byte after its spelling in `spelled`


def caption(tok, text: str) -> Caption:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pieces = tok.convert_ids_to_tokens(ids)
    expansions = [bytes(_C2B[c] for c in p) for p in pieces]
    raw = text.encode("utf-8")
    assert b"".join(expansions) == raw, f"the tokens' byte expansions do not give the caption back: {text!r}"
    spellings = [p.encode("utf-8") for p in pieces]
    return Caption(text, tuple(ids), raw, b"".join(spellings), tuple(accumulate(len(e) for e in expansions)),
                   tuple(accumulate(len(s) for s in spellings)))


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
