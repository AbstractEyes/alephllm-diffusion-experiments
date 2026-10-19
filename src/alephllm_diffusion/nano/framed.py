"""alephllm_diffusion.nano.framed - a description written in the frame grammar the framed nano arms were trained on, one question
per attribute: the frames of the caption register (the nano trainer's bed_frames, sized frames sliced at 32 bytes), each attribute one
record in its own 32-byte DATA frame, then the question for one attribute and its value, read at two places: the POINTER (the
question's separator, where the arm starts to copy the value) and the CONTENT (the END after the value, the fetched value as read).

A row is built by the trainer's own frame builder (bed_frames.build_frame), so its bytes are the training layout: the register head
(MODE c, SEP q SEP), the frames back to back in the order the description writes its attributes, the question CUE key SEP right after
the last frame, the value, END. The document byte the packer puts in front is not part of the row: the extractor adds it, so every
position here is the row's own byte index plus one. Needs the nano trainer's library (geolip.nano_beatrix)."""
from __future__ import annotations

NOUNS = {"hair_colour": (" hair",), "eye_colour": (" eyes",), "expression": (" expression", " face", " look")}
KEYS = {"natural": {"hair_colour": b"hair", "expression": b"expression", "eye_colour": b"eyes"},
        "label": {"hair_colour": b"label001", "expression": b"label002", "eye_colour": b"label003"}}
BODY = 29                                      # a sized 32-byte frame's body: the open tag (2 bytes) and END around it


def value_of(pair: str, wording: str) -> bytes:
    """The record's value: the wording without its attribute noun or a leading article ('golden hair' -> 'golden', 'a warm smile'
    -> 'warm smile', 'a disapproving expression' -> 'disapproving')."""
    w = wording
    for noun in NOUNS[pair]:
        if w.endswith(noun):
            w = w[: -len(noun)]
            break
    for art in ("a ", "an "):
        if w.startswith(art):
            w = w[len(art):]
            break
    assert w and w.isascii(), wording
    return w.encode()


def fits(key: bytes, val: bytes) -> bool:
    """A record CUE key SEP value fits one 32-byte frame's body."""
    return 2 + len(key) + len(val) <= BODY


def row(pairs: list, words: list, order: list, ask: int, keys: str = "natural") -> tuple:
    """One row: the description's attributes (pairs[i] worded words[i]) as one sized 32-byte DATA frame each, in the written `order`,
    then the question for pairs[ask] and its value, END. -> (row bytes, {'pointer', 'content', 'value': (first, last)} as positions
    in the extractor's sequence, the document byte in front)."""
    import numpy as np
    from geolip.alephllm.data.special_tokens import CUE, END, SEP
    from geolip.nano_beatrix import bed_frames as bf
    state = dict(frames=[], defs={}, facts=None, key_iter=iter(()), fact_iter=None, sized=True, sliced=32)
    rng = np.random.default_rng(0)                                  # nothing random is drawn: the keys and values are given
    fill = bf.Filler(b" " * 4096, 0)                                # spaces inside a frame, as the note and carry rows
    out = bytearray(bf.tag_bytes("caption", "mode") + bytes([SEP]) + b"q" + bytes([SEP]))
    base = 1 + len(out)                                             # build_frame's base counts the document byte
    for i in order:
        key, val = KEYS[keys][pairs[i]], value_of(pairs[i], words[i])
        assert fits(key, val), (key, val)
        sp = bf.spec("data", 32, records=1, keys=[key], values=[val])
        f = bf.build_frame(rng, sp, base, fill, "caption", state)
        assert len(f) == 32, len(f)
        out += f
        base += len(f)
    key, val = KEYS[keys][pairs[ask]], value_of(pairs[ask], words[ask])
    out += bytes([CUE]) + key + bytes([SEP])
    pointer = len(out)                                              # the SEP: the extractor's sequence index (DOC in front)
    out += val
    out += bytes([END])
    return bytes(out), {"pointer": pointer, "content": len(out), "value": (pointer + 1, pointer + len(val)),
                        "key": key.decode(), "val": val.decode()}
