"""alephllm_diffusion.nano.framed - a description written in the frame grammar the framed nano arms were trained on, one question
per attribute: the frames of the caption register (the nano trainer's bed_frames, sized frames sliced at 32 bytes), each attribute one
record in its own 32-byte DATA frame, then the question for one attribute and its value, read at two places: the POINTER (the
question's separator, where the arm starts to copy the value) and the CONTENT (the END after the value, the fetched value as read).

A row is built by the trainer's own frame builder (bed_frames.build_frame), so its bytes are the training layout: the register head
(MODE c, SEP q SEP), the frames back to back in the order the description writes its attributes, the question CUE key SEP right after
the last frame, the value, END. The document byte the packer puts in front is not part of the row: the extractor adds it, so every
position here is the row's own byte index plus one. Needs the nano trainer's library (geolip.nano_beatrix).

caption_row writes the same description in the caption phase's two reading layouts (keys as paths of symbols, one sized frame for
the records, the caption and the question after it), read at the same two places."""
from __future__ import annotations

import re

NOUNS ={"hair_colour": (" hair",), "eye_colour": (" eyes",), "expression": (" expression", " face", " look")}
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


# The caption phase's reading layouts (the nano trainer's caption reads): a record's key is a path of symbols, the system first and
# the attribute second; one system's records share one sized frame; the caption and the question follow.
PATHS = {"hair_colour": b"look hair", "expression": b"look expression", "eye_colour": b"look eyes"}
REC_MAX, VAL_MAX = 27, 20                      # key + value within 27 bytes (one 32-byte slice holds the record), the value within 20
SCAFFOLD = re.compile(r"^(?:with an? |with |wearing an? |wearing |has an? |has |having an? |having |in an? |in |an? |the )+")
LEAF_NOUNS = {"hair": (" hair",), "eyes": (" eyes", " eye"), "expression": (" expression", " look", " face")}
LAYOUTS = ("extraction", "copy")


def caption_value(pair: str, wording: str) -> bytes:
    """The record's value in the caption layouts: the wording in lower case without its scaffolding ('a', 'with a', 'wearing')
    and without the attribute's noun, cut at a word boundary to the record's room ('a pleased look' -> 'pleased'; 'a cheerful
    grin' -> 'cheerful' under 'look expression', whose record leaves 12 bytes for the value)."""
    key = PATHS[pair].decode()
    s = re.sub(r"\s+", " ", wording).strip().lower().strip(" .,;:")
    s = SCAFFOLD.sub("", s)
    for noun in LEAF_NOUNS[key.split(" ")[-1]]:
        if s.endswith(noun) and len(s) - len(noun) >= 2:
            s = s[: -len(noun)].strip()
            break
    b = s.strip(" .,;:").encode("ascii")
    room = min(VAL_MAX, REC_MAX - len(key))
    if len(b) > room:
        b = b[:room].rsplit(b" ", 1)[0].rstrip(b" ,;:.")
        assert len(b) >= 3, wording
    assert 1 <= len(b) <= room and all(32 <= c <= 126 for c in b), wording
    return b


def caption_slices(recs: list, room: int = BODY, cap: int = 4) -> list:
    """The records packed into 32-byte slices in order, as many a slice as fit (a record never split)."""
    slices, cur, used = [], [], 0
    for k, v in recs:
        n = 2 + len(k) + len(v)
        assert n <= room, (k, v)
        if used + n > room:
            slices.append(cur)
            cur, used = [], 0
        cur.append((k, v))
        used += n
    slices += [cur] if cur else []
    assert len(slices) <= cap, recs
    return slices


def caption_row(pairs: list, words: list, order: list, ask: int, text: str, layout: str = "extraction") -> tuple:
    """One row in a caption reading layout: the caption register's head; for 'copy' the description's records (pairs[i] worded
    words[i], keyed by path, in the written `order`) as one sized frame (32, 64 or 128 bytes by the slices they need) and a
    newline; then the description `text` as the caption, a newline, the question for pairs[ask], its value, END ('extraction':
    the value has to come from the caption's words). -> (row bytes, positions as row(): the document byte in front)."""
    import numpy as np
    from geolip.alephllm.data.special_tokens import CUE, END, SEP
    from geolip.nano_beatrix import bed_frames as bf
    assert layout in LAYOUTS, layout
    out = bytearray(bf.tag_bytes("caption", "mode") + bytes([SEP]) + b"q" + bytes([SEP]))
    if layout == "copy":
        recs = [(PATHS[pairs[i]], caption_value(pairs[i], words[i])) for i in order]
        n = len(caption_slices(recs))
        size = 32 if n <= 1 else 64 if n == 2 else 128
        state = dict(frames=[], defs={}, facts=None, key_iter=iter(()), fact_iter=None, sized=True, sliced=32)
        sp = bf.spec("data", size, records=len(recs), keys=[k for k, _ in recs], values=[v for _, v in recs])
        out += bf.build_frame(np.random.default_rng(0), sp, 0, bf.Filler(b" " * 4096, 0), "caption", state) + b"\n"
    out += text.encode("ascii") + b"\n"
    key, val = PATHS[pairs[ask]], caption_value(pairs[ask], words[ask])
    out += bytes([CUE]) + key + bytes([SEP])
    pointer = len(out)
    out += val + bytes([END])
    return bytes(out), {"pointer": pointer, "content": len(out), "value": (pointer + 1, pointer + len(val)),
                        "key": key.decode(), "val": val.decode()}
