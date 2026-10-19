"""A description in the framed arms' grammar: the record values, the frame budget, and (where the nano trainer's library is
installed; skipped elsewhere) a row's layout: the register head, one 32-byte record frame per attribute in the written order, the
question, the value and the end byte, with the read positions on the separator and the end byte. Milliseconds, no download."""
import pytest

from alephllm_diffusion.nano import framed as FR


def test_a_value_is_the_wording_without_its_noun_or_article():
    assert FR.value_of("hair_colour", "golden hair") == b"golden"
    assert FR.value_of("expression", "a warm smile") == b"warm smile"
    assert FR.value_of("expression", "a disapproving expression") == b"disapproving"
    assert FR.value_of("expression", "an open face") == b"open"
    assert FR.value_of("eye_colour", "crimson eyes") == b"crimson"


def test_a_record_fits_one_frame_only_within_its_body():
    assert FR.fits(b"expression", b"x" * (FR.BODY - 2 - len(b"expression")))
    assert not FR.fits(b"expression", b"x" * (FR.BODY - 1 - len(b"expression")))
    assert all(FR.fits(k, b"disapproving") for form in FR.KEYS.values() for k in form.values())


@pytest.fixture(scope="module")
def nano_lib():
    pytest.importorskip("geolip.nano_beatrix.bed_frames", reason="the nano trainer's library is not installed here")


@pytest.mark.parametrize("keys", ["natural", "label"])
def test_a_row_is_the_head_three_record_frames_and_the_question(nano_lib, keys):
    from geolip.alephllm.data.special_tokens import CUE, DATA, DOC, END, MODE, SEP
    pairs, words = ["hair_colour", "expression", "eye_colour"], ["golden hair", "a warm smile", "crimson eyes"]
    b, at = FR.row(pairs, words, [2, 0, 1], ask=1, keys=keys)
    k = {p: FR.KEYS[keys][p] for p in pairs}
    question = bytes([CUE]) + k["expression"] + bytes([SEP]) + b"warm smile" + bytes([END])
    head = b[:5]
    assert head == bytes([MODE]) + b"c" + bytes([SEP]) + b"q" + bytes([SEP])
    assert len(b) == len(head) + 3 * 32 + len(question) and b.endswith(question)
    for i, (p, v) in enumerate([("eye_colour", b"crimson"), ("hair_colour", b"golden"), ("expression", b"warm smile")]):
        f = b[5 + 32 * i: 5 + 32 * (i + 1)]
        record = bytes([CUE]) + k[p] + bytes([SEP]) + v
        assert f[:2] == bytes([DATA]) + b"a" and f[-1] == END
        assert f[2:-1] == record + b" " * (FR.BODY - len(record))
    seq = bytes([DOC]) + b                                   # the extractor puts the document byte in front
    assert seq[at["pointer"]] == SEP and seq[at["content"]] == END
    assert seq[at["value"][0]: at["value"][1] + 1] == b"warm smile" and at["val"] == "warm smile"
    assert at["key"] == k["expression"].decode()
