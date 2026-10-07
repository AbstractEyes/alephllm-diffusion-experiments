"""Stage 1's padded batches, on the CPU with the library's tiny test trunk (3 blocks, every block a hub with 16-byte chunks, so
captions cross chunk boundaries): the padded form gives every caption the states the equal-length form gives it (the trunk is
causal: the padding after a caption never reaches its positions), batches fill a byte budget in length order, and the
blackboard tap (a sum over every position) is refused in the padded form. Seconds, no download."""
import pytest
import torch

from alephllm_diffusion.beatrix import extract as bx

TEXTS = ["a cat.", "a red bus on a hill.", "two dogs", "x", "a long caption that crosses three hub chunks of sixteen.",
         "a cat?", "snow over the quiet harbour at dusk", "an owl"]
LAYERS = [-1, 0, 1, 2, 32]


@pytest.fixture(scope="module")
def tiny():
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.alephllm.tests.test_arm_mount import TINY
    torch.manual_seed(0)
    return AlephLM(TINY).eval()


def test_padded_form_gives_every_caption_its_own_states(tiny):
    eq = bx.extract(tiny, TEXTS, LAYERS, taps=("pool", "last", "byte"), device="cpu", amp=False)
    pd = bx.extract(tiny, TEXTS, LAYERS, taps=("pool", "last", "byte"), device="cpu", amp=False, pad=True, max_tokens=96,
                    pad_max_batch=3)
    for l in LAYERS:
        for t in ("pool", "last"):
            assert torch.allclose(pd[t][l], eq[t][l], rtol=1e-5, atol=1e-6), (t, l, float((pd[t][l] - eq[t][l]).abs().max()))
        for a, b in zip(pd["byte"][l], eq["byte"][l]):
            assert a.shape == b.shape and torch.allclose(a, b, rtol=1e-5, atol=1e-6), (l, a.shape, b.shape)


def test_padded_groups_fill_the_budget_in_length_order():
    lens = [5, 50, 7, 7, 30, 12, 3, 49]
    g = bx.padded_groups(lens, max_tokens=100, max_batch=3)
    assert sorted(i for grp in g for i in grp) == list(range(len(lens)))
    assert [lens[i] for grp in g for i in grp] == sorted(lens)
    for grp in g:
        assert len(grp) <= 3 and max(lens[i] for i in grp) * len(grp) <= 100
    assert [[lens[i] for i in grp] for grp in g] == [[3, 5, 7], [7, 12, 30], [49, 50]]
    assert bx.padded_groups([500, 2], max_tokens=100) == [[1], [0]]        # a caption over the budget gets a batch of its own


def test_padded_form_refuses_the_blackboard(tiny):
    with pytest.raises(AssertionError):
        bx.extract(tiny, TEXTS[:2], [0], taps=("bb",), device="cpu", amp=False, pad=True)
