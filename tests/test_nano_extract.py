"""The nano extractor where the nano trainer's library is installed (its own environment; skipped elsewhere): every hub's
blackboard reproduces the hub's own output and a byte's state equals a prefix pass's, on each kind of body, untrained, on the CPU.
Seconds, no download."""
import pytest
import torch


@pytest.fixture(scope="module")
def nano_lib():
    pytest.importorskip("geolip.nano_beatrix.nano_v1", reason="the nano trainer's library is not installed here")


@pytest.mark.parametrize("body,names", [("castor", set()), ("pollux-canon", {"0", "2", "4"}),
                                        ("pollux-dual", {"0r", "0p", "2r", "2p", "4r", "4p"})])
def test_every_hub_reads_as_the_hub_and_a_byte_as_its_prefix(nano_lib, body, names):
    from alephllm_diffusion.nano import bodies as NB
    from alephllm_diffusion.nano import extract as NX
    torch.manual_seed(0)
    model = NB.build(body, "birth", device="cpu")
    assert set(NX.hubs(model)) == names
    texts = [b"a red bus on a quiet street.", b"two dogs run."]
    pos = [[5, 12, len(texts[0])], [4, len(texts[1])]]
    S = NX.read(model, texts, pos, device="cpu", r=5, log=lambda *_: None)
    assert S["n"] == 5 and all(v < NX.PARITY_MAX for v in S["parity"].values())
    c = NX.causality(model, texts, pos, [(0, 0), (0, 1), (1, 1)], device="cpu")
    assert max(c.values()) < 1e-4, c
