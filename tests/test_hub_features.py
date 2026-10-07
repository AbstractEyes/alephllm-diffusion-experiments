"""The slider features file on the CPU with the library's tiny test trunk: the rows of a reading are hubs.read's part-B
features in the table's order; the 'both' rows give, through anima-trainer's own connector_axis, exactly the mean of the hub's
and the stream's slider values; the file carries the phrase table and the reading. Seconds, no download."""
import json

import pytest
import torch

from alephllm_diffusion.hubs import features as HF
from alephllm_diffusion.hubs import read as H


@pytest.fixture(scope="module")
def tiny():
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.alephllm.tests.test_arm_mount import TINY
    torch.manual_seed(0)
    return AlephLM(TINY).eval()


def test_parse():
    assert HF.parse("close/hub/18") == ("close", "hub", [18])
    assert HF.parse("last/both/rec") == ("last", "both", list(H.RECORD_BLOCKS))
    with pytest.raises(AssertionError):
        HF.parse("frame/hub/18")


def test_rows_are_the_reads_features(tiny):
    got = HF.rows_for(tiny, "cpu", "close/stream/1")
    L = H.features_b(tiny, "cpu", "close", "", [1])
    want = H.zscored({1: L[("stream", 1)]}, [1])[[H.STRINGS_B.index(t) for t in H.TEXTS]].float()
    assert got.shape == (74, tiny.nf.weight.numel()) and torch.equal(got, want)


def test_both_gives_the_mean_slider_value_through_the_connectors_axis(tiny):
    from geolip_anima_trainer.anima_runner import connector_axis
    pool = {c: [i for i, r in enumerate(H.ROWS) if r[0] == c and r[1] == "train"] for c in ("up", "down", "neutral")}

    def slider(Fm):
        p = connector_axis(Fm, pool)
        return ((Fm - p["mu"]) @ p["V"].T / p["scale"])[:, 0].double()
    a_hub = slider(HF.rows_for(tiny, "cpu", "close/hub/2"))
    a_stream = slider(HF.rows_for(tiny, "cpu", "close/stream/2"))
    a_both = slider(HF.rows_for(tiny, "cpu", "close/both/2"))
    assert torch.allclose(a_both, (a_hub + a_stream) / 2, atol=1e-4), float((a_both - (a_hub + a_stream) / 2).abs().max())


def test_the_file_carries_the_table_and_the_reading(tiny, tmp_path):
    from safetensors import safe_open
    feats = {k: HF.rows_for(tiny, "cpu", "last/stream/0") for k in HF.CONDITIONS}
    p = HF.write("last/stream/0", feats, out_dir=str(tmp_path))
    with safe_open(p, framework="pt") as f:
        m = f.metadata()
        assert sorted(f.keys()) == sorted(HF.CONDITIONS)
        assert [x["text"] for x in json.loads(m["phrases"])] == H.TEXTS and m["reading"] == "last/stream/0"
        assert set(json.loads(m["reads"])) == set(HF.CONDITIONS)
