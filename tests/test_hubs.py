"""The hub read on the CPU with the library's tiny test trunk (3 blocks, every block a hub): the blackboard Grams computed on the
device equal the gauges' path on the extractor's own rows, for any number of blocks a pass; the earlier probe's two numbers
behave as defined; the slider reads recover a planted mood axis; the decision follows its registered rule; both parts run end to
end. Seconds, no download."""
import pytest
import torch

from alephllm_diffusion.beatrix import extract as bx
from alephllm_diffusion.beatrix import gauges as G
from alephllm_diffusion.hubs import read as H

TEXTS = ["a cat.", "a red bus on a hill.", "two dogs", "x", "a long caption that crosses three hub chunks of sixteen.",
         "a cat?", "snow over the quiet harbour at dusk", "an owl", "a dog.", "two cats", "a blue van parked by a wall.", "an elk"]


@pytest.fixture(scope="module")
def tiny():
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.alephllm.tests.test_arm_mount import TINY
    torch.manual_seed(0)
    return AlephLM(TINY).eval()


@pytest.fixture(autouse=True)
def cpu_gauges(monkeypatch):
    monkeypatch.setattr(G, "dev", "cpu")


def test_equal_groups_are_the_extractors_batches():
    lens = [5, 3, 5, 7, 3, 5, 5]
    assert H.equal_groups(lens, max_batch=2) == [[1, 4], [0, 2], [5, 6], [3]]


@pytest.mark.parametrize("per_pass", [1, 2, 3])
def test_blackboard_grams_equal_the_gauges_path_on_the_extractors_rows(tiny, per_pass):
    blocks = [0, 1, 2]
    ex = bx.extract(tiny, TEXTS, blocks, taps=("bb",), device="cpu", amp=False)
    got = H.blackboard_grams(tiny, TEXTS, blocks, "cpu", per_pass)
    for b in blocks:
        K, S = got[b]
        assert torch.equal(K, G._gram_chunked(ex["bb"][b]).cpu()), b
        Xn = torch.nn.functional.normalize(ex["bb"][b].float(), dim=-1)
        assert torch.allclose(S, Xn @ Xn.T, atol=1e-6), (b, float((S - Xn @ Xn.T).abs().max()))


def test_prep_from_the_gram_is_the_gauges_prep(tiny):
    ex = bx.extract(tiny, TEXTS, [1], taps=("bb",), device="cpu", amp=False)["bb"][1]
    a = H.prepped_from_gram(G._gram_chunked(ex), "cpu")
    b = G.prep(ex)
    assert torch.allclose(a @ a.T, b @ b.T, rtol=1e-9, atol=1e-9)


def test_the_earlier_probes_numbers():
    g = torch.Generator().manual_seed(3)
    rand = torch.randn(64, 512, generator=g)
    ii, jj = torch.randint(0, 64, (100_000,), generator=g), torch.randint(0, 64, (100_000,), generator=g)
    r2, i2, j2 = H.v1_draws(64)
    assert torch.equal(rand, r2) and torch.equal(ii[ii != jj], i2) and torch.equal(jj[ii != jj], j2)
    S = H.cosine_matrix(rand)
    assert H.overlap10(H.top10(S), H.top10(S)) == 1.0
    a = torch.randn(1000, generator=g)
    assert abs(H.spearman(a, a) - 1) < 1e-12 and abs(H.spearman(a, -a) + 1) < 1e-12


def _planted(sep=2.0, noise=0.1, width=64):
    """z-scored-like rows for STRINGS_B with a planted mood direction: up rows at +sep/2, down rows at -sep/2 along e0."""
    g = torch.Generator().manual_seed(1)
    Z = torch.randn(len(H.STRINGS_B), width, generator=g, dtype=torch.float64) * noise
    cls = {p: c for c, _, p in H.ROWS}
    for i, s in enumerate(H.STRINGS_B):
        c = cls.get(s, "up" if s in H.PROBE[:30] else "down" if s in H.PROBE[30:] else None)
        if c in ("up", "down"):
            Z[i, 0] += sep / 2 if c == "up" else -sep / 2
    return Z


def test_slider_reads_recover_a_planted_axis():
    r = H.slider_reads(_planted(), P=_planted().float())
    assert r["loo_signs"] == [60, 60] and r["heldout_signs"] == [8, 8]
    assert 1.6 < r["separation"] < 2.4 and r["balance"] < 0.5 and r["probe_acc"] > 0.9


def test_the_combined_read_of_one_source_with_itself_is_that_source():
    r = H.slider_reads(_planted())
    c = H.combined_reads(r, r)
    for k in ("loo_signs", "heldout_signs", "separation", "sd_up", "sd_down", "balance", "gloomy_negative", "ful_right"):
        assert c[k] == pytest.approx(r[k]), k
    other = dict(r, a_rows=[-x for x in r["a_rows"]], a_tests=[-x for x in r["a_tests"]], loo_a=[-x for x in r["loo_a"]])
    assert H.combined_reads(r, other)["separation"] == pytest.approx(0.0, abs=1e-12)


def _cands(hub, stream):
    """part B rows with the given (separation, balance, eligible) for one hub and one stream feature."""
    row = lambda sep, bal, ok: {"separation": sep, "balance": bal, "heldout_signs": [8 if ok else 5, 8],  # noqa: E731
                                "loo_signs": [55, 60]}
    return {"last/hub/14": row(*hub), "last/stream/16": row(*stream)}


@pytest.mark.parametrize("hub,stream,verdict", [
    ((1.5, 0.2, True), (1.2, 0.2, True), "HUB BETTER"),
    ((1.2, 0.2, True), (1.5, 0.2, True), "STREAM BETTER"),
    ((1.50, 0.20, True), (1.47, 0.25, True), "TIED"),
    ((1.50, 0.10, True), (1.47, 0.40, True), "HUB BETTER"),
    ((1.5, 0.2, True), (1.2, 0.2, False), "HUB BETTER"),
    ((1.5, 0.2, False), (1.2, 0.2, False), "NONE QUALIFIES")])
def test_the_decision_follows_its_rule(hub, stream, verdict):
    d = H.decide(_cands(hub, stream), {})
    assert d["verdict"] == verdict, d


def test_both_parts_run_end_to_end(tiny):
    caps = [f"{t} {i}" for i in range(4) for t in TEXTS][:40]
    g = torch.Generator().manual_seed(2)
    refs = {"clip_b32": torch.randn(40, 16, generator=g), "t5_xxl": torch.randn(40, 48, generator=g),
            "adapter_pool": torch.randn(40, 8, generator=g)}
    _, ii, jj = H.v1_draws(40)
    R = H.Refs(refs, (ii, jj), "cpu")
    A = H.part_a(tiny, "cpu", caps, R, ridge=True, per_pass=2, blocks=[0, 1, 2], label="tiny")
    assert set(A) == {f"{t}@L{l}" for t in ("pool", "last") for l in (0, 1, 2, 32)} | {f"bb@L{l}" for l in (0, 1, 2)}
    assert "r2_rep2ref" in A["bb@L1"]["t5_xxl"] and "r2_rep2ref" not in A["bb@L1"]["clip_b32"]
    B, keep = H.part_b(tiny, "cpu", "", blocks=[0, 1, 2], forms=("last", "close"), record=(1, 2))
    assert set(B) == {f"{f}/{k}/{b}" for f in ("last", "close") for k in ("stream", "hub", "both") for b in (0, 1, 2, "rec")}
    assert keep["stream_last_rec"].shape == (74, 2 * tiny.nf.weight.numel())
    assert "probe_acc" in B["last/hub/rec"] and "probe_acc" not in B["close/hub/rec"]
    d = H.decide(B, A)
    assert d["verdict"] in ("HUB BETTER", "STREAM BETTER", "TIED", "NONE QUALIFIES")
