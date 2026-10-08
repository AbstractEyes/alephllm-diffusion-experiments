"""The slider features read through a surface arm (triangulate.arm_features) on the CPU: the texts and their order, the z-score
rule (hubs.read's), the library's reader on the tiny test trunk with Anima's tokenizer (every text's last site is the token the
dual-extraction read's mood rows use: the one closing at the full stop), and the file. Seconds, no download."""
import json
import os

import pytest
import torch

from alephllm_diffusion import settings
from alephllm_diffusion.hubs import read as H
from alephllm_diffusion.triangulate import arm_features as AF


@pytest.fixture(scope="module")
def tiny():
    """the library's tiny test trunk with room for a framed phrase (about 90 bytes; its own context is 64)"""
    import dataclasses
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.alephllm.tests.test_arm_mount import TINY
    torch.manual_seed(0)
    return AlephLM(dataclasses.replace(TINY, context=256)).eval()


def _have_tokenizer():
    return bool(settings.DP) and os.path.isdir(os.path.join(settings.DP, "configs", "qwen3_06b"))


def test_the_texts_the_names_and_the_frame():
    assert AF.STRINGS[:74] == H.TEXTS and set(H.REF) <= set(AF.STRINGS) and len(AF.STRINGS) == len(set(AF.STRINGS))
    assert AF.reading("qwen3", 20) == "frame/qwen3-arm/20"
    assert AF.file_name("qwen3", 20) == "mood_phrases_frame-qwen3-arm-20_mini-beatrix-3_step245674.safetensors"
    assert AF.framed("gloomy") == "masterpiece, best quality, score_7, safe, an illustration of a quiet street, gloomy."


def test_zscored_is_the_reads_rule_in_the_tables_order():
    torch.manual_seed(1)
    X = torch.randn(len(AF.STRINGS), 6) * 3 + 1
    Z = AF.zscored(X)
    ref = X[[AF.STRINGS.index(s) for s in H.REF]]
    want = ((X - ref.mean(0)) / (ref.std(0) + 1e-6)).double()[[AF.STRINGS.index(t) for t in H.TEXTS]].float()
    assert Z.shape == (74, 6) and torch.equal(Z, want)
    assert len(AF.STRINGS) == 74                                      # every reference text is one of the table's rows


@pytest.mark.skipif(not _have_tokenizer(), reason="Anima's tokenizer (the diffusion-pipe fork's configs) is not here")
def test_the_reader_closes_every_text_at_its_full_stop(tiny):
    from geolip.alephllm.train.surface import sites
    tok = AF.tokenizer()
    kw = {"convention": "gpt2", "a_text": "written", "site_fn": sites}
    for surface in ("A", "B"):
        X = AF.read_closing(tiny, tok, 1, surface, kw)           # check_rows inside: the dual-extraction read's mood rows
        assert X.shape == (len(AF.STRINGS), tiny.nf.weight.numel()) and torch.isfinite(X).all()


def test_the_file_carries_the_tensors_the_table_and_the_reading(tmp_path):
    from safetensors import safe_open
    torch.manual_seed(2)
    feats = {k: AF.zscored(torch.randn(len(AF.STRINGS), 8)) for k in AF.TENSORS}
    p = AF.write("qwen3", 20, feats, out_dir=str(tmp_path))
    assert os.path.basename(p) == AF.file_name("qwen3", 20)
    with safe_open(p, framework="pt") as f:
        m = f.metadata()
        assert sorted(f.keys()) == sorted(AF.TENSORS)
        assert [x["text"] for x in json.loads(m["phrases"])] == H.TEXTS and m["reading"] == "frame/qwen3-arm/20"
        assert set(json.loads(m["reads"])) == set(AF.TENSORS) and json.loads(m["surface"])["member"] == "s10_qwen"
        assert "untrained" in m["trunk"]
    two = {k: feats[k] for k in ("qwen", "plain")}                    # without the control: the file says so
    with safe_open(AF.write("qwen3", 20, two, out_dir=str(tmp_path / "b")), framework="pt") as f:
        assert sorted(f.keys()) == ["plain", "qwen"] and "untrained" not in f.metadata()["trunk"]
