"""The read of record on a surface arm (triangulate.arm_read), on the CPU without a model: the reader's sites reduced to e038's rows,
the row check, and the registered decisions (B3, B4, the slider's block) on hand-made gauges."""
import pytest
import torch

from alephllm_diffusion.triangulate import arm_read as AR


def _D(rows, ids, n_mood):
    n = len(rows)
    return {"rows": rows, "ids": torch.tensor(ids), "mood": torch.arange(n) >= n - n_mood}


def test_row_index_keeps_every_caption_site_and_each_mood_texts_last():
    # texts 0-1 captions, 2-3 mood texts (each with three sites); the reader lays the sites out text by text
    table = [(0, 1, 11), (0, 2, 12), (1, 1, 21), (2, 1, 31), (2, 2, 32), (2, 3, 33), (3, 1, 41), (3, 2, 42)]
    idx = AR.row_index(table, n_caps=2, n_texts=4)
    assert idx.tolist() == [0, 1, 2, 5, 7]
    with pytest.raises(AssertionError, match="gave no site"):
        AR.row_index(table[:6], n_caps=2, n_texts=4)          # text 3 has no site


def test_check_rows_wants_e038s_rows_and_ids_in_order():
    D = _D([(0, 1), (0, 2), (1, 3)], [11, 12, 33], 1)
    AR.check_rows([(0, 1, 11), (0, 2, 12), (1, 3, 33)], D)
    with pytest.raises(AssertionError, match="token ids"):
        AR.check_rows([(0, 1, 11), (0, 2, 12), (1, 3, 34)], D)
    with pytest.raises(AssertionError, match="not e038's rows"):
        AR.check_rows([(0, 2, 11), (0, 1, 12), (1, 3, 33)], D)


def test_read_set_reduces_the_readers_sites_to_e038s_rows(monkeypatch):
    from geolip.alephllm.train import surface as SF
    table = torch.tensor([(0, 1, 11), (0, 2, 12), (1, 1, 31), (1, 2, 32)])          # one caption, one mood text
    states = {20: torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)}
    seen = {}

    def fake(model, tok, texts, blocks, surface="B", amp=True, **kw):
        seen.update(surface=surface, amp=amp, kw=kw)
        return {"states": states, "sites": table}
    monkeypatch.setattr(SF, "read_spelled", fake)
    D = _D([(0, 1), (0, 2), (1, 2)], [11, 12, 32], 1)
    out = AR.read_set(None, None, ["a caption", "a mood text."], D, "B", [20], {"convention": "gpt2"})
    assert seen == {"surface": "B", "amp": False, "kw": {"convention": "gpt2"}}       # fp32, the arm's own reader arguments
    assert torch.equal(out[20], states[20][[0, 1, 3]])


def _R(armed24, plain24, triangle_gaps, axis, plain_axis, untrained_axis, gloomy_right):
    C1 = {f"B8q|{b}": {"Q": {"alignment": 0.5 + 0.001 * b}} for b in AR.SLIDER_BAND}
    C1[f"B8q|{AR.B3_BLOCK}"] = {"Q": {"alignment": armed24}}
    C1[f"A8|{AR.B3_BLOCK}"] = {"Q": {"alignment": plain24}}
    C6 = {}
    for b in AR.SLIDER_BAND:
        C6[f"B8q|{b}"] = {"cos_axis": axis[b], "mapped_down": [gloomy_right[b], 4]}
        C6[f"A8|{b}"] = {"cos_axis": plain_axis}
        C6[f"UB0|{b}"] = {"cos_axis": untrained_axis}
    tri = {f"A8->B8q->Q|{i}": {"gap": g} for i, g in enumerate(triangle_gaps)}
    tri["A8->B8->Q|0"] = {"gap": 0.5}                                     # the before-arm path is not a detour of the arm
    return {"C1": C1, "C6": C6, "triangles": tri, "pairs": {}}


def test_the_registered_decisions():
    band = AR.SLIDER_BAND
    good = _R(0.55, 0.56, [0.01, -0.02], {b: 0.45 for b in band}, 0.5, 0.1, {b: 4 for b in band})
    dec = AR.decide(good, good, list(band) + [AR.B3_BLOCK])
    assert dec["B3"]["draw1"]["verdict"] == "MET" and dec["B3"]["draw1"]["paths"] == 2
    assert dec["B3"]["draw1"]["detours_within"] == 2
    assert dec["B4"]["draw1"]["verdict"] == "MET"                        # .45 >= half of .50 at every block of the band
    assert dec["slider_block"]["block"] == AR.SLIDER_BLOCK
    low = _R(0.49, 0.56, [0.20], {b: 0.2 for b in band}, 0.5, 0.1, {b: 4 for b in band})
    d2 = AR.decide(low, low, list(band) + [AR.B3_BLOCK])
    assert d2["B3"]["draw1"]["verdict"] == "NOT MET" and d2["B3"]["draw1"]["detours_within"] == 0
    assert d2["B4"]["draw1"]["verdict"] == "NOT MET"                     # .2 < half of .5
    # block 20 fails on both draws (too few unseen gloomy phrases); 16 and 18 pass; the higher alignment to Q wins (18)
    right = {12: 2, 14: 2, 16: 4, 18: 3, 20: 2}
    moved = _R(0.55, 0.56, [0.0], {b: 0.45 for b in band}, 0.5, 0.1, right)
    d3 = AR.decide(moved, moved, list(band) + [AR.B3_BLOCK])
    assert d3["slider_block"]["block"] == 18
    # one draw passing at block 20 keeps the registered block
    other = _R(0.55, 0.56, [0.0], {b: 0.45 for b in band}, 0.5, 0.1, {b: 4 for b in band})
    assert AR.decide(moved, other, list(band) + [AR.B3_BLOCK])["slider_block"]["block"] == AR.SLIDER_BLOCK
    none = _R(0.55, 0.56, [0.0], {b: 0.15 for b in band}, 0.5, 0.1, {b: 4 for b in band})   # axis gain .05: no block passes
    d4 = AR.decide(none, none, list(band) + [AR.B3_BLOCK])
    assert d4["slider_block"]["block"] is None and "waits" in d4["slider_block"]["why"]


def _ledger(smoke=False):
    """a full ledger's shape with made-up numbers: every set, pair and triangle at two blocks of the band and block 24"""
    blocks = [AR.SLIDER_BLOCK, AR.B3_BLOCK]
    names = [n for n in AR.SETS if n != "UBq"]

    def draw(shift):
        R = {"C1": {}, "C6": {}, "pairs": {}, "triangles": {}}
        for i, n in enumerate(names):
            for b in blocks:
                R["C1"][f"{n}|{b}"] = {"Q": {"alignment": 0.3 + 0.03 * i + shift}}
                R["C6"][f"{n}|{b}"] = {"cos_axis": 0.4 - 0.04 * i, "mapped_down": [3, 4]}
        for p in AR.PAIRS:
            for b in blocks:
                R["pairs"][f"{p}|{b}"] = {"alignment": 0.5 if p == "gap before" else 0.96}
        for t in AR.TRIANGLES:
            for b in blocks:
                g = 0.15 if t.startswith("A8->B8->") else 0.005
                R["triangles"][f"{t}|{b}"] = {"direct": 0.5, "composed": 0.5 - g, "gap": g}
        return R
    L = {"_meta": {"surface": "qwen3", "registry_row": {"file": "surface/qwen3/x/s10_qwen.safetensors", "member": "s10_qwen",
                                                         "base": ["gXA", 8]},
                   "step": 245674, "k": 128, "n_caps": 512, "blocks": blocks, "smoke": smoke,
                   "rows": {"draw1": {"captions": 4920, "mood": 74}, "draw2": {"captions": 5086, "mood": 74}},
                   "anchor_against_e038": {"A0": 9e-5, "B0": 1e-4}, "seconds": 788.0},
         "draw1": draw(0.0), "draw2": draw(0.002)}
    L["decisions"] = AR.decide(L["draw1"], L["draw2"], blocks)
    return L


def test_the_reads_page_comes_from_its_ledger(tmp_path):
    import json
    from alephllm_diffusion.triangulate import arm_experiment as AE
    from geolip_anima_trainer import anima_experiments as ax
    assert AE.EXPERIMENT_ID == ax.ARM_READ_ID and AE.EXPERIMENT_ID.startswith("e045_")
    L = _ledger()
    p = tmp_path / "arm_read_qwen3_step245674.json"
    p.write_text(json.dumps(L), encoding="utf-8")
    mt = AE.write(str(p), str(tmp_path / "mirror"), code="the code at commit abc1234")
    folder = tmp_path / "mirror" / "experiments" / AE.EXPERIMENT_ID
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(f"# {AE.EXPERIMENT_ID}: ") and "The slider's block: 20" in readme and "abc1234" in readme
    assert f"**{L['decisions']['B3']['draw1']['verdict']}**" in readme and f"**{L['decisions']['B4']['draw2']['verdict']}**" in readme
    assert json.loads((folder / "result.json").read_text(encoding="utf-8")) == json.loads(json.dumps(L))   # the ledger itself
    assert mt["kind"] == "beatrix_read" and mt["result"]["decisions"]["slider_block"]["block"] == AR.SLIDER_BLOCK
    for bad, msg in ((_ledger(smoke=True), "small run"), ({k: v for k, v in L.items() if k != "decisions"}, "no decisions")):
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            AE.write(str(p), str(tmp_path / "mirror2"))
