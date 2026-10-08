"""The dual-extraction read on the CPU: the token rows (Anima's Qwen3 tokenizer from the fork's configs: the spelling, the closing
bytes, the round trip), the arithmetic on planted geometry (a planted rotation reads as aligned and unrelated spaces near zero, the
triangle closes when the rotations compose and opens through an unrelated middle, the context read sees shared context beyond the
token's identity, the class directions follow a planted rotation), the Gram coordinates, her prefix pass on the library's tiny
test trunk against the extractor, and the comparisons and decisions end to end on synthetic sets. Seconds, no download."""
import os

import pytest
import torch
import torch.nn.functional as F

from alephllm_diffusion import settings
from alephllm_diffusion.beatrix import extract as bx
from alephllm_diffusion.beatrix import gauges as G
from alephllm_diffusion.hubs import read as H
from alephllm_diffusion.triangulate import align as AL
from alephllm_diffusion.triangulate import read as TR
from alephllm_diffusion.triangulate import rows as RW

QWEN_DIR = os.path.join(settings.DP, "configs", "qwen3_06b")
F64 = torch.float64


@pytest.fixture(scope="module")
def tok():
    if not os.path.isdir(QWEN_DIR):
        pytest.skip("the diffusion-pipe fork's Qwen3 tokenizer is not checked out")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN_DIR, local_files_only=True)


@pytest.fixture(scope="module")
def tiny():
    from geolip.alephllm.model.alephlm import AlephLM
    from geolip.alephllm.tests.test_arm_mount import TINY
    torch.manual_seed(0)
    return AlephLM(TINY).eval()


@pytest.fixture(autouse=True)
def cpu_gauges(monkeypatch):
    monkeypatch.setattr(G, "dev", "cpu")


# ---------------------------------------------------------------------------------------------------- the rows
def test_the_byte_map_is_gpt2s():
    m = RW.char_to_byte()
    assert sorted(m.values()) == list(range(256)) and m["Ġ"] == 0x20 and m["Ċ"] == 0x0A and m["a"] == ord("a")


def _old_caption(tok, text):
    """the instrument's own renderer before it moved to the library's (alephllm-diffusion-experiments 9d57056-b1fff88), kept
    verbatim as the reference the library's must equal byte for byte"""
    from itertools import accumulate

    def c2b():
        bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
              + list(range(ord("\xae"), ord("\xff") + 1)))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {chr(c): b for c, b in zip(cs, bs)}
    m = c2b()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pieces = tok.convert_ids_to_tokens(ids)
    expansions = [bytes(m[c] for c in p) for p in pieces]
    raw = text.encode("utf-8")
    assert b"".join(expansions) == raw
    spellings = [p.encode("utf-8") for p in pieces]
    return (tuple(ids), raw, b"".join(spellings), tuple(accumulate(len(e) for e in expansions)),
            tuple(accumulate(len(s) for s in spellings)))


def _same(tok, text):
    c = RW.caption(tok, text)
    ids, raw, spelled, a_end, b_end = _old_caption(tok, text)
    return (tuple(c.ids), c.raw, c.spelled, tuple(c.a_end), tuple(c.b_end)) == (ids, raw, spelled, a_end, b_end) and all(
        RW.prefixes(c, t) == (raw[: a_end[t] + 1], spelled[: b_end[t] + 1]) for t in RW.token_positions(c))


def test_the_librarys_renderer_is_the_instruments_byte_for_byte(tok):
    from geolip_anima_trainer import anima_experiments as ax
    from alephllm_diffusion.mount import read as MR
    texts = ["a taco on a plate.", "café crème", "A clock hangs in a bathroom.\nTwo dogs run on a beach.",
             "naïve façade — 東京, 3½ apples"] + [ax.PREFIX + MR.PREFIX_FRAME.replace("{w}", p) for _, _, p in H.ROWS]
    assert all(_same(tok, t) for t in texts)


def test_the_librarys_renderer_on_both_caption_draws(tok):
    """the dual-extraction read's two COCO draws (all 2 x 2,048 captions) when the hub read's data is on this machine"""
    import json
    p = os.path.join(settings.DATA_DIR, "coco_caps_2x2048.json")
    if not os.path.exists(p):
        pytest.skip("the caption draws are not here (settings.DATA_DIR)")
    caps = json.load(open(p, encoding="utf-8"))
    bad = [(d, i) for d in ("draw1", "draw2") for i, t in enumerate(caps[d]) if not _same(tok, t)]
    assert sum(len(caps[d]) for d in ("draw1", "draw2")) == 4096 and bad == []


def test_a_caption_spelled_and_closed(tok):
    c = RW.caption(tok, "a taco on a plate.")
    assert c.raw == b"a taco on a plate." and len(c.ids) == 6
    assert tok.convert_ids_to_tokens(c.ids[1]) == "Ġtaco"
    assert c.spelled == "aĠtacoĠonĠaĠplate.".encode("utf-8") and len(c.spelled) == 22
    assert RW.token_positions(c) == [1, 2, 3, 4]
    a, b = RW.prefixes(c, 1)                      # ' taco': A closes at the space before ' on', B at the first byte of 'Ġon'
    assert a == b"a taco " and b == "aĠtaco".encode("utf-8") + b"\xc4"


def test_the_round_trip_holds_beyond_ascii(tok):
    c = RW.caption(tok, "café crème")
    assert c.a_end[-1] == len(c.raw) and c.b_end[-1] == len(c.spelled) and len(c.spelled) > len(c.raw)


def test_a_phrase_ends_on_its_own_token(tok):
    c = RW.caption(tok, "an illustration of a quiet street, gloomy and downbeat.")
    t = RW.phrase_end_token(c, len(c.raw) - 1)
    assert tok.convert_ids_to_tokens(c.ids[t]) == "beat" and t == len(c.ids) - 2
    a, _ = RW.prefixes(c, t)
    assert a.endswith(b"downbeat.")


def test_every_mood_row_has_its_phrase_token(tok):
    from geolip_anima_trainer import anima_experiments as ax
    from alephllm_diffusion.mount import read as MR
    mood = [ax.PREFIX + MR.PREFIX_FRAME.replace("{w}", p) for _, _, p in H.ROWS]
    D = TR.draw_rows(tok, ["A clock hangs in a bathroom.", "Two dogs run on a beach."], mood)
    assert int(D["mood"].sum()) == 74 and len(D["cls"]) == 74
    for (i, t), m in zip(D["rows"], D["mood"].tolist()):
        if m:
            assert D["caps"][i].raw[D["caps"][i].a_end[t]:] == b"."      # the phrase's last token closes on the full stop


# ---------------------------------------------------------------------------------------------------- the arithmetic
def _planted(n=600, d=40, caps=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g, dtype=F64) * torch.linspace(3, 0.5, d, dtype=F64)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=F64))
    return X, Q, torch.arange(n) % caps, g


def test_folds_split_by_caption():
    groups = torch.arange(100) % 10
    (fa, ha), (fb, hb) = AL.folds(groups)
    assert torch.equal(fa, hb) and torch.equal(ha, fb) and int(fa.sum()) == 50
    assert not set(groups[fa].tolist()) & set(groups[ha].tolist())


def test_a_planted_rotation_aligns_and_an_unrelated_space_does_not():
    X, Q, groups, g = _planted()
    r = AL.compare(X, X @ Q + 0.01 * torch.randn(X.shape, generator=g, dtype=F64), groups, 16)
    assert r["alignment"] > 0.95 and r["ridge_r2"] > 0.9
    r0 = AL.compare(X, torch.randn(X.shape, generator=g, dtype=F64), groups, 16)
    assert abs(r0["alignment"]) < 0.15 and r0["ridge_r2"] < 0.05


def test_the_triangle_closes_when_the_rotations_compose():
    X, Q, groups, g = _planted()
    Q2, _ = torch.linalg.qr(torch.randn(40, 40, generator=g, dtype=F64))
    Y = X @ Q
    t = AL.triangle(X, Y, Y @ Q2, groups, 16)
    assert t["direct"] > 0.95 and abs(t["gap"]) < 0.02
    t2 = AL.triangle(X, torch.randn(X.shape, generator=g, dtype=F64), Y @ Q2, groups, 16)
    assert t2["direct"] > 0.95 and t2["gap"] > 0.5


def test_the_context_read_sees_shared_context_beyond_identity():
    g = torch.Generator().manual_seed(1)
    n, d = 1200, 24
    ids = torch.randint(0, 40, (n,), generator=g)
    E1, E2 = 5 * torch.randn(40, d, generator=g, dtype=F64), 5 * torch.randn(40, d, generator=g, dtype=F64)
    ctx, other = torch.randn(n, 8, generator=g, dtype=F64), torch.randn(n, 8, generator=g, dtype=F64)
    A, B = torch.randn(8, d, generator=g, dtype=F64), torch.randn(8, d, generator=g, dtype=F64)
    noise = 0.01 * torch.randn(n, d, generator=g, dtype=F64)
    groups = torch.arange(n) % 100
    shared = AL.compare(E1[ids] + ctx @ A + noise, E2[ids] + ctx @ B + noise, groups, 8, ids=ids, ridge=False)
    apart = AL.compare(E1[ids] + ctx @ A + noise, E2[ids] + other @ B + noise, groups, 8, ids=ids, ridge=False)
    assert shared["context_alignment"] > 0.9 and abs(apart["context_alignment"]) < 0.15
    flat = AL.compare(E1[ids] + ctx @ A, E2[ids], groups, 8, ids=ids, ridge=False)   # identity only on one side: nothing to align
    assert flat["context_alignment"] == 0.0


def test_the_directions_follow_a_planted_rotation():
    X, Q, groups, g = _planted(d=20)
    u, w = torch.zeros(20, dtype=F64), torch.zeros(20, dtype=F64)
    u[0], w[1] = 1.0, 1.0
    cls = ["up"] * 10 + ["down"] * 10 + ["neutral"] * 6
    split = (["train"] * 6 + ["heldout"] * 4) * 2 + ["train"] * 4 + ["heldout"] * 2
    centre = {"up": 3 * u, "down": -3 * u + 2 * w, "neutral": torch.zeros(20, dtype=F64)}
    S = torch.stack([centre[c] for c in cls]) + 0.1 * torch.randn(len(cls), 20, generator=g, dtype=F64)
    r = AL.directions(X, X @ Q, S, S @ Q, cls, split, 20)
    assert r["cos_up"] > 0.95 and r["cos_down"] > 0.95 and r["cos_axis"] > 0.95 and r["cos_shared"] > 0.95
    assert r["mapped_up"] == (4, 4) and r["mapped_down"] == (4, 4) and r["q_own_down"] == (4, 4)
    # her class rows sit off the caption rows' mean by a shift Qwen's do not share: the rotation cannot carry the shift, so the
    # held-out read starts from her own mapped centre and still finds every side
    shift = 6 * torch.randn(20, generator=g, dtype=F64)
    r2 = AL.directions(X, X @ Q, S + shift, S @ Q, cls, split, 20)
    assert r2["mapped_up"] == (4, 4) and r2["mapped_down"] == (4, 4) and r2["cos_axis"] > 0.95


def test_gram_coordinates_are_an_isometry():
    g = torch.Generator().manual_seed(2)
    W = torch.randn(50, 500, generator=g, dtype=F64)
    Wc = W - W.mean(0)
    C, share = AL.gram_coordinates(Wc @ Wc.T, 50)
    assert torch.allclose(C @ C.T, Wc @ Wc.T, atol=1e-8) and share > 0.999999
    Y = W @ torch.randn(500, 10, generator=g, dtype=F64)
    groups = torch.arange(50) % 10
    a = AL.compare(W, Y, groups, 5, ridge=False)["alignment"]
    b = AL.compare(C, Y, groups, 5, ridge=False)["alignment"]
    assert abs(a - b) < 1e-6


def test_the_centred_gram_matches_the_dense_form():
    X = torch.randn(70, 300, generator=torch.Generator().manual_seed(3)).half()
    Xc = X.float() - X.float().mean(0)
    assert float((TR.centred_gram(X, chunk=16) - (Xc @ Xc.T).double()).abs().max()) < 1e-3


# ---------------------------------------------------------------------------------------------------- her prefix pass
def test_her_rows_equal_the_extractor(tiny):
    texts = ["a cat.", "two dogs", "a red bus on a hill.", "an owl", "x"]
    S = TR.her_rows(tiny, [t.encode() for t in texts], "cpu", [0, 1, 2], [1, 2], per_pass=1, r=4)
    ex = bx.extract(tiny, texts, [0, 1, 2], taps=("last", "bb"), device="cpu", amp=False)
    d = tiny.nf.weight.numel()
    for b in (0, 1, 2):
        assert torch.allclose(S["stream"][b], F.layer_norm(ex["last"][b], (d,)), atol=1e-5)
    for b in (1, 2):
        C, _ = AL.gram_coordinates(TR.centred_gram(ex["bb"][b]), 4)
        H_ = S["hub"][b].double()
        assert torch.allclose(H_ @ H_.T, C @ C.T, atol=1e-3)
    assert torch.isfinite(S["size"]).all() and (S["size"] > 0).all()


# ---------------------------------------------------------------------------------------------------- end to end, synthetic
def _synthetic_draw(seed, untrained=True):
    g = torch.Generator().manual_seed(seed)
    n_cap, n_mood, d = 400, 74, 12
    n = n_cap + n_mood
    D = {"group": torch.cat([torch.arange(n_cap) // 10, 1000 + torch.arange(n_mood)]),
         "ids": torch.cat([torch.randint(0, 30, (n_cap,), generator=g), torch.randint(0, 30, (n_mood,), generator=g)]),
         "mood": torch.arange(n) >= n_cap, "cls": [c for c, _, _ in H.ROWS], "split": [s for _, s, _ in H.ROWS]}
    base = torch.randn(n, d, generator=g)
    Q = {0: torch.randn(n, d, generator=g), 16: base + 0.5 * torch.randn(n, d, generator=g), "28pre": 3 * base,
         "28post": base}
    rot = torch.linalg.qr(torch.randn(d, d, generator=g))[0]

    def her(scale):
        return {"stream": {16: base @ rot + scale * torch.randn(n, d, generator=g), 18: torch.randn(n, d, generator=g)},
                "hub": {18: base @ rot.T + scale * torch.randn(n, d, generator=g), 22: torch.randn(n, d, generator=g)},
                "size": torch.rand(n, generator=g), "share": {18: 1.0, 22: 1.0}}
    sets = {("trained", "A"): her(0.3), ("trained", "B"): her(0.5)}
    if untrained:
        sets[("untrained", "A")], sets[("untrained", "B")] = her(5.0), her(5.0)
    return D, Q, sets


def test_the_comparisons_and_decisions_end_to_end():
    D1, Q1, s1 = _synthetic_draw(4)
    D2, Q2, s2 = _synthetic_draw(5, untrained=False)
    R1, R2 = TR.gauges(s1, Q1, D1, 6), TR.gauges(s2, Q2, D2, 6)
    dec = TR.decide(R1, R2)
    assert set(dec) == {"D1", "D2", "D3", "D4", "D5", "D6", "PICK"}
    assert dec["D5"]["A-trunk-16"]["verdict"] == "LEARNED"
    assert all(v["verdict"] in ("LEARNED", "NOT LEARNED") for v in dec["D5"].values())
    assert dec["D2"]["A|alignment"]["trunk"][0] == "trained|A-trunk-16"
    assert set(dec["D6"]["A-trunk-16"]) == {"axis", "shared", "up", "down"}
    L = {"_meta": {"step": 0, "k": 6}, "draw1": R1, "draw2": R2, "decisions": dec}
    md = TR.report(L)
    assert "THE PICK" in md and "| A-hub-18 |" in md


def test_the_experiment_page_from_a_ledger(tmp_path):
    import json
    from alephllm_diffusion.triangulate import experiment as EX
    D1, Q1, s1 = _synthetic_draw(4)
    D2, Q2, s2 = _synthetic_draw(5, untrained=False)
    R1, R2 = TR.gauges(s1, Q1, D1, 6), TR.gauges(s2, Q2, D2, 6)
    meta = {"step": 245674, "k": 6, "r_hub": 12, "n_caps": 40, "stream_blocks": [16, 18], "hub_blocks": [18, 22],
            "rows": {"draw1": {"captions": 400, "mood": 74}, "draw2": {"captions": 400, "mood": 74}}, "commit": "abcdef1234",
            "anchor": {"block": 18, "rows": 10, "max_abs_diff": 1e-5}, "outliers": {"qwen|draw1": 0},
            "shares": {"trained|A|draw1": {"18": 0.99, "22": 0.98}}, "started_utc": "2026-10-07 05:51:00", "seconds": 1200.0,
            "finished_utc": "2026-10-07 06:12:00"}
    L = json.loads(json.dumps({"_meta": meta, "draw1": R1, "draw2": R2, "decisions": TR.decide(R1, R2)}))
    path = tmp_path / "triangulation_step245674.json"
    path.write_text(json.dumps(L), encoding="utf-8")
    mt = EX.write(str(path), str(tmp_path / "mirror"))
    folder = tmp_path / "mirror" / "experiments" / EX.EXPERIMENT_ID
    assert {"README.md", "meta.json", "result.json"} <= {p.name for p in folder.iterdir()}
    page = (folder / "README.md").read_text(encoding="utf-8")
    assert EX.EXPERIMENT_ID.startswith("e038_") and "THE PICK" in page and "aĠtacoĠonĠaĠplate." in page
    assert mt["id"] == EX.EXPERIMENT_ID and mt["status"] == "done" and mt["seconds"] == 1200 and mt["summary"]
    L["_meta"]["smoke"] = True
    path.write_text(json.dumps(L), encoding="utf-8")
    with pytest.raises(ValueError):
        EX.write(str(path), str(tmp_path / "mirror2"))
