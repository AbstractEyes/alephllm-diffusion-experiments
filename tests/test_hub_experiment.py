"""hubs.experiment on a small made-up ledger (CPU, no network): the folder the Anima experiments repo takes (README, meta.json
with the fields its index reads, the ledger, the tables, the small run), the summary from the decision and part A, the arms the
decision set, and the refusals (a small run's ledger, an unfinished read)."""
import json

import pytest

from alephllm_diffusion.hubs import experiment as ex

BLOCKS = [14, 22]
REFS = ("clip_b32", "adapter_pool")


def _tap(o, sp, cka, r2):
    return {"clip_b32": {"overlap10": o, "spearman": sp, "cka_deb": 0.5},
            "adapter_pool": {"overlap10": 0.1, "spearman": 0.1, "cka_deb": cka, "r2_rep2ref": r2}, "pr": 1.0}


def _row(sep, bal, loo, held, ha):
    return {"separation": sep, "balance": bal, "loo_signs": [loo, 60], "heldout_signs": [held, 8], "heldout_a": ha}


def ledger(smoke=False, decided=True):
    held = {"merry and glad": 1.0, "elated": 0.8, "jubilant and gleeful": 1.1, "blissful": 0.5, "mournful": -0.9,
            "forlorn and desolate": -1.0, "dismal": -1.2, "despairing and grim": -0.8, "typical": 0.0, "unremarkable": 0.1}
    A0 = {}
    for b in BLOCKS:
        A0[f"bb@L{b}"] = _tap(0.40 if b == 14 else 0.33, 0.45, 0.82, 0.63)
        A0[f"pool@L{b}"] = _tap(0.32, 0.30, 0.80, 0.56)
        A0[f"last@L{b}"] = _tap(0.11, 0.05, 0.40, 0.30)
    rows = {"close/stream/18": _row(2.2, 0.41, 56, 8, held), "close/hub/22": _row(1.8, 0.13, 55, 8, held),
            "close/both/18": _row(2.0, 0.33, 57, 8, held)}
    L = {"_meta": {"step": 245674, "sweep": BLOCKS, "n_captions": 64, "made": "2026-10-07 13:35:04 PDT", "smoke": smoke},
         "A": {"M0": A0, "M0_draw2": json.loads(json.dumps(A0)), "U0": A0, "M9_A": A0},
         "B": {"M0": rows, "U0": {k: _row(0.2, 0.3, 40, 5, held) for k in rows}}}
    if decided:
        ch = {"key": "close/stream/18", "kind": "stream", "sep": 2.2, "bal": 0.41, "cka": 0.43, "eligible": True}
        hub = {"key": "close/hub/22", "kind": "hub", "sep": 1.8, "bal": 0.13, "cka": 0.81, "eligible": True}
        L["decision"] = {"verdict": "STREAM BETTER", "chosen": ch, "best_hub": hub, "best_stream": ch, "candidates": [ch, hub]}
        L["anchor_smoke"] = {"trained": {"max_rel": 6e-6}, "random": {"max_rel": 9e-7}}
    return L


def _write(tmp_path, L, name="hub_read.json"):
    p = tmp_path / name
    p.write_text(json.dumps(L), encoding="utf-8")
    return str(p)


def test_the_folder_the_experiments_repo_takes(tmp_path):
    led = _write(tmp_path, ledger())
    (tmp_path / "hub_read.md").write_text("# The hub read (made up)\n", encoding="utf-8")     # the read's tables beside it
    small = _write(tmp_path, ledger(smoke=True), "hub_read_smoke.json")
    mt = ex.write(led, str(tmp_path / "mirror"), small=small, seconds=6094, finished_utc="2026-10-07 22:16")
    folder = tmp_path / "mirror" / "experiments" / ex.EXPERIMENT_ID
    assert sorted(p.name for p in folder.iterdir()) == sorted(ex.FILES)
    assert ex.EXPERIMENT_ID == "e030_beatrix_hub_read"
    on_disk = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(mt))
    for k in ("id", "title", "date", "kind", "status", "summary", "recipe", "result", "seconds", "finished_utc"):
        assert k in mt, k                                     # the fields the repo index and its README read
    assert mt["status"] == "done" and mt["date"] == "2026-10-07" and mt["seconds"] == 6094
    assert mt["summary"].startswith("STREAM BETTER: her stream at the phrase's closing full stop, block 18 (separation 2.20")
    assert "the blackboards lead on caption geometry (CLIP-B/32 overlap 0.40 against the pooled stream's 0.32 at block 14" \
        in mt["summary"]
    assert mt["result"]["block_leads"] == {"clip": 2, "ridge": 2, "cka": 2}
    assert mt["result"]["arms"] == [a.id for a in ex.hub_arms()] and len(mt["result"]["arms"]) == 7
    assert json.loads((folder / "result.json").read_text(encoding="utf-8")) == ledger()
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(f"# {ex.EXPERIMENT_ID}: ")
    for s in ("## Question", "## Design", "## Recipe", "## The rule (fixed before the run)", "## Result", "## What it set",
              "## Files", "**STREAM BETTER**", "| +2.20; 56; 8; 0.41 |", "the untrained trunk", "6.0e-06 (her step 212,000"):
        assert s in readme, s
    assert all(a.id in readme for a in ex.hub_arms())
    assert (folder / "tables.md").read_text(encoding="utf-8").startswith("# The hub read")


def test_the_page_names_readings_in_words():
    assert ex.words("close/stream/18") == "her stream at the phrase's closing full stop, block 18"
    assert ex.words("last/hub/rec") == "the hub's blackboard after the phrase alone, blocks 16, 18, 21 and 24 together"
    assert set(ex.READINGS) == {"close/stream/18", "close/hub/22", "close/both/18"}


def test_a_small_run_or_an_unfinished_read_is_refused(tmp_path):
    with pytest.raises(ValueError, match="small run"):
        ex.write(_write(tmp_path, ledger(smoke=True)), str(tmp_path / "m"))
    with pytest.raises(ValueError, match="no decision"):
        ex.write(_write(tmp_path, ledger(decided=False)), str(tmp_path / "m"))
