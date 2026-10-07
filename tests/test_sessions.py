"""The session runner's bookkeeping, on the CPU: file names, grid states, markers, the reference manifest."""
import json
import os
import time

import pytest

from alephllm_diffusion import sessions, settings, storage


def test_every_step_has_its_function():
    for st in sessions.STEPS:
        assert callable(getattr(sessions._Session, f"do_{st.name}", None)), st.name
    assert {st.session for st in sessions.STEPS} == {1, 2, 3, 4}
    assert set(sessions.RESTORE) == set(sessions.SAVE) == {1, 2, 3, 4}


def test_the_hub_sliders_name_their_readings_and_arms():
    from alephllm_diffusion.hubs import features
    for r in sessions.HUB_READINGS:
        features.parse(r)                                  # every reading is one the features file can be built for
    from geolip_anima_trainer import anima_experiments as ax
    hub = [a for a in ax.CONNECTOR_ARMS if a.features in ax.HUB_FEATURES.values()]
    assert list(sessions.SLIDER_ARMS) == [a.id for a in hub]           # the trainer's hub arms, in its run order
    assert set(sessions.HUB_READINGS) == set(ax.HUB_FEATURES)            # the files session 4 builds are the ones they read
    assert [a[:4] for a in sessions.SLIDER_ARMS] == [f"e03{i}" for i in range(1, 8)]
    assert ax.HUB_READ_ID[:4] == "e030" and ax.HUB_READ_ID not in sessions.SLIDER_ARMS


def test_only_the_first_three_sessions_need_the_reference_data(monkeypatch):
    from alephllm_diffusion import storage
    s = object.__new__(sessions._Session)                 # no card, no workspace: only what reference() reads
    s.store, s.tok, s.say = False, None, lambda *a: None
    monkeypatch.setattr(storage, "missing_reference", lambda: ["s0.pt"])
    monkeypatch.setattr(storage, "check_reference", lambda: pytest.fail("session 4 checked the reference data"))
    s.reference(4)                                        # the hub sliders start without sessions 1-3's inputs
    for n in sessions.REFERENCE_SESSIONS:
        with pytest.raises(sessions.StepFailed, match="reference data missing"):
            s.reference(n)


def test_stage1_file_names_follow_the_stage1_program():
    f = sessions.stage1_file
    assert f("245674") == "s1_step245674.pt"
    assert f("random:1") == "s1_random1.pt"
    assert f("245674", mount="gCA") == "s1_step245674_gCA.pt"
    assert f("245674", mount="gCB", frame="cap") == "s1_step245674_gCB_fcap.pt"
    assert f("245674", mount="gCA", mood=True, limit=80) == "s1_mood_step245674_gCA_limit80.pt"
    assert f("random:0", mood=True) == "s1_mood_random0.pt"


def test_grid_state_and_resume_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "OUT_DIR", str(tmp_path))
    assert sessions.grid_state("record") == "absent" and sessions.resume_dir("record") == ""
    (tmp_path / "stitch_record_full.json").write_text(json.dumps({"partial": {"cells_done": 3}}))
    assert sessions.grid_state("record") == "partial"
    for name, n in (("stitch_record_cells_20261006_100000", 2), ("stitch_record_cells_20261006_110000", 5),
                    ("stitch_record_control_outside_cells_20261006_120000", 9)):
        d = tmp_path / name
        d.mkdir()
        for i in range(n):
            (d / f"{i:04d}.pt").write_bytes(b"")
    assert os.path.basename(sessions.resume_dir("record")) == "stitch_record_cells_20261006_110000"
    (tmp_path / "stitch_record_full.json").write_text(json.dumps({"cells": {}}))
    assert sessions.grid_state("record") == "done"
    (tmp_path / "stitch_smoke_m0.json").write_text(json.dumps({"cells": {}}))
    assert sessions.grid_state("smoke_m0") == "done"


def test_the_reference_manifest_covers_every_reference_file():
    man = storage.reference_manifest()
    assert set(man) == set(storage.REFERENCE)
    assert all(len(v["sha256"]) == 64 and v["bytes"] > 0 for v in man.values())


def test_progress_line_reads_cells_and_time_left(capsys):
    s = object.__new__(sessions._Session)
    s.progress(754, "[stitch s2] cell 120/612 record|step245674|b16|close|k8: ...; 3600 s spent, about 11160 s left")
    out = capsys.readouterr().out
    assert "cell 120 of 612" in out and "about 3 h 6 min left" in out and "13 min in" in out


def test_progress_line_counts_down_from_the_newest_progress_line(capsys):
    s = object.__new__(sessions._Session)
    s.progress(600, "[arms] restored ['s1_perspective']", ("[mount read 135.3s] M0 read: 1/7 conditions, about 812 s left",
                                                            time.time() - 200))
    out = capsys.readouterr().out
    assert "10 min in" in out and "about 10 min left" in out and "[arms]" not in out


@pytest.mark.parametrize("s,text", [(45, "45 s"), (600, "10 min"), (11160, "3 h 6 min")])
def test_durations(s, text):
    assert sessions._dur(s) == text


def test_restart_test_counts_nan_in_the_same_places_as_equal():
    import torch
    from alephllm_diffusion.stitch.resume_test import equal
    a = torch.tensor([1.0, float("nan"), 3.0])
    assert equal(a, a.clone()) and not torch.equal(a, a.clone())
    assert not equal(a, torch.tensor([1.0, 2.0, 3.0]))
    assert not equal(a, torch.tensor([1.0, float("nan"), 3.5]))
    assert equal(torch.tensor([1, 2]), torch.tensor([1, 2])) and not equal(torch.tensor([1, 2]), torch.tensor([1.0, 2.0]))
