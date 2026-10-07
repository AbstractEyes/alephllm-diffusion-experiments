"""The environment the experiments run in: the pins, the libraries, the fork, the card."""
import pytest

from alephllm_diffusion import environment
from alephllm_diffusion.paths import repo_root


def test_requirements_pin_the_libraries_by_commit():
    pins = environment.read_pins(repo_root() / "requirements.txt")
    assert {"geolip-alephllm", "amoe-lora", "geolip-anima-trainer"} <= set(pins["commits"])
    assert pins["versions"]["torch"].endswith("+cu128")


def test_the_libraries_import():
    import amoe
    import geolip.alephllm as al
    import geolip_anima_trainer  # noqa: F401
    from geolip.alephllm.arm_mount import mount_group  # noqa: F401
    assert al.__version__ == "0.10.7"
    assert amoe.__version__ == "0.2.11"


def test_installed_versions_match_the_pins():
    rep = environment.check(verbose=False)
    assert rep["pin_differences"] == []


def test_the_fork_is_present_at_its_pin():
    d = environment.check(verbose=False)["diffusion_pipe"]
    assert d["present"] and d["commit"] == d["pinned"] and d["comfyui_commit"]


def test_cuda_runs_a_kernel():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA card on this machine")
    assert environment.check(verbose=False)["cuda_math_ok"]
