"""Where a nano craft's checkpoint lives: the release repo for the kept crafts, the training repo for the superseded arms moved there.
No download, milliseconds."""
from alephllm_diffusion.nano import bodies as NB


def test_a_moved_arm_resolves_to_the_training_repo_and_a_kept_craft_to_the_release_repo():
    assert NB.location("pollux-dual-canon", "cf") == (
        NB.TRAINING_REPO, "castor-pollux-v1.1/nano-pollux-dual-canon-v1.1-cf/checkpoints/step_00065237.safetensors")
    assert NB.location("castor", "fr-cf") == (
        NB.TRAINING_REPO, "castor-pollux-v1.1/nano-castor-v1.1-fr-cf/checkpoints/step_00069052.safetensors")
    assert NB.location("pollux-dual", "mx-cp") == (NB.REPO, "nano-pollux-dual-v1.1-mx-cp/checkpoints/step_00072868.safetensors")
    assert NB.location("castor", "end") == (NB.REPO, "nano-castor-v1.1/checkpoints/step_00061422.safetensors")
    assert NB.MOVED <= set(NB.VERSIONS) and not {"end", "mx", "mx-cp", "mx-cpx"} & NB.MOVED


def test_a_v12_boundary_resolves_to_its_own_craft_in_the_release_repo_for_pollux_dual_only():
    import pytest
    assert NB.location("pollux-dual", "v12-1145") == (NB.REPO, "nano-pollux-dual-v1.2/checkpoints/step_00001145.safetensors")
    assert NB.location("pollux-dual", "v12-61422") == (NB.REPO, "nano-pollux-dual-v1.2/checkpoints/step_00061422.safetensors")
    assert len(NB.CRAFTS) == 13 and not set(NB.CRAFTS) & set(NB.VERSIONS)
    with pytest.raises(KeyError):
        NB.craft("castor", "v12-1145")
