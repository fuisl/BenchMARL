# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Audit Gate A0 step-1 contracts for the parallel LeWM reference profile."""

from types import SimpleNamespace

import pytest
import torch

from examples.world_model.models import (
    MultiAgentWorldModel,
    ReferenceMultiAgentWorldModel,
)
from examples.world_model.train import (
    load_model,
    observation_statistics,
    reference_lr_factor,
    reference_training_view,
    world_model_from_config,
)
from omegaconf import OmegaConf


def config(profile="legacy_compact"):
    model = {
        "kind": "relational",
        "dim": 16,
        "hidden_dim": 24,
        "conditioner_budget": None,
        "depth": 2,
        "heads": 2,
        "dim_head": 8,
        "mlp_dim": 32,
        "dropout": 0.0,
    }
    if profile == "lewm_reference":
        model.update(
            profile=profile,
            history_size=3,
            projector_hidden_dim=32,
        )
    return OmegaConf.create({"model": model})


def shapes():
    return {"agents": 2, "obs_dim": 6, "action_dim": 10, "frames": 6}


def test_factory_keeps_missing_profile_backward_compatible():
    model = world_model_from_config(
        config(), shapes(), torch.zeros(6), torch.ones(6), "cpu"
    )
    assert type(model) is MultiAgentWorldModel
    assert model.profile == "legacy_compact"
    assert not hasattr(model, "projector")


def test_factory_builds_parallel_reference_profile():
    model = world_model_from_config(
        config("lewm_reference"),
        shapes(),
        torch.zeros(6),
        torch.ones(6),
        "cpu",
    )
    assert isinstance(model, ReferenceMultiAgentWorldModel)
    assert model.profile == "lewm_reference"
    assert model.predictor.pos_embedding.shape[1] == 3


def test_reference_training_view_is_exact_shifted_four_frame_window():
    model = SimpleNamespace(profile="lewm_reference", history_size=3)
    observation = torch.arange(2 * 6 * 2 * 3).reshape(2, 6, 2, 3)
    action = torch.arange(2 * 5 * 2 * 4).reshape(2, 5, 2, 4)
    batch = {
        "observation": observation,
        "observation_valid": torch.ones(2, 6, dtype=torch.bool),
        "action": action,
        "valid": torch.ones(2, 5, dtype=torch.bool),
        "episode_id": torch.tensor([4, 5]),
    }
    view = reference_training_view(model, batch)
    assert view["observation"].shape[1] == 4
    assert view["action"].shape[1] == 3
    torch.testing.assert_close(view["observation"][:, :3], observation[:, :3])
    torch.testing.assert_close(view["observation"][:, 1:4], observation[:, 1:4])
    torch.testing.assert_close(view["action"], action[:, :3])
    assert torch.equal(view["episode_id"], batch["episode_id"])


def test_legacy_training_view_is_identity():
    batch = {"observation": torch.randn(2, 6, 2, 3)}
    model = SimpleNamespace(profile="legacy_compact")
    assert reference_training_view(model, batch) is batch


def test_reference_normalization_uses_only_its_training_window():
    observation = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]], [[100.0]]]])
    loader = [
        {
            "observation": observation,
            "observation_valid": torch.ones(1, 5, dtype=torch.bool),
        }
    ]
    mean, _std = observation_statistics(loader, max_frames=4)
    torch.testing.assert_close(mean, torch.tensor([1.5]))


def test_reference_warmup_then_cosine_schedule():
    assert reference_lr_factor(0, 100, 10) == 0.0
    assert reference_lr_factor(5, 100, 10) == 0.5
    assert reference_lr_factor(10, 100, 10) == pytest.approx(1.0)
    assert reference_lr_factor(55, 100, 10) == pytest.approx(0.5)
    assert reference_lr_factor(100, 100, 10) == pytest.approx(0.0)


@pytest.mark.parametrize("profile", ["legacy_compact", "lewm_reference"])
def test_checkpoint_round_trip_uses_recorded_profile(tmp_path, profile):
    cfg = config(profile)
    model = world_model_from_config(
        cfg, shapes(), torch.zeros(6), torch.ones(6), "cpu"
    ).eval()
    checkpoint = tmp_path / f"{profile}.pt"
    payload = {
        "kind": "relational",
        "state_dict": model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "observation_mean": torch.zeros(6),
        "observation_std": torch.ones(6),
        "shapes": shapes(),
    }
    # Historical payloads have neither a top-level nor config profile.
    if profile == "lewm_reference":
        payload["profile"] = profile
    torch.save(payload, checkpoint)
    restored = load_model(checkpoint)
    assert restored.profile == profile
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])


def test_checkpoint_rejects_profile_disagreement(tmp_path):
    cfg = config("lewm_reference")
    model = world_model_from_config(
        cfg, shapes(), torch.zeros(6), torch.ones(6), "cpu"
    )
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {
            "kind": "relational",
            "profile": "legacy_compact",
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "observation_mean": torch.zeros(6),
            "observation_std": torch.ones(6),
            "shapes": shapes(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="disagrees"):
        load_model(checkpoint)
