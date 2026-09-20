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
    action_statistics,
    load_model,
    observation_statistics,
    reference_lr_factor,
    reference_training_view,
    world_model_from_config,
)
from _lewm_pinned_8edfeb33 import (
    ARPredictor as PinnedARPredictor,
    Embedder as PinnedEmbedder,
    MLP as PinnedMLP,
)
from _stable_pretraining_pinned_v017 import (
    LinearWarmupCosineAnnealingLR as PinnedReferenceScheduler,
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


def test_reference_action_normalization_is_train_only_sample_std():
    action = torch.tensor(
        [[[[1.0, 10.0]], [[3.0, 20.0]], [[5.0, 30.0]], [[1000.0, 2000.0]]]]
    )
    loader = [{"action": action, "valid": torch.ones(1, 4, dtype=torch.bool)}]
    mean, std = action_statistics(loader, max_blocks=3)
    torch.testing.assert_close(mean, torch.tensor([3.0, 20.0]))
    # torch.std's default correction, as used by pinned LeWM.
    torch.testing.assert_close(std, torch.tensor([2.0, 10.0]))


def test_reference_action_statistics_are_shared_across_block_positions():
    """Pinned LeWM normalizes the action column before frameskip concatenation.

    One statistic per primitive coordinate is therefore reused at every block
    position.  The two positions here have deliberately different column means,
    so fitting the blocked width directly would not produce this answer.
    """
    action = torch.tensor([[[[0.0, 0.0, 10.0, 100.0]], [[2.0, 20.0, 4.0, 40.0]]]])
    loader = [{"action": action, "valid": torch.ones(1, 2, dtype=torch.bool)}]
    mean, std = action_statistics(loader, action_block=2)
    primitive = torch.tensor([[0.0, 0.0], [10.0, 100.0], [2.0, 20.0], [4.0, 40.0]])
    torch.testing.assert_close(mean, primitive.mean(0).repeat(2))
    torch.testing.assert_close(std, primitive.std(0).repeat(2))
    torch.testing.assert_close(mean[:2], mean[2:])
    torch.testing.assert_close(std[:2], std[2:])


def test_reference_warmup_then_cosine_schedule():
    assert reference_lr_factor(0, 100, 10) == 0.0
    assert reference_lr_factor(5, 100, 10) == 0.5
    assert reference_lr_factor(10, 100, 10) == pytest.approx(1.0)
    assert reference_lr_factor(55, 100, 10) == pytest.approx(0.5)
    assert reference_lr_factor(100, 100, 10) == pytest.approx(0.0)


def test_reference_scheduler_matches_release_at_pinned_lewm_cutoff():
    """Match stable-pretraining v0.1.7 through every optimizer update.

    LeWM says ``interval=epoch``, but v0.1.7 manually steps the returned
    scheduler in every training batch.  250 updates also pins the factory's
    floor semantics: ``int(.01 * 250) == 2``.
    """
    total_steps = 250
    warmup_steps = max(1, int(0.01 * total_steps))
    ours_parameter = torch.nn.Parameter(torch.tensor(0.0))
    pinned_parameter = torch.nn.Parameter(torch.tensor(0.0))
    ours_optimizer = torch.optim.AdamW([ours_parameter], lr=5e-5)
    pinned_optimizer = torch.optim.AdamW([pinned_parameter], lr=5e-5)
    ours = torch.optim.lr_scheduler.LambdaLR(
        ours_optimizer,
        lambda step: reference_lr_factor(step, total_steps, warmup_steps),
    )
    pinned = PinnedReferenceScheduler(
        pinned_optimizer,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        warmup_start_lr=0.0,
        eta_min=0.0,
    )
    ours_trace = [ours_optimizer.param_groups[0]["lr"]]
    pinned_trace = [pinned_optimizer.param_groups[0]["lr"]]
    for _ in range(total_steps):
        ours_optimizer.step()
        pinned_optimizer.step()
        ours.step()
        pinned.step()
        ours_trace.append(ours_optimizer.param_groups[0]["lr"])
        pinned_trace.append(pinned_optimizer.param_groups[0]["lr"])
    torch.testing.assert_close(
        torch.tensor(ours_trace), torch.tensor(pinned_trace), rtol=0, atol=0
    )


def _pinned_single_agent_modules(model):
    # Our port fixes LeWM's pointwise smoothing width to the action width; the
    # pinned class defaults it to 10, so state it explicitly rather than
    # comparing two differently shaped Conv1d stacks.
    action_encoder = PinnedEmbedder(
        input_dim=model.action_mean.numel(),
        smoothed_dim=model.action_mean.numel(),
        emb_dim=model.dim,
    )
    action_encoder.load_state_dict(model.action_encoder.state_dict())
    predictor = PinnedARPredictor(
        num_frames=model.history_size,
        input_dim=model.dim,
        hidden_dim=model.dim,
        output_dim=model.dim,
        depth=len(model.predictor.layers),
        heads=model.predictor.layers[0].attn.heads,
        dim_head=model.predictor.layers[0].attn.to_qkv.out_features
        // (3 * model.predictor.layers[0].attn.heads),
        mlp_dim=model.predictor.layers[0].mlp.net[1].out_features,
        dropout=model.predictor.layers[0].attn.dropout,
        emb_dropout=0.0,
    )
    mapped = {}
    for name, value in model.predictor.state_dict().items():
        if name.startswith("layers."):
            name = "transformer." + name
        elif name.startswith("norm."):
            name = "transformer." + name
        mapped[name] = value
    predictor.load_state_dict(mapped)
    pred_proj = PinnedMLP(
        model.dim,
        model.pred_proj.net[0].out_features,
        model.dim,
        norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj.load_state_dict(model.pred_proj.state_dict())
    return action_encoder.eval(), predictor.eval(), pred_proj.eval()


def test_single_agent_output_matches_vendored_pinned_implementation():
    """External N=1 comparison with LeWM's own module structure and einops."""
    torch.manual_seed(41)
    action_mean = torch.tensor([0.2, -0.5, 1.0, 2.0])
    action_std = torch.tensor([0.5, 2.0, 0.25, 4.0])
    model = ReferenceMultiAgentWorldModel(
        "relational",
        obs_dim=3,
        action_dim=4,
        agents=1,
        dim=16,
        hidden_dim=24,
        history_size=3,
        depth=2,
        heads=2,
        dim_head=8,
        mlp_dim=32,
        dropout=0.0,
        projector_hidden_dim=32,
        action_mean=action_mean,
        action_std=action_std,
    ).eval()
    # AdaLN-zero correctly ignores actions at initialization. Wake its gates so
    # this comparison also verifies the native -> normalized action boundary.
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for block in model.predictor.layers:
            block.adaLN_modulation[-1].weight.normal_(
                mean=0.0, std=0.03, generator=generator
            )
            block.adaLN_modulation[-1].bias.normal_(
                mean=0.0, std=0.03, generator=generator
            )
    pinned_action, pinned_predictor, pinned_proj = _pinned_single_agent_modules(model)

    latent = torch.randn(2, 3, 1, 16, generator=generator)
    past = torch.randn(2, 2, 1, 4, generator=generator)
    future = torch.randn(2, 4, 1, 4, generator=generator)
    expected = []
    pinned_history = latent.squeeze(2)
    pinned_actions = past.squeeze(2)
    with torch.no_grad():
        ours = model.rollout_from_context(latent, past, future)
        for current in future.squeeze(2).split(1, dim=1):
            action_context = torch.cat([pinned_actions, current], dim=1)
            normalized = (action_context - action_mean) / action_std
            action_embedding = pinned_action(normalized)
            predicted = pinned_predictor(pinned_history, action_embedding)
            predicted = pinned_proj(predicted.reshape(-1, model.dim)).reshape(
                predicted.shape
            )
            newest = predicted[:, -1:]
            expected.append(newest.unsqueeze(2))
            pinned_history = torch.cat([pinned_history[:, 1:], newest], dim=1)
            pinned_actions = action_context[:, 1:]
    expected = torch.cat(expected, dim=1)
    torch.testing.assert_close(ours, expected, rtol=0, atol=0)


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
        payload["action_mean"] = torch.zeros(shapes()["action_dim"])
        payload["action_std"] = torch.ones(shapes()["action_dim"])
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
            "action_mean": torch.zeros(shapes()["action_dim"]),
            "action_std": torch.ones(shapes()["action_dim"]),
            "shapes": shapes(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="disagrees"):
        load_model(checkpoint)
