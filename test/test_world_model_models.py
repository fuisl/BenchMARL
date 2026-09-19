#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""M4 model invariants.

The scientific content of the three baselines is *which information can reach
agent i's prediction*. These tests assert exactly that, because a silent wiring
mistake would make the independent control leak cross-agent information (or the
relational model fail to use it) while every loss curve still looked plausible.
"""

import pytest
import torch

from examples.world_model.models import (
    effective_rank,
    latent_variance,
    masked_mean,
    masked_sum_count,
    MultiAgentWorldModel,
    parameter_counts,
    ReferenceMultiAgentWorldModel,
    ReferenceProjector,
    SIGReg,
)

AGENTS, OBS, ACT, FRAMES, DIM = 4, 11, 10, 6, 32


def build(kind, seed=0, wake=False):
    torch.manual_seed(seed)
    model = MultiAgentWorldModel(
        kind,
        OBS,
        ACT,
        AGENTS,
        dim=DIM,
        hidden_dim=48,
        frames=FRAMES,
        depth=2,
        heads=2,
        dim_head=16,
        mlp_dim=48,
        dropout=0.0,
    )
    if wake:
        wake_conditioning(model)
    return model.eval()


def build_reference(agents=AGENTS):
    torch.manual_seed(0)
    return ReferenceMultiAgentWorldModel(
        "relational",
        OBS,
        ACT,
        agents,
        dim=DIM,
        hidden_dim=48,
        history_size=3,
        depth=2,
        heads=2,
        dim_head=16,
        mlp_dim=48,
        dropout=0.1,
        projector_hidden_dim=64,
    ).eval()


def wake_conditioning(model):
    """Move the AdaLN-zero gates off zero so conditioning can be probed.

    The reference initialises ``adaLN_modulation``'s last layer to exactly zero,
    which makes every block the identity: at initialisation the predictor ignores
    its conditioning in both value and gradient, and all three baselines are the
    same function. That is intended for training stability, but it means a test
    of the information path has to look at a model whose gates have moved, as
    they do after the first optimiser steps.
    """
    generator = torch.Generator().manual_seed(7)
    for block in model.predictor.layers:
        final = block.adaLN_modulation[-1]
        with torch.no_grad():
            final.weight.normal_(0.0, 0.05, generator=generator)
            final.bias.normal_(0.0, 0.05, generator=generator)


def sample(batch=3, seed=1):
    generator = torch.Generator().manual_seed(seed)
    observation = torch.randn(batch, FRAMES, AGENTS, OBS, generator=generator)
    action = torch.rand(batch, FRAMES - 1, AGENTS, ACT, generator=generator) * 2 - 1
    return observation, action


def predict(model, observation, action):
    with torch.no_grad():
        return model.predict(model.encode(observation)[:, :-1], action)


@pytest.mark.parametrize("kind", ["independent", "joint", "relational"])
def test_prediction_shape(kind):
    model = build(kind)
    observation, action = sample()
    assert predict(model, observation, action).shape == (3, FRAMES - 1, AGENTS, DIM)


def test_reference_profile_has_pinned_projectors_and_predictor_shape():
    model = build_reference()
    assert model.profile == "lewm_reference"
    assert model.history_size == 3
    assert len(model.predictor.layers) == 2
    assert model.predictor.layers[0].attn.heads == 2
    assert isinstance(model.projector, ReferenceProjector)
    assert isinstance(model.projector.net[1], torch.nn.BatchNorm1d)
    assert model.projector.net[0].in_features == DIM
    assert model.projector.net[0].out_features == 64
    assert model.projector.net[-1].out_features == DIM
    assert isinstance(model.pred_proj.net[1], torch.nn.BatchNorm1d)
    assert model.predictor.pos_embedding.shape[1] == 3


def test_reference_profile_rejects_context_longer_than_history():
    model = build_reference()
    latent = torch.randn(2, 4, AGENTS, DIM)
    action = torch.randn(2, 4, AGENTS, ACT)
    with pytest.raises(ValueError, match="exceeds history_size"):
        model.predict(latent, action)


def test_reference_single_agent_bypasses_multi_agent_conditioner():
    model = build_reference(agents=1)

    class Forbidden(torch.nn.Module):
        def forward(self, *_args):
            raise AssertionError("single-agent reference must use action embeddings")

    model.conditioner = Forbidden()
    observation = torch.randn(2, 3, 1, OBS)
    action = torch.randn(2, 3, 1, ACT)
    with torch.no_grad():
        result = model.predict(model.encode(observation), action)
    assert result.shape == (2, 3, 1, DIM)


def test_reference_rollout_never_exceeds_registered_context():
    model = build_reference()
    lengths = []
    original = model.predict

    def record(latent, action):
        lengths.append((latent.size(1), action.size(1)))
        return original(latent, action)

    model.predict = record
    initial = torch.randn(2, 1, AGENTS, DIM)
    actions = torch.randn(2, 5, AGENTS, ACT)
    with torch.no_grad():
        result = model.rollout(initial, actions)
    assert result.shape == (2, 5, AGENTS, DIM)
    assert lengths == [(1, 1), (2, 2), (3, 3), (3, 3), (3, 3)]


def test_independent_has_no_cross_agent_path():
    """Changing agent 1's action must not move agent 0's prediction at all."""
    model = build(wake=True, kind="independent")
    observation, action = sample()
    other = action.clone()
    other[:, :, 1] = torch.randn_like(other[:, :, 1])
    base = predict(model, observation, action)
    changed = predict(model, observation, other)
    assert torch.equal(base[:, :, 0], changed[:, :, 0])
    assert not torch.equal(base[:, :, 1], changed[:, :, 1])


def test_independent_ignores_other_agents_observations():
    model = build(wake=True, kind="independent")
    observation, action = sample()
    other = observation.clone()
    other[:, :, 1] = torch.randn_like(other[:, :, 1])
    base = predict(model, observation, action)
    changed = predict(model, other, action)
    assert torch.equal(base[:, :, 0], changed[:, :, 0])


@pytest.mark.parametrize("kind", ["joint", "relational"])
def test_cross_agent_action_reaches_other_agents(kind):
    """Both interaction models must actually use agent j's action for agent i."""
    model = build(kind, wake=True)
    observation, action = sample()
    other = action.clone()
    other[:, :, 1] = torch.randn_like(other[:, :, 1])
    base = predict(model, observation, action)
    changed = predict(model, observation, other)
    assert not torch.allclose(base[:, :, 0], changed[:, :, 0], atol=1e-7)


def test_relational_is_permutation_equivariant():
    """Sum pooling means relabelling agents relabels predictions, nothing else."""
    model = build(wake=True, kind="relational")
    observation, action = sample()
    order = [2, 0, 3, 1]
    base = predict(model, observation, action)
    permuted = predict(model, observation[:, :, order], action[:, :, order])
    torch.testing.assert_close(permuted, base[:, :, order], atol=1e-5, rtol=1e-5)


def test_joint_is_not_permutation_equivariant():
    """The fixed-order concatenation is the control: it must lack that symmetry."""
    model = build(wake=True, kind="joint")
    observation, action = sample()
    order = [2, 0, 3, 1]
    base = predict(model, observation, action)
    permuted = predict(model, observation[:, :, order], action[:, :, order])
    assert not torch.allclose(permuted, base[:, :, order], atol=1e-5)


@pytest.mark.parametrize("kind", ["independent", "joint", "relational"])
def test_prediction_is_causal_in_time(kind):
    """A later action must not change an earlier frame's prediction."""
    model = build(kind, wake=True)
    observation, action = sample()
    later = action.clone()
    later[:, -1] = torch.randn_like(later[:, -1])
    base = predict(model, observation, action)
    changed = predict(model, observation, later)
    torch.testing.assert_close(base[:, :-1], changed[:, :-1], atol=1e-6, rtol=1e-6)


def test_rollout_matches_teacher_forcing_on_first_step():
    """The first rolled step has no predicted input yet, so it must agree exactly."""
    model = build("relational", wake=True)
    observation, action = sample()
    with torch.no_grad():
        latent = model.encode(observation)
        forced = model.predict(latent[:, :-1], action)
        rolled = model.rollout(latent[:, :1], action)
    torch.testing.assert_close(rolled[:, 0], forced[:, 0], atol=1e-6, rtol=1e-6)


def test_readout_shapes_and_termination_is_permutation_invariant():
    model = build(wake=True, kind="relational")
    observation, action = sample()
    with torch.no_grad():
        latent = model.encode(observation)
        predicted = model.predict(latent[:, :-1], action)
        reward, terminated = model.readout(latent[:, :-1], predicted)
        order = [2, 0, 3, 1]
        _, permuted = model.readout(latent[:, :-1][:, :, order], predicted[:, :, order])
    assert reward.shape == (3, FRAMES - 1, AGENTS, 1)
    assert terminated.shape == (3, FRAMES - 1)
    torch.testing.assert_close(permuted, terminated, atol=1e-6, rtol=1e-6)


def test_masked_reductions_ignore_padding():
    values = torch.tensor([[1.0, 100.0], [3.0, 100.0]])
    mask = torch.tensor([[True, False], [True, False]])
    total, count = masked_sum_count(values, mask)
    assert float(total) == 4.0 and float(count) == 2.0
    assert float(masked_mean(values, mask)) == 2.0


def test_masked_mean_rejects_empty_mask():
    with pytest.raises(ValueError):
        masked_mean(torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.bool))


def test_sigreg_prefers_isotropic_over_collapsed_latents():
    """The anti-collapse term must actually penalise collapse."""
    sigreg = SIGReg(knots=17, num_proj=256)
    torch.manual_seed(0)
    isotropic = torch.randn(1, 4096, 16)
    collapsed = isotropic.clone()
    collapsed[..., 1:] = 0.0
    assert float(sigreg(collapsed)) > float(sigreg(isotropic))


def test_latent_variance_and_rank_detect_collapse():
    population = torch.randn(1, 512, 16)
    mask = torch.ones(1, 512, dtype=torch.bool)
    collapsed = population.clone()
    collapsed[..., 1:] = 0.0
    assert float(latent_variance(collapsed, mask)) < float(
        latent_variance(population, mask)
    )
    assert effective_rank(collapsed, mask) < effective_rank(population, mask)


@pytest.mark.parametrize("kind", ["independent", "joint", "relational"])
def test_parameter_counts_are_reported_per_group(kind):
    counts = parameter_counts(build(kind))
    assert counts["total"] == counts["dynamics_total"] + counts["readout"]
    assert counts["dynamics_total"] > 0


def test_only_the_conditioner_differs_between_baselines():
    """Encoder/predictor/readout capacity must be identical across baselines, so a
    difference in results cannot be explained by a bigger backbone."""
    counts = {
        kind: parameter_counts(build(kind))
        for kind in ("independent", "joint", "relational")
    }
    for group in ("encoder", "action_encoder", "predictor", "readout"):
        sizes = {counts[kind][group] for kind in counts}
        assert len(sizes) == 1, f"{group} differs across baselines: {sizes}"


def test_reference_parameter_report_includes_both_projectors():
    counts = parameter_counts(build_reference())
    assert counts["projector"] > 0
    assert counts["pred_proj"] == counts["projector"]
    assert counts["total"] == counts["dynamics_total"] + counts["readout"]
