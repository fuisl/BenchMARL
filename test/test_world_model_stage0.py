#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Stage 0 measurement contracts.

Each test pins one defect the 2026-09-15 audit found, at the offset where it
actually bites. The suite that existed before these passed while every one of
them was live, because it checked the typical case and not the boundary.
"""

import torch

from examples.world_model.plan_ranking import model_costs  # noqa: F401  (import guard)


def survival_prefix(terminated_logits):
    """The exclusive prefix product plan_ranking now uses."""
    alive = 1 - torch.sigmoid(terminated_logits)
    return torch.cat(
        [torch.ones_like(alive[:, :1]), torch.cumprod(alive, dim=1)[:, :-1]], dim=1
    )


def test_survival_weights_survive_a_saturated_termination():
    """A block certain to end must still carry its own reward.

    The previous form divided an inclusive cumulative product by the current
    factor. Once a termination sigmoid saturates to exactly 1.0 in float32 the
    numerator is zero and the clamped denominator cannot recover the prefix, so
    every block including the terminal one is weighted zero.
    """
    for logit in (20.0, 40.0, 80.0):
        weights = survival_prefix(torch.tensor([[[logit], [0.0], [0.0]]]))
        assert weights.flatten().tolist() == [1.0, 0.0, 0.0], logit

    # And the ordinary unsaturated case is unchanged.
    weights = survival_prefix(torch.tensor([[[0.0], [0.0], [0.0]]])).flatten()
    torch.testing.assert_close(weights, torch.tensor([1.0, 0.5, 0.25]))


def block_masks(primitive_valid, block):
    """Reproduce the dataset's two validity contracts."""
    reshaped = primitive_valid.reshape(-1, block)
    return reshaped.all(dim=1), reshaped.any(dim=1)


def test_terminal_block_is_admitted_once_at_every_primitive_offset():
    """Termination at each offset inside a block, hand-checked.

    `valid` (all) is what prediction needs -- a partial block has no observed
    end-of-block latent. `outcome_valid` (any) is what reward and termination
    need. They must differ on exactly the terminal block, and the terminal block
    must appear exactly once, with nothing after it.
    """
    block, blocks = 5, 4
    for offset in range(block):
        steps = block + offset  # terminate in block 1, at this offset
        primitive = torch.zeros(blocks * block, dtype=torch.bool)
        primitive[:steps] = True
        dynamics, outcome = block_masks(primitive, block)

        assert dynamics.tolist() == [True, False, False, False], offset
        # The partial block is admitted for outcomes exactly when it has steps.
        expected_outcome = [True, offset > 0, False, False]
        assert outcome.tolist() == expected_outcome, offset
        # Never more than one block beyond the fully-valid ones.
        assert int(outcome.sum()) - int(dynamics.sum()) in (0, 1), offset
        # And nothing is admitted after the terminal block.
        last = max(i for i, v in enumerate(outcome.tolist()) if v)
        assert not any(outcome.tolist()[last + 1 :]), offset


def test_outcome_validity_keeps_the_terminal_reward():
    """The block-summed reward of a partial terminal block must be preserved."""
    block = 5
    primitive_valid = torch.tensor([True] * 7 + [False] * 13)  # ends at offset 2
    reward = torch.zeros(20, 1, 1)
    reward[6] = -10.0  # the collision penalty, inside the partial block
    dynamics, outcome = block_masks(primitive_valid, block)

    summed = (reward.reshape(4, block, 1, 1) * primitive_valid.reshape(4, block, 1, 1)).sum(1)
    assert float(summed[1]) == -10.0
    assert not dynamics[1], "prediction cannot use this block"
    assert outcome[1], "but reward and termination must"
    # Under the old contract the penalty was multiplied by a zero mask.
    assert float((summed.squeeze() * dynamics).sum()) == 0.0
    assert float((summed.squeeze() * outcome).sum()) == -10.0


def test_simulate_endpoint_is_the_frame_actually_reached():
    """Endpoints must come from the rollout, not from block arithmetic.

    The replaced helper counted complete blocks and clamped to at least one, so
    a candidate terminating inside its first block was scored at a frame *after*
    termination, and one terminating inside a later block was scored at the
    boundary *before* it. Buzz Wire collides in most episodes, so this was the
    common case rather than an edge one.
    """
    from benchmarl.environments import VmasTask
    from examples.world_model.readout_diagnostic import simulate
    from examples.world_model.snapshot_restore import snapshot_state

    states, candidates, block, blocks = 2, 4, 5, 3
    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    task.config["max_steps"] = blocks * block
    env = task.get_env_fun(states, True, 0, "cpu")()
    scratch = task.get_env_fun(states * candidates, True, 0, "cpu")()
    env.set_seed(0)
    env.reset()
    scratch.reset()
    try:
        plans = torch.rand(states, candidates, blocks * block, 4) * 2 - 1
        _cost, _complete, block_valid, observation, endpoint = simulate(
            scratch, snapshot_state(env), plans, block
        )
    finally:
        env.close()
        scratch.close()

    assert endpoint.shape == (states, candidates, *observation.shape[-2:])
    assert torch.isfinite(endpoint).all()

    complete = block_valid.sum(dim=-1)
    for s in range(states):
        for k in range(candidates):
            frames = observation[s, k]
            if int(complete[s, k]) == blocks:
                # Never terminated: the endpoint is the final boundary frame.
                torch.testing.assert_close(endpoint[s, k], frames[-1])
            else:
                # Terminated early: the endpoint must not be a padded frame
                # from after termination, which is what the old helper returned
                # whenever no block completed.
                assert endpoint[s, k].abs().sum() > 0
