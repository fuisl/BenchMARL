#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Feed a planner's model the input construction its encoder was fitted on.

Job 1261/1262/1263 planned with models that cannot see the object the task is
about. Buzz Wire's agents observe ``[pos, vel, pos - goal]`` and nothing else;
the ball jointed between them, whose contact with the wire *is* the failure the
reward punishes, appears in no agent's observation. Job 1203 measured the
consequence directly -- the reward readout ranks plans at Spearman ~ -0.25 even
when handed the simulator's own latents -- and Gate 4 paid it: every learned
policy collided more often than random actions (0.77-0.94 against 0.62) and died
by step 38 of 100.

That is not a dynamics failure and more seeds cannot reach it. A planner
maximising predicted reward will drive into a constraint its model cannot
represent, every time, because nothing in the predicted return falls when it
does. Pixel-based world models never meet this: the obstacle is in the frame, so
predicting the frame means predicting the obstacle. Hand-picking a vector
observation picked the constraint out, and the self-supervised objective then
had no way to put it back.

`dataset.py` already builds the three input conditions job 1223 trained on.
This module builds the same three from a *live* simulator, so a checkpoint can
be planned with rather than only scored offline:

``observation``  what the agents see -- the established baseline, unchanged
``history``      legacy concatenation of the last ``history_frames`` before one
                 encoding; not LeWM's temporal latent/action context
``physical``     the tracked entity states appended, identical for every agent

The constructions must match `Dataset._apply_state_input` exactly. Where they
cannot (a history whose decision cadence differs from the training cadence) this
raises rather than feeding the encoder a distribution it never saw.
"""

from dataclasses import dataclass

import torch

from examples.world_model.dataset import STATE_INPUTS
from examples.world_model.snapshot_restore import (
    agent_observations,
    physical_state,
    tracked_entities,
)

__all__ = [
    "STATE_INPUTS",
    "ObservationBuilder",
    "PlanningContext",
    "ReferenceHistory",
    "builder_for",
    "entity_frame",
    "reference_builder_for",
]


@dataclass(frozen=True)
class PlanningContext:
    """Real LeWM context at one MPC decision.

    Frames are oldest-to-newest ``[o[t-2], o[t-1], o[t]]``.  Actions are the
    two completed blocked actions ``[a[t-2], a[t-1]]``.  Future candidate
    actions are deliberately absent and are appended only inside model scoring.
    """

    observations: torch.Tensor
    past_actions: torch.Tensor

    def __post_init__(self):
        if self.observations.ndim != 4:
            raise ValueError("Context observations must have shape (B,T,N,O)")
        if self.past_actions.ndim != 4:
            raise ValueError("Context actions must have shape (B,T-1,N,A)")
        if self.observations.shape[1] != self.past_actions.shape[1] + 1:
            raise ValueError("Context needs one more observation than past actions")
        observation_axes = (self.observations.shape[0], self.observations.shape[2])
        action_axes = (self.past_actions.shape[0], self.past_actions.shape[2])
        if observation_axes != action_axes:
            raise ValueError("Context observation/action batch or agents differ")

    @property
    def current(self):
        return self.observations[:, -1]


class ReferenceHistory:
    """Stateful real observation/action history for reference LeWM MPC.

    One instance belongs to one policy evaluation.  At episode start the first
    observation is repeated and historical actions are zero, following the
    stable-worldmodel convention.  Thereafter an action block must be recorded
    before the next observation can be appended, preventing silent context
    resets or temporally misaligned pairs.
    """

    requires_receding_horizon_one = True

    def __init__(self, frame_builder, history_size=3, action_block=5):
        if history_size != 3:
            raise ValueError("The registered LeWM reference history size is 3")
        if action_block < 1:
            raise ValueError("action_block must be positive")
        if getattr(frame_builder, "state_input", None) == "history":
            raise ValueError(
                "Reference LeWM history encodes frames independently; it cannot "
                "wrap the legacy concatenated state_input=history condition"
            )
        self.frame_builder = frame_builder
        self.history_size = history_size
        self.action_block = action_block
        self.reset()

    def reset(self):
        if hasattr(self.frame_builder, "reset"):
            self.frame_builder.reset()
        self._observations = None
        self._actions = None
        self._pending_action = None

    def __call__(self, env):
        observation = self.frame_builder(env)
        if observation.ndim != 3:
            raise ValueError("Reference frame builder must return (B,N,O)")
        if self._observations is None:
            batch, agents = observation.shape[:2]
            primitive = env.full_action_spec_unbatched["agents", "action"].shape[-1]
            action_width = self.action_block * primitive
            self._observations = [observation.clone()] * self.history_size
            zero = observation.new_zeros(batch, agents, action_width)
            self._actions = [zero.clone() for _ in range(self.history_size - 1)]
        else:
            if self._pending_action is None:
                raise RuntimeError(
                    "A completed action block must be recorded before the next frame"
                )
            self._observations = self._observations[1:] + [observation.clone()]
            self._actions = self._actions[1:] + [self._pending_action]
            self._pending_action = None
        return PlanningContext(
            torch.stack(self._observations, dim=1),
            torch.stack(self._actions, dim=1),
        )

    def record_action(self, blocked_action):
        """Record one actually executed block as ``(B,N,block*action_dim)``."""
        if self._observations is None:
            raise RuntimeError("Observe the initial frame before recording an action")
        if self._pending_action is not None:
            raise RuntimeError("The previous action has not been paired with a frame")
        expected = self._actions[-1].shape
        if blocked_action.shape != expected:
            raise ValueError(
                f"Executed block has shape {tuple(blocked_action.shape)}, "
                f"expected {tuple(expected)}"
            )
        self._pending_action = blocked_action.detach().clone()


def entity_frame(env):
    """(B, E * 6) -- every tracked entity's ``[pos, vel, rot, ang_vel]``.

    `collect.tracked_entities` selects the bodies the bank recorded under
    ``package_state``, so this is the same schema the dataset appends, in the
    same world order. Reusing both helpers is what keeps the live construction
    and the stored one from drifting apart.
    """
    return physical_state(tracked_entities(env)).flatten(1)


class ObservationBuilder:
    """Live observations in one checkpoint's input format: ``(B, N, obs_dim)``.

    Stateful only for ``history``, which needs the frames a decision follows.
    Construct one per evaluated episode set and call `reset` before the first
    decision, or the buffer carries the previous episode's frames into this one.
    """

    def __init__(self, state_input, history_frames, obs_dim, action_block_stride=1):
        if state_input not in STATE_INPUTS:
            raise ValueError(f"state_input must be one of {STATE_INPUTS}")
        if state_input == "history" and action_block_stride != 1:
            # Training stacks consecutive *block boundaries*. A planner that
            # executes k blocks between observations would stack frames k blocks
            # apart and call them adjacent, which is a different input.
            raise ValueError(
                "history inputs require --execute-blocks 1 so that the stacked "
                f"frames are one block apart as in training; got stride "
                f"{action_block_stride}"
            )
        self.state_input = state_input
        self.history_frames = history_frames
        self.obs_dim = obs_dim
        self._past = None

    def reset(self):
        self._past = None

    def __call__(self, env):
        agent_obs = agent_observations(env)
        observation = self._build(env, agent_obs)
        if observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"state_input={self.state_input} produced {observation.shape[-1]} "
                f"dimensions, but the checkpoint was trained on {self.obs_dim}"
            )
        return observation

    def _build(self, env, agent_obs):
        if self.state_input == "observation":
            return agent_obs
        if self.state_input == "physical":
            batch, agents = agent_obs.shape[:2]
            entities = entity_frame(env).view(batch, 1, -1).expand(batch, agents, -1)
            return torch.cat([agent_obs, entities], dim=-1)

        # History: frame t carries [t, t-1, ..., t-k+1], clamped at the episode
        # start exactly as the dataset clamps at the snippet start.
        if self._past is None:
            self._past = [agent_obs] * self.history_frames
        else:
            self._past = [agent_obs] + self._past[: self.history_frames - 1]
        return torch.cat(self._past, dim=-1)


def builder_for(checkpoint_config, obs_dim, action_block_stride=1):
    """The builder a checkpoint's own recorded data config asks for."""
    data = checkpoint_config.data
    return ObservationBuilder(
        data.state_input,
        data.get("history_frames", 3),
        obs_dim,
        action_block_stride=action_block_stride,
    )


def reference_builder_for(checkpoint_config, obs_dim, action_block):
    """Build the real temporal observer required by ``lewm_reference``."""
    data = checkpoint_config.data
    model = checkpoint_config.model
    frame_builder = ObservationBuilder(
        data.state_input,
        data.get("history_frames", 3),
        obs_dim,
        action_block_stride=1,
    )
    return ReferenceHistory(
        frame_builder,
        history_size=model.get("history_size", 3),
        action_block=action_block,
    )


def compose_frames(state_input, agent_obs, entity_obs, history_frames=3):
    """Model-space frames from a recorded trajectory: ``(B, T, N, obs_dim)``.

    The offline counterpart of `ObservationBuilder`, for paths that already hold
    a rolled trajectory rather than a live simulator -- the readout diagnostic
    scores the *simulator's own* block boundaries, so it has frames, not a world
    to read. ``agent_obs`` is (B,T,N,O) and ``entity_obs`` is (B,T,E*6).

    This mirrors `Dataset._apply_state_input`; the contract tests compare both
    constructions against that method's own output, which is what keeps three
    copies of one rule from drifting apart.
    """
    if state_input not in STATE_INPUTS:
        raise ValueError(f"state_input must be one of {STATE_INPUTS}")
    if state_input == "observation":
        return agent_obs
    batch, steps, agents = agent_obs.shape[:3]
    if state_input == "physical":
        entities = entity_obs.reshape(batch, steps, 1, -1).expand(
            batch, steps, agents, entity_obs.shape[-1]
        )
        return torch.cat([agent_obs, entities], dim=-1)

    # Frame t carries [t, t-1, ..., t-k+1], clamped at the start of the record.
    offsets = torch.arange(history_frames)
    index = (torch.arange(steps)[:, None] - offsets[None, :]).clamp_min(0)
    past = agent_obs[:, index]  # (B,T,k,N,O)
    return past.permute(0, 1, 3, 2, 4).reshape(batch, steps, agents, -1)
