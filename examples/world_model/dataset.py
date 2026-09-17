# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Episode-disjoint, action-aligned offline snippets for the M4 predictors."""

import hashlib
import json
from pathlib import Path

import torch

from tensordict import TensorDict
from torch.utils.data import Dataset


STATE_INPUTS = ("observation", "history", "physical")


class OfflineSequences(Dataset):
    """Read one controlled action regime and one episode split into CPU memory.

    Each item has observations ``(L+1,N,O)`` and actions ``(L,N,block*A)``.
    Within each agent's action feature, primitive time precedes action coordinate.
    ``primitive_action`` keeps ``(L,block,N,A)`` for joint-action reconstruction;
    flatten its last three dimensions for the MPC's time-then-agent ordering.
    ``valid`` marks blocks whose every primitive transition is valid, which is
    the contract prediction needs; ``outcome_valid`` marks blocks that start
    from an observed state, which is the contract reward and termination need.
    They differ exactly on partial terminal blocks, which retain their primitive
    rewards, masks and termination flags.

    Simulator snapshots never appear in model samples. Normalization is left to
    M4 and must be fitted using valid training transitions only.
    """

    def __init__(
        self,
        root,
        regime,
        split,
        action_block=1,
        state_input="observation",
        history_frames=3,
    ):
        if regime not in ("independent", "correlated"):
            raise ValueError("regime must be independent or correlated")
        if state_input not in STATE_INPUTS:
            raise ValueError(f"state_input must be one of {STATE_INPUTS}")
        if state_input == "history" and (
            not isinstance(history_frames, int) or history_frames < 2
        ):
            raise ValueError("history_frames must be an integer of at least 2")
        splits = {"train": 0, "validation": 1, "test": 2}
        if split not in splits:
            raise ValueError("split must be train, validation or test")
        if not isinstance(action_block, int) or action_block < 1:
            raise ValueError("action_block must be a positive integer")
        self.root = Path(root)
        self.regime = regime
        self.split = split
        self.action_block = action_block
        self.state_input = state_input
        self.history_frames = history_frames
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema_version"] != 1:
            raise ValueError("Unsupported offline dataset schema_version")
        anchors = self._load("anchors.pt")
        self.samples = self._load(f"samples_{regime}.pt")
        self._validate(anchors)
        self._apply_state_input()
        self.episode_id = anchors["episode_id"][self.samples["anchor_id"]]
        self.indices = (
            anchors["split"][self.samples["anchor_id"]] == splits[split]
        ).nonzero(as_tuple=True)[0]


    def _apply_state_input(self):
        """Rewrite observation/next_observation to the Stage 2 input condition.

        Buzz Wire's agents see only their own position, velocity and offset to
        the goal, while the task is defined on a ball jointed to both of them
        that nobody observes. That is an information concern, not a proof of
        irreducible failure: joint observations and history may infer the
        hidden state (the audit's geometric check ruled out the ball being
        exactly the agents' midpoint, mean discrepancy 0.0408, but not all
        inference). These three conditions separate the possibilities.

        ``observation``  what the agents actually see -- the established baseline
        ``history``      the last ``history_frames`` observed frames, so hidden
                         state can be inferred from motion rather than supplied
        ``physical``     the recorded entity states appended outright, the upper
                         bound where the hidden variable is simply given

        Rewriting the stored keys rather than adding new ones keeps everything
        downstream unchanged: normalisation is fitted from the loader, and the
        model reads obs_dim off the sample.
        """
        if self.state_input == "observation":
            return
        observation = self.samples["observation"]
        next_observation = self.samples["next_observation"]
        count, steps, agents, _ = observation.shape

        if self.state_input == "physical":
            # One shared world state, given identically to every agent.
            def entities(key):
                flat = self.samples[key].reshape(count, steps, 1, -1)
                return flat.expand(count, steps, agents, flat.shape[-1])

            observation = torch.cat([observation, entities("package_state")], dim=-1)
            next_observation = torch.cat(
                [next_observation, entities("next_package_state")], dim=-1
            )
        else:
            # Frame t carries [t, t-1, ..., t-k+1], clamped at the snippet start
            # because an anchor's earlier frames are not stored. next_observation
            # at t is the frame at t+1, so its tail is [t, t-1, ...]; that is what
            # keeps observation[:, 1:] == next_observation[:, :-1] true, which the
            # dynamics target relies on.
            offsets = torch.arange(self.history_frames)
            index = (torch.arange(steps)[:, None] - offsets[None, :]).clamp_min(0)
            past = observation[:, index]  # (S,T,k,N,D)
            flat = past.permute(0, 1, 3, 2, 4).reshape(count, steps, agents, -1)
            tail = flat[..., : -observation.shape[-1]]
            observation = flat
            next_observation = torch.cat([next_observation, tail], dim=-1)

        self.samples["observation"] = observation.contiguous()
        self.samples["next_observation"] = next_observation.contiguous()
        adjacent = self.samples["valid"][:, 1:]
        if not torch.equal(
            self.samples["observation"][:, 1:][adjacent],
            self.samples["next_observation"][:, :-1][adjacent],
        ):
            raise ValueError(
                f"state_input={self.state_input} broke observation alignment"
            )

    def _load(self, filename):
        path = self.root / filename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.manifest["files"][filename]:
            raise ValueError(f"Checksum mismatch: {filename}")
        return torch.load(path, map_location="cpu", weights_only=True)

    def _validate(self, anchors):
        samples = self.samples
        observation = samples["observation"]
        action = samples["action"]
        if observation.ndim != 4 or action.ndim != 4:
            raise ValueError("observation/action must have shape (S,T,N,features)")
        count, steps, agents, obs_dim = observation.shape
        action_dim = action.shape[-1]
        if not count or not steps or not agents or not obs_dim or not action_dim:
            raise ValueError("Empty snippet dimensions")
        if action.shape[:3] != (count, steps, agents):
            raise ValueError("Actions and observations are not aligned")
        if steps % self.action_block:
            raise ValueError(
                "Primitive snippet length must be divisible by action_block"
            )

        anchor_count = anchors["episode_id"].numel()
        if anchor_count != count:
            raise ValueError("Each regime must contain one snippet per anchor")
        for key in ("episode_id", "split", "source_regime", "source_step"):
            if (
                anchors[key].shape != (anchor_count,)
                or anchors[key].dtype != torch.long
            ):
                raise ValueError(f"anchors {key} must be a long vector")
        if (anchors["episode_id"] < 0).any() or (anchors["source_step"] < 0).any():
            raise ValueError("Episode IDs and source steps must be nonnegative")
        if not torch.isin(anchors["split"], torch.tensor([0, 1, 2])).all():
            raise ValueError("Invalid episode split")
        if not torch.isin(anchors["source_regime"], torch.tensor([0, 1, 2])).all():
            raise ValueError("Invalid source action regime")
        for episode in anchors["episode_id"].unique():
            if anchors["split"][anchors["episode_id"] == episode].unique().numel() != 1:
                raise ValueError("An episode appears in multiple splits")
        ids = samples["anchor_id"]
        if (
            ids.shape != (count,)
            or ids.dtype != torch.long
            or (ids < 0).any()
            or (ids >= anchor_count).any()
            or ids.unique().numel() != count
        ):
            raise ValueError("Invalid or duplicated sample anchor_id")

        float_shapes = {
            "observation": (count, steps, agents, obs_dim),
            "next_observation": (count, steps, agents, obs_dim),
            "action": (count, steps, agents, action_dim),
            "reward": (count, steps, agents, 1),
            "agent_state": (count, steps, agents, 6),
            "next_agent_state": (count, steps, agents, 6),
        }
        packages = samples["package_state"]
        if packages.ndim != 4:
            raise ValueError("package_state must have shape (S,T,P,6)")
        for key in ("package_state", "next_package_state"):
            float_shapes[key] = (count, steps, packages.shape[2], 6)
        for key, shape in float_shapes.items():
            value = samples[key]
            if value.shape != shape or not value.is_floating_point():
                raise ValueError(f"Invalid shape or dtype for {key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite {key}")
        for key in ("done", "terminated", "truncated", "valid"):
            if samples[key].shape != (count, steps) or samples[key].dtype != torch.bool:
                raise ValueError(f"{key} must be bool (S,T)")
        valid, done = samples["valid"], samples["done"]
        if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("valid must be a nonempty contiguous prefix")
        if (done & ~valid).any() or (done[:, :-1] & valid[:, 1:]).any():
            raise ValueError("Terminal transition must be valid and end the snippet")
        if not torch.equal(done, samples["terminated"] | samples["truncated"]):
            raise ValueError("done must equal terminated | truncated")
        if (samples["terminated"] & samples["truncated"]).any():
            raise ValueError("Task termination takes precedence over timeout")
        adjacent = valid[:, 1:]
        for key in ("observation", "agent_state", "package_state"):
            if not torch.equal(
                samples[key][:, 1:][adjacent], samples[f"next_{key}"][:, :-1][adjacent]
            ):
                raise ValueError(f"Misaligned consecutive {key}")
        low = torch.as_tensor(self.manifest["action_low"])
        high = torch.as_tensor(self.manifest["action_high"])
        if (
            low.shape != (agents, action_dim)
            or high.shape != low.shape
            or not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or not (low < high).all()
        ):
            raise ValueError("Invalid native action bounds")
        if ((action[valid] < low) | (action[valid] > high)).any():
            raise ValueError("Action outside native bounds")

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, index):
        row = int(self.indices[index])
        samples = self.samples
        block = self.action_block
        steps, agents, action_dim = samples["action"].shape[1:]
        length = steps // block
        primitive_action = samples["action"][row].reshape(
            length, block, agents, action_dim
        )
        primitive_valid = samples["valid"][row].reshape(length, block)
        return TensorDict(
            {
                "observation": torch.cat(
                    [
                        samples["observation"][row, :1],
                        samples["next_observation"][row, block - 1 :: block],
                    ]
                ),
                # Physical state at the same block boundaries as observation.
                # This is a diagnostic target, not an input unless the caller
                # explicitly selects state_input=physical above.
                "agent_state": torch.cat(
                    [
                        samples["agent_state"][row, :1],
                        samples["next_agent_state"][row, block - 1 :: block],
                    ]
                ),
                "package_state": torch.cat(
                    [
                        samples["package_state"][row, :1],
                        samples["next_package_state"][row, block - 1 :: block],
                    ]
                ),
                "action": primitive_action.permute(0, 2, 1, 3).reshape(
                    length, agents, block * action_dim
                ),
                "primitive_action": primitive_action,
                "primitive_reward": samples["reward"][row].reshape(
                    length, block, agents, 1
                ),
                "primitive_valid": primitive_valid,
                "valid": primitive_valid.all(dim=1),
                # Dynamics validity and outcome validity are different
                # contracts. A block that terminates part-way through has no
                # observed end-of-block latent, so it cannot supply a prediction
                # target -- but its reward and termination flag are complete and
                # correct, and on Buzz Wire the terminal block is where the -10
                # collision penalty lives. Masking the readout with `valid`
                # discarded 79% of training terminations together with their
                # penalties. `valid` is a contiguous prefix, so `any` selects
                # exactly the blocks that start from an observed state, which
                # admits each terminal block once and nothing after it.
                "outcome_valid": primitive_valid.any(dim=1),
                "observation_valid": torch.cat(
                    [torch.ones(1, dtype=torch.bool), primitive_valid.all(dim=1)]
                ),
                **{
                    key: samples[key][row].reshape(length, block).any(dim=1)
                    for key in ("done", "terminated", "truncated")
                },
                "episode_id": self.episode_id[row],
                "anchor_id": samples["anchor_id"][row],
            },
            batch_size=[],
        )
