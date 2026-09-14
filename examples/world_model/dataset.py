# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Episode-disjoint, action-aligned offline snippets for the M4 predictors."""

import hashlib
import json
from pathlib import Path

import torch

from tensordict import TensorDict
from torch.utils.data import Dataset


class OfflineSequences(Dataset):
    """Read one controlled action regime and one episode split into CPU memory.

    Each item has observations ``(L+1,N,O)`` and actions ``(L,N,block*A)``.
    Within each agent's action feature, primitive time precedes action coordinate.
    ``primitive_action`` keeps ``(L,block,N,A)`` for joint-action reconstruction;
    flatten its last three dimensions for the MPC's time-then-agent ordering.
    A block is valid only when all its primitive transitions are valid. Partial
    terminal blocks retain primitive rewards, masks and termination flags.

    Simulator snapshots never appear in model samples. Normalization is left to
    M4 and must be fitted using valid training transitions only.
    """

    def __init__(self, root, regime, split, action_block=1):
        if regime not in ("independent", "correlated"):
            raise ValueError("regime must be independent or correlated")
        splits = {"train": 0, "validation": 1, "test": 2}
        if split not in splits:
            raise ValueError("split must be train, validation or test")
        if not isinstance(action_block, int) or action_block < 1:
            raise ValueError("action_block must be a positive integer")
        self.root = Path(root)
        self.regime = regime
        self.split = split
        self.action_block = action_block
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema_version"] != 1:
            raise ValueError("Unsupported offline dataset schema_version")
        anchors = self._load("anchors.pt")
        self.samples = self._load(f"samples_{regime}.pt")
        self._validate(anchors)
        self.episode_id = anchors["episode_id"][self.samples["anchor_id"]]
        self.indices = (
            anchors["split"][self.samples["anchor_id"]] == splits[split]
        ).nonzero(as_tuple=True)[0]

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
                "action": primitive_action.permute(0, 2, 1, 3).reshape(
                    length, agents, block * action_dim
                ),
                "primitive_action": primitive_action,
                "primitive_reward": samples["reward"][row].reshape(
                    length, block, agents, 1
                ),
                "primitive_valid": primitive_valid,
                "valid": primitive_valid.all(dim=1),
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
