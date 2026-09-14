# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Offline dataset alignment, episode separation and artifact identity checks."""

import hashlib
import json

import pytest
import torch

from examples.world_model.dataset import OfflineSequences
from torch.utils.data import DataLoader


@pytest.fixture
def dataset_files(tmp_path):
    count, steps, agents = 4, 6, 2
    sequence = torch.arange(count * (steps + 1) * agents * 3, dtype=torch.float32)
    sequence = sequence.reshape(count, steps + 1, agents, 3)
    state = torch.cat([sequence, sequence], dim=-1)
    packages = state[:, :, :1]
    anchors = {
        "snapshot": {"steps": torch.zeros(count)},
        "episode_id": torch.tensor([10, 10, 11, 12]),
        "split": torch.tensor([0, 0, 1, 2]),
        "source_regime": torch.tensor([0, 1, 0, 1]),
        "source_step": torch.tensor([0, 3, 0, 0]),
    }
    samples = {
        "anchor_id": torch.arange(count),
        "observation": sequence[:, :-1].clone(),
        "next_observation": sequence[:, 1:].clone(),
        "action": torch.arange(count * steps * agents * 2, dtype=torch.float32).reshape(
            count, steps, agents, 2
        )
        / 100,
        "reward": torch.ones(count, steps, agents, 1),
        "done": torch.zeros(count, steps, dtype=torch.bool),
        "terminated": torch.zeros(count, steps, dtype=torch.bool),
        "truncated": torch.zeros(count, steps, dtype=torch.bool),
        "valid": torch.ones(count, steps, dtype=torch.bool),
        "agent_state": state[:, :-1].clone(),
        "next_agent_state": state[:, 1:].clone(),
        "package_state": packages[:, :-1].clone(),
        "next_package_state": packages[:, 1:].clone(),
    }

    def write():
        manifest = {
            "schema_version": 1,
            "action_low": [[-1, -1], [-1, -1]],
            "action_high": [[1, 1], [1, 1]],
            "files": {},
        }
        for name, value in (
            ("anchors.pt", anchors),
            ("samples_independent.pt", samples),
            ("samples_correlated.pt", samples),
        ):
            path = tmp_path / name
            torch.save(value, path)
            manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    write()
    return tmp_path, anchors, samples, write


def test_action_blocks_preserve_agent_and_primitive_order(dataset_files):
    root, anchors, samples, write = dataset_files
    dataset = OfflineSequences(root, "independent", "train", action_block=3)
    item = dataset[0]
    assert len(dataset) == 2
    assert item["observation"].shape == (3, 2, 3)
    assert item["action"].shape == (2, 2, 6)
    assert item["episode_id"].item() == 10
    assert "snapshot" not in item
    torch.testing.assert_close(item["observation"][0], samples["observation"][0, 0])
    torch.testing.assert_close(
        item["observation"][1:], samples["next_observation"][0, [2, 5]]
    )
    for block in range(2):
        for agent in range(2):
            expected = samples["action"][0, 3 * block : 3 * block + 3, agent].flatten()
            torch.testing.assert_close(item["action"][block, agent], expected)
    torch.testing.assert_close(
        item["primitive_action"].flatten(1), samples["action"][0].reshape(2, -1)
    )


def test_partial_terminal_block_keeps_terminal_reward(dataset_files):
    root, anchors, samples, write = dataset_files
    samples["done"][0, 4] = True
    samples["terminated"][0, 4] = True
    samples["valid"][0, 5] = False
    samples["reward"][0, 4] = 7
    samples["reward"][0, 5] = 0
    write()
    item = OfflineSequences(root, "correlated", "train", action_block=3)[0]
    assert item["valid"].tolist() == [True, False]
    assert item["observation_valid"].tolist() == [True, True, False]
    assert item["primitive_valid"].tolist() == [[True, True, True], [True, True, False]]
    assert item["done"].tolist() == [False, True]
    assert item["terminated"].tolist() == [False, True]
    assert not item["truncated"].any()
    assert item["primitive_reward"][1, 1].sum().item() == 14


def test_split_selection_follows_anchor_ids_and_rejects_leakage(dataset_files):
    root, anchors, samples, write = dataset_files
    permutation = torch.tensor([3, 1, 0, 2])
    for key in samples:
        samples[key] = samples[key][permutation]
    write()
    for split, ids in (("train", {0, 1}), ("validation", {2}), ("test", {3})):
        dataset = OfflineSequences(root, "independent", split)
        assert {int(item["anchor_id"]) for item in dataset} == ids
        for item in dataset:
            assert int(item["episode_id"]) == int(
                anchors["episode_id"][item["anchor_id"]]
            )
    anchors["split"][1] = 2
    write()
    with pytest.raises(ValueError, match="multiple splits"):
        OfflineSequences(root, "independent", "train")


@pytest.mark.parametrize("filename", ["anchors.pt", "samples_independent.pt"])
def test_checksum_corruption_fails_before_loading(dataset_files, filename):
    root, anchors, samples, write = dataset_files
    with (root / filename).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        OfflineSequences(root, "independent", "train")


@pytest.mark.parametrize(
    "fault", ["alignment", "hole", "post_terminal", "bounds", "nan", "duplicate"]
)
def test_invalid_transitions_fail_near_source(dataset_files, fault):
    root, anchors, samples, write = dataset_files
    if fault == "alignment":
        samples["observation"][0, 1] += 1
    elif fault == "hole":
        samples["valid"][0, 1] = False
    elif fault == "post_terminal":
        samples["done"][0, 1] = samples["terminated"][0, 1] = True
    elif fault == "bounds":
        samples["action"][0, 0, 0, 0] = 2
    elif fault == "nan":
        samples["reward"][0, 0, 0, 0] = float("nan")
    else:
        samples["anchor_id"][1] = 0
    write()
    with pytest.raises(ValueError):
        OfflineSequences(root, "independent", "train")


def test_action_block_must_divide_snippet(dataset_files):
    root, anchors, samples, write = dataset_files
    with pytest.raises(ValueError, match="divisible"):
        OfflineSequences(root, "independent", "train", action_block=4)


def test_m4_batches_keep_episode_identity_and_masks(dataset_files):
    root, anchors, samples, write = dataset_files
    anchors["source_regime"][1] = 2
    write()
    dataset = OfflineSequences(root, "correlated", "train", action_block=3)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=torch.stack)))
    assert batch.batch_size == torch.Size([2])
    assert batch["observation"].shape == (2, 3, 2, 3)
    assert batch["action"].shape == (2, 2, 2, 6)
    assert batch["observation_valid"].all()
    assert batch["episode_id"].tolist() == [10, 10]
    samples["done"][0, -1] = True
    samples["terminated"][0, -1] = samples["truncated"][0, -1] = True
    write()
    with pytest.raises(ValueError, match="precedence"):
        OfflineSequences(root, "correlated", "train")
