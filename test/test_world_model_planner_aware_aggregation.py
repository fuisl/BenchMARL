import json

import pytest
import torch

from examples.world_model.planner_aware_aggregation import (
    assert_disjoint_from_frozen_test_roots,
    write_gate5_source,
)


class StubModel:
    """A minimal stand-in for StructuredSurrogate: only `state_dict()` matters
    to `write_gate5_source`, which never runs the model."""

    def state_dict(self):
        return {"dummy": torch.zeros(1)}


def test_disjoint_check_passes_when_collection_avoids_frozen_roots():
    collection_ids = torch.tensor([0, 1, 2, 10, 11])
    assert_disjoint_from_frozen_test_roots(collection_ids, [3, 5, 6])


def test_disjoint_check_raises_when_collection_touches_a_frozen_root():
    collection_ids = torch.tensor([0, 1, 3])
    with pytest.raises(ValueError, match="disjoint"):
        assert_disjoint_from_frozen_test_roots(collection_ids, [3, 5, 6])


def test_write_gate5_source_produces_a_schema_planner_tail_failure_accepts(tmp_path):
    coverage_source = tmp_path / "job1331_coverage"
    coverage_source.mkdir()
    torch.save({"marker": "frozen-oracle-plans"}, coverage_source / "oracle_plans.pt")

    frozen = {
        "horizon": 5, "num_samples": 300, "num_elites": 30, "num_iters": 30,
        "control_seed": 8700, "test_root_ids": [3, 5, 6],
    }
    seeds = (9100, 9101, 9102)
    models = {seed: (StubModel(), {"validation_loss": 0.1}) for seed in seeds}
    output_dir = tmp_path / "g6a_targeted_round1"

    write_gate5_source(output_dir, seeds, models, frozen, coverage_source)

    result = json.loads((output_dir / "result/result.json").read_text())
    assert result["selected_objective"]["full"] == "probability"
    assert sorted(row["seed"] for row in result["fits"]) == list(seeds)
    assert result["test_root_ids"] == [3, 5, 6]
    assert result["config"]["control_seed"] == 8700
    for seed in seeds:
        assert (output_dir / f"result/model_full_{seed}.pt").exists()
    restored = torch.load(
        output_dir / "coverage/oracle_plans.pt", map_location="cpu", weights_only=False
    )
    assert restored["marker"] == "frozen-oracle-plans"
