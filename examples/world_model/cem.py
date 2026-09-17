#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Centralised CEM trajectory optimiser for multi-agent planning (M2).

Ported from ``stable_worldmodel.planning.solver.CEMSolver`` (MIT, galilai-group),
the planner LeWorldModel uses. Conventions kept from the reference:

* the sampling distribution is parameterised by a **standard deviation**, not a
  variance (the reference names it ``var`` but samples ``randn * var + mean``
  and updates it with ``.std(correction=0)``);
* the first candidate of every iteration is forced to the current mean, so the
  incumbent plan is always among those scored;
* elites are the ``num_elites`` lowest-cost candidates, and their mean and
  population std parameterise the next iteration.

Two deliberate deviations, both recorded in
docs/paper/experiments/02_oracle_validation.md:

* we plan in the environment's own action units and clamp to its bounds; the
  reference plans in ``StandardScaler``-normalised action space and does not
  clamp (its iCEM variant does);
* the joint action of ``N`` agents is flattened into ``action_dim``, so this
  optimiser stays agent-agnostic and needs no modification for the multi-agent
  case -- all cross-agent structure lives in the dynamics that ``cost_fn``
  calls. This is "centralised CEM with factorised proposals": the proposal is
  diagonal, but scoring and elite selection are jointly coupled.

Run directly for the self-contained correctness checks:
``python examples/world_model/cem.py``
"""

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class CEMConfig:
    """Defaults mirror LeWorldModel's shipped planner config (config/eval/solver/cem.yaml)."""

    horizon: int = 5
    num_samples: int = 300
    num_elites: int = 30
    num_iters: int = 30
    init_std: float = 1.0

    def __post_init__(self):
        if min(self.horizon, self.num_samples, self.num_iters, self.num_elites) < 1:
            raise ValueError(
                "CEM horizon, samples, iterations and elites must be positive"
            )
        if self.num_elites > self.num_samples:
            raise ValueError("CEM elites cannot exceed samples")
        if not math.isfinite(self.init_std) or self.init_std <= 0:
            raise ValueError(
                "CEM initial standard deviation must be finite and positive"
            )


@dataclass
class CEMResult:
    """Plan plus the diagnostics the plan-ranking metrics need.

    The elite set is returned because "elite agreement" -- does the model keep
    the candidates the simulator would keep? -- is defined on exactly these
    candidates, so they must be re-scorable by another model afterwards.
    """

    plan: torch.Tensor  # (B, H, action_dim) final mean
    candidates: torch.Tensor  # (B, S, H, action_dim) final-iteration candidates
    costs: torch.Tensor  # (B, S) their costs under the planning model
    elite_idx: torch.Tensor  # (B, num_elites) indices into candidates
    elite_cost_history: torch.Tensor  # (num_iters, B) mean elite cost per iteration
    best_cost_history: torch.Tensor  # (num_iters, B) best cost seen so far
    candidate_history: torch.Tensor | None = None  # (I,B,S,H,A), optional


@torch.no_grad()
def cem_plan(
    cost_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    action_dim: int,
    action_low: float,
    action_high: float,
    config: CEMConfig,
    batch_size: int = 1,
    init_mean: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    record_candidates: bool = False,
) -> CEMResult:
    """Optimise a joint action sequence by the cross-entropy method.

    Args:
        cost_fn: maps candidates ``(B, S, H, action_dim)`` to costs ``(B, S)``.
            Lower is better. For the multi-agent case ``action_dim`` is the
            flattened joint action ``N * d_a``.
        action_dim: flattened action dimension per planning step.
        action_low/action_high: bounds candidates are clamped to.
        config: CEM hyperparameters.
        batch_size: number of independent planning problems (e.g. eval states).
        init_mean: optional warm start ``(B, H, action_dim)``.

    Returns:
        A :class:`CEMResult`.
    """
    horizon = config.horizon
    shape = (batch_size, horizon, action_dim)
    if batch_size < 1 or action_dim < 1:
        raise ValueError("Batch size and action dimension must be positive")
    if not action_low < action_high:
        raise ValueError("Action lower bound must be below upper bound")
    if init_mean is not None and init_mean.shape != shape:
        raise ValueError(f"init_mean must have shape {shape}")

    mean = (
        torch.zeros(shape, device=device)
        if init_mean is None
        else init_mean.to(device).clone()
    )
    std = torch.full(shape, config.init_std, device=device)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    elite_cost_history = []
    best_cost_history = []
    candidate_history = []
    best_cost = torch.full((batch_size,), float("inf"), device=device)

    for _ in range(config.num_iters):
        noise = torch.randn(
            batch_size,
            config.num_samples,
            horizon,
            action_dim,
            generator=generator,
            device=device,
        )
        candidates = noise * std.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean  # keep the incumbent (reference convention)
        candidates = candidates.clamp(action_low, action_high)
        if record_candidates:
            candidate_history.append(candidates.detach().cpu())

        costs = cost_fn(candidates)  # (B, S)
        if costs.shape != (batch_size, config.num_samples):
            raise ValueError(
                f"cost_fn must return {(batch_size, config.num_samples)}, got {tuple(costs.shape)}"
            )
        if not torch.isfinite(costs).all():
            raise ValueError("CEM cost_fn returned non-finite costs")

        elite_costs, elite_idx = torch.topk(
            costs, config.num_elites, dim=1, largest=False
        )
        elites = candidates[batch_idx, elite_idx]  # (B, num_elites, H, action_dim)

        mean = elites.mean(dim=1)
        # correction=0 keeps this finite when a caller intentionally uses a
        # single elite; the sample std of one point would be NaN.
        std = elites.std(dim=1, correction=0)
        elite_cost_history.append(elite_costs.mean(dim=1))
        best_cost = torch.minimum(best_cost, elite_costs[:, 0])
        best_cost_history.append(best_cost)

    return CEMResult(
        plan=mean,
        candidates=candidates,
        costs=costs,
        elite_idx=elite_idx,
        elite_cost_history=torch.stack(elite_cost_history),
        best_cost_history=torch.stack(best_cost_history),
        candidate_history=(
            torch.stack(candidate_history) if record_candidates else None
        ),
    )


def _quadratic_cost_to(target: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Cost whose optimum is `target` -- known in closed form, so any failure is the optimiser's."""

    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        err = candidates - target.unsqueeze(1)
        return err.pow(2).sum(dim=(2, 3))

    return cost_fn


if __name__ == "__main__":
    device = "cpu"
    action_dim, horizon = 6, 5
    config = CEMConfig(horizon=horizon)
    bounds = {"action_low": -1.0, "action_high": 1.0}

    # -- L0: recover a known optimum, with no environment and no world model.
    torch.manual_seed(0)
    target = torch.rand(1, horizon, action_dim) * 1.2 - 0.6  # inside [-1, 1]
    cost_fn = _quadratic_cost_to(target)
    result = cem_plan(
        cost_fn, action_dim=action_dim, **bounds, config=config, device=device
    )
    err = (result.plan - target).abs().max().item()
    assert err < 1e-3, f"CEM failed to recover the known optimum (max err {err})"
    print(f"PASS: recovered the known quadratic optimum (max abs err {err:.2e}).")

    # -- L0: bounds hold even when the optimum is unreachable.
    far_cost = _quadratic_cost_to(torch.full((1, horizon, action_dim), 5.0))
    far = cem_plan(
        far_cost, action_dim=action_dim, **bounds, config=config, device=device
    )
    assert far.plan.max().item() <= 1.0 + 1e-6, "plan exceeded the action upper bound"
    assert far.plan.min().item() >= 0.9, f"plan did not saturate: {far.plan.min()}"
    print(
        f"PASS: respected action bounds (range [{far.plan.min():.3f}, {far.plan.max():.3f}])."
    )

    # -- L1: CEM must beat random shooting at an equal rollout budget. This is
    # the test that catches a reversed topk, a mis-gathered elite set, or a
    # collapsed std -- all of which still return a plausible-looking plan.
    torch.manual_seed(0)
    budget = config.num_iters * config.num_samples
    random_candidates = (
        torch.randn(1, budget, horizon, action_dim) * config.init_std
    ).clamp(-1.0, 1.0)
    random_best = cost_fn(random_candidates).min().item()
    cem_best = cost_fn(result.plan.unsqueeze(1))[0, 0].item()
    assert cem_best < random_best, (
        f"CEM ({cem_best:.4f}) did not beat random shooting ({random_best:.4f}) "
        f"at an equal budget of {budget} rollouts"
    )
    print(
        f"PASS: CEM beat random shooting at equal budget "
        f"({cem_best:.2e} vs {random_best:.2e}, {random_best / max(cem_best, 1e-12):.0f}x better)."
    )

    # -- L1: the elite cost must actually come down over iterations.
    history = result.elite_cost_history[:, 0]
    assert torch.isfinite(
        history
    ).all(), "elite cost history contains non-finite values"
    assert (
        history[-1] < history[0]
    ), f"elite cost did not improve: {history[0]} -> {history[-1]}"
    print(f"PASS: elite cost decreased ({history[0]:.3f} -> {history[-1]:.2e}).")

    # -- L1: same seed, same plan.
    def run_seeded():
        gen = torch.Generator(device=device).manual_seed(1234)
        return cem_plan(
            cost_fn,
            action_dim=action_dim,
            **bounds,
            config=config,
            device=device,
            generator=gen,
        ).plan

    assert torch.equal(
        run_seeded(), run_seeded()
    ), "CEM is not reproducible under a fixed seed"
    print("PASS: identical plans under a fixed seed.")

    # -- L1: a single elite must not poison the next distribution with NaNs
    # (the reference ships this exact edge case).
    single = cem_plan(
        cost_fn,
        action_dim=action_dim,
        **bounds,
        config=CEMConfig(horizon=horizon, num_elites=1, num_iters=5),
        device=device,
    )
    assert torch.isfinite(
        single.plan
    ).all(), "single-elite update produced non-finite values"
    print("PASS: single-elite update stayed finite.")

    # -- L1: a cost that ignores the actions must not crash or drift.
    flat = cem_plan(
        lambda c: torch.zeros(c.shape[0], c.shape[1]),
        action_dim=action_dim,
        **bounds,
        config=CEMConfig(horizon=horizon, num_iters=5),
        device=device,
    )
    assert torch.isfinite(flat.plan).all(), "constant cost produced non-finite values"
    print("PASS: constant cost handled without crashing.")
