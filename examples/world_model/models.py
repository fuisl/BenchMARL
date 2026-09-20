#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Latent world-model profiles and multi-agent conditioning baselines.

The historical :class:`MultiAgentWorldModel` is the ``legacy_compact`` profile:
three baselines that differ only in how another agent's latent and action reach
the predictor.  It is intentionally frozen for checkpoint compatibility.
Audit Gate A0 adds :class:`ReferenceMultiAgentWorldModel` as a separate
``lewm_reference`` profile rather than rewriting that lineage in place.

The predictor stack, the SIGReg anti-collapse term and the action embedder are
ported from official LeWM revision 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
(MIT, lucas-maes/le-wm), files ``module.py`` and ``jepa.py``. Conventions kept
from the reference: causal autoregressive prediction over frames, AdaLN-zero
action conditioning, an undetached prediction target, and
``loss = MSE(pred, target) + weight * SIGReg(emb)``.

Deliberate legacy deviations, all recorded in experiments/04_model_baselines.md:

* the reference encodes pixels with a ViT; VMAS tasks are vector-observation, so
  a shared per-agent MLP encoder replaces it;
* ``einops`` is not a project dependency, so rearrangements use native reshapes;
* padded blocks are masked out of the prediction, readout and SIGReg terms,
  because our offline snippets end at episode boundaries while the reference
  replaces boundary values with zeros;
* the reference scores plans by terminal latent-goal distance. We plan with
  ``J = -sum_t sum_i r_i,t``, so a reward/termination readout is added here. It
  is deliberately *not* part of the dynamics objective -- see ``Readout``.
"""

import math

import torch
from torch import nn


def modulate(x, shift, scale):
    """AdaLN-zero modulation (LeWM ``module.modulate``)."""
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (LeWM ``module.SIGReg``).

    Epps-Pulley normality statistic of random 1-D projections of the latents.
    This primitive consumes ``(T,B,D)`` and averages the statistic over time.
    ``train.sigreg_loss`` owns the profile-specific population construction:
    reference models preserve time, while legacy reproduction keeps the old
    flattened population.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # trapezoidal rule
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: (T, B, D). Returns a scalar; lower means closer to isotropic."""
        directions = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        directions = directions.div_(directions.norm(p=2, dim=0))
        x_t = (proj @ directions).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class FeedForward(nn.Module):
    """LeWM ``module.FeedForward``."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Causal scaled dot-product attention over frames (LeWM ``module.Attention``)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Identity()
            if heads == 1 and dim_head == dim
            else nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        )

    def forward(self, x, causal=True):
        """x: (B, T, D)."""
        batch, frames, _ = x.shape
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            tensor.view(batch, frames, self.heads, -1).transpose(1, 2) for tensor in qkv
        )
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=causal
        )
        out = out.transpose(1, 2).reshape(batch, frames, -1)
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning (LeWM ``module.ConditionalBlock``)."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ARPredictor(nn.Module):
    """Causal next-frame latent predictor (LeWM ``module.ARPredictor``).

    Frame ``t`` attends only to frames ``<= t`` and is trained to produce frame
    ``t + 1``, so one forward pass supervises every frame in the snippet.
    """

    def __init__(self, dim, num_frames, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        self.layers = nn.ModuleList(
            ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, c):
        """x: (B, T, D) latents. c: (B, T, D) conditioning. Returns (B, T, D)."""
        x = x + self.pos_embedding[:, : x.size(1)]
        for block in self.layers:
            x = block(x, c)
        return self.norm(x)


class Embedder(nn.Module):
    """Action embedder (LeWM ``module.Embedder``): pointwise Conv1d then MLP."""

    def __init__(self, input_dim, emb_dim, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(input_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """x: (B, T, input_dim)."""
        x = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        return self.embed(x)


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Conditioner(nn.Module):
    """Build agent i's conditioning vector. The ONLY structural difference
    between the three baselines.

    ``independent``  c_i = W[a_i, z_i]                     -- no cross-agent path
    ``joint``        c_i = W[a_i, z_i, (z_j, a_j) in fixed agent order]
    ``relational``   c_i = W[a_i, z_i, sum_{j!=i} phi(z_i, z_j, a_i, a_j)]

    ``joint`` and ``relational`` receive exactly the same information; they
    differ only in inductive bias (fixed concatenation vs. permutation-equivariant
    sum pooling), which is what makes ``joint`` a control rather than a strawman.
    """

    def __init__(self, kind, dim, agents, hidden_dim):
        super().__init__()
        if kind not in ("independent", "joint", "relational"):
            raise ValueError(f"Unknown conditioner: {kind}")
        self.kind = kind
        self.agents = agents
        if kind == "independent":
            self.head = mlp(2 * dim, hidden_dim, dim)
        elif kind == "joint":
            self.head = mlp(2 * dim + 2 * agents * dim, hidden_dim, dim)
        else:
            self.pair = mlp(4 * dim, hidden_dim, dim)
            self.head = mlp(3 * dim, hidden_dim, dim)

    @staticmethod
    def hidden_for_budget(kind, dim, agents, budget):
        """Hidden width whose conditioner parameter count is closest to `budget`.

        The three conditioners consume different input widths, so a shared
        hidden width gives them different capacities (at dim=192 the joint
        conditioner is 3.7x the independent one). Solving for the width instead
        lets the baselines be compared at matched capacity, so a difference
        cannot be dismissed as one model simply being bigger.

        ``mlp(a, h, b)`` holds ``h * (a + b + 3) + b`` parameters: two linears
        plus the LayerNorm's weight and bias.
        """
        if kind == "independent":
            per_hidden, constant = 3 * dim + 3, dim
        elif kind == "joint":
            per_hidden, constant = 3 * dim + 2 * agents * dim + 3, dim
        elif kind == "relational":
            per_hidden, constant = 9 * dim + 6, 2 * dim
        else:
            raise ValueError(f"Unknown conditioner: {kind}")
        return max(1, round((budget - constant) / per_hidden))

    def forward(self, latent, action_emb):
        """latent, action_emb: (B, T, N, D). Returns (B, T, N, D)."""
        own = torch.cat([action_emb, latent], dim=-1)
        if self.kind == "independent":
            return self.head(own)

        batch, frames, agents, dim = latent.shape
        if agents != self.agents:
            raise ValueError(f"Expected {self.agents} agents, got {agents}")

        if self.kind == "joint":
            joint = torch.cat([latent, action_emb], dim=-1).reshape(
                batch, frames, 1, 2 * agents * dim
            )
            joint = joint.expand(batch, frames, agents, 2 * agents * dim)
            return self.head(torch.cat([own, joint], dim=-1))

        # Relational: sum over ordered pairs, excluding self.
        i = latent.unsqueeze(3).expand(batch, frames, agents, agents, dim)
        j = latent.unsqueeze(2).expand(batch, frames, agents, agents, dim)
        ai = action_emb.unsqueeze(3).expand(batch, frames, agents, agents, dim)
        aj = action_emb.unsqueeze(2).expand(batch, frames, agents, agents, dim)
        messages = self.pair(torch.cat([i, j, ai, aj], dim=-1))
        mask = ~torch.eye(agents, dtype=torch.bool, device=latent.device)
        pooled = (messages * mask.view(1, 1, agents, agents, 1)).sum(dim=3)
        return self.head(torch.cat([own, pooled], dim=-1))


class Readout(nn.Module):
    """Shared reward/termination readout for ``J = -sum_t sum_i r_i,t``.

    LeWM scores plans by terminal latent-goal distance and therefore has no
    reward model. Our planner needs task reward, so this head maps a latent
    transition to the block-summed per-agent reward and a termination logit.

    "Shared" means identical architecture, capacity and optimisation for all
    three baselines, not shared weights: each model learns its own latent space,
    so one set of weights cannot read all three (the same reason M4 warns
    against comparing raw MSE across separately learned latent spaces).

    It is trained in a second stage on frozen dynamics, so the reward signal
    never shapes the representation being compared.
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.reward = mlp(2 * dim, hidden_dim, 1)
        self.terminated = mlp(2 * dim, hidden_dim, 1)

    def forward(self, latent, next_latent):
        """latent, next_latent: (B, T, N, D).

        Reward is per agent because the dataset stores it per agent. Termination
        is a world event, so it is read from the agent-mean latent, which is
        permutation invariant and therefore identical for all three baselines.
        """
        pair = torch.cat([latent, next_latent], dim=-1)
        reward = self.reward(pair)
        pooled = torch.cat([latent.mean(2), next_latent.mean(2)], dim=-1)
        return reward, self.terminated(pooled).squeeze(-1)


class MultiAgentWorldModel(nn.Module):
    """Encoder + conditioner + causal predictor, with an optional readout.

    Shapes throughout: observations ``(B, T+1, N, obs_dim)``, blocked actions
    ``(B, T, N, block * act_dim)``, latents ``(B, T+1, N, dim)``.
    """

    def __init__(
        self,
        kind,
        obs_dim,
        action_dim,
        agents,
        dim=192,
        hidden_dim=512,
        conditioner_budget=None,
        frames=6,
        depth=4,
        heads=8,
        dim_head=32,
        mlp_dim=512,
        dropout=0.0,
        obs_mean=None,
        obs_std=None,
    ):
        super().__init__()
        self.profile = "legacy_compact"
        self.kind = kind
        self.agents = agents
        self.dim = dim
        self.encoder = mlp(obs_dim, hidden_dim, dim)
        self.action_encoder = Embedder(action_dim, dim)
        conditioner_hidden = (
            hidden_dim
            if conditioner_budget is None
            else Conditioner.hidden_for_budget(kind, dim, agents, conditioner_budget)
        )
        self.conditioner_hidden = conditioner_hidden
        self.conditioner = Conditioner(kind, dim, agents, conditioner_hidden)
        self.predictor = ARPredictor(
            dim, frames, depth, heads, dim_head, mlp_dim, dropout
        )
        self.readout = Readout(dim, hidden_dim)
        mean = torch.zeros(obs_dim) if obs_mean is None else obs_mean
        std = torch.ones(obs_dim) if obs_std is None else obs_std
        self.register_buffer("obs_mean", mean.clone())
        self.register_buffer("obs_std", std.clone())

    def encode(self, observation):
        """observation: (B, T+1, N, obs_dim) -> (B, T+1, N, dim)."""
        normalized = (observation - self.obs_mean) / self.obs_std
        return self.encoder(normalized)

    def predict(self, latent, action):
        """Teacher-forced next-latent prediction.

        latent: (B, T, N, dim), action: (B, T, N, action_dim).
        Returns (B, T, N, dim): position t is the prediction of latent t+1.
        """
        action_emb = self.action_encoder(
            action.transpose(1, 2).reshape(-1, action.size(1), action.size(-1))
        )
        batch, frames, agents = latent.shape[:3]
        action_emb = action_emb.view(batch, agents, frames, self.dim).transpose(1, 2)
        conditioning = self.conditioner(latent, action_emb)
        tokens = latent.transpose(1, 2).reshape(batch * agents, frames, self.dim)
        cond = conditioning.transpose(1, 2).reshape(batch * agents, frames, self.dim)
        predicted = self.predictor(tokens, cond)
        return predicted.view(batch, agents, frames, self.dim).transpose(1, 2)

    def rollout(self, latent, actions):
        """Autoregressive multi-step rollout for planning.

        latent: (B, 1, N, dim) initial encoded frame.
        actions: (B, H, N, action_dim) blocked joint actions.
        Returns predicted latents (B, H, N, dim) for steps 1..H.
        """
        history = latent
        for step in range(actions.size(1)):
            predicted = self.predict(history, actions[:, : step + 1])
            history = torch.cat([history, predicted[:, -1:]], dim=1)
        return history[:, 1:]


class ReferenceProjector(nn.Module):
    """LeWM's two-layer projector with hidden BatchNorm.

    This is a literal vector analogue of ``module.MLP`` in pinned LeWM
    revision ``8edfeb33``.  The vector encoder is the unavoidable modality
    adaptation; projector geometry is not.
    """

    def __init__(self, dim=192, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, value):
        shape = value.shape
        return self.net(value.reshape(-1, shape[-1])).reshape(shape)


class ReferenceMultiAgentWorldModel(MultiAgentWorldModel):
    """Reference-compatible LeWM vector profile.

    The encoder is necessarily an MLP rather than LeWM's image ViT.  For one
    agent, action conditioning bypasses the multi-agent conditioner and matches
    the pinned predictor interface exactly.  With multiple agents the existing
    conditioner is retained as an explicit multi-agent adaptation.

    Real observation/action history is intentionally not managed here.  That
    belongs to the planner interface and is introduced by Audit Gate A0 step 2.
    This class only establishes the reference architecture and bounded temporal
    context while keeping legacy checkpoints loadable by the original class.
    """

    def __init__(
        self,
        kind,
        obs_dim,
        action_dim,
        agents,
        dim=192,
        hidden_dim=512,
        conditioner_budget=None,
        history_size=3,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
        projector_hidden_dim=2048,
        obs_mean=None,
        obs_std=None,
    ):
        if history_size < 1:
            raise ValueError("history_size must be positive")
        super().__init__(
            kind,
            obs_dim,
            action_dim,
            agents,
            dim=dim,
            hidden_dim=hidden_dim,
            conditioner_budget=conditioner_budget,
            frames=history_size,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        self.profile = "lewm_reference"
        self.history_size = history_size
        self.projector = ReferenceProjector(dim, projector_hidden_dim)
        self.pred_proj = ReferenceProjector(dim, projector_hidden_dim)

    def encode(self, observation):
        normalized = (observation - self.obs_mean) / self.obs_std
        return self.projector(self.encoder(normalized))

    def predict(self, latent, action):
        if latent.size(1) != action.size(1):
            raise ValueError("Reference latent/action contexts must have equal length")
        if latent.size(1) > self.history_size:
            raise ValueError(
                f"Reference context exceeds history_size={self.history_size}"
            )
        action_emb = self.action_encoder(
            action.transpose(1, 2).reshape(-1, action.size(1), action.size(-1))
        )
        batch, frames, agents = latent.shape[:3]
        action_emb = action_emb.view(batch, agents, frames, self.dim).transpose(1, 2)
        conditioning = (
            action_emb
            if agents == 1
            else self.conditioner(latent, action_emb)
        )
        tokens = latent.transpose(1, 2).reshape(batch * agents, frames, self.dim)
        cond = conditioning.transpose(1, 2).reshape(batch * agents, frames, self.dim)
        predicted = self.predictor(tokens, cond)
        predicted = self.pred_proj(predicted)
        return predicted.view(batch, agents, frames, self.dim).transpose(1, 2)

    def rollout(self, latent, actions):
        raise RuntimeError(
            "lewm_reference cannot roll out from one current frame; use "
            "rollout_from_context with three real frames and two past actions"
        )

    def rollout_from_context(self, latent_history, past_actions, future_actions):
        """LeWM rollout from real temporal state/action context.

        ``latent_history`` contains ``z[t-2:t]`` and ``past_actions`` contains
        the two actions that produced the latter two real observations.  At
        each imagined step, the current candidate action completes the
        three-position predictor window.  Only the newest three latent/action
        pairs survive to the next step, exactly as in pinned LeWM ``JEPA.rollout``.
        """
        if latent_history.size(1) != self.history_size:
            raise ValueError(
                f"Reference rollout requires {self.history_size} latent frames, "
                f"got {latent_history.size(1)}"
            )
        if past_actions.size(1) != self.history_size - 1:
            raise ValueError(
                f"Reference rollout requires {self.history_size - 1} past actions, "
                f"got {past_actions.size(1)}"
            )
        if future_actions.size(1) < 1:
            raise ValueError("Reference rollout needs at least one future action")
        expected = latent_history.shape[0], latent_history.shape[2]
        for name, value in (
            ("past_actions", past_actions),
            ("future_actions", future_actions),
        ):
            if (value.shape[0], value.shape[2]) != expected:
                raise ValueError(f"{name} batch/agent dimensions do not match history")
        history = latent_history
        actions = past_actions
        predictions = []
        for current in future_actions.split(1, dim=1):
            action_context = torch.cat([actions, current], dim=1)
            predicted = self.predict(history, action_context)[:, -1:]
            predictions.append(predicted)
            history = torch.cat([history[:, 1:], predicted], dim=1)
            actions = action_context[:, 1:]
        return torch.cat(predictions, dim=1)


class SharedAgentPositionModel(nn.Module):
    """Direct next-position baseline with one predictor shared by every agent.

    This model deliberately removes the learned-latent/probe ambiguity from the
    first validation question. It predicts each agent's physical displacement
    over one action block and is trained directly against simulator position.
    The three ``kind`` values use the same information contracts as
    :class:`Conditioner`:

    ``independent``
        Agent ``i`` sees only its own observation and blocked action.
    ``joint``
        Agent ``i`` also sees every agent in a fixed order.
    ``relational``
        Agent ``i`` receives a sum of pairwise messages from the other agents.

    Encoder, message and prediction weights are shared across agent indices.
    Therefore this is a single-agent transition function applied to every
    agent, not one separately fitted network per agent.
    """

    def __init__(
        self,
        kind,
        obs_dim,
        action_dim,
        agents,
        hidden_dim=128,
        obs_mean=None,
        obs_std=None,
        action_mean=None,
        action_std=None,
        delta_mean=None,
        delta_std=None,
        conditioner_budget=100000,
    ):
        super().__init__()
        if kind not in ("independent", "joint", "relational"):
            raise ValueError(f"Unknown position model kind: {kind}")
        self.kind = kind
        self.agents = agents
        self.observation_encoder = mlp(obs_dim, hidden_dim, hidden_dim)
        self.action_encoder = mlp(action_dim, hidden_dim, hidden_dim)
        conditioner_hidden = Conditioner.hidden_for_budget(
            kind, hidden_dim, agents, conditioner_budget
        )
        self.conditioner_hidden = conditioner_hidden
        self.conditioner = Conditioner(
            kind, hidden_dim, agents, conditioner_hidden
        )
        self.head = mlp(hidden_dim, hidden_dim, 2)

        def value_or_default(value, width, default):
            value = torch.full((width,), default) if value is None else value
            return value.clone().float()

        self.register_buffer(
            "obs_mean", value_or_default(obs_mean, obs_dim, 0.0)
        )
        self.register_buffer(
            "obs_std", value_or_default(obs_std, obs_dim, 1.0)
        )
        self.register_buffer(
            "action_mean", value_or_default(action_mean, action_dim, 0.0)
        )
        self.register_buffer(
            "action_std", value_or_default(action_std, action_dim, 1.0)
        )
        self.register_buffer(
            "delta_mean", value_or_default(delta_mean, 2, 0.0)
        )
        self.register_buffer(
            "delta_std", value_or_default(delta_std, 2, 1.0)
        )

    def forward(self, observation, action):
        """Predict physical displacement ``(dx, dy)`` for every agent.

        Args:
            observation: ``(..., N, obs_dim)`` at the block start.
            action: ``(..., N, block * action_dim)`` for the same block.

        Returns:
            Displacement in simulator units with shape ``(..., N, 2)``.
        """
        if observation.shape[-2] != self.agents:
            raise ValueError(
                f"Expected {self.agents} agents, got {observation.shape[-2]}"
            )
        normalized_observation = (
            observation - self.obs_mean
        ) / self.obs_std
        normalized_action = (action - self.action_mean) / self.action_std
        obs = self.observation_encoder(normalized_observation)
        act = self.action_encoder(normalized_action)
        normalized_delta = self.head(self.conditioner(obs, act))
        return normalized_delta * self.delta_std + self.delta_mean


def parameter_counts(model):
    """Report capacity so the three baselines can be compared honestly."""
    modules = [
        ("encoder", model.encoder),
        ("action_encoder", model.action_encoder),
        ("conditioner", model.conditioner),
        ("predictor", model.predictor),
    ]
    for name in ("projector", "pred_proj"):
        if hasattr(model, name):
            modules.append((name, getattr(model, name)))
    modules.append(("readout", model.readout))
    groups = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in modules
    }
    groups["dynamics_total"] = sum(
        value for key, value in groups.items() if key != "readout"
    )
    groups["total"] = groups["dynamics_total"] + groups["readout"]
    return groups


def conditioning_gate_scale(model):
    """Mean absolute AdaLN-zero modulation weight.

    The reference initialises these to exactly zero, which makes the predictor
    the identity and its conditioning inert in both value and gradient. If this
    stays at zero after training then all three baselines are the same function
    and any comparison between them is meaningless, so it is reported with the
    validation metrics rather than left implicit.
    """
    weights = [
        block.adaLN_modulation[-1].weight.detach().abs().mean()
        for block in model.predictor.layers
    ]
    return float(torch.stack(weights).mean())


def masked_sum_count(values, mask):
    """Sum of `values` over `mask`, with the number of entries summed.

    Returned separately so metrics can be pooled correctly across batches:
    averaging per-batch means would weight batches equally even though their
    valid-entry counts differ.
    """
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values).float()
    return (values * mask).sum(), mask.sum()


def masked_mean(values, mask):
    """Mean of `values` over entries selected by `mask`, broadcast over trailing dims."""
    total, count = masked_sum_count(values, mask)
    if count == 0:
        raise ValueError("Masked reduction received no valid entries")
    return total / count


def latent_variance(latent, valid):
    """Per-dimension variance of valid latents, averaged. Collapse detector."""
    flat = latent[valid]
    if flat.numel() == 0:
        raise ValueError("No valid latents for variance check")
    return flat.reshape(-1, flat.size(-1)).var(dim=0, unbiased=False).mean()


def effective_rank(latent, valid):
    """Entropy-based effective rank of the valid latent population.

    A model that collapses to a low-dimensional subspace keeps nonzero variance
    but loses rank, so this catches a failure the variance alone does not.
    """
    flat = latent[valid].reshape(-1, latent.size(-1))
    flat = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(flat.float())
    spectrum = singular / singular.sum().clamp_min(1e-12)
    entropy = -(spectrum * spectrum.clamp_min(1e-12).log()).sum()
    return math.exp(entropy.item())
