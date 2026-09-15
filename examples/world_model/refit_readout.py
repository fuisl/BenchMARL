#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Refit the reward/termination readout on frozen dynamics.

The readouts in every existing checkpoint were trained under a mask that
required all five primitive transitions of a block to be valid. On the Buzz Wire
development bank that discarded 79% of training terminations together with their
-10 collision penalties, so the readout was never shown most of the events
carrying the dominant cost term.

That confounds the diagnosis those checkpoints produced. A readout correlation
of -0.25 against the simulator's own plan costs was reported as evidence that
Buzz Wire's reward is unrecoverable from observations -- a task property. It is
equally consistent with a readout that never saw the penalties.

Refitting isolates the bug's contribution: the dynamics weights are untouched,
so any change in the readout diagnostic comes from the corrected mask alone. If
the correlation stays negative, the observability explanation survives on its
own. If it does not, the earlier attribution was wrong.

Run:
    python -m examples.world_model.refit_readout outputs/buzz_wire_1196/baselines \\
        --data outputs/buzz_wire_1196/data --output outputs/refit
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

from examples.world_model.dataset import OfflineSequences
from examples.world_model.train import (
    evaluate,
    load_model,
    readout_losses,
    run_stage,
    write_json,
)
from examples.world_model.models import SIGReg
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


class Silent:
    """run_stage logs per epoch; a refit over many checkpoints does not need it."""

    def log_scalar(self, key, value, step=None):
        pass


def loaders(root, regime, block, batch_size, state_input, history_frames):
    return {
        split: DataLoader(
            OfflineSequences(
                root,
                regime,
                split,
                action_block=block,
                state_input=state_input,
                history_frames=history_frames,
            ),
            batch_size=batch_size,
            shuffle=split == "train",
            collate_fn=torch.stack,
        )
        for split in ("train", "validation")
    }


def refit(directory, data_root, output, device):
    """Retrain one checkpoint's readout; return its before/after metrics."""
    config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    cfg = OmegaConf.create(config)
    model = load_model(directory / "model.pt", device)
    sigreg = SIGReg(cfg.train.sigreg_knots, cfg.train.sigreg_projections).to(device)

    splits = loaders(
        data_root,
        cfg.data.regime,
        cfg.data.action_block,
        cfg.train.batch_size,
        cfg.data.get("state_input", "observation"),
        cfg.data.get("history_frames", 3),
    )
    before = evaluate(model, sigreg, splits["validation"], cfg, device)

    # Dynamics stay exactly as trained; only the readout moves.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("readout."))
    readout_parameters = [p for p in model.parameters() if p.requires_grad]
    run_stage(
        model,
        readout_parameters,
        splits["train"],
        lambda batch: readout_losses(model, batch),
        cfg.train.readout_epochs,
        cfg,
        device,
        Silent(),
        "readout",
    )
    after = evaluate(model, sigreg, splits["validation"], cfg, device)

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy(directory / "resolved_config.yaml", output / "resolved_config.yaml")
    checkpoint = torch.load(directory / "model.pt", map_location=device, weights_only=False)
    checkpoint["state_dict"] = model.state_dict()
    torch.save(checkpoint, output / "model.pt")
    write_json(output / "metrics.json", after)
    write_json(
        output / "refit.json",
        {
            "source": str(directory),
            "before": {k: before[k] for k in before if "reward" in k or "terminated" in k},
            "after": {k: after[k] for k in after if "reward" in k or "terminated" in k},
        },
    )
    return before, after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, help="sweep directory of trained models")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    checkpoints = [
        directory
        for directory in sorted(args.runs.glob("[0-9]*"))
        if (directory / "model.pt").exists()
    ][: args.max_runs]
    print(f"refitting {len(checkpoints)} readouts on frozen dynamics", flush=True)

    summary = []
    for directory in checkpoints:
        before, after = refit(
            directory, args.data, args.output / directory.name, args.device
        )
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        row = {
            "run": directory.name,
            "kind": config["model"]["kind"],
            "regime": config["data"]["regime"],
            "seed": config["seed"],
            "reward_relative_error_before": before["reward_relative_error"],
            "reward_relative_error_after": after["reward_relative_error"],
            "terminated_rate_before": before["terminated_rate"],
            "terminated_rate_after": after["terminated_rate"],
        }
        summary.append(row)
        print(
            f"  {row['kind']:11s} {row['regime']:11s} seed {row['seed']}  "
            f"reward rel. error {row['reward_relative_error_before']:.4f} -> "
            f"{row['reward_relative_error_after']:.4f}  "
            f"termination rate {row['terminated_rate_before']:.4f} -> "
            f"{row['terminated_rate_after']:.4f}",
            flush=True,
        )

    (args.output / "refit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.output}/refit_summary.json")


if __name__ == "__main__":
    main()
