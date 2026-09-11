# Packed Hydra sweeps on Slurm

The packed launcher keeps the normal BenchMARL/Hydra multirun syntax. Hydra
expands the Cartesian sweep first; the launcher then submits one or two Slurm
jobs and runs several experiment processes inside each allocated GPU/MIG.

## 1. Install

```bash
uv sync --frozen --extra vmas
```

Run this on the login node from the repository root. The repository and its
`.venv` must be visible from the worker nodes.

## 2. Test locally

```bash
uv run python benchmarl/run.py \
  --config-name sweep/vmas_smoke \
  --multirun \
  hydra/launcher=packed_local
```

This submits a real, non-interactive Slurm batch job to the local partition.
Use `squeue --me` to monitor it. Results and Submitit logs are under
`multirun/YYYY-MM-DD/HH-MM-SS/`.

## 3. Submit a lab sweep

Use the same multirun shown in the BenchMARL README, adding only the launcher:

```bash
uv run python benchmarl/run.py --multirun \
  hydra/launcher=packed_mig \
  algorithm=mappo,ippo,masac \
  task=vmas/balance \
  seed=0,1,2,3,4,5
```

That command contains 3 algorithms x 1 task x 6 seeds = 18 experiments. With
the checked-in `packed_mig` policy it produces:

- two Slurm array elements, each requesting exactly one 3g.40gb MIG;
- nine experiment configurations in each element, assigned round-robin;
- eight concurrent Python workers per element because the two jobs split the
  16-CPU user limit; and
- one queued configuration in each element, started immediately when any
  worker finishes (there is no wave barrier).

The Slurm resource settings are in
`benchmarl/conf/hydra/launcher/packed_mig.yaml`. The generated batch script uses
`sbatch --array=0-1%2` and launches each allocation with `srun` through
Submitit. If only one MIG is currently free, the other element waits in the
queue; it does not duplicate the sweep.

## Policy and tuning

`policy: balanced` is the default:

- 1-16 experiments: one MIG;
- more than 16 experiments: two MIGs, split as evenly as possible; and
- more configurations than workers: workers dynamically take the next config
  as they finish.

For 18 experiments, `balanced` is the throughput/headroom-first choice. It is
faster only when sharing one MIG is a meaningful GPU bottleneck. Because the
cluster still caps the two jobs at 16 CPUs in total, both layouts have at most
16 active experiment processes.

For light experiments, `compact` is usually the allocation-cost-first starting
point: keep all 18 on one MIG and let two wait for a worker:

```bash
uv run python benchmarl/run.py --multirun \
  hydra/launcher=packed_mig \
  hydra.launcher.policy=compact \
  algorithm=mappo,ippo,masac \
  task=vmas/balance \
  seed=0,1,2,3,4,5
```

Compact mode still caps concurrency at 16; it runs the remaining two configs
as workers become free. Running 18 simultaneously would exceed the stated
16-CPU quota. Benchmark a representative subset with both policies: use
`balanced` only if its reduced run time justifies occupying the second MIG.

The number `16` is a scheduling limit, not proof that a real training config
fits efficiently. GPU compute can saturate before VRAM does. Start lower when
moving beyond the smoke settings, for example:

```bash
hydra.launcher.max_workers_per_mig=4
```

Then increase only after measuring peak VRAM, GPU utilization, throughput, and
host RAM. `one_mig_threshold` controls when a second MIG is requested;
`max_workers_per_mig` independently controls process concurrency.

Submitit waits in the invoking shell so Hydra can collect all results, but the
Slurm jobs are batch jobs and continue after an SSH disconnect once submitted.
Use `squeue --me` and the Submitit logs to reconnect to their status; use
`tmux` if retaining the launcher's final exit status matters.
