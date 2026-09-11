# Packed Hydra sweeps on Slurm

Each named Hydra config defines the complete sweep. Slurm only allocates one
GPU/MIG; Hydra Joblib runs several configurations inside that allocation.

## 1. Install

```bash
uv sync --frozen --extra vmas
```

## 2. Run the local smoke sweep

```bash
sbatch scripts/slurm/packed_local.sbatch sweep/vmas_smoke
squeue --me
```

The sweep definition is in `benchmarl/conf/sweep/vmas_smoke.yaml`. Results are
written under `multirun/YYYY-MM-DD/HH-MM-SS/`; Slurm output is written to
`slurm-JOB_ID.out`.

## 3. Define and schedule another sweep

Copy `vmas_smoke.yaml`, then edit its `hydra.sweeper.params`,
`hydra.launcher.n_jobs`, and `experiment` sections. For example:

```yaml
hydra:
  launcher:
    n_jobs: 4
  sweeper:
    params:
      algorithm: mappo,qmix,masac
      task: vmas/balance,vmas/sampling
      seed: 0,1,2,3
```

Submit any number of named sweeps. Slurm queues them and enforces your resource
limits:

```bash
sbatch scripts/slurm/packed_local.sbatch sweep/first_sweep
sbatch scripts/slurm/packed_local.sbatch sweep/second_sweep
```

## 4. Submit on one lab MIG

Override only the Slurm allocation fields; the experiment remains entirely in
the Hydra config:

```bash
sbatch \
  --partition=mig \
  --account=normal \
  --qos=normal \
  --cpus-per-task=16 \
  --mem=128G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  scripts/slurm/packed_local.sbatch sweep/my_sweep
```

Every submission is one Slurm job requesting one MIG. Submit multiple sweep
configs normally; jobs beyond your concurrent MIG limit remain queued.

## 5. Example: 16 configurations

`sweep/vmas_16` extends the smoke config and changes only the sweep grid,
concurrency, and small off-policy test settings:

```yaml
defaults:
  - vmas_smoke
  - _self_

hydra:
  launcher:
    n_jobs: 4
  sweeper:
    params:
      algorithm: mappo,ippo,masac,isac
      task: vmas/balance
      seed: 0,1,2,3
```

This is 4 algorithms × 4 seeds = 16 configurations, but only four Python
processes run concurrently. Submit it with:

```bash
sbatch scripts/slurm/packed_local.sbatch sweep/vmas_16
```

For a new sweep, copy `vmas_16.yaml`, keep the `vmas_smoke` default, and edit
the values under `hydra.sweeper.params`. Set `hydra.launcher.n_jobs` according
to measured GPU memory and throughput, not the total number of configurations.
