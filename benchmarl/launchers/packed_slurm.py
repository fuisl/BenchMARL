"""Pack a Hydra multirun into a small number of Slurm GPU allocations."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
)
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from joblib import Parallel, delayed
from omegaconf import DictConfig, open_dict

log = logging.getLogger(__name__)

IndexedOverride = Tuple[int, List[str]]


def _execute_hydra_job(
    indexed_override: IndexedOverride,
    hydra_context: HydraContext,
    config: DictConfig,
    task_function: TaskFunction,
    singleton_state: Dict[Any, Any],
    slurm_job_id: str,
) -> Tuple[int, JobReturn]:
    """Run one concrete Hydra configuration in a Joblib child process."""
    job_num, overrides = indexed_override
    setup_globals()
    Singleton.set_state(singleton_state)

    sweep_config = hydra_context.config_loader.load_sweep_config(config, overrides)
    with open_dict(sweep_config):
        sweep_config.hydra.job.id = f"{slurm_job_id}_{job_num}"
        sweep_config.hydra.job.num = job_num
    HydraConfig.instance().set_config(sweep_config)

    result = run_job(
        hydra_context=hydra_context,
        config=sweep_config,
        task_function=task_function,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
    )
    return job_num, result


@dataclass
class _PackedSlurmTask:
    """Callable serialized by Submitit and executed once per Slurm allocation."""

    hydra_context: HydraContext
    config: DictConfig
    task_function: TaskFunction
    singleton_state: Dict[Any, Any]
    workers: int
    threads_per_worker: int

    def __call__(self, bundle: List[IndexedOverride]) -> List[Tuple[int, JobReturn]]:
        import submitit

        setup_globals()
        Singleton.set_state(self.singleton_state)
        os.environ["OMP_NUM_THREADS"] = str(self.threads_per_worker)
        os.environ["MKL_NUM_THREADS"] = str(self.threads_per_worker)

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        slurm_job_id = submitit.JobEnvironment().job_id
        log.info(
            "Slurm allocation %s is running %d Hydra jobs with %d workers",
            slurm_job_id,
            len(bundle),
            self.workers,
        )
        for job_num, overrides in bundle:
            log.info("\t#%d : %s", job_num, " ".join(filter_overrides(overrides)))

        return Parallel(
            n_jobs=min(self.workers, len(bundle)),
            backend="loky",
            prefer="processes",
        )(
            delayed(_execute_hydra_job)(
                indexed_override,
                self.hydra_context,
                self.config,
                self.task_function,
                self.singleton_state,
                slurm_job_id,
            )
            for indexed_override in bundle
        )


class PackedSlurmLauncher(Launcher):
    """Submit one MIG per Slurm job and run several Hydra jobs inside each MIG.

    ``balanced`` uses one allocation up to ``one_mig_threshold`` Hydra jobs and
    then distributes work across at most ``max_migs`` allocations. ``compact``
    always uses one allocation and is useful when allocation-hours matter more
    than sweep completion time.
    """

    def __init__(
        self,
        *,
        submitit_folder: str,
        partition: str,
        gres: str,
        timeout_min: int,
        total_cpus: int,
        mem_gb: int,
        max_migs: int = 2,
        one_mig_threshold: int = 16,
        max_workers_per_mig: int = 16,
        policy: str = "balanced",
        account: Optional[str] = None,
        qos: Optional[str] = None,
        name: str = "${hydra.job.name}",
        array_parallelism: int = 2,
        signal_delay_s: int = 120,
        additional_parameters: Optional[Dict[str, Any]] = None,
        setup: Optional[List[str]] = None,
    ) -> None:
        self.submitit_folder = submitit_folder
        self.partition = partition
        self.gres = gres
        self.timeout_min = timeout_min
        self.total_cpus = total_cpus
        self.mem_gb = mem_gb
        self.max_migs = max_migs
        self.one_mig_threshold = one_mig_threshold
        self.max_workers_per_mig = max_workers_per_mig
        self.policy = policy
        self.account = account
        self.qos = qos
        self.name = name
        self.array_parallelism = array_parallelism
        self.signal_delay_s = signal_delay_s
        self.additional_parameters = additional_parameters or {}
        self.setup_commands = setup

        self.config: Optional[DictConfig] = None
        self.hydra_context: Optional[HydraContext] = None
        self.task_function: Optional[TaskFunction] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.hydra_context = hydra_context
        self.task_function = task_function
        self.config = config

    def _allocation_count(self, experiment_count: int) -> int:
        if self.policy not in {"balanced", "compact"}:
            raise ValueError("hydra.launcher.policy must be 'balanced' or 'compact'")
        if min(self.total_cpus, self.max_migs, self.max_workers_per_mig) < 1:
            raise ValueError(
                "total_cpus, max_migs, and max_workers_per_mig must be positive"
            )
        if self.one_mig_threshold < 1:
            raise ValueError("one_mig_threshold must be positive")
        if self.policy == "compact" or experiment_count <= self.one_mig_threshold:
            return 1
        return min(self.max_migs, experiment_count, self.total_cpus)

    @staticmethod
    def _make_bundles(
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int,
        allocation_count: int,
    ) -> List[List[IndexedOverride]]:
        # Round-robin assignment is deterministic and avoids placing an entire
        # slow algorithm block in one allocation when sweep order is grouped.
        bundles: List[List[IndexedOverride]] = [[] for _ in range(allocation_count)]
        for offset, overrides in enumerate(job_overrides):
            bundles[offset % allocation_count].append(
                (initial_job_idx + offset, list(overrides))
            )
        return bundles

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        import submitit

        if (
            self.config is None
            or self.hydra_context is None
            or self.task_function is None
        ):
            raise RuntimeError("Hydra called the launcher before setup")
        if not job_overrides:
            return []

        allocation_count = self._allocation_count(len(job_overrides))
        bundles = self._make_bundles(job_overrides, initial_job_idx, allocation_count)
        cpus_per_allocation = max(1, self.total_cpus // allocation_count)
        workers = min(
            self.max_workers_per_mig,
            cpus_per_allocation,
            max(len(bundle) for bundle in bundles),
        )
        threads_per_worker = max(1, cpus_per_allocation // workers)

        executor = submitit.AutoExecutor(folder=self.submitit_folder, cluster="slurm")
        parameters: Dict[str, Any] = {
            "timeout_min": self.timeout_min,
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": cpus_per_allocation,
            "mem_gb": self.mem_gb,
            "name": self.name,
            "slurm_partition": self.partition,
            "slurm_gres": self.gres,
            "slurm_array_parallelism": min(self.array_parallelism, allocation_count),
            "slurm_signal_delay_s": self.signal_delay_s,
            "slurm_additional_parameters": self.additional_parameters,
        }
        if self.account:
            parameters["slurm_account"] = self.account
        if self.qos:
            parameters["slurm_qos"] = self.qos
        if self.setup_commands:
            parameters["slurm_setup"] = self.setup_commands
        executor.update_parameters(**parameters)

        log.info(
            "Packed Slurm plan: %d Hydra jobs -> %d allocation(s), bundle sizes=%s, "
            "workers/allocation=%d, CPUs/allocation=%d",
            len(job_overrides),
            allocation_count,
            [len(bundle) for bundle in bundles],
            workers,
            cpus_per_allocation,
        )

        Path(str(self.config.hydra.sweep.dir)).mkdir(parents=True, exist_ok=True)
        task = _PackedSlurmTask(
            hydra_context=self.hydra_context,
            config=self.config,
            task_function=self.task_function,
            singleton_state=Singleton.get_state(),
            workers=workers,
            threads_per_worker=threads_per_worker,
        )
        jobs = executor.map_array(task, bundles)

        indexed_results: List[Tuple[int, JobReturn]] = []
        for job in jobs:
            indexed_results.extend(job.result())
        indexed_results.sort(key=lambda item: item[0])
        return [result for _, result in indexed_results]
