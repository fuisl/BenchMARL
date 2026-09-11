"""Hydra-discoverable entry point for BenchMARL's packed Slurm launcher."""

from benchmarl.launchers.packed_slurm import PackedSlurmLauncher as _PackedSlurmLauncher


class PackedSlurmLauncher(_PackedSlurmLauncher):
    """Expose the launcher from Hydra's required plugin namespace."""
