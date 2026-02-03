"""
Generic partition assignment coordinator for distributed dataloading.

Assigns partitions to workers for parallel processing.
"""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slaf.distributed.data_source import DataSource


class PartitionAssignment:
    """Assignment of partitions to a worker."""

    def __init__(self, worker_id: str, partition_indices: list[int]):
        self.worker_id = worker_id
        self.partition_indices = partition_indices


class Coordinator:
    """
    Generic partition assignment coordinator.

    Assigns partitions to workers for parallel processing.
    """

    def __init__(self, data_source: "DataSource", n_workers: int):
        """
        Initialize coordinator.

        Args:
            data_source: Data source to get partition count from
            n_workers: Number of workers to assign partitions to
        """
        self.data_source = data_source
        self.n_workers = n_workers
        self.total_partitions = data_source.get_partition_count()

    def assign_partitions(self, seed: int = 42) -> dict[str, PartitionAssignment]:
        """
        Randomly assign partitions to workers.

        Args:
            seed: Random seed for partition shuffling

        Returns:
            Dictionary mapping worker_id -> PartitionAssignment
        """
        random.seed(seed)
        partition_indices = list(range(self.total_partitions))
        random.shuffle(partition_indices)

        # Calculate partitions per worker
        partitions_per_worker = self.total_partitions // self.n_workers
        remainder = self.total_partitions % self.n_workers

        assignments = {}
        start_idx = 0

        for worker_idx in range(self.n_workers):
            worker_id = f"worker_{worker_idx}"

            # Distribute remainder partitions across first workers
            worker_partition_count = partitions_per_worker
            if worker_idx < remainder:
                worker_partition_count += 1

            end_idx = start_idx + worker_partition_count
            worker_partitions = partition_indices[start_idx:end_idx]

            assignments[worker_id] = PartitionAssignment(
                worker_id=worker_id,
                partition_indices=worker_partitions,
            )

            start_idx = end_idx

        return assignments
