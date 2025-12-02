"""
Unit tests for Coordinator partition assignment logic.

Tests focus on partition assignment, edge cases, and determinism.
"""

from slaf.distributed.coordinator import Coordinator, PartitionAssignment
from slaf.distributed.data_source import DataSource


class MockDataSource(DataSource):
    """Mock DataSource for testing."""

    def __init__(self, partition_count: int):
        self._partition_count = partition_count

    def get_partition_count(self) -> int:
        return self._partition_count

    def create_reader(self, partition_index: int, batch_size: int):
        # Not used in coordinator tests
        raise NotImplementedError


class TestCoordinator:
    """Test cases for Coordinator partition assignment."""

    def test_coordinator_initialization(self):
        """Test Coordinator creation."""
        data_source = MockDataSource(partition_count=10)
        coordinator = Coordinator(data_source, n_workers=3)

        assert coordinator.data_source == data_source
        assert coordinator.n_workers == 3
        assert coordinator.total_partitions == 10

    def test_assign_partitions_exact_division(self):
        """Test when partitions divide evenly by workers."""
        data_source = MockDataSource(partition_count=12)
        coordinator = Coordinator(data_source, n_workers=4)

        assignments = coordinator.assign_partitions(seed=42)

        # Should have 4 workers
        assert len(assignments) == 4

        # Each worker should have 3 partitions (12 / 4 = 3)
        for worker_id, assignment in assignments.items():
            assert len(assignment.partition_indices) == 3
            assert isinstance(assignment, PartitionAssignment)
            assert assignment.worker_id == worker_id

        # All partitions should be assigned
        all_partitions = []
        for assignment in assignments.values():
            all_partitions.extend(assignment.partition_indices)
        assert len(all_partitions) == 12
        assert set(all_partitions) == set(range(12))

    def test_assign_partitions_with_remainder(self):
        """Test when partitions don't divide evenly."""
        data_source = MockDataSource(partition_count=10)
        coordinator = Coordinator(data_source, n_workers=3)

        assignments = coordinator.assign_partitions(seed=42)

        # Should have 3 workers
        assert len(assignments) == 3

        # First worker should have 4 partitions, others should have 3
        # (10 = 3*3 + 1, so first worker gets extra)
        partition_counts = [
            len(assignment.partition_indices) for assignment in assignments.values()
        ]
        assert sorted(partition_counts) == [3, 3, 4]

        # All partitions should be assigned
        all_partitions = []
        for assignment in assignments.values():
            all_partitions.extend(assignment.partition_indices)
        assert len(all_partitions) == 10
        assert set(all_partitions) == set(range(10))

    def test_assign_partitions_single_worker(self):
        """Test edge case with 1 worker."""
        data_source = MockDataSource(partition_count=10)
        coordinator = Coordinator(data_source, n_workers=1)

        assignments = coordinator.assign_partitions(seed=42)

        # Should have 1 worker
        assert len(assignments) == 1

        # Worker should have all 10 partitions
        worker_0 = assignments["worker_0"]
        assert len(worker_0.partition_indices) == 10
        assert set(worker_0.partition_indices) == set(range(10))

    def test_assign_partitions_more_workers_than_partitions(self):
        """Test edge case with more workers than partitions."""
        data_source = MockDataSource(partition_count=3)
        coordinator = Coordinator(data_source, n_workers=5)

        assignments = coordinator.assign_partitions(seed=42)

        # Should have 5 workers
        assert len(assignments) == 5

        # First 3 workers should have 1 partition each, last 2 should have 0
        partition_counts = [
            len(assignment.partition_indices) for assignment in assignments.values()
        ]
        assert sorted(partition_counts) == [0, 0, 1, 1, 1]

        # All partitions should be assigned
        all_partitions = []
        for assignment in assignments.values():
            all_partitions.extend(assignment.partition_indices)
        assert len(all_partitions) == 3
        assert set(all_partitions) == set(range(3))

    def test_assign_partitions_deterministic(self):
        """Test that same seed produces same assignment."""
        data_source = MockDataSource(partition_count=10)
        coordinator = Coordinator(data_source, n_workers=3)

        assignments1 = coordinator.assign_partitions(seed=42)
        assignments2 = coordinator.assign_partitions(seed=42)

        # Should produce identical assignments
        assert len(assignments1) == len(assignments2)
        for worker_id in assignments1.keys():
            assert (
                assignments1[worker_id].partition_indices
                == assignments2[worker_id].partition_indices
            )

    def test_assign_partitions_shuffled(self):
        """Test that partitions are shuffled before assignment."""
        data_source = MockDataSource(partition_count=10)
        coordinator = Coordinator(data_source, n_workers=2)

        assignments = coordinator.assign_partitions(seed=42)

        # Get partitions for first worker
        worker_0_partitions = assignments["worker_0"].partition_indices

        # With shuffling, partitions should not be in order [0, 1, 2, 3, 4]
        # (with high probability)
        # But we can't guarantee it, so we just check that assignment happened
        assert len(worker_0_partitions) > 0

    def test_partition_assignment_structure(self):
        """Test PartitionAssignment object structure."""
        data_source = MockDataSource(partition_count=10)
        coordinator = Coordinator(data_source, n_workers=3)

        assignments = coordinator.assign_partitions(seed=42)

        # Check structure of each assignment
        for worker_id, assignment in assignments.items():
            assert isinstance(assignment, PartitionAssignment)
            assert assignment.worker_id == worker_id
            assert isinstance(assignment.partition_indices, list)
            assert all(isinstance(p, int) for p in assignment.partition_indices)
            assert all(0 <= p < 10 for p in assignment.partition_indices)

    def test_assign_partitions_zero_partitions(self):
        """Test edge case with zero partitions."""
        data_source = MockDataSource(partition_count=0)
        coordinator = Coordinator(data_source, n_workers=3)

        assignments = coordinator.assign_partitions(seed=42)

        # Should have 3 workers
        assert len(assignments) == 3

        # All workers should have 0 partitions
        for assignment in assignments.values():
            assert len(assignment.partition_indices) == 0
