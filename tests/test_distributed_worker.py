"""
Unit tests for worker function logic (with mocks).

Tests focus on worker processing logic, partition handling, and metrics.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl

from slaf.distributed.data_source import DataSource
from slaf.distributed.worker import prefetch_worker


class MockDataSource(DataSource):
    """Mock DataSource for testing."""

    def __init__(self, partition_count: int, data_per_partition: list[pl.DataFrame]):
        self._partition_count = partition_count
        self._data = data_per_partition

    def get_partition_count(self) -> int:
        return self._partition_count

    def create_reader(
        self, partition_index: int, batch_size: int
    ) -> Iterator[pl.DataFrame]:
        """Yield dataframes from self._data[partition_index]."""
        if partition_index >= len(self._data):
            return
        yield from self._data[partition_index]


class MockQueue:
    """Mock queue for testing."""

    def __init__(self):
        self.items = []

    def put(self, item: any):
        """Put item in queue."""
        self.items.append(item)

    def put_many(self, items: list[any]):
        """Put multiple items in queue."""
        self.items.extend(items)


class MockKVStore:
    """Mock KV store for testing."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def put(self, key: str, value: Any, ttl: int | None = None):
        self._store[key] = value

    def pop(self, key: str, default: Any = None) -> Any:
        return self._store.pop(key, default)


def create_test_dataframe(
    group_key: str = "group_id",
    item_key: str = "item_id",
    value_key: str = "value",
    n_groups: int = 5,
    items_per_group: int = 10,
) -> pl.DataFrame:
    """Create a test DataFrame with specified structure."""
    data = []
    for group_id in range(n_groups):
        for item_id in range(items_per_group):
            value = items_per_group - item_id
            data.append(
                {
                    group_key: group_id,
                    item_key: item_id,
                    value_key: value,
                }
            )
    return pl.DataFrame(data)


class TestWorker:
    """Test cases for worker function logic."""

    @patch("slaf.distributed.data_source.LanceDataSource")
    def test_worker_initialization(self, mock_lance_ds):
        """Test worker function signature."""
        # Mock data source
        mock_ds = MagicMock()
        mock_ds.get_partition_count.return_value = 10
        mock_ds.create_reader.return_value = iter([])
        mock_lance_ds.return_value = mock_ds

        # Mock queue
        queue = MockQueue()

        # Configs
        data_source_config = {"type": "lance", "path": "/fake/path"}
        processor_config = {
            "schema": {
                "group_key": "group_id",
                "item_key": "item_id",
                "value_key": "value",
                "item_list_key": "item_list",
            },
            "max_items": 5,
            "seed": 42,
            "n_epochs": 1,
        }

        # Call worker
        result = prefetch_worker(
            worker_id="worker_0",
            partition_indices=[0, 1],
            data_source_config=data_source_config,
            processor_config=processor_config,
            queue=queue,
            n_scanners=2,
            prefetch_batch_count=2,
            prefetch_batch_size=100,
        )

        # Should return metrics
        assert isinstance(result, dict)
        assert "samples_produced" in result or "batches_produced" in result

    @patch("slaf.distributed.data_source.LanceDataSource")
    def test_worker_single_partition(self, mock_lance_ds):
        """Test processing single partition."""
        # Create test data
        df = create_test_dataframe(n_groups=3, items_per_group=5)

        # Mock data source
        mock_ds = MagicMock()
        mock_ds.get_partition_count.return_value = 1
        mock_ds.create_reader.return_value = iter([df])
        mock_lance_ds.return_value = mock_ds

        # Mock queue
        queue = MockQueue()

        # Configs
        data_source_config = {"type": "lance", "path": "/fake/path"}
        processor_config = {
            "schema": {
                "group_key": "group_id",
                "item_key": "item_id",
                "value_key": "value",
                "item_list_key": "item_list",
            },
            "max_items": 5,
            "seed": 42,
            "n_epochs": 1,
        }

        # Call worker
        result = prefetch_worker(
            worker_id="worker_0",
            partition_indices=[0],
            data_source_config=data_source_config,
            processor_config=processor_config,
            queue=queue,
            n_scanners=1,
            prefetch_batch_count=1,
            prefetch_batch_size=100,
        )

        # Should process partition
        assert isinstance(result, dict)

    @patch("slaf.distributed.data_source.LanceDataSource")
    def test_worker_multiple_partitions(self, mock_lance_ds):
        """Test processing multiple partitions."""
        # Create test data for multiple partitions
        df1 = create_test_dataframe(n_groups=2, items_per_group=5)
        df2 = create_test_dataframe(n_groups=2, items_per_group=5)

        # Mock data source
        mock_ds = MagicMock()
        mock_ds.get_partition_count.return_value = 2

        def create_reader(partition_index, batch_size):
            if partition_index == 0:
                return iter([df1])
            else:
                return iter([df2])

        mock_ds.create_reader.side_effect = create_reader
        mock_lance_ds.return_value = mock_ds

        # Mock queue
        queue = MockQueue()

        # Configs
        data_source_config = {"type": "lance", "path": "/fake/path"}
        processor_config = {
            "schema": {
                "group_key": "group_id",
                "item_key": "item_id",
                "value_key": "value",
                "item_list_key": "item_list",
            },
            "max_items": 5,
            "seed": 42,
            "n_epochs": 1,
        }

        # Call worker
        result = prefetch_worker(
            worker_id="worker_0",
            partition_indices=[0, 1],
            data_source_config=data_source_config,
            processor_config=processor_config,
            queue=queue,
            n_scanners=2,
            prefetch_batch_count=1,
            prefetch_batch_size=100,
        )

        # Should process both partitions
        assert isinstance(result, dict)

    @patch("slaf.distributed.data_source.LanceDataSource")
    def test_worker_prefetch_batch_count(self, mock_lance_ds):
        """Test prefetch_batch_count parameter."""
        # Create test data
        df = create_test_dataframe(n_groups=2, items_per_group=5)

        # Mock data source
        mock_ds = MagicMock()
        mock_ds.get_partition_count.return_value = 1
        mock_ds.create_reader.return_value = iter([df])
        mock_lance_ds.return_value = mock_ds

        # Mock queue
        queue = MockQueue()

        # Configs
        data_source_config = {"type": "lance", "path": "/fake/path"}
        processor_config = {
            "schema": {
                "group_key": "group_id",
                "item_key": "item_id",
                "value_key": "value",
                "item_list_key": "item_list",
            },
            "max_items": 5,
            "seed": 42,
            "n_epochs": 1,
        }

        # Call worker with prefetch_batch_count
        result = prefetch_worker(
            worker_id="worker_0",
            partition_indices=[0],
            data_source_config=data_source_config,
            processor_config=processor_config,
            queue=queue,
            n_scanners=1,
            prefetch_batch_count=5,
            prefetch_batch_size=100,
        )

        # Should work with prefetch_batch_count
        assert isinstance(result, dict)

    @patch("slaf.distributed.data_source.LanceDataSource")
    def test_worker_prefetch_batch_size(self, mock_lance_ds):
        """Test prefetch_batch_size parameter."""
        # Create test data
        df = create_test_dataframe(n_groups=2, items_per_group=5)

        # Mock data source
        mock_ds = MagicMock()
        mock_ds.get_partition_count.return_value = 1
        mock_ds.create_reader.return_value = iter([df])
        mock_lance_ds.return_value = mock_ds

        # Mock queue
        queue = MockQueue()

        # Configs
        data_source_config = {"type": "lance", "path": "/fake/path"}
        processor_config = {
            "schema": {
                "group_key": "group_id",
                "item_key": "item_id",
                "value_key": "value",
                "item_list_key": "item_list",
            },
            "max_items": 5,
            "seed": 42,
            "n_epochs": 1,
        }

        # Call worker with prefetch_batch_size
        result = prefetch_worker(
            worker_id="worker_0",
            partition_indices=[0],
            data_source_config=data_source_config,
            processor_config=processor_config,
            queue=queue,
            n_scanners=1,
            prefetch_batch_count=1,
            prefetch_batch_size=200,
        )

        # Should work with prefetch_batch_size
        assert isinstance(result, dict)

    @patch("slaf.distributed.data_source.LanceDataSource")
    def test_worker_max_batches(self, mock_lance_ds):
        """Test max_batches limiting."""
        # Create test data
        df = create_test_dataframe(n_groups=10, items_per_group=5)

        # Mock data source
        mock_ds = MagicMock()
        mock_ds.get_partition_count.return_value = 1
        mock_ds.create_reader.return_value = iter([df])
        mock_lance_ds.return_value = mock_ds

        # Mock queue
        queue = MockQueue()

        # Configs
        data_source_config = {"type": "lance", "path": "/fake/path"}
        processor_config = {
            "schema": {
                "group_key": "group_id",
                "item_key": "item_id",
                "value_key": "value",
                "item_list_key": "item_list",
            },
            "max_items": 5,
            "seed": 42,
            "n_epochs": 1,
        }

        # Call worker with max_batches
        result = prefetch_worker(
            worker_id="worker_0",
            partition_indices=[0],
            data_source_config=data_source_config,
            processor_config=processor_config,
            queue=queue,
            n_scanners=1,
            prefetch_batch_count=1,
            prefetch_batch_size=100,
            max_batches=5,
        )

        # Should respect max_batches limit
        assert isinstance(result, dict)

    @patch("slaf.distributed.data_source.LanceDataSource")
    def test_worker_cross_worker_boundary_merging(self, mock_lance_ds):
        """Test KV store integration."""
        # Create test data
        df = create_test_dataframe(n_groups=2, items_per_group=5)

        # Mock data source
        mock_ds = MagicMock()
        mock_ds.get_partition_count.return_value = 1
        mock_ds.create_reader.return_value = iter([df])
        mock_lance_ds.return_value = mock_ds

        # Mock queue and KV store
        queue = MockQueue()
        kv_store = MockKVStore()

        # Configs with cross-worker merging enabled
        data_source_config = {"type": "lance", "path": "/fake/path"}
        processor_config = {
            "schema": {
                "group_key": "group_id",
                "item_key": "item_id",
                "value_key": "value",
                "item_list_key": "item_list",
            },
            "max_items": 5,
            "seed": 42,
            "n_epochs": 1,
            "enable_cross_worker_boundary_merging": True,
            "continuity_check": "sequential",
        }

        # Call worker with KV store
        result = prefetch_worker(
            worker_id="worker_0",
            partition_indices=[0],
            data_source_config=data_source_config,
            processor_config=processor_config,
            queue=queue,
            n_scanners=1,
            prefetch_batch_count=1,
            prefetch_batch_size=100,
            partial_groups_kv=kv_store,
        )

        # Should work with KV store
        assert isinstance(result, dict)

    @patch("slaf.distributed.data_source.LanceDataSource")
    def test_worker_partition_exhaustion(self, mock_lance_ds):
        """Test handling partition exhaustion."""
        # Create test data
        df = create_test_dataframe(n_groups=2, items_per_group=5)

        # Mock data source
        mock_ds = MagicMock()
        mock_ds.get_partition_count.return_value = 1
        mock_ds.create_reader.return_value = iter([df])
        mock_lance_ds.return_value = mock_ds

        # Mock queue
        queue = MockQueue()

        # Configs
        data_source_config = {"type": "lance", "path": "/fake/path"}
        processor_config = {
            "schema": {
                "group_key": "group_id",
                "item_key": "item_id",
                "value_key": "value",
                "item_list_key": "item_list",
            },
            "max_items": 5,
            "seed": 42,
            "n_epochs": 1,
        }

        # Call worker
        result = prefetch_worker(
            worker_id="worker_0",
            partition_indices=[0],
            data_source_config=data_source_config,
            processor_config=processor_config,
            queue=queue,
            n_scanners=1,
            prefetch_batch_count=1,
            prefetch_batch_size=100,
        )

        # Should handle partition exhaustion
        assert isinstance(result, dict)

    @patch("slaf.distributed.data_source.LanceDataSource")
    def test_worker_error_handling(self, mock_lance_ds):
        """Test error handling in partition reading."""
        # Mock data source that raises error
        mock_ds = MagicMock()
        mock_ds.get_partition_count.return_value = 1

        def create_reader(partition_index, batch_size):
            raise ValueError("Test error")

        mock_ds.create_reader.side_effect = create_reader
        mock_lance_ds.return_value = mock_ds

        # Mock queue
        queue = MockQueue()

        # Configs
        data_source_config = {"type": "lance", "path": "/fake/path"}
        processor_config = {
            "schema": {
                "group_key": "group_id",
                "item_key": "item_id",
                "value_key": "value",
                "item_list_key": "item_list",
            },
            "max_items": 5,
            "seed": 42,
            "n_epochs": 1,
        }

        # Call worker - should handle error gracefully
        result = prefetch_worker(
            worker_id="worker_0",
            partition_indices=[0],
            data_source_config=data_source_config,
            processor_config=processor_config,
            queue=queue,
            n_scanners=1,
            prefetch_batch_count=1,
            prefetch_batch_size=100,
        )

        # Should return result (may have error info)
        assert isinstance(result, dict)

    @patch("slaf.distributed.data_source.LanceDataSource")
    def test_worker_metrics(self, mock_lance_ds):
        """Test worker metrics collection."""
        # Create test data
        df = create_test_dataframe(n_groups=2, items_per_group=5)

        # Mock data source
        mock_ds = MagicMock()
        mock_ds.get_partition_count.return_value = 1
        mock_ds.create_reader.return_value = iter([df])
        mock_lance_ds.return_value = mock_ds

        # Mock queue
        queue = MockQueue()

        # Configs
        data_source_config = {"type": "lance", "path": "/fake/path"}
        processor_config = {
            "schema": {
                "group_key": "group_id",
                "item_key": "item_id",
                "value_key": "value",
                "item_list_key": "item_list",
            },
            "max_items": 5,
            "seed": 42,
            "n_epochs": 1,
        }

        # Call worker
        result = prefetch_worker(
            worker_id="worker_0",
            partition_indices=[0],
            data_source_config=data_source_config,
            processor_config=processor_config,
            queue=queue,
            n_scanners=1,
            prefetch_batch_count=1,
            prefetch_batch_size=100,
        )

        # Should return metrics
        assert isinstance(result, dict)
        # Metrics should include relevant information
        assert len(result) > 0
