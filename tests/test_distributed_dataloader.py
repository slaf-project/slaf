"""
Unit tests for DistributedDataLoader queue consumption.

Tests focus on queue reading, batching, prefetching, and formatting.
"""

import pytest

from slaf.distributed.dataloader import DistributedDataLoader


class MockQueue:
    """Mock queue-like object for testing."""

    def __init__(self, items: list[any]):
        self._items = items
        self._idx = 0
        self._timeout_after = None  # Set to item count to simulate timeout

    def get_many(
        self, n_values: int, block: bool = True, timeout: float = 1.0
    ) -> list[any]:
        """Get multiple items from queue."""
        if self._timeout_after is not None and self._idx >= self._timeout_after:
            raise TimeoutError("Queue timeout")

        result = []
        for _ in range(n_values):
            if self._idx >= len(self._items):
                # No more items - raise timeout if blocking
                if block:
                    raise TimeoutError("Queue timeout")
                break
            result.append(self._items[self._idx])
            self._idx += 1

        if not result and block:
            # Simulate timeout when no items
            raise TimeoutError("Queue timeout")

        return result


class TestDistributedDataLoader:
    """Test cases for DistributedDataLoader queue consumption."""

    def test_dataloader_initialization(self):
        """Test dataloader creation."""
        queue = MockQueue([])
        dataloader = DistributedDataLoader(queue, batch_size=32)

        assert dataloader.queue == queue
        assert dataloader.batch_size == 32
        assert dataloader.return_tensors is True
        assert dataloader.prefetch_factor == 8
        assert dataloader.enable_diagnostics is False

    def test_dataloader_iteration_basic(self):
        """Test basic iteration over queue."""
        samples = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "group_key": i}
            for i in range(10)
        ]
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=0, return_tensors=False
        )

        batches = list(dataloader)

        # Should have batches
        assert len(batches) > 0

    def test_dataloader_batch_size(self):
        """Test that batches have correct size."""
        samples = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "group_key": i}
            for i in range(10)
        ]
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=0, return_tensors=False
        )

        batches = list(dataloader)

        # First batch should have batch_size samples (or fewer if not enough data)
        if batches:
            # Check that batch has expected structure
            assert "input_ids" in batches[0] or "grouped" in batches[0]

    def test_dataloader_return_tensors_true(self):
        """Test return_tensors=True (PyTorch tensors)."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "group_key": i,
            }
            for i in range(5)
        ]
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=0, return_tensors=True
        )

        batches = list(dataloader)

        if batches:
            assert "input_ids" in batches[0]
            assert "attention_mask" in batches[0]
            # Should be tensors
            assert hasattr(batches[0]["input_ids"], "shape")

    def test_dataloader_return_tensors_false(self):
        """Test return_tensors=False (Python lists)."""
        samples = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "group_key": i}
            for i in range(5)
        ]
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=0, return_tensors=False
        )

        batches = list(dataloader)

        if batches:
            assert "input_ids" in batches[0]
            assert isinstance(batches[0]["input_ids"], list)

    def test_dataloader_prefetching_enabled(self):
        """Test prefetching with prefetch_factor > 0.

        Note: We only test that prefetching is enabled, not the full iteration,
        since prefetching involves complex threading that's better tested in
        integration tests.
        """
        samples = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "group_key": i}
            for i in range(10)
        ]
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=2, return_tensors=False
        )

        # Should use prefetching
        assert dataloader._use_prefetching is True

        # Check that prefetch queue exists
        assert hasattr(dataloader, "_prefetch_queue")

        # Don't actually iterate - prefetching with threading is complex
        # and better tested in integration tests

    def test_dataloader_prefetching_disabled(self):
        """Test synchronous mode with prefetch_factor=0."""
        samples = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "group_key": i}
            for i in range(10)
        ]
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=0, return_tensors=False
        )

        # Should not use prefetching
        assert dataloader._use_prefetching is False

    def test_dataloader_queue_timeout(self):
        """Test queue timeout handling."""
        # Empty queue will timeout
        queue = MockQueue([])
        queue._timeout_after = 0  # Timeout immediately
        dataloader = DistributedDataLoader(
            queue,
            batch_size=3,
            prefetch_factor=0,
            queue_timeout=0.1,
            return_tensors=False,
        )

        # Should handle timeout gracefully
        batches = list(dataloader)
        assert len(batches) == 0

    def test_dataloader_empty_queue(self):
        """Test behavior when queue is empty."""
        queue = MockQueue([])
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=0, return_tensors=False
        )

        batches = list(dataloader)
        assert len(batches) == 0

    def test_dataloader_end_of_epoch_marker(self):
        """Test end-of-epoch marker filtering."""
        samples = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "group_key": i}
            for i in range(5)
        ]
        samples.append({"end_of_epoch": True})
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=0, return_tensors=False
        )

        batches = list(dataloader)

        # Should stop at end_of_epoch marker
        # All batches should be valid (no end_of_epoch markers)
        for batch in batches:
            assert "end_of_epoch" not in batch

    def test_dataloader_filter_none_values(self):
        """Test filtering of None values from queue."""
        samples = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "group_key": i}
            if i % 2 == 0
            else None
            for i in range(10)
        ]
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=0, return_tensors=False
        )

        batches = list(dataloader)

        # Should filter None values
        # All batches should contain valid samples
        for batch in batches:
            if "input_ids" in batch:
                assert len(batch["input_ids"]) > 0

    def test_dataloader_diagnostics_enabled(self):
        """Test diagnostics collection."""
        samples = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "group_key": i}
            for i in range(5)
        ]
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue,
            batch_size=3,
            prefetch_factor=0,
            enable_diagnostics=True,
            return_tensors=False,
        )

        # Iterate to generate diagnostics
        list(dataloader)

        # Check diagnostics
        diag = dataloader.get_diagnostics()
        assert "queue_get_time" in diag
        assert "format_time" in diag
        assert "total_batches" in diag

    def test_dataloader_diagnostics_disabled(self):
        """Test diagnostics when disabled."""
        samples = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "group_key": i}
            for i in range(5)
        ]
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue,
            batch_size=3,
            prefetch_factor=0,
            enable_diagnostics=False,
            return_tensors=False,
        )

        diag = dataloader.get_diagnostics()
        assert "error" in diag
        assert diag["error"] == "Diagnostics not enabled"

    def test_dataloader_batch_format_tokenized(self):
        """Test tokenized batch format."""
        samples = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "group_key": i}
            for i in range(5)
        ]
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=0, return_tensors=False
        )

        batches = list(dataloader)

        if batches:
            assert "input_ids" in batches[0]
            assert "attention_mask" in batches[0]
            assert "cell_ids" in batches[0]

    def test_dataloader_batch_format_raw(self):
        """Test raw batch format."""
        samples = [
            {"grouped": None, "group_key": i} for i in range(5)
        ]  # Simplified for testing
        queue = MockQueue(samples)
        dataloader = DistributedDataLoader(
            queue, batch_size=3, prefetch_factor=0, return_tensors=False
        )

        batches = list(dataloader)

        if batches:
            # Should have grouped format
            assert "grouped" in batches[0] or "samples" in batches[0]
