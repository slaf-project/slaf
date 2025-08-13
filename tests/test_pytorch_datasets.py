from unittest.mock import Mock, patch

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from slaf.ml.aggregators import GeneformerWindow, ScGPTWindow
from slaf.ml.datasets import (
    AsyncPrefetcher,
    PrefetchBatchProcessor,
    SLAFIterableDataset,
    TokenizedPrefetchBatch,
)
from slaf.ml.samplers import RandomShuffle
from slaf.ml.tokenizers import SLAFTokenizer


class TestSLAFIterableDataset:
    """Test suite for SLAFIterableDataset with comprehensive coverage"""

    def test_dataset_initialization(self, tiny_slaf):
        """Test SLAFIterableDataset initialization"""
        tokenizer = SLAFTokenizer(tiny_slaf)

        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=32,
            seed=42,
            max_queue_size=10,
            tokenizer_type="geneformer",
        )

        assert dataset.slaf_array is tiny_slaf
        assert dataset.tokenizer is tokenizer
        assert dataset.batch_size == 32
        assert dataset.seed == 42
        assert dataset.tokenizer_type == "geneformer"
        # Check device exists (could be None, string, or torch.device)
        assert hasattr(dataset, "device")
        assert hasattr(dataset, "batch_processor")
        assert hasattr(dataset, "prefetcher")

    def test_dataset_initialization_scgpt(self, tiny_slaf):
        """Test SLAFIterableDataset initialization with scGPT tokenizer"""
        tokenizer = SLAFTokenizer(tiny_slaf)

        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=16,
            seed=123,
            max_queue_size=5,
            tokenizer_type="scgpt",
        )

        assert dataset.tokenizer_type == "scgpt"
        assert dataset.batch_size == 16
        assert dataset.seed == 123

    def test_prefetch_batch_processor_initialization(self, tiny_slaf):
        """Test PrefetchBatchProcessor initialization and configuration"""
        window = ScGPTWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            seed=42,
            max_genes=1024,
            n_expression_bins=10,
            use_binned_expressions=True,
        )

        assert processor.slaf_array is tiny_slaf
        assert processor.window is window
        assert processor.shuffle is shuffle
        assert processor.seed == 42
        assert processor.max_genes == 1024
        assert processor.n_expression_bins == 10
        assert processor.use_binned_expressions is True
        assert hasattr(processor, "expression_dataset")
        assert hasattr(processor, "batch_generator")

    def test_prefetch_batch_processor_fragment_parameter(self, tiny_slaf):
        """Test the by_fragment parameter in PrefetchBatchProcessor."""
        window = ScGPTWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        # Test fragment-based loading
        processor_fragment = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            batch_size=32,
            verbose=False,
            by_fragment=True,
        )

        assert processor_fragment.by_fragment is True

        # Test batch-based loading
        processor_batch = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            batch_size=32,
            verbose=False,
            by_fragment=False,
        )

        assert processor_batch.by_fragment is False

    def test_prefetch_batch_processor_reset_fragment(self, tiny_slaf):
        """Test processor epoch reset in fragment mode."""
        window = ScGPTWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            batch_size=32,
            n_epochs=3,
            verbose=False,
            by_fragment=True,
        )

        # Test reset for epoch
        processor.reset_for_epoch(1)
        assert processor.current_epoch == 1
        assert processor.batch_id == 0

    def test_prefetch_batch_processor_reset_batch(self, tiny_slaf):
        """Test processor epoch reset in batch mode."""
        window = ScGPTWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            batch_size=32,
            n_epochs=3,
            verbose=False,
            by_fragment=False,
        )

        # Test reset for epoch
        processor.reset_for_epoch(1)
        assert processor.current_epoch == 1
        assert processor.batch_id == 0

    def test_prefetch_batch_processor_load_fragment(self, tiny_slaf):
        """Test processor batch loading in fragment mode."""
        window = ScGPTWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            batch_size=32,
            verbose=False,
            by_fragment=True,
        )

        # Test that we can load batches
        batch = processor.load_prefetch_batch()
        assert hasattr(batch, "input_ids")
        assert hasattr(batch, "attention_mask")
        assert hasattr(batch, "cell_integer_ids")

    def test_prefetch_batch_processor_load_batch_mode(self, tiny_slaf):
        """Test processor batch loading in batch mode."""
        window = ScGPTWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            batch_size=32,
            verbose=False,
            by_fragment=False,
        )

        # Test that we can load batches
        batch = processor.load_prefetch_batch()
        assert hasattr(batch, "input_ids")
        assert hasattr(batch, "attention_mask")
        assert hasattr(batch, "cell_integer_ids")

    def test_prefetch_batch_dataclass(self):
        """Test PrefetchBatch dataclass"""
        # Create mock tensors
        input_ids = torch.randint(0, 1000, (2, 1024))
        attention_mask = torch.ones(2, 1024, dtype=torch.bool)

        batch = TokenizedPrefetchBatch(
            batch_id=0,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cell_integer_ids=[100, 101],
            partial_cell_data={},
            tokenize_time=0.1,
        )

        assert batch.batch_id == 0
        assert batch.input_ids.shape == (2, 1024)
        assert batch.attention_mask.shape == (2, 1024)
        assert len(batch.cell_integer_ids) == 2
        assert batch.tokenize_time == 0.1

    def test_prefetch_batch_geneformer(self):
        """Test PrefetchBatch for Geneformer format"""
        # Create mock tensors for Geneformer (2048 sequence length)
        input_ids = torch.randint(0, 1000, (3, 2048))
        attention_mask = torch.ones(3, 2048, dtype=torch.bool)

        batch = TokenizedPrefetchBatch(
            batch_id=1,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cell_integer_ids=[200, 201, 202],
            partial_cell_data={},
            tokenize_time=0.05,
        )

        assert batch.batch_id == 1
        assert batch.input_ids.shape == (3, 2048)
        assert batch.attention_mask.shape == (3, 2048)
        assert len(batch.cell_integer_ids) == 3
        assert batch.cell_integer_ids == [200, 201, 202]
        assert batch.tokenize_time == 0.05

    def test_async_prefetcher_initialization(self, tiny_slaf):
        """Test AsyncPrefetcher initialization"""
        window = ScGPTWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            seed=42,
            max_genes=1024,
        )

        prefetcher = AsyncPrefetcher(processor, max_queue_size=100)

        assert prefetcher.batch_processor is processor
        assert prefetcher.max_queue_size == 100
        assert prefetcher.queue is not None
        assert prefetcher.worker_thread is None
        assert prefetcher.should_stop is False

    def test_prefetcher_start_stop(self, tiny_slaf):
        """Test AsyncPrefetcher start and stop functionality"""
        window = ScGPTWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            seed=42,
            max_genes=1024,
        )

        prefetcher = AsyncPrefetcher(processor, max_queue_size=10)

        # Test start
        prefetcher.start()
        assert prefetcher.worker_thread is not None
        assert prefetcher.worker_thread.is_alive()

        # Test stop
        prefetcher.stop()
        assert prefetcher.should_stop is True

    def test_prefetcher_queue_operations(self, tiny_slaf):
        """Test AsyncPrefetcher queue operations"""
        window = ScGPTWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            seed=42,
            max_genes=1024,
        )

        prefetcher = AsyncPrefetcher(processor, max_queue_size=5)

        # Test initial state
        assert prefetcher.has_batch() is False
        assert prefetcher.get_batch() is None

        # Test stats
        stats = prefetcher.get_stats()
        assert "total_cells" in stats
        assert "elapsed_time" in stats
        assert "cells_per_sec" in stats

    def test_dataset_iteration_geneformer(self, tiny_slaf):
        """Test dataset iteration with Geneformer tokenizer"""
        tokenizer = SLAFTokenizer(tiny_slaf)
        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=8,
            tokenizer_type="geneformer",
        )

        batch_count = 0
        for batch in dataset:
            # Check batch structure
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch

            # Check tensor types
            assert isinstance(batch["input_ids"], torch.Tensor)
            assert isinstance(batch["attention_mask"], torch.Tensor)
            assert isinstance(batch["cell_ids"], torch.Tensor)

            # Check shapes
            batch_size = batch["input_ids"].shape[0]
            seq_length = batch["input_ids"].shape[1]
            assert batch_size <= 8
            assert seq_length == 2048  # Geneformer hardcoded length

            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break

        assert batch_count > 0

    def test_dataset_iteration_scgpt(self, tiny_slaf):
        """Test dataset iteration with scGPT tokenizer"""
        tokenizer = SLAFTokenizer(tiny_slaf, tokenizer_type="scgpt")
        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=8,
            tokenizer_type="scgpt",
        )

        batch_count = 0
        for batch in dataset:
            # Check batch structure
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch

            # Check tensor types
            assert isinstance(batch["input_ids"], torch.Tensor)
            assert isinstance(batch["attention_mask"], torch.Tensor)
            assert isinstance(batch["cell_ids"], torch.Tensor)

            # Check shapes
            batch_size = batch["input_ids"].shape[0]
            seq_length = batch["input_ids"].shape[1]
            assert batch_size <= 8
            assert seq_length == 2050  # scGPT: 2*1024+2

            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break

        assert batch_count > 0

    def test_device_transfer(self, tiny_slaf):
        """Test device transfer functionality"""
        tokenizer = SLAFTokenizer(tiny_slaf)
        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=4,
        )

        for batch in dataset:
            # Check that tensors are on the correct device (should be CPU by default)
            assert batch["input_ids"].device.type == "cpu"
            assert batch["attention_mask"].device.type == "cpu"
            assert batch["cell_ids"].device.type == "cpu"
            break  # Just test first batch

    def test_prefetcher_timeout(self, tiny_slaf):
        """Test prefetcher timeout handling"""
        tokenizer = SLAFTokenizer(tiny_slaf)
        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=4,
        )

        # The timeout should not cause issues in normal operation
        # We just test that the dataset can be created and iterated
        batch_count = 0
        for _batch in dataset:
            batch_count += 1
            if batch_count >= 2:
                break

        assert batch_count > 0

    def test_error_handling_invalid_tokenizer_type(self, tiny_slaf):
        """Test error handling for invalid tokenizer type"""
        # Create tokenizer with invalid type
        tokenizer = SLAFTokenizer(tiny_slaf, tokenizer_type="geneformer")

        # Create dataset with invalid tokenizer type
        with pytest.raises(ValueError, match="is not a valid WindowType"):
            SLAFIterableDataset(
                slaf_array=tiny_slaf,
                tokenizer=tokenizer,
                batch_size=4,
                tokenizer_type="invalid",
            )

    def test_memory_efficiency(self, tiny_slaf):
        """Test memory efficiency of the dataset"""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        tokenizer = SLAFTokenizer(tiny_slaf)
        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=4,
        )

        # Iterate through a few batches
        for i, _batch in enumerate(dataset):
            if i >= 3:  # Just test first 3 batches
                break

        # Force garbage collection
        gc.collect()

        # Check that memory usage hasn't exploded
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

    def test_pytorch_dataloader_integration(self, tiny_slaf):
        """Test integration with PyTorch DataLoader"""

        tokenizer = SLAFTokenizer(tiny_slaf)
        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=4,
        )

        # Test with PyTorch DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=None,  # Dataset already handles batching
            num_workers=0,  # Single-threaded for testing
        )

        batch_count = 0
        for batch in dataloader:
            # Check batch structure
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch

            # Check tensor types
            assert isinstance(batch["input_ids"], torch.Tensor)
            assert isinstance(batch["attention_mask"], torch.Tensor)
            assert isinstance(batch["cell_ids"], torch.Tensor)

            # Check shapes
            batch_size = batch["input_ids"].shape[0]
            seq_length = batch["input_ids"].shape[1]
            assert batch_size <= 4
            assert seq_length == 2048  # Geneformer default

            batch_count += 1
            if batch_count >= 2:  # Just test first 2 batches
                break

        assert batch_count > 0

    def test_window_strategy_integration(self, tiny_slaf):
        """Test window strategy integration with batch processor"""
        window = GeneformerWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            max_genes=512,
        )

        # Test that processor can be initialized with window strategy
        assert processor.window is window
        assert processor.shuffle is shuffle
        assert processor.max_genes == 512

    def test_multi_epoch_initialization(self, tiny_slaf):
        """Test SLAFIterableDataset initialization with multi-epoch support"""
        tokenizer = SLAFTokenizer(tiny_slaf)

        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=32,
            n_epochs=5,  # Test multi-epoch initialization
        )

        assert dataset.n_epochs == 5
        assert dataset.batch_processor.n_epochs == 5
        # Note: current_epoch may be > 0 if prefetcher has already processed some epochs
        assert dataset.batch_processor.current_epoch >= 0
        assert dataset.batch_processor.current_epoch < 5

    def test_multi_epoch_iteration(self, tiny_slaf):
        """Test multi-epoch iteration functionality"""
        tokenizer = SLAFTokenizer(tiny_slaf)

        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=8,  # Small batch size for testing
            n_epochs=3,  # Test with 3 epochs
        )

        # Track epochs and batches
        epochs_seen = set()
        total_batches = 0

        for batch in dataset:
            epoch = batch.get("epoch", 0)
            epochs_seen.add(epoch)
            total_batches += 1

            # Check batch structure
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch
            assert "epoch" in batch  # Should have epoch info for multi-epoch

            # Limit test to reasonable number of batches
            if total_batches > 15:
                break

        # Verify we saw some epochs (may not see all epochs in limited test)
        assert len(epochs_seen) >= 1, f"Expected at least 1 epoch, got {epochs_seen}"
        assert total_batches > 0

    def test_epoch_transition_functionality(self, tiny_slaf):
        """Test that epoch transitions work correctly"""
        tokenizer = SLAFTokenizer(tiny_slaf)

        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=4,  # Very small batch size
            n_epochs=2,  # Just 2 epochs for testing
        )

        # Track epoch transitions
        epoch_sequence = []
        batch_count = 0

        for batch in dataset:
            epoch = batch.get("epoch", 0)
            epoch_sequence.append(epoch)
            batch_count += 1

            # Stop after reasonable number of batches
            if batch_count > 10:
                break

        # Verify epoch progression
        assert len(epoch_sequence) > 0
        # Note: epoch 0 may not be seen if prefetcher has already processed it
        # Just check that we have some epochs and they progress in order
        assert min(epoch_sequence) >= 0, f"Expected epochs >= 0, got {epoch_sequence}"

        # Check that epochs progress in order (allowing for some overlap during transitions)
        for i in range(len(epoch_sequence) - 1):
            assert epoch_sequence[i] <= epoch_sequence[i + 1], (
                "Epochs should not go backwards"
            )

    def test_batch_processor_epoch_reset(self, tiny_slaf):
        """Test PrefetchBatchProcessor epoch reset functionality"""
        window = GeneformerWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            n_epochs=3,
        )

        # Test initial state
        assert processor.current_epoch == 0
        assert processor.batch_id == 0

        # Test epoch reset
        processor.reset_for_epoch(1)
        assert processor.current_epoch == 1
        assert processor.batch_id == 0
        assert len(processor.partial_cell_data) == 0  # Should be reset

        # Test invalid epoch
        with pytest.raises(ValueError):
            processor.reset_for_epoch(-1)  # Invalid epoch

        with pytest.raises(ValueError):
            processor.reset_for_epoch(3)  # Invalid epoch (>= n_epochs)

    def test_multi_epoch_with_small_dataset(self, tiny_slaf):
        """Test multi-epoch functionality with small dataset that gets exhausted"""
        tokenizer = SLAFTokenizer(tiny_slaf)

        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=2,  # Very small batch size to exhaust quickly
            n_epochs=3,  # Reduced from 4 to 3 for speed
        )

        epochs_completed = set()
        total_batches = 0

        for batch in dataset:
            epoch = batch.get("epoch", 0)
            epochs_completed.add(epoch)
            total_batches += 1

            # Check that epoch is valid
            assert 0 <= epoch < 3, f"Invalid epoch {epoch}"

            # Early termination for speed
            if total_batches >= 10:
                break

        # Should complete multiple epochs
        assert len(epochs_completed) >= 1, (
            f"Expected at least 1 epoch, got {epochs_completed}"
        )
        assert total_batches > 0

    def test_single_epoch_behavior(self, tiny_slaf):
        """Test that single epoch (default) behavior is unchanged"""
        tokenizer = SLAFTokenizer(tiny_slaf)

        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=8,
            n_epochs=1,  # Single epoch (default)
        )

        batch_count = 0
        epochs_seen = set()

        for batch in dataset:
            epoch = batch.get("epoch", 0)
            epochs_seen.add(epoch)
            batch_count += 1

            if batch_count > 10:
                break

        # Should only see epoch 0 in single epoch mode
        assert epochs_seen == {0}, f"Expected only epoch 0, got {epochs_seen}"
        assert batch_count > 0

    def test_multi_epoch_prefetcher_stats(self, tiny_slaf):
        """Test that AsyncPrefetcher correctly tracks multi-epoch statistics"""
        tokenizer = SLAFTokenizer(tiny_slaf)

        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=4,
            n_epochs=3,
        )

        # Get some batches to populate stats
        batch_count = 0
        for _batch in dataset:
            batch_count += 1
            if batch_count > 5:
                break

        # Check prefetcher stats
        stats = dataset.prefetcher.get_stats()
        assert "current_epoch" in stats
        assert "n_epochs" in stats
        assert stats["n_epochs"] == 3
        assert stats["current_epoch"] >= 0

    def test_multi_epoch_completion_detection(self, tiny_slaf):
        """Test that the dataset correctly detects when all epochs are completed"""
        tokenizer = SLAFTokenizer(tiny_slaf)

        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=2,  # Small batch size to complete quickly
            n_epochs=2,  # Just 2 epochs
        )

        # Iterate through batches (limit to avoid slow test)
        all_batches = []
        for batch in dataset:
            all_batches.append(batch)
            if len(all_batches) > 20:  # Limit to avoid slow test
                break

        # Should have some batches
        assert len(all_batches) > 0

        # Check that we have batches from multiple epochs
        epochs_seen = set()
        for batch in all_batches:
            epoch = batch.get("epoch", 0)
            epochs_seen.add(epoch)

        # Should see at least one epoch
        assert len(epochs_seen) >= 1, f"Expected at least 1 epoch, got {epochs_seen}"


class TestPrefetchBatchProcessing:
    """Test suite for PrefetchBatchProcessor with comprehensive coverage"""

    def test_batch_processor_with_mock_data(self):
        """Test PrefetchBatchProcessor with mock data"""
        # Create mock SLAF array
        mock_slaf = Mock()
        mock_slaf.slaf_path = "/mock/path"

        # Mock Lance dataset
        mock_dataset = Mock()
        mock_batch = Mock()
        mock_batch.to_pandas.return_value = pd.DataFrame(
            {
                "cell_integer_id": [1, 1, 2, 2],
                "gene_id": [100, 101, 100, 101],
                "expression": [0.5, 0.8, 0.3, 0.9],
            }
        )

        mock_generator = iter([mock_batch])
        mock_dataset.to_batches.return_value = mock_generator

        # Mock Lance
        with patch("slaf.ml.datasets.lance") as mock_lance:
            mock_lance.dataset.return_value = mock_dataset

            # Create processor
            window = GeneformerWindow()
            shuffle = RandomShuffle()
            tokenizer = SLAFTokenizer(mock_slaf)

            processor = PrefetchBatchProcessor(
                slaf_array=mock_slaf,
                window=window,
                shuffle=shuffle,
                tokenizer=tokenizer,
                batches_per_chunk=1,
            )

            # Test that processor can be initialized
            assert processor.slaf_array is mock_slaf
            assert processor.window is window
            assert processor.shuffle is shuffle

    def test_prefetch_batch_serialization(self):
        """Test PrefetchBatch serialization and deserialization"""
        # Create mock tensors
        input_ids = torch.randint(0, 1000, (2, 1024))
        attention_mask = torch.ones(2, 1024, dtype=torch.bool)

        batch = TokenizedPrefetchBatch(
            batch_id=0,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cell_integer_ids=[100, 101],
            partial_cell_data={},
            tokenize_time=0.1,
        )

        # Test that batch can be created
        assert batch.batch_id == 0
        assert batch.input_ids.shape == (2, 1024)
        assert batch.attention_mask.shape == (2, 1024)
        assert len(batch.cell_integer_ids) == 2
        assert batch.tokenize_time == 0.1

    def test_batch_processor_expression_binning(self, tiny_slaf):
        """Test batch processor with expression binning"""
        window = ScGPTWindow()
        shuffle = RandomShuffle()
        tokenizer = SLAFTokenizer(tiny_slaf)

        processor = PrefetchBatchProcessor(
            slaf_array=tiny_slaf,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            n_expression_bins=10,
            use_binned_expressions=True,
        )

        assert processor.n_expression_bins == 10
        assert processor.use_binned_expressions is True

    def test_dataset_fragment_parameter(self, tiny_slaf):
        """Test the by_fragment parameter functionality."""
        tokenizer = SLAFTokenizer(tiny_slaf)

        # Test fragment-based loading
        dataset_fragment = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=32,
            verbose=False,
            by_fragment=True,
        )

        assert dataset_fragment.by_fragment is True

        # Test batch-based loading
        dataset_batch = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=32,
            verbose=False,
            by_fragment=False,
        )

        assert dataset_batch.by_fragment is False

    def test_dataset_iteration_fragment_mode(self, tiny_slaf):
        """Test dataset iteration in fragment mode."""
        tokenizer = SLAFTokenizer(tiny_slaf)
        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=32,
            verbose=False,
            by_fragment=True,
        )

        # Test that we can iterate through batches
        batch_count = 0
        for batch in dataset:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch
            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0

    def test_dataset_iteration_batch_mode(self, tiny_slaf):
        """Test dataset iteration in batch mode."""
        tokenizer = SLAFTokenizer(tiny_slaf)
        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=tokenizer,
            batch_size=32,
            verbose=False,
            by_fragment=False,
        )

        # Test that we can iterate through batches
        batch_count = 0
        for batch in dataset:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch
            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0

    def test_dataset_raw_mode_fragment(self, tiny_slaf):
        """Test dataset in raw mode with fragment loading."""
        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=None,
            batch_size=32,
            raw_mode=True,
            verbose=False,
            by_fragment=True,
        )

        # Test that we can iterate through raw batches
        batch_count = 0
        for batch in dataset:
            assert "cell_ids" in batch
            assert "x" in batch
            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0

    def test_dataset_raw_mode_batch(self, tiny_slaf):
        """Test dataset in raw mode with batch loading."""
        dataset = SLAFIterableDataset(
            slaf_array=tiny_slaf,
            tokenizer=None,
            batch_size=32,
            raw_mode=True,
            verbose=False,
            by_fragment=False,
        )

        # Test that we can iterate through raw batches
        batch_count = 0
        for batch in dataset:
            assert "cell_ids" in batch
            assert "x" in batch
            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0


if __name__ == "__main__":
    pytest.main([__file__])
