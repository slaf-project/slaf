from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from slaf.ml.aggregators import GeneformerWindow, ScGPTWindow
from slaf.ml.datasets import (
    AsyncPrefetcher,
    PrefetchBatch,
    PrefetchBatchProcessor,
    SLAFIterableDataset,
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

    def test_prefetch_batch_dataclass(self):
        """Test PrefetchBatch dataclass"""
        # Create mock tensors
        input_ids = torch.randint(0, 1000, (2, 1024))
        attention_mask = torch.ones(2, 1024, dtype=torch.bool)

        batch = PrefetchBatch(
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

        batch = PrefetchBatch(
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


class TestPrefetchBatchProcessing:
    """Test prefetch batch processing functionality"""

    def test_batch_processor_with_mock_data(self):
        """Test PrefetchBatchProcessor with mock data"""
        # Mock SLAFArray
        mock_slaf_array = Mock()
        mock_slaf_array.slaf_path = "/mock/path"

        # Mock Lance dataset
        with patch("slaf.ml.datasets.lance") as mock_lance:
            mock_dataset = Mock()
            mock_batches = [Mock(), Mock()]
            mock_dataset.to_batches.return_value = iter(mock_batches)
            mock_lance.dataset.return_value = mock_dataset

            # Mock Polars DataFrame
            with patch("slaf.ml.datasets.pl") as mock_pl:
                mock_df = Mock()
                mock_pl.from_arrow.return_value = mock_df
                mock_pl.concat.return_value = mock_df
                mock_df.with_columns.return_value = mock_df
                mock_df.filter.return_value = mock_df
                mock_df.group_by.return_value.agg.return_value = Mock()
                mock_df.columns = ["cell_integer_id", "gene_sequence", "expr_sequence"]

                window = ScGPTWindow()
                shuffle = RandomShuffle()
                tokenizer = SLAFTokenizer(mock_slaf_array)
                processor = PrefetchBatchProcessor(
                    mock_slaf_array,
                    window,
                    shuffle,
                    tokenizer=tokenizer,
                    seed=42,
                    max_genes=1024,
                )

                # Test that we can create the processor
                assert processor.slaf_array is mock_slaf_array
                assert processor.window is window
                assert processor.shuffle is shuffle
                assert processor.seed == 42
                assert processor.max_genes == 1024

    def test_prefetch_batch_serialization(self):
        """Test PrefetchBatch serialization and comparison"""
        # Create mock tensors
        input_ids1 = torch.randint(0, 1000, (2, 1024))
        attention_mask1 = torch.ones(2, 1024, dtype=torch.bool)
        input_ids2 = torch.randint(0, 1000, (1, 1024))
        attention_mask2 = torch.ones(1, 1024, dtype=torch.bool)

        batch1 = PrefetchBatch(
            batch_id=0,
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            cell_integer_ids=[100, 101],
            partial_cell_data={},
            tokenize_time=0.1,
        )

        batch2 = PrefetchBatch(
            batch_id=0,
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            cell_integer_ids=[100, 101],
            partial_cell_data={},
            tokenize_time=0.1,
        )

        batch3 = PrefetchBatch(
            batch_id=1,
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            cell_integer_ids=[102],
            partial_cell_data={},
            tokenize_time=0.2,
        )

        # Test equality
        assert batch1 == batch2
        assert batch1 != batch3

        # Test string representation
        assert "PrefetchBatch" in str(batch1)
        assert "batch_id=0" in str(batch1)

    def test_batch_processor_expression_binning(self, tiny_slaf):
        """Test PrefetchBatchProcessor with expression binning"""
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

        assert processor.n_expression_bins == 10
        assert processor.use_binned_expressions is True

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
            seed=42,
            max_genes=2048,
        )

        assert processor.window is window
        assert processor.shuffle is shuffle
        assert processor.max_genes == 2048


if __name__ == "__main__":
    pytest.main([__file__])
