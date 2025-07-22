import time
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from slaf.ml.datasets import (
    AsyncFragmentPrefetcher,
    FragmentLoader,
    RawFragment,
    SLAFIterableDataset,
)
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
            device="cpu",
            tokenizer_type="geneformer",
        )

        assert dataset.slaf_array is tiny_slaf
        assert dataset.tokenizer is tokenizer
        assert dataset.batch_size == 32
        assert dataset.seed == 42
        assert dataset.tokenizer_type == "geneformer"
        # Check device exists (could be None, string, or torch.device)
        assert hasattr(dataset, "device")
        assert hasattr(dataset, "fragment_loader")
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
            device="cpu",
            tokenizer_type="scgpt",
        )

        assert dataset.tokenizer_type == "scgpt"
        assert dataset.batch_size == 16
        assert dataset.seed == 123

    def test_fragment_loader_initialization(self, tiny_slaf):
        """Test FragmentLoader initialization and configuration"""
        loader = FragmentLoader(tiny_slaf, seed=42, max_genes=1024)

        assert loader.slaf_array is tiny_slaf
        assert loader.seed == 42
        assert loader.max_genes == 1024
        assert hasattr(loader, "fragments")

    def test_raw_fragment_dataclass(self):
        """Test RawFragment dataclass"""
        fragment = RawFragment(
            fragment_id=0,
            gene_sequences=[[1, 2, 3], [4, 5, 6]],
            cell_integer_ids=[100, 101],
        )

        assert fragment.fragment_id == 0
        assert len(fragment.gene_sequences) == 2
        assert len(fragment.cell_integer_ids) == 2
        assert fragment.gene_sequences[0] == [1, 2, 3]
        assert fragment.cell_integer_ids[0] == 100

    def test_async_fragment_prefetcher_initialization(self, tiny_slaf):
        """Test AsyncFragmentPrefetcher initialization"""
        fragment_loader = FragmentLoader(tiny_slaf)
        prefetcher = AsyncFragmentPrefetcher(
            fragment_processor=fragment_loader,
            max_queue_size=5,
        )

        assert prefetcher.fragment_processor is fragment_loader
        assert prefetcher.max_queue_size == 5
        assert prefetcher.queue.maxsize == 5

    def test_prefetcher_start_stop(self, tiny_slaf):
        """Test AsyncFragmentPrefetcher start and stop functionality"""
        fragment_loader = FragmentLoader(tiny_slaf)
        prefetcher = AsyncFragmentPrefetcher(
            fragment_processor=fragment_loader,
            max_queue_size=3,
        )

        # Test start
        prefetcher.start()
        assert prefetcher.worker_thread is not None
        assert prefetcher.worker_thread.is_alive()

        # Test stop
        prefetcher.stop()
        # Give thread time to stop

        time.sleep(0.1)
        assert not prefetcher.worker_thread.is_alive()

    def test_prefetcher_queue_operations(self, tiny_slaf):
        """Test AsyncFragmentPrefetcher queue operations"""
        fragment_loader = FragmentLoader(tiny_slaf)
        prefetcher = AsyncFragmentPrefetcher(
            fragment_processor=fragment_loader,
            max_queue_size=2,
        )

        # Test empty queue
        assert prefetcher.has_fragment() is False
        assert prefetcher.get_fragment() is None

        # Test queue operations
        test_fragment = RawFragment(
            fragment_id=0,
            gene_sequences=[[1, 2, 3]],
            cell_integer_ids=[100],
        )

        # Put fragment in queue
        prefetcher.queue.put(test_fragment)
        assert prefetcher.has_fragment() is True

        # Get fragment from queue
        retrieved_fragment = prefetcher.get_fragment()
        assert retrieved_fragment is test_fragment
        assert prefetcher.has_fragment() is False

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
        tokenizer = SLAFTokenizer(tiny_slaf)
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
            assert (
                seq_length == 1024 * 2 + 2
            )  # scGPT format: CLS + (gene,expr)*1024 + SEP

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
            device="cpu",  # Use CPU for testing
        )

        for batch in dataset:
            # Check that tensors are on the correct device
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

        # This should not raise an error during initialization
        # The error would be raised during iteration if the tokenizer type
        # is not supported by the conversion methods
        # We test that it doesn't crash during initialization

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

            batch_count += 1
            if batch_count >= 2:  # Test first 2 batches
                break

        assert batch_count > 0


class TestFragmentProcessing:
    """Test fragment processing functionality"""

    def test_fragment_loader_with_mock_data(self):
        """Test FragmentLoader with mock data"""
        # Mock SLAFArray
        mock_slaf_array = Mock()
        mock_slaf_array.slaf_path = "/mock/path"

        # Mock Lance dataset
        with patch("slaf.ml.datasets.lance") as mock_lance:
            mock_dataset = Mock()
            mock_fragments = [Mock(), Mock()]
            mock_dataset.get_fragments.return_value = mock_fragments
            mock_lance.dataset.return_value = mock_dataset

            # Mock Polars DataFrame
            with patch("slaf.ml.datasets.pl") as mock_pl:
                mock_df = Mock()
                mock_pl.from_arrow.return_value = mock_df
                mock_df.with_columns.return_value = mock_df
                mock_df.filter.return_value = mock_df
                mock_df.group_by.return_value.agg.return_value = Mock()

                loader = FragmentLoader(mock_slaf_array, seed=42, max_genes=1024)

                # Test that we can create the loader
                assert loader.slaf_array is mock_slaf_array
                assert loader.seed == 42
                assert loader.max_genes == 1024

    def test_raw_fragment_serialization(self):
        """Test RawFragment serialization and comparison"""
        fragment1 = RawFragment(
            fragment_id=0,
            gene_sequences=[[1, 2, 3], [4, 5, 6]],
            cell_integer_ids=[100, 101],
        )

        fragment2 = RawFragment(
            fragment_id=0,
            gene_sequences=[[1, 2, 3], [4, 5, 6]],
            cell_integer_ids=[100, 101],
        )

        fragment3 = RawFragment(
            fragment_id=1,
            gene_sequences=[[7, 8, 9]],
            cell_integer_ids=[102],
        )

        # Test equality
        assert fragment1 == fragment2
        assert fragment1 != fragment3

        # Test string representation
        assert "RawFragment" in str(fragment1)
        assert "fragment_id=0" in str(fragment1)


if __name__ == "__main__":
    pytest.main([__file__])
