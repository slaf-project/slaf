from unittest.mock import patch

import numpy as np
import pytest

from slaf.ml.dataloaders import SLAFDataLoader, get_device_info, get_optimal_device


class TestSLAFDataLoader:
    """Test suite for SLAFDataLoader class with streaming implementation"""

    def test_dataloader_initialization(self, tiny_slaf):
        """Test SLAFDataLoader initialization with new streaming implementation"""
        dataloader = SLAFDataLoader(tiny_slaf)

        # Check basic attributes
        assert dataloader.slaf_array is tiny_slaf
        assert dataloader.tokenizer_type == "geneformer"
        assert dataloader.batch_size == 32
        assert dataloader.max_genes == 2048
        assert dataloader.num_workers == 4

        # Check tokenizer initialization
        assert dataloader.tokenizer is not None
        assert dataloader.tokenizer.slaf_array is tiny_slaf

        # Check special tokens
        assert dataloader.special_tokens is not None
        assert "PAD" in dataloader.special_tokens

        # Check that we're using the new dataset implementation
        assert dataloader._use_new_dataset is True
        assert hasattr(dataloader, "_dataset")

    def test_dataloader_initialization_custom_params(self, tiny_slaf):
        """Test SLAFDataLoader initialization with custom parameters"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            tokenizer_type="scgpt",
            batch_size=16,
            max_genes=1024,
            num_workers=2,
            vocab_size=1000,
            n_expression_bins=5,
            chunk_size=512,
        )

        assert dataloader.tokenizer_type == "scgpt"
        assert dataloader.batch_size == 16
        assert dataloader.max_genes == 1024
        assert dataloader.num_workers == 2
        assert dataloader.tokenizer.vocab_size == 1000
        assert dataloader.tokenizer.n_expression_bins == 5
        assert dataloader.tokenizer.chunk_size == 512

    def test_geneformer_iteration(self, tiny_slaf):
        """Test dataloader iteration with Geneformer tokenizer"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            tokenizer_type="geneformer",
            batch_size=5,
            max_genes=10,  # This is ignored by the dataset implementation
        )

        # Test that we can iterate (streaming, so we don't know exact length)
        batch_count = 0
        for batch in dataloader:
            # Check batch structure
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch

            # Check tensor shapes
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            cell_ids = batch["cell_ids"]

            assert input_ids.shape[0] == attention_mask.shape[0]
            assert input_ids.shape[0] == cell_ids.shape[0]
            # Geneformer format: just gene tokens, padded to hardcoded max_genes (2048)
            assert input_ids.shape[1] == 2048  # Hardcoded in dataset implementation

            # Check data types (handle both PyTorch tensors and numpy arrays)
            try:
                import torch

                if isinstance(input_ids, torch.Tensor):
                    assert input_ids.dtype in [torch.int64, torch.int32]
                    assert attention_mask.dtype in [torch.bool, torch.uint8]
                    assert cell_ids.dtype in [torch.int64, torch.int32]
                else:
                    assert input_ids.dtype in [np.int64, np.int32]
                    assert attention_mask.dtype in [np.bool_, np.uint8]
                    assert cell_ids.dtype in [np.int64, np.int32]
            except ImportError:
                # PyTorch not available, should be numpy arrays
                assert input_ids.dtype in [np.int64, np.int32]
                assert attention_mask.dtype in [np.bool_, np.uint8]
                assert cell_ids.dtype in [np.int64, np.int32]

            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0

    def test_scgpt_iteration(self, tiny_slaf):
        """Test dataloader iteration with scGPT tokenizer"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            tokenizer_type="scgpt",
            batch_size=5,
            max_genes=10,  # This is ignored by the dataset implementation
        )

        # Test that we can iterate (streaming, so we don't know exact length)
        batch_count = 0
        for batch in dataloader:
            # Check batch structure
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch

            # Check tensor shapes
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            cell_ids = batch["cell_ids"]

            assert input_ids.shape[0] == attention_mask.shape[0]
            assert input_ids.shape[0] == cell_ids.shape[0]
            # scGPT format: CLS + (gene,expr)*max_genes + SEP
            # Uses hardcoded max_genes (1024 for scGPT)
            expected_length = 1024 * 2 + 2  # CLS + (gene,expr)*1024 + SEP
            assert input_ids.shape[1] == expected_length

            # Check data types (handle both PyTorch tensors and numpy arrays)
            try:
                import torch

                if isinstance(input_ids, torch.Tensor):
                    assert input_ids.dtype in [torch.int64, torch.int32]
                    assert attention_mask.dtype in [torch.bool, torch.uint8]
                    assert cell_ids.dtype in [torch.int64, torch.int32]
                else:
                    assert input_ids.dtype in [np.int64, np.int32]
                    assert attention_mask.dtype in [np.bool_, np.uint8]
                    assert cell_ids.dtype in [np.int64, np.int32]
            except ImportError:
                # PyTorch not available, should be numpy arrays
                assert input_ids.dtype in [np.int64, np.int32]
                assert attention_mask.dtype in [np.bool_, np.uint8]
                assert cell_ids.dtype in [np.int64, np.int32]

            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0

    def test_invalid_tokenizer_type(self, tiny_slaf):
        """Test that invalid tokenizer type raises ValueError during iteration"""
        # The error is only raised in the legacy path, not the new dataset path
        # So we need to force the legacy path by disabling the new dataset
        dataloader = SLAFDataLoader(
            tiny_slaf, tokenizer_type="invalid", use_new_dataset=False
        )

        # The error should be raised during iteration in the legacy path
        with pytest.raises(ValueError, match="Unknown tokenizer type"):
            for _batch in dataloader:
                break  # Just try to get first batch

    def test_dataloader_length(self, tiny_slaf):
        """Test that dataloader length returns -1 for streaming"""
        dataloader = SLAFDataLoader(tiny_slaf)
        assert dataloader.__len__() == -1  # Streaming datasets have unknown length

    def test_consistent_batch_sizes(self, tiny_slaf):
        """Test that batches have consistent sizes"""
        dataloader = SLAFDataLoader(tiny_slaf, batch_size=8)

        batch_sizes = []
        for batch in dataloader:
            batch_size = batch["input_ids"].shape[0]
            batch_sizes.append(batch_size)
            if len(batch_sizes) >= 5:  # Test first 5 batches
                break

        # All batches should have the same size (except possibly the last one)
        if len(batch_sizes) > 1:
            # All but the last batch should have the expected size
            for size in batch_sizes[:-1]:
                assert size == dataloader.batch_size

    def test_cell_id_mapping(self, tiny_slaf):
        """Test that cell IDs are properly mapped"""
        dataloader = SLAFDataLoader(tiny_slaf, batch_size=5)

        for batch in dataloader:
            cell_ids = batch["cell_ids"]

            # Check that cell IDs are within expected range
            max_cell_id = int(tiny_slaf.obs["cell_integer_id"].astype(int).max())
            assert all(0 <= cell_id <= max_cell_id for cell_id in cell_ids)

            # Check that cell IDs are unique within a batch
            assert len(set(cell_ids)) == len(cell_ids)

            break  # Just test first batch

    def test_tokenizer_integration(self, tiny_slaf):
        """Test that tokenizer is properly integrated"""
        dataloader = SLAFDataLoader(tiny_slaf)

        # Check that tokenizer has expected attributes
        assert hasattr(dataloader.tokenizer, "gene_vocab")
        assert hasattr(dataloader.tokenizer, "special_tokens")
        assert hasattr(dataloader.tokenizer, "vocab_size")

        # Check that special tokens are properly set
        assert dataloader.special_tokens == dataloader.tokenizer.special_tokens

    def test_memory_efficiency(self, tiny_slaf):
        """Test that dataloader is memory efficient"""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        dataloader = SLAFDataLoader(tiny_slaf, batch_size=4)

        # Iterate through a few batches
        for i, _batch in enumerate(dataloader):
            if i >= 3:  # Just test first 3 batches
                break

        # Force garbage collection
        gc.collect()

        # Check that memory usage hasn't exploded
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

    def test_dataloader_without_pytorch(self, tiny_slaf):
        """Test dataloader behavior when PyTorch is not available"""
        # Mock PyTorch not being available
        with patch("slaf.ml.dataloaders.TORCH_AVAILABLE", False):
            with patch("slaf.ml.dataloaders.DATASETS_AVAILABLE", False):
                # Should fall back to legacy implementation
                dataloader = SLAFDataLoader(tiny_slaf, use_new_dataset=False)

                assert dataloader._use_new_dataset is False
                assert hasattr(dataloader, "cell_integer_ranges")

    def test_dataloader_legacy_mode(self, tiny_slaf):
        """Test dataloader in legacy mode (without new dataset)"""
        dataloader = SLAFDataLoader(tiny_slaf, use_new_dataset=False)

        assert dataloader._use_new_dataset is False
        assert hasattr(dataloader, "cell_integer_ranges")

        # Test that we can still iterate
        batch_count = 0
        for batch in dataloader:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch
            batch_count += 1
            if batch_count >= 2:
                break

        assert batch_count > 0

    def test_dataloader_cleanup(self, tiny_slaf):
        """Test dataloader cleanup functionality"""
        dataloader = SLAFDataLoader(tiny_slaf)

        # Test that cleanup methods exist and don't crash
        dataloader.stop_streaming()

        # Test destructor
        dataloader.__del__()

    def test_dataloader_with_custom_device(self, tiny_slaf):
        """Test dataloader with custom device specification"""
        dataloader = SLAFDataLoader(tiny_slaf, device="cpu")

        assert dataloader.device is not None
        # Device should be a string or torch.device
        assert isinstance(dataloader.device, str) or hasattr(dataloader.device, "type")


class TestDeviceDetection:
    """Test device detection and optimization"""

    def test_device_info_and_optimal_device(self):
        """Test device information and optimal device detection"""
        device_info = get_device_info()

        # Check that device info has expected keys
        assert "torch_available" in device_info
        assert "cuda_available" in device_info
        assert "mps_available" in device_info
        assert "optimal_device" in device_info

        # Check that optimal device is set
        optimal_device = get_optimal_device()
        if device_info["torch_available"]:
            assert optimal_device is not None
        else:
            assert optimal_device is None

    @pytest.mark.skipif(
        not get_device_info()["torch_available"], reason="PyTorch not available"
    )
    def test_tensor_on_optimal_device(self):
        """Test that tensors are created on the optimal device"""
        import torch

        optimal_device = get_optimal_device()
        if optimal_device is not None:
            tensor = torch.tensor([1, 2, 3], device=optimal_device)
            # Compare device types, not exact device objects
            assert tensor.device.type == optimal_device.type

    @pytest.mark.skipif(
        not get_device_info()["torch_available"], reason="PyTorch not available"
    )
    def test_dataloader_device(self, tiny_slaf):
        """Test that dataloader uses the correct device"""
        dataloader = SLAFDataLoader(tiny_slaf)

        # Check that device is set
        if dataloader.device is not None:
            assert isinstance(dataloader.device, str) or hasattr(
                dataloader.device, "type"
            )

            # Test that batches are on the correct device
            for batch in dataloader:
                if hasattr(batch["input_ids"], "device"):
                    # Compare device types, not exact device objects
                    assert batch["input_ids"].device.type == dataloader.device.type
                break  # Just test first batch
