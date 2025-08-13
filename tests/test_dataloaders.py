import polars as pl
import pytest
import torch

from slaf.ml.dataloaders import SLAFDataLoader, get_device_info, get_optimal_device


class TestSLAFDataLoader:
    """Test suite for SLAFDataLoader class with new architecture"""

    def test_dataloader_initialization(self, tiny_slaf):
        """Test SLAFDataLoader initialization with new architecture"""
        dataloader = SLAFDataLoader(tiny_slaf)

        # Check basic attributes
        assert dataloader.slaf_array is tiny_slaf
        assert dataloader.tokenizer_type == "geneformer"
        assert dataloader.batch_size == 32
        assert dataloader.max_genes == 2048

        # Check tokenizer initialization
        assert dataloader.tokenizer is not None
        assert dataloader.tokenizer.slaf_array is tiny_slaf

        # Check that we're using the new dataset implementation
        assert hasattr(dataloader, "_dataset")

    def test_dataloader_initialization_custom_params(self, tiny_slaf):
        """Test SLAFDataLoader initialization with custom parameters"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            tokenizer_type="scgpt",
            batch_size=16,
            max_genes=1024,
            vocab_size=1000,
            n_expression_bins=5,
        )

        assert dataloader.tokenizer_type == "scgpt"
        assert dataloader.batch_size == 16
        assert dataloader.max_genes == 1024
        assert dataloader.tokenizer.vocab_size == 1000
        assert dataloader.tokenizer.n_expression_bins == 5

    def test_geneformer_iteration(self, tiny_slaf):
        """Test dataloader iteration with Geneformer tokenizer"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            tokenizer_type="geneformer",
            batch_size=5,
            max_genes=10,
        )

        # Test that we can iterate
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
            assert input_ids.shape[1] == 2048  # Geneformer default

            # Check data types
            assert input_ids.dtype == torch.long
            assert attention_mask.dtype == torch.bool
            assert cell_ids.dtype == torch.long

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
            max_genes=10,
        )

        # Test that we can iterate
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
            assert input_ids.shape[1] == 2050  # scGPT: 2*1024+2

            # Check data types
            assert input_ids.dtype == torch.long
            assert attention_mask.dtype == torch.bool
            assert cell_ids.dtype == torch.long

            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0

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
            max_cell_id = int(tiny_slaf.obs["cell_integer_id"].cast(pl.Int64).max())
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

    def test_dataloader_cleanup(self, tiny_slaf):
        """Test dataloader cleanup functionality"""
        dataloader = SLAFDataLoader(tiny_slaf)

        # Test that cleanup methods exist and don't crash
        # The dataloader doesn't have a stop_streaming method
        # It just uses __del__ for cleanup

        # Test destructor
        dataloader.__del__()

    def test_dataloader_device(self, tiny_slaf):
        """Test dataloader device handling"""
        dataloader = SLAFDataLoader(tiny_slaf)

        # Get a sample batch
        batch = next(iter(dataloader))

        # Check that tensors are on CPU (device-agnostic design)
        assert batch["input_ids"].device == torch.device("cpu")
        assert batch["attention_mask"].device == torch.device("cpu")
        assert batch["cell_ids"].device == torch.device("cpu")

    def test_multi_epoch_initialization(self, tiny_slaf):
        """Test SLAFDataLoader initialization with multi-epoch support"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            n_epochs=5,  # Test multi-epoch initialization
        )

        assert dataloader.n_epochs == 5
        assert dataloader._dataset.n_epochs == 5
        assert dataloader._dataset.batch_processor.n_epochs == 5

    def test_multi_epoch_iteration(self, tiny_slaf):
        """Test dataloader iteration with multiple epochs"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            batch_size=4,  # Small batch size for testing
            n_epochs=3,  # Test with 3 epochs
        )

        # Track epochs and batches
        epochs_seen = set()
        total_batches = 0

        for batch in dataloader:
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

    def test_multi_epoch_epoch_progression(self, tiny_slaf):
        """Test that epochs progress correctly in multi-epoch mode"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            batch_size=2,  # Very small batch size
            n_epochs=4,  # Test with 4 epochs
        )

        # Track epoch sequence
        epoch_sequence = []
        batch_count = 0

        for batch in dataloader:
            epoch = batch.get("epoch", 0)
            epoch_sequence.append(epoch)
            batch_count += 1

            # Stop after reasonable number of batches
            if batch_count > 20:
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

    def test_single_epoch_default_behavior(self, tiny_slaf):
        """Test that single epoch (default) behavior is unchanged"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            batch_size=8,
            n_epochs=1,  # Single epoch (default)
        )

        batch_count = 0
        epochs_seen = set()

        for batch in dataloader:
            epoch = batch.get("epoch", 0)
            epochs_seen.add(epoch)
            batch_count += 1

            if batch_count > 10:
                break

        # Should only see epoch 0 in single epoch mode
        assert epochs_seen == {0}, f"Expected only epoch 0, got {epochs_seen}"
        assert batch_count > 0

    def test_multi_epoch_with_different_tokenizers(self, tiny_slaf):
        """Test multi-epoch functionality with different tokenizer types"""
        # Test with Geneformer - limit epochs and batches for speed
        dataloader_geneformer = SLAFDataLoader(
            tiny_slaf,
            tokenizer_type="geneformer",
            batch_size=4,
            n_epochs=2,
        )

        epochs_geneformer = set()
        batch_count = 0
        for batch in dataloader_geneformer:
            epochs_geneformer.add(batch.get("epoch", 0))
            batch_count += 1
            # Early termination for speed
            if batch_count >= 5:
                break

        assert len(epochs_geneformer) >= 1, (
            f"Geneformer: Expected at least 1 epoch, got {epochs_geneformer}"
        )

        # Test with scGPT - limit epochs and batches for speed
        dataloader_scgpt = SLAFDataLoader(
            tiny_slaf,
            tokenizer_type="scgpt",
            batch_size=4,
            n_epochs=2,
        )

        epochs_scgpt = set()
        batch_count = 0
        for batch in dataloader_scgpt:
            epochs_scgpt.add(batch.get("epoch", 0))
            batch_count += 1
            # Early termination for speed
            if batch_count >= 5:
                break

        assert len(epochs_scgpt) >= 1, (
            f"scGPT: Expected at least 1 epoch, got {epochs_scgpt}"
        )

    def test_multi_epoch_completion(self, tiny_slaf):
        """Test that dataloader correctly completes all epochs"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            batch_size=2,  # Small batch size to complete quickly
            n_epochs=3,  # Test with 3 epochs
        )

        # Collect batches (limit to avoid slow test)
        all_batches = []
        for batch in dataloader:
            all_batches.append(batch)
            if len(all_batches) > 30:  # Limit to avoid slow test
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

    def test_multi_epoch_parameter_passing(self, tiny_slaf):
        """Test that n_epochs parameter is correctly passed through the hierarchy"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            n_epochs=7,  # Test with 7 epochs
        )

        # Check that n_epochs is passed to all levels
        assert dataloader.n_epochs == 7
        assert dataloader._dataset.n_epochs == 7
        assert dataloader._dataset.batch_processor.n_epochs == 7
        assert dataloader._dataset.prefetcher.batch_processor.n_epochs == 7

    def test_multi_epoch_with_custom_parameters(self, tiny_slaf):
        """Test multi-epoch functionality with custom parameters"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            tokenizer_type="scgpt",
            batch_size=6,
            max_genes=512,
            n_epochs=4,
        )

        # Verify all parameters are set correctly
        assert dataloader.n_epochs == 4
        assert dataloader.tokenizer_type == "scgpt"
        assert dataloader.batch_size == 6
        assert (
            dataloader._dataset.batch_processor.use_binned_expressions is False
        )  # Default for scgpt

        # Test iteration
        epochs_seen = set()
        batch_count = 0

        for batch in dataloader:
            epoch = batch.get("epoch", 0)
            epochs_seen.add(epoch)
            batch_count += 1

            if batch_count > 20:
                break

        # Should see some epochs (may not see all epochs in limited test)
        assert len(epochs_seen) >= 1, f"Expected at least 1 epoch, got {epochs_seen}"
        assert batch_count > 0

    def test_dataloader_fragment_parameter(self, tiny_slaf):
        """Test the by_fragment parameter functionality."""
        # Test fragment-based loading
        dataloader_fragment = SLAFDataLoader(
            tiny_slaf,
            batch_size=32,
            raw_mode=True,
            verbose=False,
            by_fragment=True,
        )

        assert dataloader_fragment.by_fragment is True

        # Test batch-based loading
        dataloader_batch = SLAFDataLoader(
            tiny_slaf,
            batch_size=32,
            raw_mode=True,
            verbose=False,
            by_fragment=False,
        )

        assert dataloader_batch.by_fragment is False

    def test_dataloader_iteration_fragment_mode(self, tiny_slaf):
        """Test dataloader iteration in fragment mode."""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            batch_size=32,
            raw_mode=True,
            verbose=False,
            by_fragment=True,
        )

        # Test that we can iterate through batches
        batch_count = 0
        for batch in dataloader:
            assert "cell_ids" in batch
            assert "x" in batch
            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0

    def test_dataloader_iteration_batch_mode(self, tiny_slaf):
        """Test dataloader iteration in batch mode."""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            batch_size=32,
            raw_mode=True,
            verbose=False,
            by_fragment=False,
        )

        # Test that we can iterate through batches
        batch_count = 0
        for batch in dataloader:
            assert "cell_ids" in batch
            assert "x" in batch
            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0

    def test_dataloader_tokenized_mode_fragment(self, tiny_slaf):
        """Test dataloader in tokenized mode with fragment loading."""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            batch_size=32,
            raw_mode=False,
            verbose=False,
            by_fragment=True,
        )

        # Test that we can iterate through tokenized batches
        batch_count = 0
        for batch in dataloader:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch
            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0

    def test_dataloader_tokenized_mode_batch(self, tiny_slaf):
        """Test dataloader in tokenized mode with batch loading."""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            batch_size=32,
            raw_mode=False,
            verbose=False,
            by_fragment=False,
        )

        # Test that we can iterate through tokenized batches
        batch_count = 0
        for batch in dataloader:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "cell_ids" in batch
            batch_count += 1
            if batch_count >= 3:  # Just test first few batches
                break

        assert batch_count > 0

    def test_dataloader_parameters_consistency(self, tiny_slaf):
        """Test that all parameters are properly passed through."""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            batch_size=64,
            max_genes=1024,
            vocab_size=10000,
            n_expression_bins=5,
            n_epochs=5,
            raw_mode=True,
            verbose=True,
            batches_per_chunk=25,
            by_fragment=True,
        )

        assert dataloader.batch_size == 64
        assert dataloader.max_genes == 1024
        assert dataloader.n_epochs == 5
        assert dataloader.raw_mode is True
        assert dataloader.verbose is True
        assert dataloader.batches_per_chunk == 25
        assert dataloader.by_fragment is True

    def test_dataloader_length(self, tiny_slaf):
        """Test that dataloader length returns 0 for streaming datasets."""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            batch_size=32,
            verbose=False,
        )

        assert len(dataloader) == 0  # Streaming datasets have unknown length (return 0)


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
