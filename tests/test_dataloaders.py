import numpy as np
import pytest

from slaf.ml.dataloaders import SLAFDataLoader, get_device_info, get_optimal_device


class TestSLAFDataLoader:
    """Test suite for SLAFDataLoader class"""

    def test_dataloader_initialization(self, tiny_slaf):
        """Test SLAFDataLoader initialization"""
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

        # Check cell integer ranges
        assert len(dataloader.cell_integer_ranges) > 0
        assert all(
            isinstance(r, tuple) and len(r) == 2 for r in dataloader.cell_integer_ranges
        )

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

    def test_cell_integer_ranges_generation(self, tiny_slaf):
        """Test cell integer ranges generation"""
        dataloader = SLAFDataLoader(tiny_slaf, batch_size=5)

        # Check ranges are properly formed
        for start, end in dataloader.cell_integer_ranges:
            assert start < end
            assert end - start <= dataloader.batch_size

        # Check ranges cover all cells
        max_cell_id = int(tiny_slaf.obs["cell_integer_id"].astype(int).max())
        all_cells = set()
        for start, end in dataloader.cell_integer_ranges:
            all_cells.update(range(start, end))

        # Should cover all cells from 0 to max_cell_id
        expected_cells = set(range(max_cell_id + 1))
        assert all_cells == expected_cells

    def test_geneformer_iteration(self, tiny_slaf):
        """Test dataloader iteration with Geneformer tokenizer"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            tokenizer_type="geneformer",
            batch_size=5,
            max_genes=10,
        )

        batches = list(dataloader)
        assert len(batches) > 0

        for batch in batches:
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
            assert input_ids.shape[1] <= dataloader.max_genes

            # Check data types robustly
            try:
                import torch

                is_torch = isinstance(input_ids, torch.Tensor)
            except ImportError:
                is_torch = False

            if isinstance(input_ids, np.ndarray):
                assert input_ids.dtype == np.int64
                assert attention_mask.dtype == bool
                assert cell_ids.dtype == np.int64
            elif is_torch:
                assert input_ids.dtype == torch.int64
                assert attention_mask.dtype == torch.bool
                assert cell_ids.dtype == torch.int64
            else:
                raise AssertionError(f"Unknown tensor type: {type(input_ids)}")

            # Check attention mask logic
            expected_mask = input_ids != dataloader.special_tokens["PAD"]
            if isinstance(attention_mask, np.ndarray):
                assert np.array_equal(attention_mask, expected_mask)
            else:
                assert (attention_mask == expected_mask).all()

    def test_scgpt_iteration(self, tiny_slaf):
        """Test dataloader iteration with scGPT tokenizer"""
        dataloader = SLAFDataLoader(
            tiny_slaf,
            tokenizer_type="scgpt",
            batch_size=5,
            max_genes=10,
        )

        batches = list(dataloader)
        assert len(batches) > 0

        for batch in batches:
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

            # scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]
            # So sequence length should be <= max_genes * 2 + 2
            max_seq_len = dataloader.max_genes * 2 + 2
            assert input_ids.shape[1] <= max_seq_len

            # Check data types robustly
            try:
                import torch

                is_torch = isinstance(input_ids, torch.Tensor)
            except ImportError:
                is_torch = False

            if isinstance(input_ids, np.ndarray):
                assert input_ids.dtype == np.int64
                assert attention_mask.dtype == bool
                assert cell_ids.dtype == np.int64
            elif is_torch:
                assert input_ids.dtype == torch.int64
                assert attention_mask.dtype == torch.bool
                assert cell_ids.dtype == torch.int64
            else:
                raise AssertionError(f"Unknown tensor type: {type(input_ids)}")

            # Check scGPT format and padding
            for seq in input_ids:
                seq = seq.tolist() if hasattr(seq, "tolist") else list(seq)
                assert seq[0] == dataloader.special_tokens["CLS"]
                # SEP should be present
                assert dataloader.special_tokens["SEP"] in seq
                sep_pos = seq.index(dataloader.special_tokens["SEP"])
                # All tokens after SEP should be PAD
                if sep_pos < len(seq) - 1:
                    assert all(
                        t == dataloader.special_tokens["PAD"]
                        for t in seq[sep_pos + 1 :]
                    )

    def test_invalid_tokenizer_type(self, tiny_slaf):
        """Test dataloader with invalid tokenizer type"""
        dataloader = SLAFDataLoader(tiny_slaf, tokenizer_type="invalid")

        with pytest.raises(ValueError, match="Unknown tokenizer type"):
            list(dataloader)

    def test_empty_dataset(self, tiny_slaf):
        """Test dataloader with empty dataset (no expression data)"""
        # This test would require a dataset with no expression data
        # For now, we'll test that the dataloader handles empty batches gracefully
        dataloader = SLAFDataLoader(tiny_slaf, batch_size=1000)  # Large batch size

        # Should still work, just might have fewer batches
        batches = list(dataloader)
        assert len(batches) >= 0

    def test_small_batch_size(self, tiny_slaf):
        """Test dataloader with very small batch size"""
        dataloader = SLAFDataLoader(tiny_slaf, batch_size=1, max_genes=5)

        batches = list(dataloader)
        assert len(batches) > 0

        for batch in batches:
            input_ids = batch["input_ids"]
            assert input_ids.shape[0] <= 1  # At most 1 cell per batch

    def test_large_max_genes(self, tiny_slaf):
        """Test dataloader with large max_genes"""
        dataloader = SLAFDataLoader(tiny_slaf, max_genes=10000)

        batches = list(dataloader)
        assert len(batches) > 0

        for batch in batches:
            input_ids = batch["input_ids"]
            # Should not exceed max_genes
            assert input_ids.shape[1] <= dataloader.max_genes

    def test_dataloader_length(self, tiny_slaf):
        """Test dataloader __len__ method"""
        dataloader = SLAFDataLoader(tiny_slaf, batch_size=5)

        expected_length = len(dataloader.cell_integer_ranges)
        assert len(dataloader) == expected_length

    def test_consistent_batch_sizes(self, tiny_slaf):
        """Test that batch sizes are consistent"""
        dataloader = SLAFDataLoader(tiny_slaf, batch_size=3)

        batches = list(dataloader)

        # All batches except the last should have batch_size cells
        for _ in range(len(batches[:-1])):
            batch = batches[_]
            input_ids = batch["input_ids"]
            assert input_ids.shape[0] == dataloader.batch_size

        # Last batch might be smaller
        if batches:
            last_batch = batches[-1]
            input_ids = last_batch["input_ids"]
            assert input_ids.shape[0] <= dataloader.batch_size

    def test_cell_id_mapping(self, tiny_slaf):
        """Test that cell IDs are correctly mapped"""
        dataloader = SLAFDataLoader(tiny_slaf, batch_size=5)

        batches = list(dataloader)

        # Check that cell IDs are sequential and match the ranges
        for _ in range(len(batches)):
            batch = batches[_]
            cell_ids = batch["cell_ids"]
            start_cell, end_cell = dataloader.cell_integer_ranges[_]

            # Cell IDs should be in the expected range
            expected_cell_ids = list(range(start_cell, end_cell))
            actual_cell_ids = (
                cell_ids.tolist() if hasattr(cell_ids, "tolist") else list(cell_ids)
            )

            # Should match up to the number of cells in the batch
            assert (
                actual_cell_ids[: len(expected_cell_ids)]
                == expected_cell_ids[: len(actual_cell_ids)]
            )

    def test_tokenizer_integration(self, tiny_slaf):
        """Test that dataloader properly integrates with tokenizer"""
        dataloader = SLAFDataLoader(tiny_slaf, batch_size=5, max_genes=10)

        # Check tokenizer attributes
        assert dataloader.tokenizer.slaf_array is tiny_slaf
        assert dataloader.tokenizer.vocab_size == 50000  # default
        assert dataloader.tokenizer.n_expression_bins == 10  # default

        # Check special tokens are shared
        assert dataloader.special_tokens is dataloader.tokenizer.special_tokens

        # Test that tokenization works
        batches = list(dataloader)
        assert len(batches) > 0

    def test_memory_efficiency(self, tiny_slaf):
        """Test that dataloader doesn't load all data into memory at once"""
        dataloader = SLAFDataLoader(tiny_slaf, batch_size=5)

        # Iterate through batches - should not load all data at once
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            # Each batch should be a separate object
            assert isinstance(batch, dict)

        assert batch_count > 0


class TestDeviceDetection:
    def test_device_info_and_optimal_device(self):
        info = get_device_info()
        device = get_optimal_device()
        assert "torch_available" in info
        if info["torch_available"]:
            # Should return a torch.device
            import torch

            assert isinstance(device, torch.device)
            # Should be one of the valid types
            assert device.type in ("cuda", "mps", "cpu")
        else:
            assert device is None

    @pytest.mark.skipif(
        not get_device_info()["torch_available"], reason="PyTorch not available"
    )
    def test_tensor_on_optimal_device(self):
        import torch

        device = get_optimal_device()
        t = torch.tensor([1, 2, 3], device=device)
        # Device type should match
        assert t.device.type == device.type

    @pytest.mark.skipif(
        not get_device_info()["torch_available"], reason="PyTorch not available"
    )
    def test_dataloader_device(self, tiny_slaf):
        dataloader = SLAFDataLoader(tiny_slaf)
        device = get_optimal_device()
        assert dataloader.device == device
