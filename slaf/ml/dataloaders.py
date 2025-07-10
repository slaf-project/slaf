from typing import Optional

import numpy as np

from slaf.core.slaf import SLAFArray

from .tokenizers import SLAFTokenizer

# Try to import torch, but make it optional
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Tensor operations will be disabled.")


# Define device utility functions
def get_optimal_device():
    """Get the optimal device for PyTorch operations (CUDA > MPS > CPU)"""
    if not TORCH_AVAILABLE:
        return None

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info():
    """Get comprehensive device information for debugging"""
    if not TORCH_AVAILABLE:
        return {
            "torch_available": False,
            "cuda_available": False,
            "mps_available": False,
            "optimal_device": None,
        }

    info = {
        "torch_available": True,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "optimal_device": str(get_optimal_device()),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_capability"] = torch.cuda.get_device_capability(0)

    return info


# Get the optimal device once at module level
OPTIMAL_DEVICE = get_optimal_device()


class SLAFDataLoader:
    """
    High-performance DataLoader for SLAF data optimized for ML training.

    SLAFDataLoader provides efficient batching and tokenization of single-cell data
    for machine learning applications. It supports multiple tokenization strategies
    and automatic device optimization for PyTorch training.

    Key Features:
        - Multiple tokenization strategies (GeneFormer, SCPGPT)
        - Automatic device optimization (CUDA > MPS > CPU)
        - Efficient batching with integer ID ranges
        - PyTorch tensor output with attention masks
        - Memory-efficient lazy loading

    Examples:
        >>> # Basic usage with default settings
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> dataloader = SLAFDataLoader(slaf_array)
        >>> for batch in dataloader:
        ...     print(f"Batch shape: {batch['input_ids'].shape}")
        ...     print(f"Cell IDs: {batch['cell_ids']}")
        ...     break
        Batch shape: torch.Size([32, 2048])
        Cell IDs: tensor([0, 1, 2, ..., 29, 30, 31])

        >>> # Custom configuration for training
        >>> dataloader = SLAFDataLoader(
        ...     slaf_array=slaf_array,
        ...     tokenizer_type="geneformer",
        ...     batch_size=64,
        ...     max_genes=1024,
        ...     vocab_size=30000
        ... )
        >>> print(f"Number of batches: {len(dataloader)}")
        Number of batches: 42

        >>> # Training loop example
        >>> for batch_idx, batch in enumerate(dataloader):
        ...     input_ids = batch["input_ids"]
        ...     attention_mask = batch["attention_mask"]
        ...     cell_ids = batch["cell_ids"]
        ...     # Your training code here
        ...     if batch_idx >= 2:  # Just show first few batches
        ...         break
        >>> print("Training loop completed")
        Training loop completed
    """

    device: Optional["torch.device"]  # type: ignore

    def __init__(
        self,
        slaf_array: SLAFArray,
        tokenizer_type: str = "geneformer",
        batch_size: int = 32,
        max_genes: int = 2048,
        num_workers: int = 4,
        vocab_size: int = 50000,
        n_expression_bins: int = 10,
        chunk_size: int = 1024,
        device: str | None = None,  # Allow manual device override
    ):
        """
        Initialize the SLAF DataLoader with training configuration.

        Args:
            slaf_array: SLAFArray instance containing the single-cell data.
            tokenizer_type: Tokenization strategy to use. Options: "geneformer", "scgpt".
            batch_size: Number of cells per batch. Larger batches use more memory.
            max_genes: Maximum number of genes to include in each cell's tokenization.
            num_workers: Number of worker processes for data loading (unused in current implementation).
            vocab_size: Size of the tokenizer vocabulary.
            n_expression_bins: Number of expression level bins for discretization.
            chunk_size: Size of chunks for processing large datasets.
            device: PyTorch device to use. If None, automatically selects optimal device.

        Raises:
            ValueError: If tokenizer_type is not supported.
            RuntimeError: If PyTorch is not available and device is specified.

        Examples:
            >>> # Basic initialization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> dataloader = SLAFDataLoader(slaf_array)
            >>> print(f"Batch size: {dataloader.batch_size}")
            Batch size: 32

            >>> # Custom configuration
            >>> dataloader = SLAFDataLoader(
            ...     slaf_array=slaf_array,
            ...     tokenizer_type="scgpt",
            ...     batch_size=64,
            ...     max_genes=1024
            ... )
            >>> print(f"Tokenizer type: {dataloader.tokenizer_type}")
            Tokenizer type: scgpt

            >>> # With custom device
            >>> dataloader = SLAFDataLoader(
            ...     slaf_array=slaf_array,
            ...     device="cuda:0"
            ... )
            >>> print(f"Device: {dataloader.device}")
            Device: cuda:0
        """
        self.slaf_array = slaf_array
        self.tokenizer_type = tokenizer_type
        self.batch_size = batch_size
        self.max_genes = max_genes
        self.num_workers = num_workers

        # Set device - use provided device, optimal device, or CPU
        if device is not None and TORCH_AVAILABLE:
            self.device = torch.device(device)
        elif TORCH_AVAILABLE:
            self.device = OPTIMAL_DEVICE
        else:
            self.device = None

        # Initialize tokenizer
        self.tokenizer = SLAFTokenizer(
            slaf_array=slaf_array,
            vocab_size=vocab_size,
            n_expression_bins=n_expression_bins,
            chunk_size=chunk_size,
        )

        # Get special tokens from tokenizer
        self.special_tokens = self.tokenizer.special_tokens

        # Pre-compute cell integer ID ranges for efficient batching
        self.cell_integer_ranges = self._get_cell_integer_ranges()

    def _get_cell_integer_ranges(self) -> list[tuple[int, int]]:
        """Get cell integer ID ranges for batching"""
        # Get the maximum cell integer ID from the obs DataFrame
        max_cell_id = int(self.slaf_array.obs["cell_integer_id"].astype(int).max())

        # Create ranges based on batch size
        ranges = []
        for start in range(0, max_cell_id + 1, self.batch_size):
            end = min(start + self.batch_size, max_cell_id + 1)
            ranges.append((start, end))

        return ranges

    def __iter__(self):
        """
        Iterate through batches of tokenized single-cell data.

        Yields batches of tokenized data suitable for machine learning training.
        Each batch contains input_ids, attention_mask, and cell_ids for the
        cells in that batch.

        Yields:
            dict: Batch dictionary containing:
                - input_ids: Tokenized gene expression data (torch.Tensor or np.ndarray)
                - attention_mask: Boolean mask indicating valid tokens (torch.Tensor or np.ndarray)
                - cell_ids: Integer IDs of cells in the batch (torch.Tensor or np.ndarray)

        Raises:
            ValueError: If the tokenizer type is not supported.
            RuntimeError: If tokenization fails for a batch.

        Examples:
            >>> # Basic iteration
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> dataloader = SLAFDataLoader(slaf_array, batch_size=16)
            >>> for batch in dataloader:
            ...     print(f"Batch keys: {list(batch.keys())}")
            ...     print(f"Input shape: {batch['input_ids'].shape}")
            ...     print(f"Cell IDs: {batch['cell_ids']}")
            ...     break
            Batch keys: ['input_ids', 'attention_mask', 'cell_ids']
            Input shape: (16, 2048)
            Cell IDs: tensor([0, 1, 2, ..., 13, 14, 15])

            >>> # Training loop with error handling
            >>> for batch_idx, batch in enumerate(dataloader):
            ...     try:
            ...         input_ids = batch["input_ids"]
            ...         attention_mask = batch["attention_mask"]
            ...         cell_ids = batch["cell_ids"]
            ...         # Your training code here
            ...         print(f"Processed batch {batch_idx}")
            ...     except Exception as e:
            ...         print(f"Error in batch {batch_idx}: {e}")
            ...         continue
            Processed batch 0
            Processed batch 1
            Processed batch 2
        """
        for cell_range in self.cell_integer_ranges:
            if self.tokenizer_type == "geneformer":
                tokens = self.tokenizer.tokenize_geneformer(
                    cell_integer_id_range=cell_range, max_genes=self.max_genes
                )
            elif self.tokenizer_type == "scgpt":
                tokens = self.tokenizer.tokenize_scgpt(
                    cell_integer_id_range=cell_range, max_genes=self.max_genes
                )
            else:
                raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

            if not tokens:
                continue

            # Convert to tensors if torch is available
            if TORCH_AVAILABLE:
                batch_tensors = torch.tensor(
                    tokens, dtype=torch.long, device=self.device
                )
                attention_mask = batch_tensors != self.special_tokens["PAD"]

                # Get cell IDs for this range
                start_cell, end_cell = cell_range
                cell_ids = list(range(start_cell, end_cell))

                yield {
                    "input_ids": batch_tensors,
                    "attention_mask": attention_mask,
                    "cell_ids": torch.tensor(
                        cell_ids[: len(tokens)], dtype=torch.long, device=self.device
                    ),
                }
            else:
                # Return as numpy arrays if torch is not available
                batch_tensors = np.array(tokens, dtype=np.int64)
                attention_mask = batch_tensors != self.special_tokens["PAD"]

                # Get cell IDs for this range
                start_cell, end_cell = cell_range
                cell_ids = list(range(start_cell, end_cell))

                yield {
                    "input_ids": batch_tensors,
                    "attention_mask": attention_mask,
                    "cell_ids": np.array(cell_ids[: len(tokens)], dtype=np.int64),
                }

    def __len__(self):
        """Return number of batches"""
        return len(self.cell_integer_ranges)
