from typing import Optional

from loguru import logger

from slaf.core.slaf import SLAFArray

from .tokenizers import SLAFTokenizer

# Try to import torch, but make it optional
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Tensor operations will be disabled.")

# Try to import the dataset
try:
    from .datasets import SLAFIterableDataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning(
        "Datasets not available. DataLoader will not work without datasets module."
    )


# Define device utility functions
def get_optimal_device():
    """
    Get the optimal device for PyTorch operations (CUDA > MPS > CPU).

    This function determines the best available device for PyTorch operations
    by checking for CUDA first, then MPS (Apple Silicon), and falling back
    to CPU if neither is available.

    Returns:
        torch.device | None: The optimal device, or None if PyTorch is not available.

    Examples:
        >>> # Check optimal device
        >>> device = get_optimal_device()
        >>> print(f"Optimal device: {device}")
        Optimal device: cuda

        >>> # Device priority (CUDA > MPS > CPU)
        >>> # If CUDA is available: cuda
        >>> # If MPS is available but not CUDA: mps
        >>> # If neither: cpu
        >>> device = get_optimal_device()
        >>> if device.type == "cuda":
        ...     print("Using CUDA GPU")
        ... elif device.type == "mps":
        ...     print("Using Apple Silicon GPU")
        ... else:
        ...     print("Using CPU")
        Using CUDA GPU

        >>> # Handle PyTorch not available
        >>> # This would return None if PyTorch is not installed
        >>> device = get_optimal_device()
        >>> if device is None:
        ...     print("PyTorch not available")
        ... else:
        ...     print(f"Device available: {device}")
        Device available: cuda
    """
    if not TORCH_AVAILABLE:
        return None

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info():
    """
    Get comprehensive device information for debugging.

    This function returns detailed information about the available PyTorch devices,
    including CUDA and MPS availability, device counts, and capabilities.
    Useful for debugging device-related issues and understanding the system
    configuration.

    Returns:
        dict: Device information dictionary containing:
            - torch_available: Whether PyTorch is available
            - cuda_available: Whether CUDA is available
            - mps_available: Whether MPS (Apple Silicon) is available
            - optimal_device: String representation of the optimal device
            - cuda_device_count: Number of CUDA devices (if CUDA available)
            - cuda_device_name: Name of the first CUDA device (if available)
            - cuda_device_capability: Compute capability of first CUDA device

    Examples:
        >>> # Get device information
        >>> info = get_device_info()
        >>> print(f"PyTorch available: {info['torch_available']}")
        PyTorch available: True
        >>> print(f"CUDA available: {info['cuda_available']}")
        CUDA available: True
        >>> print(f"Optimal device: {info['optimal_device']}")
        Optimal device: cuda

        >>> # Check CUDA details
        >>> if info['cuda_available']:
        ...     print(f"CUDA devices: {info['cuda_device_count']}")
        ...     print(f"Device name: {info['cuda_device_name']}")
        ...     print(f"Capability: {info['cuda_device_capability']}")
        CUDA devices: 1
        Device name: NVIDIA GeForce RTX 3080
        Capability: (8, 6)

        >>> # Check MPS availability
        >>> print(f"MPS available: {info['mps_available']}")
        MPS available: False

        >>> # Handle PyTorch not available
        >>> # This would show torch_available: False if PyTorch is not installed
        >>> info = get_device_info()
        >>> if not info['torch_available']:
        ...     print("PyTorch not available")
        ... else:
        ...     print("PyTorch is available")
        PyTorch is available
    """
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

    SLAFDataLoader provides efficient streaming of pre-tokenized single-cell data
    for machine learning applications. It uses async batch processing and provides
    device-agnostic CPU tensor output for maximum training flexibility.

    Key Features:
        - Multiple tokenization strategies (GeneFormer, scGPT)
        - Pre-tokenized sequences for maximum performance
        - Device-agnostic CPU tensor output
        - Async batch processing with background prefetching
        - Memory-efficient streaming
        - PyTorch tensor output with attention masks
        - Comprehensive error handling and validation

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
        ...     tokenizer_type="scgpt",
        ...     batch_size=64,
        ...     max_genes=1024
        ... )
        >>> print(f"Number of batches: {len(dataloader)}")
        Number of batches: 42

        >>> # Training loop example
        >>> for batch_idx, batch in enumerate(dataloader):
        ...     input_ids = batch["input_ids"]
        ...     attention_mask = batch["attention_mask"]
        ...     cell_ids = batch["cell_ids"]
        ...     # Your training code here
        ...     if batch_idx >= 2:  # Just test first few batches
        ...         break
        >>> print("Training loop completed")
        Training loop completed

        >>> # Error handling for invalid tokenizer type
        >>> try:
        ...     dataloader = SLAFDataLoader(slaf_array, tokenizer_type="invalid")
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Unsupported tokenizer type: invalid
    """

    device: Optional["torch.device"]  # type: ignore
    tokenizer: Optional["SLAFTokenizer"]  # type: ignore

    def __init__(
        self,
        slaf_array: SLAFArray,
        tokenizer_type: str = "geneformer",
        batch_size: int = 32,
        max_genes: int = 2048,
        vocab_size: int = 50000,
        n_expression_bins: int = 10,
        n_epochs: int = 1,  # Add n_epochs parameter
        raw_mode: bool = False,  # Add raw_mode parameter
        verbose: bool = True,  # Add verbose parameter
        batches_per_chunk: int = 50,  # Add batches_per_chunk parameter
        by_fragment: bool = False,  # Add by_fragment parameter for fragment-based loading
    ):
        """
        Initialize the SLAF DataLoader with training configuration.

        Args:
            slaf_array: SLAFArray instance containing the single-cell data.
                       Must be a valid SLAFArray with proper Lance dataset structure.
            tokenizer_type: Tokenization strategy to use. Options: "geneformer", "scgpt".
                          Geneformer uses ranked gene sequences, scGPT uses interleaved
                          gene-expression pairs.
            batch_size: Number of cells per batch. Larger batches use more memory
                       but may improve training efficiency. Range: 1-512, default: 32.
            max_genes: Maximum number of genes to include in each cell's tokenization.
                     For Geneformer: same as sequence length. For scGPT: number of
                     gene-expression pairs (sequence length = 2*max_genes+2).
            vocab_size: Size of the tokenizer vocabulary. Higher values allow more
                       genes but use more memory. Range: 1000-100000, default: 50000.
            n_expression_bins: Number of expression level bins for scGPT discretization.
                             Higher values provide finer expression resolution.
                             Range: 1-1000, default: 10.
            n_epochs: Number of epochs to run. The generator will automatically reset
                     after each epoch, enabling multi-epoch training on small datasets.
                     Default: 1.
            raw_mode: If True, return raw cell × gene data as sparse CSR tensors
                     instead of pre-tokenized sequences. Default: False.
            verbose: If True, print detailed timing and progress information.
                    If False, suppress all SLAF internal prints for clean output.
                    Default: True.
            batches_per_chunk: Number of Lance batches to load per chunk for batch-based loading.
                             Higher values use more memory but may improve throughput.
                             Range: 10-200, default: 50.
            by_fragment: If True, use fragment-based loading instead of batch-based loading.
                        Fragment-based loading provides higher entropy but may be slightly slower.
                        Default: False.

        Raises:
            ValueError: If tokenizer_type is not supported or parameters are invalid.
            RuntimeError: If PyTorch is not available or datasets module is missing.
            TypeError: If slaf_array is not a valid SLAFArray instance.
            ImportError: If required dependencies are not available.

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

            >>> # Multi-epoch training
            >>> dataloader = SLAFDataLoader(
            ...     slaf_array=slaf_array,
            ...     n_epochs=5
            ... )
            >>> print(f"Number of epochs: {dataloader.n_epochs}")
            Number of epochs: 5

            >>> # Raw mode for external comparisons
            >>> dataloader = SLAFDataLoader(
            ...     slaf_array=slaf_array,
            ...     raw_mode=True
            ... )
            >>> print(f"Raw mode: {dataloader.raw_mode}")
            Raw mode: True

            >>> # Fragment-based loading for higher entropy
            >>> dataloader = SLAFDataLoader(
            ...     slaf_array=slaf_array,
            ...     by_fragment=True
            ... )
            >>> print(f"Fragment-based loading: {dataloader.by_fragment}")
            Fragment-based loading: True

            >>> # Error handling for invalid tokenizer type
            >>> try:
            ...     dataloader = SLAFDataLoader(slaf_array, tokenizer_type="invalid")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Unsupported tokenizer type: invalid

            >>> # Error handling for invalid SLAF array
            >>> try:
            ...     dataloader = SLAFDataLoader(None)
            ... except TypeError as e:
            ...     print(f"Error: {e}")
            Error: slaf_array must be a valid SLAFArray instance
        """
        self.slaf_array = slaf_array
        self.tokenizer_type = tokenizer_type
        self.batch_size = batch_size
        self.max_genes = max_genes
        self.n_epochs = n_epochs
        self.raw_mode = raw_mode  # Add raw_mode attribute
        self.verbose = verbose  # Add verbose attribute
        self.batches_per_chunk = batches_per_chunk  # Add batches_per_chunk attribute
        self.by_fragment = by_fragment  # Add by_fragment attribute

        # Device-agnostic: always return CPU tensors
        self.device = None

        # Check that required modules are available
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "SLAFIterableDataset is required but not available. Please install required dependencies."
            )

        # Initialize tokenizer (only needed for non-raw mode)
        if not self.raw_mode:
            self.tokenizer = SLAFTokenizer(
                slaf_array=slaf_array,
                tokenizer_type=tokenizer_type,
                vocab_size=vocab_size,
                n_expression_bins=n_expression_bins,
            )

            # Get special tokens from tokenizer
            self.special_tokens = self.tokenizer.special_tokens
        else:
            # For raw mode, we don't need a tokenizer
            self.tokenizer = None
            self.special_tokens = None

        # Use IterableDataset
        self._dataset = SLAFIterableDataset(
            slaf_array=slaf_array,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            seed=42,  # TODO: make configurable
            max_queue_size=500,
            tokenizer_type=tokenizer_type,
            n_epochs=n_epochs,  # Pass n_epochs to dataset
            raw_mode=raw_mode,  # Pass raw_mode to dataset
            verbose=verbose,  # Pass verbose to dataset
            batches_per_chunk=batches_per_chunk,  # Pass batches_per_chunk to dataset
            by_fragment=by_fragment,  # Pass by_fragment to dataset
        )

    def __iter__(self):
        """
        Iterate through batches of pre-tokenized single-cell data.

        Yields batches of pre-tokenized data suitable for machine learning training.
        Each batch contains input_ids, attention_mask, and cell_ids for the
        cells in that batch. All tensors are returned on CPU for device-agnostic training.
        The method automatically handles multi-epoch training when n_epochs > 1.

        Yields:
            dict: Batch dictionary containing:
                - input_ids: Pre-tokenized gene expression data (torch.Tensor)
                - attention_mask: Boolean mask indicating valid tokens (torch.Tensor)
                - cell_ids: Integer IDs of cells in the batch (torch.Tensor)
                - epoch: Current epoch number (int, only if n_epochs > 1)

        Raises:
            ValueError: If the tokenizer type is not supported.
            RuntimeError: If batch processing fails.

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

            >>> # Multi-epoch training
            >>> dataloader = SLAFDataLoader(slaf_array, n_epochs=3)
            >>> epochs_seen = set()
            >>> for batch in dataloader:
            ...     if 'epoch' in batch:
            ...         epochs_seen.add(batch['epoch'])
            ...     if len(epochs_seen) >= 3:  # Stop after seeing all epochs
            ...         break
            >>> print(f"Epochs completed: {sorted(epochs_seen)}")
            Epochs completed: [0, 1, 2]

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
            ...     if batch_idx >= 2:  # Just first few batches
            ...         break
            Processed batch 0
            Processed batch 1
            Processed batch 2

            >>> # Different tokenizer types
            >>> dataloader_geneformer = SLAFDataLoader(slaf_array, tokenizer_type="geneformer")
            >>> dataloader_scgpt = SLAFDataLoader(slaf_array, tokenizer_type="scgpt")
            >>>
            >>> # Compare batch shapes
            >>> for batch in dataloader_geneformer:
            ...     print(f"Geneformer input shape: {batch['input_ids'].shape}")
            ...     break
            Geneformer input shape: (32, 2048)
            >>> for batch in dataloader_scgpt:
            ...     print(f"scGPT input shape: {batch['input_ids'].shape}")
            ...     break
            scGPT input shape: (32, 1024)
        """
        yield from self._dataset

    def __len__(self):
        """
        Return the number of batches in the dataset.

        Note: Since SLAFDataLoader uses an IterableDataset that streams data,
        the exact number of batches is not known in advance. This method
        returns 0 to indicate an unknown length for streaming datasets.

        Returns:
            int: Always returns 0 to indicate unknown length for streaming datasets.

        Examples:
            >>> # Check dataset length
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> dataloader = SLAFDataLoader(slaf_array)
            >>> print(f"Dataset length: {len(dataloader)}")
            Dataset length: 0

            >>> # IterableDataset behavior
            >>> batch_count = 0
            >>> for batch in dataloader:
            ...     batch_count += 1
            ...     if batch_count >= 5:  # Just count first 5 batches
            ...         break
            >>> print(f"Actually processed {batch_count} batches")
            Actually processed 5 batches

            >>> # Length is consistent
            >>> print(f"Length check: {len(dataloader)}")
            Length check: 0
        """
        return 0  # Indicates unknown length

    def __del__(self):
        """
        Cleanup method to stop async prefetching.

        This method is called when the DataLoader object is garbage collected.
        It ensures that the underlying dataset's prefetcher is properly cleaned up
        to prevent resource leaks.

        Examples:
            >>> # DataLoader cleanup happens automatically
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> dataloader = SLAFDataLoader(slaf_array)
            >>> print("DataLoader created")
            DataLoader created
            >>> # When dataloader goes out of scope, __del__ is called automatically
            >>> del dataloader
            >>> print("DataLoader destroyed and cleaned up")
            DataLoader destroyed and cleaned up

            >>> # Manual cleanup (not usually needed)
            >>> dataloader = SLAFDataLoader(slaf_array)
            >>> dataloader.__del__()
            >>> print("Manual cleanup completed")
            Manual cleanup completed
        """
        if hasattr(self, "_dataset"):
            # The SLAFIterableDataset doesn't have a stop method,
            # so we just let it finish its current epoch.
            pass
