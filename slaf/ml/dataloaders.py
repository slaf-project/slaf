import numpy as np
from typing import List, Tuple
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
    """High-performance DataLoader for SLAF data"""

    def __init__(
        self,
        slaf_array,
        tokenizer_type="geneformer",
        batch_size=32,
        max_genes=2048,
        num_workers=4,
        vocab_size=50000,
        n_expression_bins=10,
        chunk_size=1024,
        device=None,  # Allow manual device override
    ):
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

    def _get_cell_integer_ranges(self) -> List[Tuple[int, int]]:
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
        """Iterate through batches"""
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
