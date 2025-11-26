"""
Generic distributed dataloader that reads from a queue-like object.

Each queue item is a single sample (one group). The dataloader collects
batch_size samples to form a training batch.

All GPUs in the cluster can concurrently read from the same FIFO queue,
making it directly compatible with DDP or FSDP code.

Uses get_many for efficient batch reads from the queue.

Framework-agnostic - accepts any queue-like object with get_many() method.
"""

from collections.abc import Iterator
from typing import Any


class DistributedDataLoader:
    """
    Generic distributed dataloader that reads from a queue-like object.

    Each queue item is a single sample (one group). The dataloader collects
    batch_size samples to form a training batch.

    The queue is a FIFO queue, and all GPUs in the cluster concurrently
    read from it within their respective training loops. This is directly
    compatible with DDP or FSDP code.

    Framework-agnostic - works with any queue-like object that has:
    - get(timeout: float) -> Any | None method
    - Raises TimeoutError when timeout is reached

    Args:
        queue: Queue-like object to read samples from (e.g., Modal Queue)
        batch_size: Number of samples to collect per training batch
    """

    def __init__(
        self,
        queue: Any,
        batch_size: int = 32,
        return_tensors: bool = True,
    ):
        """
        Initialize distributed dataloader.

        Args:
            queue: Queue-like object with get_many() method
            batch_size: Number of samples to collect per training batch
            return_tensors: If True, return torch.Tensor objects (matches SLAFDataLoader).
                          If False, return Python lists/objects. Default: True.
        """
        self.queue = queue
        self.batch_size = batch_size
        self.return_tensors = return_tensors

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over batches from the queue.

        Collects batch_size samples from the queue using get_many (single round trip)
        and reshapes them directly into a training batch.

        Yields:
            Dictionary with processed batch data (matches SLAFDataLoader format):
            - input_ids: Pre-tokenized sequences (torch.Tensor if return_tensors=True, else list)
            - attention_mask: Boolean mask for valid tokens (torch.Tensor if return_tensors=True, else list)
            - cell_ids: Cell integer IDs (torch.Tensor if return_tensors=True, else list)
        """
        stop_iteration = False

        while not stop_iteration:
            # Use get_many for batch_size samples (single round trip)
            # This is much more efficient than multiple get() calls
            try:
                samples = self.queue.get_many(
                    n_values=self.batch_size, block=True, timeout=30.0
                )
            except TimeoutError:
                stop_iteration = True
                break
            except Exception as e:
                print(f"Error reading from queue: {e}")
                continue

            if not samples:
                stop_iteration = True
                break

            # Check for end-of-epoch marker
            end_of_epoch_samples = [
                s for s in samples if s is not None and s.get("end_of_epoch", False)
            ]
            if end_of_epoch_samples:
                stop_iteration = True
                # Filter out end-of-epoch markers
                samples = [
                    s
                    for s in samples
                    if s is not None and not s.get("end_of_epoch", False)
                ]
                if not samples:
                    break

            # Yield batch
            if samples:
                yield from self._format_batch(samples)

    def _format_batch(self, samples: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """
        Format samples into a training batch.

        Optimized tensor stacking - pre-extracts lists then stacks once.
        """
        if not samples:
            return

        # Check format and reshape accordingly
        if "input_ids" in samples[0] and "attention_mask" in samples[0]:
            # Tokenized format - matches SLAFDataLoader output
            if self.return_tensors:
                try:
                    import torch

                    # Optimized: extract all tensors first, then stack once
                    # This is faster than stacking one-by-one
                    input_ids_list = [
                        (
                            s["input_ids"]
                            if isinstance(s["input_ids"], torch.Tensor)
                            else torch.tensor(s["input_ids"])
                        )
                        for s in samples
                    ]
                    attention_mask_list = [
                        (
                            s["attention_mask"]
                            if isinstance(s["attention_mask"], torch.Tensor)
                            else torch.tensor(s["attention_mask"])
                        )
                        for s in samples
                    ]

                    input_ids = torch.stack(input_ids_list)
                    attention_mask = torch.stack(attention_mask_list)
                    # Convert group_key to cell_ids (torch.Tensor) to match SLAFDataLoader
                    cell_ids = torch.tensor([s.get("group_key", 0) for s in samples])

                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "cell_ids": cell_ids,  # Matches SLAFDataLoader format
                        **{
                            key: [s[key] for s in samples]
                            for key in samples[0].keys()
                            if key not in ["input_ids", "attention_mask", "group_key"]
                        },
                    }
                except ImportError:
                    # PyTorch not available - fall back to lists
                    yield {
                        "input_ids": [s["input_ids"] for s in samples],
                        "attention_mask": [s["attention_mask"] for s in samples],
                        "cell_ids": [s.get("group_key", 0) for s in samples],
                    }
            else:
                # Return Python objects (lists)
                yield {
                    "input_ids": [s["input_ids"] for s in samples],
                    "attention_mask": [s["attention_mask"] for s in samples],
                    "cell_ids": [s.get("group_key", 0) for s in samples],
                }
        elif "grouped" in samples[0]:
            # Raw format - return list of DataFrames
            yield {
                "grouped": [s["grouped"] for s in samples],
                "group_keys": [s.get("group_key") for s in samples],
            }
        else:
            # Unknown format - return as-is
            yield {"samples": samples}
