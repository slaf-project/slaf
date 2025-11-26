"""
Generic distributed dataloader that reads from a queue-like object.

Each queue item is a single sample (one group). The dataloader collects
batch_size samples to form a training batch.

All GPUs in the cluster can concurrently read from the same FIFO queue,
making it directly compatible with DDP or FSDP code.

Uses multithreading to read from queue concurrently for better performance.

Framework-agnostic - accepts any queue-like object with get() method.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
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
        n_queue_readers: int = 4,
        return_tensors: bool = True,
    ):
        """
        Initialize distributed dataloader.

        Args:
            queue: Queue-like object with get(timeout) method
            batch_size: Number of samples to collect per training batch
            n_queue_readers: Number of concurrent threads to read from queue (default: 4)
            return_tensors: If True, return torch.Tensor objects (matches SLAFDataLoader).
                          If False, return Python lists/objects. Default: True.
        """
        self.queue = queue
        self.batch_size = batch_size
        self.n_queue_readers = n_queue_readers
        self.return_tensors = return_tensors

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over batches from the queue.

        Collects batch_size samples from the queue using multithreaded reads
        and reshapes them directly into a training batch.

        Yields:
            Dictionary with processed batch data (matches SLAFDataLoader format):
            - input_ids: Pre-tokenized sequences (torch.Tensor if return_tensors=True, else list)
            - attention_mask: Boolean mask for valid tokens (torch.Tensor if return_tensors=True, else list)
            - cell_ids: Cell integer IDs (torch.Tensor if return_tensors=True, else list)
        """
        stop_iteration = False

        with ThreadPoolExecutor(max_workers=self.n_queue_readers) as executor:
            while not stop_iteration:
                # Submit batch_size concurrent read tasks
                futures = [
                    executor.submit(self.queue.get, timeout=30.0)
                    for _ in range(self.batch_size)
                ]

                # Get all results at once (vectorized)
                try:
                    samples = [f.result() for f in futures]
                except TimeoutError:
                    stop_iteration = True
                    break
                except Exception as e:
                    print(f"Error reading from queue: {e}")
                    continue

                # Check for end-of-epoch marker
                if any(s is None or s.get("end_of_epoch", False) for s in samples):
                    stop_iteration = True
                    # Filter out end-of-epoch markers
                    samples = [
                        s
                        for s in samples
                        if s is not None and not s.get("end_of_epoch", False)
                    ]
                    if not samples:
                        break

                # Reshape directly into batch format (no separate combine function)
                if not samples:
                    continue

                # Check format and reshape accordingly
                if "input_ids" in samples[0] and "attention_mask" in samples[0]:
                    # Tokenized format - matches SLAFDataLoader output
                    if self.return_tensors:
                        try:
                            import torch

                            input_ids = torch.stack(
                                [
                                    (
                                        s["input_ids"]
                                        if isinstance(s["input_ids"], torch.Tensor)
                                        else torch.tensor(s["input_ids"])
                                    )
                                    for s in samples
                                ]
                            )
                            attention_mask = torch.stack(
                                [
                                    (
                                        s["attention_mask"]
                                        if isinstance(s["attention_mask"], torch.Tensor)
                                        else torch.tensor(s["attention_mask"])
                                    )
                                    for s in samples
                                ]
                            )
                            # Convert group_key to cell_ids (torch.Tensor) to match SLAFDataLoader
                            cell_ids = torch.tensor(
                                [s.get("group_key", 0) for s in samples]
                            )

                            yield {
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                                "cell_ids": cell_ids,  # Matches SLAFDataLoader format
                                **{
                                    key: [s[key] for s in samples]
                                    for key in samples[0].keys()
                                    if key
                                    not in ["input_ids", "attention_mask", "group_key"]
                                },
                            }
                        except ImportError:
                            # PyTorch not available - fall back to lists
                            yield {
                                "input_ids": [s["input_ids"] for s in samples],
                                "attention_mask": [
                                    s["attention_mask"] for s in samples
                                ],
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
