"""
Generic distributed dataloader that reads from a queue-like object.

Each queue item is a single sample (one group). The dataloader collects
batch_size samples to form a training batch.

All GPUs in the cluster can concurrently read from the same FIFO queue,
making it directly compatible with DDP or FSDP code.

Uses get_many for efficient batch reads from the queue.
Uses background prefetching to overlap queue I/O and tensor formatting.

Framework-agnostic - accepts any queue-like object with get_many() method.
"""

import threading
from collections.abc import Iterator
from queue import Empty, Queue
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
        prefetch_factor: int = 2,
    ):
        """
        Initialize distributed dataloader.

        Args:
            queue: Queue-like object with get_many() method
            batch_size: Number of samples to collect per training batch
            return_tensors: If True, return torch.Tensor objects (matches SLAFDataLoader).
                          If False, return Python lists/objects. Default: True.
            prefetch_factor: Number of batches to prefetch in background thread.
                           Higher values use more memory but improve throughput.
                           Default: 2 (similar to PyTorch DataLoader).
        """
        self.queue = queue
        self.batch_size = batch_size
        self.return_tensors = return_tensors
        self.prefetch_factor = prefetch_factor

        # Internal queue for prefetched formatted batches
        self._prefetch_queue: Queue[dict[str, Any] | None] = Queue(
            maxsize=prefetch_factor
        )
        self._worker_thread: threading.Thread | None = None
        self._should_stop = False
        self._stop_iteration = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over batches from the queue with background prefetching.

        Uses a background thread to prefetch and format batches while the main
        thread consumes pre-formatted batches. This overlaps queue I/O and tensor
        operations for better throughput.

        Yields:
            Dictionary with processed batch data (matches SLAFDataLoader format):
            - input_ids: Pre-tokenized sequences (torch.Tensor if return_tensors=True, else list)
            - attention_mask: Boolean mask for valid tokens (torch.Tensor if return_tensors=True, else list)
            - cell_ids: Cell integer IDs (torch.Tensor if return_tensors=True, else list)
        """
        # Start background prefetching thread
        self._should_stop = False
        self._stop_iteration = False
        self._worker_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True
        )
        self._worker_thread.start()

        try:
            while True:
                try:
                    # Get prefetched formatted batch (non-blocking with timeout)
                    batch = self._prefetch_queue.get(timeout=1.0)
                    if batch is None:  # Stop iteration marker
                        break
                    yield batch
                except Empty:
                    # Check if worker thread is still alive
                    if (
                        not self._worker_thread.is_alive()
                        and self._prefetch_queue.empty()
                    ):
                        break
                    continue
        finally:
            # Signal worker to stop and wait for it
            self._should_stop = True
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)

    def _prefetch_worker(self):
        """Background worker thread that prefetches and formats batches."""
        while not self._should_stop:
            try:
                # Get samples from queue (this is the I/O operation)
                samples = self.queue.get_many(
                    n_values=self.batch_size, block=True, timeout=1.0
                )
            except TimeoutError:
                # Queue timeout - check if we should continue
                if self._should_stop:
                    break
                continue
            except Exception as e:
                print(f"Error reading from queue: {e}")
                continue

            if not samples:
                # Queue is empty - signal end of iteration
                self._prefetch_queue.put(None)
                break

            # Check for end-of-epoch marker and filter in single pass
            filtered_samples = []
            for s in samples:
                if s is not None:
                    if s.get("end_of_epoch", False):
                        self._stop_iteration = True
                    else:
                        filtered_samples.append(s)

            if self._stop_iteration:
                if not filtered_samples:
                    self._prefetch_queue.put(None)
                    break
                samples = filtered_samples
            else:
                samples = filtered_samples if filtered_samples else samples

            # Format batch (this is the CPU-bound operation)
            if samples:
                # Format batch and put in prefetch queue
                for formatted_batch in self._format_batch(samples):
                    if self._should_stop:
                        break
                    try:
                        self._prefetch_queue.put(formatted_batch, timeout=1.0)
                    except Exception:
                        # Queue full or timeout - continue anyway
                        break

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
