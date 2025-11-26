"""
Generic distributed dataloader that reads from a queue-like object.

Simple iterator that reads batches from a queue.
All GPUs in the cluster can concurrently read from the same FIFO queue,
making it directly compatible with DDP or FSDP code.

Framework-agnostic - accepts any queue-like object with get() method.
"""

from collections.abc import Iterator
from typing import Any


class DistributedDataLoader:
    """
    Generic distributed dataloader that reads from a queue-like object.

    The queue is a FIFO queue, and all GPUs in the cluster concurrently
    read from it within their respective training loops. This is directly
    compatible with DDP or FSDP code.

    Framework-agnostic - works with any queue-like object that has:
    - get(timeout: float) -> Any | None method
    - Raises TimeoutError when timeout is reached

    Args:
        queue: Queue-like object to read batches from (e.g., Modal Queue)
    """

    def __init__(self, queue: Any):
        """
        Initialize distributed dataloader.

        Args:
            queue: Queue-like object with get(timeout) method
        """
        self.queue = queue

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over batches from the queue.

        Yields:
            Dictionary with processed batch data (format depends on tokenizer)
        """
        while True:
            try:
                batch = self.queue.get(timeout=30.0)
                # Check for end-of-epoch marker
                if batch is None or batch.get("end_of_epoch", False):
                    break
                yield batch
            except TimeoutError:
                # Queue timeout - check if workers are still running
                # In production, this would check worker status
                # For now, we'll break on timeout (workers finished)
                break
