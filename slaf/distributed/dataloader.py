"""
Generic distributed dataloader that reads from Modal Queue.

Simple iterator that reads batches from a Modal Queue.
All GPUs in the cluster can concurrently read from the same FIFO queue,
making it directly compatible with DDP or FSDP code.
"""

from collections.abc import Iterator
from typing import Any

import modal


class DistributedDataLoader:
    """
    Generic distributed dataloader that reads from Modal Queue.

    The queue is a FIFO queue, and all GPUs in the cluster concurrently
    read from it within their respective training loops. This is directly
    compatible with DDP or FSDP code.

    Args:
        queue_name: Name of the Modal Queue to read from
    """

    def __init__(self, queue_name: str):
        self.queue_name = queue_name
        self.queue = modal.Queue.from_name(queue_name, create_if_missing=False)

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
