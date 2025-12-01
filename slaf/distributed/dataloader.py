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
import time
from collections.abc import Iterator
from queue import Empty, Queue
from typing import Any

import polars as pl


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
        prefetch_factor: int = 8,
        enable_diagnostics: bool = False,
        queue_prefetch_multiplier: int = 1,
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
            enable_diagnostics: If True, collect timing statistics for bottleneck analysis.
                              Default: False.
            queue_prefetch_multiplier: Multiplier for queue.get_many() n_values to reduce
                                     network round-trips. E.g., if batch_size=32 and
                                     queue_prefetch_multiplier=4, we request 128 samples
                                     per queue call. Default: 1 (no prefetching).
        """
        self.queue = queue
        self.batch_size = batch_size
        self.return_tensors = return_tensors
        self.prefetch_factor = prefetch_factor
        self._use_prefetching = prefetch_factor > 0
        self.enable_diagnostics = enable_diagnostics
        self.queue_prefetch_multiplier = queue_prefetch_multiplier
        # Calculate how many samples to request per queue call
        self._queue_batch_size = batch_size * queue_prefetch_multiplier

        # Internal queue for prefetched formatted batches (only if prefetching enabled)
        if self._use_prefetching:
            self._prefetch_queue: Queue[dict[str, Any] | None] = Queue(
                maxsize=prefetch_factor
            )
        self._worker_thread: threading.Thread | None = None
        self._should_stop = False
        self._stop_iteration = False

        # Diagnostic statistics
        if self.enable_diagnostics:
            self._diagnostics = {
                "queue_get_time": 0.0,
                "format_time": 0.0,
                "filter_time": 0.0,
                "total_batches": 0,
                "queue_get_count": 0,
                "format_count": 0,
                "start_time": time.time(),
            }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over batches from the queue.

        If prefetch_factor > 0, uses background prefetching to overlap I/O and formatting.
        Otherwise, processes batches synchronously (lower memory, simpler).

        Yields:
            Dictionary with processed batch data (matches SLAFDataLoader format):
            - input_ids: Pre-tokenized sequences (torch.Tensor if return_tensors=True, else list)
            - attention_mask: Boolean mask for valid tokens (torch.Tensor if return_tensors=True, else list)
            - cell_ids: Cell integer IDs (torch.Tensor if return_tensors=True, else list)
        """
        if self._use_prefetching:
            # Use background prefetching
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
        else:
            # Synchronous mode (no prefetching) - simpler and lower memory
            stop_iteration = False

            while not stop_iteration:
                # Use get_many with prefetch multiplier to reduce network round-trips
                # Request more samples than batch_size to amortize network latency
                queue_start = time.time() if self.enable_diagnostics else None
                try:
                    samples = self.queue.get_many(
                        n_values=self._queue_batch_size, block=True, timeout=30.0
                    )
                except TimeoutError:
                    stop_iteration = True
                    break
                except Exception as e:
                    print(f"Error reading from queue: {e}")
                    continue

                if self.enable_diagnostics and queue_start:
                    self._diagnostics["queue_get_time"] += time.time() - queue_start
                    self._diagnostics["queue_get_count"] += 1

                if not samples:
                    stop_iteration = True
                    break

                # Check for end-of-epoch marker and filter efficiently
                filter_start = time.time() if self.enable_diagnostics else None
                # Use any() with generator for early exit, then filter
                if any(s is not None and s.get("end_of_epoch", False) for s in samples):
                    stop_iteration = True
                    filtered_samples = [
                        s
                        for s in samples
                        if s is not None and not s.get("end_of_epoch", False)
                    ]
                    if not filtered_samples:
                        break
                    samples = filtered_samples
                else:
                    # Filter None values only
                    samples = [s for s in samples if s is not None]

                if self.enable_diagnostics and filter_start:
                    self._diagnostics["filter_time"] += time.time() - filter_start

                # Yield batches (format synchronously)
                # If we got more samples than batch_size, yield multiple batches
                if samples:
                    format_start = time.time() if self.enable_diagnostics else None
                    # Process samples in chunks of batch_size
                    for i in range(0, len(samples), self.batch_size):
                        batch_samples = samples[i : i + self.batch_size]
                        if batch_samples:
                            for batch in self._format_batch(batch_samples):
                                yield batch
                    if self.enable_diagnostics and format_start:
                        self._diagnostics["format_time"] += time.time() - format_start
                        batches_produced = (
                            len(samples) + self.batch_size - 1
                        ) // self.batch_size
                        self._diagnostics["format_count"] += batches_produced
                        self._diagnostics["total_batches"] += batches_produced

    def _prefetch_worker(self):
        """Background worker thread that prefetches and formats batches."""
        while not self._should_stop:
            try:
                # Get samples from queue (this is the I/O operation)
                # Use prefetch multiplier to reduce network round-trips
                samples = self.queue.get_many(
                    n_values=self._queue_batch_size, block=True, timeout=1.0
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

            # Check for end-of-epoch marker and filter efficiently
            # Use any() with generator for early exit, then filter
            if any(s is not None and s.get("end_of_epoch", False) for s in samples):
                self._stop_iteration = True
                filtered_samples = [
                    s
                    for s in samples
                    if s is not None and not s.get("end_of_epoch", False)
                ]
                if not filtered_samples:
                    self._prefetch_queue.put(None)
                    break
                samples = filtered_samples
            else:
                # Filter None values only
                samples = [s for s in samples if s is not None]

            # Format batches (this is the CPU-bound operation)
            # If we got more samples than batch_size, format multiple batches
            if samples:
                # Process samples in chunks of batch_size
                for i in range(0, len(samples), self.batch_size):
                    batch_samples = samples[i : i + self.batch_size]
                    if batch_samples:
                        # Format batch and put in prefetch queue
                        for formatted_batch in self._format_batch(batch_samples):
                            if self._should_stop:
                                break
                            try:
                                self._prefetch_queue.put(formatted_batch, timeout=1.0)
                            except Exception:
                                # Queue full or timeout - continue anyway
                                break
                    if self._should_stop:
                        break

    def _format_batch(self, samples: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """
        Format samples into a training batch.

        Optimized tensor stacking - single pass extraction, batch operations.
        """
        if not samples:
            return

        # Check format and reshape accordingly
        if "input_ids" in samples[0] and "attention_mask" in samples[0]:
            # Tokenized format - matches SLAFDataLoader output
            if self.return_tensors:
                try:
                    import torch

                    # Optimized: assume samples are already tensors (from tokenizer)
                    # Workers return tensors from SLAFTokenizer.tokenize(), so skip type checking
                    # Single pass extraction with minimal overhead
                    batch_size = len(samples)

                    # Pre-allocate lists for stacking (faster than list comprehension with checks)
                    input_ids_list = [None] * batch_size
                    attention_mask_list = [None] * batch_size
                    cell_ids_list = [0] * batch_size

                    # Single pass: extract all data
                    for i, s in enumerate(samples):
                        input_ids_list[i] = s["input_ids"]
                        attention_mask_list[i] = s["attention_mask"]
                        cell_ids_list[i] = s.get("group_key", 0)

                    # Stack tensors in single operations (very fast)
                    input_ids = torch.stack(input_ids_list)
                    attention_mask = torch.stack(attention_mask_list)
                    cell_ids = torch.tensor(cell_ids_list, dtype=torch.long)

                    # Extract other keys only if they exist (use Polars for efficiency)
                    first_sample = samples[0]
                    extra_keys = [
                        key
                        for key in first_sample.keys()
                        if key not in ["input_ids", "attention_mask", "group_key"]
                    ]
                    if extra_keys:
                        # Use Polars DataFrame for efficient column extraction
                        df = pl.DataFrame(samples)
                        extra_data = {
                            key: df[key].to_list()
                            for key in extra_keys
                            if key in df.columns
                        }
                    else:
                        extra_data = {}

                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "cell_ids": cell_ids,  # Matches SLAFDataLoader format
                        **extra_data,
                    }
                except ImportError:
                    # PyTorch not available - fall back to lists
                    yield {
                        "input_ids": [s["input_ids"] for s in samples],
                        "attention_mask": [s["attention_mask"] for s in samples],
                        "cell_ids": [s.get("group_key", 0) for s in samples],
                    }
            else:
                # Return Python objects (lists) - use Polars DataFrame for efficiency
                df = pl.DataFrame(samples)
                yield {
                    "input_ids": df["input_ids"].to_list(),
                    "attention_mask": df["attention_mask"].to_list(),
                    "cell_ids": (
                        df.get_column("group_key").fill_null(0).to_list()
                        if "group_key" in df.columns
                        else [0] * len(samples)
                    ),
                }
        elif "grouped" in samples[0]:
            # Raw format - return list of DataFrames
            # Optimized: single pass extraction
            batch_size = len(samples)
            grouped_list: list[Any] = [None] * batch_size
            group_keys_list: list[Any] = [0] * batch_size
            for i, s in enumerate(samples):
                grouped_list[i] = s["grouped"]
                group_keys_list[i] = s.get("group_key", 0)
            yield {
                "grouped": grouped_list,
                "group_keys": group_keys_list,
            }
        else:
            # Unknown format - return as-is
            yield {"samples": samples}

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Get diagnostic statistics for bottleneck analysis.

        Returns:
            Dictionary with timing statistics:
            - queue_get_time: Total time spent in queue.get_many() (seconds)
            - format_time: Total time spent formatting batches (seconds)
            - filter_time: Total time spent filtering samples (seconds)
            - total_batches: Total number of batches produced
            - queue_get_count: Number of queue.get_many() calls
            - format_count: Number of format operations
            - elapsed_time: Total elapsed time (seconds)
            - avg_queue_get_time: Average time per queue.get_many() call (ms)
            - avg_format_time: Average time per format operation (ms)
            - queue_get_pct: Percentage of time spent in queue operations
            - format_pct: Percentage of time spent formatting
        """
        if not self.enable_diagnostics:
            return {"error": "Diagnostics not enabled"}

        elapsed = time.time() - self._diagnostics["start_time"]
        diag = self._diagnostics.copy()
        diag["elapsed_time"] = elapsed

        if diag["queue_get_count"] > 0:
            diag["avg_queue_get_time"] = (
                diag["queue_get_time"] / diag["queue_get_count"]
            ) * 1000  # ms
        else:
            diag["avg_queue_get_time"] = 0.0

        if diag["format_count"] > 0:
            diag["avg_format_time"] = (
                diag["format_time"] / diag["format_count"]
            ) * 1000  # ms
        else:
            diag["avg_format_time"] = 0.0

        total_time = diag["queue_get_time"] + diag["format_time"] + diag["filter_time"]
        if total_time > 0:
            diag["queue_get_pct"] = (diag["queue_get_time"] / total_time) * 100
            diag["format_pct"] = (diag["format_time"] / total_time) * 100
            diag["filter_pct"] = (diag["filter_time"] / total_time) * 100
        else:
            diag["queue_get_pct"] = 0.0
            diag["format_pct"] = 0.0
            diag["filter_pct"] = 0.0

        return diag
