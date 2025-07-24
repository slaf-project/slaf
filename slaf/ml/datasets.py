import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Queue
from typing import Any

import polars as pl

# Try to import torch, but make it optional
try:
    import torch
    from torch.utils.data import IterableDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Tensor operations will be disabled.")

# Try to import Lance, but make it optional
try:
    import lance  # type: ignore

    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False
    print("Warning: Lance not available. Fragment loading will be disabled.")

# Try to import Polars, but make it optional
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("Warning: Polars not available. Fragment loading will be disabled.")

from slaf.core.slaf import SLAFArray
from slaf.ml.aggregators import Window
from slaf.ml.samplers import Shuffle
from slaf.ml.tokenizers import SLAFTokenizer


@dataclass
class PrefetchBatch:
    """Container for a batch of tokenized sequences"""

    batch_id: int
    input_ids: torch.Tensor  # Tokenized sequences
    attention_mask: torch.Tensor  # Attention masks
    cell_integer_ids: list[int]  # Corresponding cell integer IDs
    partial_cell_data: dict | None = (
        None  # Store partial cell data for boundary handling
    )
    tokenize_time: float = 0.0  # Time spent on tokenization


class PrefetchBatchProcessor:
    """Processes Lance fragments into prefetch batches using Window and Shuffle strategies"""

    def __init__(
        self,
        slaf_array: SLAFArray,
        window: "Window",
        shuffle: "Shuffle",
        tokenizer: SLAFTokenizer,
        seed: int = 42,
        max_genes: int = 1024,
        batches_per_chunk: int = 50,
        n_expression_bins: int = 10,
        use_binned_expressions: bool = True,
    ):
        self.slaf_array = slaf_array
        self.window = window
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_genes = max_genes
        self.batches_per_chunk = batches_per_chunk
        self.n_expression_bins = n_expression_bins
        self.use_binned_expressions = use_binned_expressions
        self.window_kwargs: dict[str, Any] = {}  # Additional kwargs for window function

        # Create Lance dataset and batch generator
        self.expression_dataset = lance.dataset(
            f"{self.slaf_array.slaf_path}/expression.lance"
        )
        self.batch_generator = self.expression_dataset.to_batches()
        self.batch_id = 0
        self.partial_cell_data: dict[
            Any, pl.DataFrame
        ] = {}  # Store partial cell data across chunks

    def load_prefetch_batch(self) -> PrefetchBatch:
        """Load a chunk of batches and apply window functions with Polars"""
        start_time = time.time()

        # Load multiple batches
        batch_dfs = []
        load_start = time.time()
        batch_sizes = []
        for _ in range(self.batches_per_chunk):
            try:
                batch = next(self.batch_generator)
                batch_df = pl.from_arrow(batch)
                batch_dfs.append(batch_df)
                batch_sizes.append(batch_df.shape[0])
            except StopIteration:
                break

        if not batch_dfs:
            raise StopIteration("No more batches available")

        # Combine all batches
        concat_start = time.time()
        combined_df = pl.concat(batch_dfs)  # type: ignore
        concat_time = time.time() - concat_start
        load_time = time.time() - load_start

        # Print detailed loading breakdown every 10 batches
        if self.batch_id % 10 == 0:
            import psutil  # type: ignore

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            print(f"🔍 Loading breakdown for batch {self.batch_id}:")
            print(
                f"   Individual batch loading: {(concat_start - load_start) * 1000:.1f}ms"
            )
            print(f"   Concatenation: {concat_time * 1000:.1f}ms")
            print(f"   Total loading: {load_time * 1000:.1f}ms")
            print(f"   Number of batches: {len(batch_dfs)}")
            print(f"   Total rows: {combined_df.shape[0]}")
            print(
                f"   Average batch size: {sum(batch_sizes) / len(batch_sizes):.0f} rows"
            )
            print(f"   Batch size range: {min(batch_sizes)} - {max(batch_sizes)} rows")
            print(f"   Memory usage: {memory_mb:.1f} MB")

        # Handle partial cells from previous chunk
        if self.partial_cell_data:
            # Combine partial data with new data
            partial_dfs = list(self.partial_cell_data.values())
            if partial_dfs:
                partial_combined = pl.concat(partial_dfs)  # type: ignore
                combined_df = pl.concat([partial_combined, combined_df])  # type: ignore

        # Handle cell boundary crossing
        # Find the last complete cell_integer_id
        cell_counts = combined_df.group_by("cell_integer_id").count()  # type: ignore
        last_complete_cell = cell_counts["cell_integer_id"].max()

        # Split into complete cells and partial cells
        complete_df = combined_df.filter(pl.col("cell_integer_id") < last_complete_cell)  # type: ignore
        partial_df = combined_df.filter(pl.col("cell_integer_id") == last_complete_cell)  # type: ignore

        # Store partial cell data for next chunk
        self.partial_cell_data = {}
        if len(partial_df) > 0:
            self.partial_cell_data[last_complete_cell] = partial_df  # type: ignore

        # Process complete cells
        if len(complete_df) > 0:
            # Apply window function with expression binning parameters
            window_start = time.time()
            window_params = {
                "n_expression_bins": self.n_expression_bins,
                "use_binned_expressions": self.use_binned_expressions,
            }
            window_params.update(self.window_kwargs)  # Add any additional kwargs
            grouped = self.window.apply(
                complete_df,  # type: ignore
                self.max_genes,
                **window_params,
            )
            window_time = time.time() - window_start

            # Apply shuffle strategy
            shuffle_start = time.time()
            cell_integer_ids = grouped["cell_integer_id"].to_list()
            shuffled_cell_integer_ids = self.shuffle.apply(
                cell_integer_ids, self.seed + self.batch_id
            )

            # Reorder grouped data based on shuffled cell IDs
            cell_id_to_index = {
                cell_id: i for i, cell_id in enumerate(cell_integer_ids)
            }
            shuffled_indices = [
                cell_id_to_index[cell_id] for cell_id in shuffled_cell_integer_ids
            ]

            shuffled_gene_sequences = [
                grouped["gene_sequence"][i].to_list() for i in shuffled_indices
            ]

            # Handle expression sequences for scGPT
            shuffled_expr_sequences = None
            if "expr_sequence" in grouped.columns:
                # scGPT format: separate gene_sequence and expr_sequence columns
                shuffled_expr_sequences = [
                    grouped["expr_sequence"][i].to_list() for i in shuffled_indices
                ]
            # For scGPT, we now use separate columns for better performance

            shuffle_time = time.time() - shuffle_start

            # Tokenize the sequences
            tokenize_start = time.time()
            input_ids, attention_mask = self.tokenizer.tokenize(
                gene_sequences=shuffled_gene_sequences,
                expr_sequences=shuffled_expr_sequences,
                max_genes=self.max_genes,
            )
            tokenize_time = time.time() - tokenize_start
            total_time = time.time() - start_time

            # Print timing breakdown every 10 batches
            if self.batch_id % 10 == 0:
                print(f"📊 Batch {self.batch_id} timing breakdown:")
                print(f"   Load: {load_time * 1000:.1f}ms")
                print(f"   Window: {window_time * 1000:.1f}ms")
                print(f"   Shuffle: {shuffle_time * 1000:.1f}ms")
                print(f"   Tokenize: {tokenize_time * 1000:.1f}ms")
                print(f"   Total: {total_time * 1000:.1f}ms")
                print(f"   Complete cells: {len(shuffled_cell_integer_ids)}")

            self.batch_id += 1
            return PrefetchBatch(
                batch_id=self.batch_id - 1,
                input_ids=input_ids,
                attention_mask=attention_mask,
                cell_integer_ids=shuffled_cell_integer_ids,
                partial_cell_data=self.partial_cell_data.copy(),
                tokenize_time=tokenize_time,
            )
        else:
            # No complete cells in this chunk
            self.batch_id += 1
            return PrefetchBatch(
                batch_id=self.batch_id - 1,
                input_ids=torch.empty((0, self.max_genes), dtype=torch.long),
                attention_mask=torch.empty((0, self.max_genes), dtype=torch.bool),
                cell_integer_ids=[],
                partial_cell_data=self.partial_cell_data.copy(),
                tokenize_time=0.0,  # No tokenization for empty batch
            )


class AsyncPrefetcher:
    """Async prefetcher for Lance batches"""

    def __init__(
        self, batch_processor: PrefetchBatchProcessor, max_queue_size: int = 500
    ):
        self.batch_processor = batch_processor
        self.max_queue_size = max_queue_size
        self.queue: Queue[PrefetchBatch] = Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.should_stop = False

        # Monitoring stats
        self.total_cells_added = 0
        self.start_time = None
        self.last_rate_print = 0
        self.total_tokenize_time = 0.0
        self.tokenize_count = 0

    def start(self):
        """Start the prefetching worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.should_stop = False
            self.start_time = time.time()
            self.worker_thread = threading.Thread(
                target=self._prefetch_worker, daemon=True
            )
            self.worker_thread.start()

    def stop(self):
        """Stop the prefetching worker thread"""
        self.should_stop = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def _prefetch_worker(self):
        """Worker thread that loads batches in background"""
        while not self.should_stop:
            try:
                # Load batch chunk
                batch = self.batch_processor.load_prefetch_batch()

                # Update monitoring stats
                self.total_cells_added += len(batch.cell_integer_ids)
                self.total_tokenize_time += batch.tokenize_time
                self.tokenize_count += 1
                elapsed = time.time() - (self.start_time or 0)
                rate = self.total_cells_added / elapsed if elapsed > 0 else 0

                # Print rate every 10 batches
                if batch.batch_id % 10 == 0 and batch.batch_id > self.last_rate_print:
                    avg_tokenize_ms = (
                        self.total_tokenize_time / self.tokenize_count
                    ) * 1000
                    print(
                        f"📊 Prefetch rate: {rate:.1f} cells/sec (total: {self.total_cells_added} cells, avg tokenize: {avg_tokenize_ms:.1f}ms)"
                    )
                    self.last_rate_print = batch.batch_id

                # Put in queue
                try:
                    self.queue.put_nowait(batch)
                except queue.Full:
                    # Queue is full, wait a bit
                    time.sleep(0.1)

            except StopIteration:
                print("Reached end of batches")
                break
            except Exception as e:
                print(f"Error loading batch: {e}")
                break

    def get_batch(self) -> PrefetchBatch | None:
        """Get next batch from queue"""
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def has_batch(self) -> bool:
        """Check if batch is available"""
        return not self.queue.empty()

    def get_stats(self) -> dict:
        """Get prefetch statistics"""
        elapsed = time.time() - (self.start_time or 0)
        rate = self.total_cells_added / elapsed if elapsed > 0 else 0
        avg_tokenize_time = (
            self.total_tokenize_time / self.tokenize_count
            if self.tokenize_count > 0
            else 0
        )
        return {
            "total_cells": self.total_cells_added,
            "elapsed_time": elapsed,
            "cells_per_sec": rate,
            "queue_size": self.queue.qsize(),
            "queue_full": self.queue.full(),
            "total_tokenize_time": self.total_tokenize_time,
            "tokenize_count": self.tokenize_count,
            "avg_tokenize_time_ms": avg_tokenize_time * 1000,
        }


class SLAFIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for streaming SLAF data with async batch prefetching.

    This dataset provides efficient streaming of tokenized single-cell data
    with background batch loading to minimize GPU idle time.
    """

    def __init__(
        self,
        slaf_array: SLAFArray,
        tokenizer: SLAFTokenizer,
        batch_size: int = 32,
        seed: int = 42,
        max_queue_size: int = 10,
        pin_memory: bool = False,
        sampler_strategy: str = "sequential",
        tokenizer_type: str = "geneformer",  # Add tokenizer type parameter
        use_binned_expressions: bool = False,  # Add binned expressions parameter
    ):
        super().__init__()
        self.slaf_array = slaf_array
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seed = seed
        self.max_queue_size = max_queue_size
        self.tokenizer_type = tokenizer_type
        self.max_genes = (
            1024 if tokenizer_type == "scgpt" else 2048
        )  # Default max_genes
        self.pin_memory = pin_memory

        # Pre-allocate cell IDs buffer for better performance
        if TORCH_AVAILABLE:
            self.cell_ids_buffer = torch.zeros(batch_size, dtype=torch.long)

        # Device-agnostic: always use CPU tensors
        self.device = torch.device("cpu")

        # Initialize processor based on loading strategy
        max_genes = 1024 if tokenizer_type == "scgpt" else 2048

        # Create window and shuffle strategies using factory functions
        from slaf.ml.aggregators import WindowType, create_window
        from slaf.ml.samplers import ShuffleType, create_shuffle

        window = create_window(WindowType(tokenizer_type))
        shuffle = create_shuffle(ShuffleType.RANDOM)

        # Get expression binning parameters from tokenizer
        n_expression_bins = tokenizer.n_expression_bins

        # Set binning based on tokenizer type
        use_binned_expressions = use_binned_expressions  # Use parameter value

        self.batch_processor = PrefetchBatchProcessor(
            slaf_array=slaf_array,
            window=window,
            shuffle=shuffle,
            tokenizer=tokenizer,
            seed=seed,
            max_genes=max_genes,
            batches_per_chunk=50,
            n_expression_bins=n_expression_bins,
            use_binned_expressions=use_binned_expressions,
        )
        self.prefetcher = AsyncPrefetcher(
            batch_processor=self.batch_processor,
            max_queue_size=500,
        )

        # Start async prefetching
        self.prefetcher.start()

        # Wait for prefetcher to initialize
        self._wait_for_prefetcher_ready()

    def _wait_for_prefetcher_ready(self, timeout: float = 10.0):
        """Wait for prefetcher to be ready with data"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.prefetcher.has_batch():
                print(f"✓ Prefetcher ready after {time.time() - start_time:.2f}s")
                return
            time.sleep(0.1)

        print(f"⚠️ Prefetcher not ready after {timeout}s, proceeding anyway...")

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate through batches of tokenized data.

        Yields:
            dict: Batch containing:
                - input_ids: Tokenized sequences (torch.Tensor)
                - attention_mask: Boolean mask for valid tokens (torch.Tensor)
                - cell_ids: Cell integer IDs (torch.Tensor)
        """
        start_time = time.time()
        batches_yielded = 0
        last_rate_time = start_time
        last_rate_batches = 0

        while True:
            # Get data from prefetcher
            data = self.prefetcher.get_batch()
            if data is None:
                while not self.prefetcher.has_batch():
                    time.sleep(0.1)
                data = self.prefetcher.get_batch()
                if data is None:
                    break

            # Process batch data - now pre-tokenized
            num_cells = len(data.cell_integer_ids)
            input_ids = data.input_ids
            attention_mask = data.attention_mask
            cell_integer_ids = data.cell_integer_ids

            # Process all cells in this data chunk
            for batch_start in range(0, num_cells, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_cells)

                # Extract batch data
                batch_input_ids = input_ids[batch_start:batch_end]
                batch_attention_mask = attention_mask[batch_start:batch_end]
                batch_cell_ids = cell_integer_ids[batch_start:batch_end]

                # Convert cell IDs to tensor (pre-allocated)
                tensor_start = time.time()
                cell_ids_tensor = self.cell_ids_buffer[: len(batch_cell_ids)].clone()
                cell_ids_tensor[:] = torch.tensor(batch_cell_ids, dtype=torch.long)
                tensor_time = time.time() - tensor_start

                # Always return CPU tensors (device-agnostic)
                # Training loop should handle device transfer

                batches_yielded += 1

                # Print detailed timing every 1000 batches
                if batches_yielded % 100 == 0:
                    print(f"🔍 Batch {batches_yielded} timing breakdown:")
                    print(f"   Tensor creation: {tensor_time * 1000:.1f}ms")
                    print("   Pre-tokenized data (no tokenization overhead)")

                # Logging
                if batches_yielded % 100 == 0:
                    current_time = time.time()
                    time_since_last_rate = current_time - last_rate_time
                    batches_since_last_rate = batches_yielded - last_rate_batches
                    if time_since_last_rate > 0:
                        instantaneous_rate = (
                            batches_since_last_rate / time_since_last_rate
                        )
                        overall_rate = batches_yielded / (current_time - start_time)
                        print(
                            f"📊 Batch {batches_yielded}: {instantaneous_rate:.1f} batches/sec (instantaneous, overall: {overall_rate:.1f})"
                        )
                    last_rate_time = current_time
                    last_rate_batches = batches_yielded

                yield {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    "cell_ids": cell_ids_tensor,
                }

    def __del__(self):
        """Cleanup when dataset is destroyed"""
        self.prefetcher.stop()

    # Tokenization is now handled at the PrefetchBatchProcessor level
    # These methods are no longer needed since we consume pre-tokenized data
