import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Queue
from typing import Any

import polars as pl

# Try to import rich for colored output
try:
    from rich.console import Console
    from rich.panel import Panel

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

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


def print_prefetch(message: str):
    """Print prefetch-related messages in cyan with a panel"""
    if RICH_AVAILABLE:
        console.print(Panel(message, border_style="cyan"))
    else:
        print(f"🔍 {message}")


def print_training(message: str):
    """Print training-related messages in green with a panel"""
    if RICH_AVAILABLE:
        console.print(Panel(message, border_style="green"))
    else:
        print(f"📊 {message}")


def print_epoch_transition(message: str):
    """Print epoch transition messages in yellow"""
    if RICH_AVAILABLE:
        console.print(f"[yellow]🔄 {message}[/yellow]")
    else:
        print(f"🔄 {message}")


def print_completion(message: str):
    """Print completion messages in bright green with a panel"""
    if RICH_AVAILABLE:
        console.print(
            Panel(
                f"[bright_green]✅ {message}[/bright_green]",
                border_style="bright_green",
            )
        )
    else:
        print(f"✅ {message}")


def print_warning(message: str):
    """Print warning messages in orange"""
    if RICH_AVAILABLE:
        console.print(f"[orange3]⚠️ {message}[/orange3]")
    else:
        print(f"⚠️ {message}")


@dataclass
class PrefetchBatch:
    """
    Container for a batch of pre-tokenized sequences.

    This dataclass holds the results of batch processing from Lance fragments,
    including tokenized sequences, attention masks, and metadata for training.
    """

    batch_id: int
    input_ids: torch.Tensor  # Tokenized sequences
    attention_mask: torch.Tensor  # Attention masks
    cell_integer_ids: list[int]  # Corresponding cell integer IDs
    partial_cell_data: dict | None = (
        None  # Store partial cell data for boundary handling
    )
    tokenize_time: float = 0.0  # Time spent on tokenization


class PrefetchBatchProcessor:
    """
    Processes Lance fragments into pre-tokenized batches using Window and Shuffle strategies.

    This processor loads Lance fragments, applies window functions to rank and filter genes,
    shuffles cells for training, and tokenizes the sequences. It handles the complete
    pipeline from raw Lance data to training-ready tensors.
    """

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
        n_epochs: int = 1,  # Add n_epochs parameter
    ):
        """
        Initialize the PrefetchBatchProcessor with processing configuration.

        Args:
            slaf_array: SLAFArray instance containing the single-cell data.
                       Must have a valid Lance dataset at slaf_path/expression.lance.
            window: Window function strategy for gene ranking and filtering.
                   Determines how genes are selected and ordered within cells.
            shuffle: Shuffle strategy for cell ordering within batches.
                    Controls the randomization of cells for training.
            tokenizer: SLAFTokenizer instance for sequence tokenization.
                      Handles conversion of gene sequences to token IDs.
            seed: Random seed for reproducible shuffling and processing.
                  Used by both window and shuffle strategies.
            max_genes: Maximum number of genes to include per cell.
                      For Geneformer: same as sequence length.
                      For scGPT: number of gene-expression pairs.
            batches_per_chunk: Number of Lance batches to process together.
                             Higher values improve throughput but use more memory.
                             Range: 10-200, default: 50.
            n_expression_bins: Number of expression bins for scGPT discretization.
                             Only used when use_binned_expressions=True.
                             Range: 1-1000, default: 10.
            use_binned_expressions: Whether to use binned expression values for scGPT.
                                   If False, raw expression values are used.
                                   Only affects scGPT tokenization.
            n_epochs: Number of epochs to run. If > 1, the generator will be reset
                     after each epoch. Default: 1.

        Raises:
            ValueError: If parameters are invalid or SLAF array is malformed.
            RuntimeError: If Lance dataset cannot be loaded.
            TypeError: If slaf_array is not a valid SLAFArray instance.

        Examples:
            >>> # Basic initialization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> window = ScGPTWindow()
            >>> shuffle = RandomShuffle()
            >>> tokenizer = SLAFTokenizer(slaf_array)
            >>> processor = PrefetchBatchProcessor(
            ...     slaf_array=slaf_array,
            ...     window=window,
            ...     shuffle=shuffle,
            ...     tokenizer=tokenizer
            ... )
            >>> print(f"Max genes: {processor.max_genes}")
            Max genes: 1024

            >>> # Custom configuration
            >>> processor = PrefetchBatchProcessor(
            ...     slaf_array=slaf_array,
            ...     window=window,
            ...     shuffle=shuffle,
            ...     tokenizer=tokenizer,
            ...     max_genes=2048,
            ...     batches_per_chunk=100,
            ...     use_binned_expressions=False
            ... )
            >>> print(f"Batches per chunk: {processor.batches_per_chunk}")
            Batches per chunk: 100

            >>> # Error handling for invalid SLAF array
            >>> try:
            ...     processor = PrefetchBatchProcessor(None, window, shuffle, tokenizer)
            ... except TypeError as e:
            ...     print(f"Error: {e}")
            Error: slaf_array must be a valid SLAFArray instance
        """
        self.slaf_array = slaf_array
        self.window = window
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_genes = max_genes
        self.batches_per_chunk = batches_per_chunk
        self.n_expression_bins = n_expression_bins
        self.use_binned_expressions = use_binned_expressions
        self.n_epochs = n_epochs
        self.window_kwargs: dict[str, Any] = {}  # Additional kwargs for window function

        # Create Lance dataset and batch generator
        self.expression_dataset = lance.dataset(
            f"{self.slaf_array.slaf_path}/expression.lance"
        )
        self.batch_generator = self.expression_dataset.to_batches()
        self.batch_id = 0
        self.current_epoch = 0
        self.partial_cell_data: dict[
            Any, pl.DataFrame
        ] = {}  # Store partial cell data across chunks

        # Initialize timing variables for consolidated reporting
        self._last_load_time = 0.0
        self._last_batch_dfs_count = 0
        self._last_total_rows = 0
        self._last_memory_mb = 0.0

    def reset_for_epoch(self, epoch: int) -> None:
        """
        Reset the batch generator for a new epoch.

        Args:
            epoch: The epoch number (0-indexed)

        Raises:
            ValueError: If epoch is invalid or exceeds n_epochs
        """
        if epoch < 0 or epoch >= self.n_epochs:
            raise ValueError(
                f"Invalid epoch {epoch}. Must be 0 <= epoch < {self.n_epochs}"
            )

        self.current_epoch = epoch
        self.batch_id = 0
        self.partial_cell_data = {}  # Reset partial cell data

        # Reinitialize the batch generator
        self.batch_generator = self.expression_dataset.to_batches()

        print_epoch_transition(f"Reset batch generator for epoch {epoch}")

    def load_prefetch_batch(self) -> PrefetchBatch:
        """
        Load and process a chunk of Lance batches into pre-tokenized sequences.

        This method loads multiple Lance batches, applies window functions to rank
        and filter genes, shuffles cells for training, and tokenizes the sequences.
        It handles cell boundary crossing and partial data management.

        Returns:
            PrefetchBatch: Container with tokenized sequences and metadata

        Raises:
            StopIteration: When no more batches are available for current epoch
        """
        # Iterative approach to handle epoch transitions
        while True:
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
                # Check if we should start a new epoch
                if self.current_epoch + 1 < self.n_epochs:
                    print_epoch_transition(
                        f"Epoch {self.current_epoch} complete, starting epoch {self.current_epoch + 1}"
                    )
                    self.reset_for_epoch(self.current_epoch + 1)
                    # Continue the loop to load the first batch of the new epoch
                    continue
                else:
                    raise StopIteration("No more epochs available")

            # Combine all batches
            combined_df = pl.concat(batch_dfs)  # type: ignore
            load_time = time.time() - load_start

            # Print detailed loading breakdown every 10 batches
            if self.batch_id % 10 == 0:
                import psutil  # type: ignore

                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024

                # Store timing info for consolidated report
                self._last_load_time = load_time
                self._last_batch_dfs_count = len(batch_dfs)
                self._last_total_rows = combined_df.shape[0]
                self._last_memory_mb = memory_mb

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
            complete_df = combined_df.filter(
                pl.col("cell_integer_id") < last_complete_cell
            )  # type: ignore
            partial_df = combined_df.filter(
                pl.col("cell_integer_id") == last_complete_cell
            )  # type: ignore

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
                    cell_integer_ids,
                    self.seed + self.batch_id + self.current_epoch * 10000,
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

                # Print consolidated timing breakdown every 10 batches
                if self.batch_id % 10 == 0:
                    # Consolidate all prefetch batch reporting into one cyan block
                    prefetch_report = f"Prefetch batch {self.batch_id} (epoch {self.current_epoch}):\n"
                    prefetch_report += f"   Lance loading: {self._last_load_time * 1000:.1f}ms ({self._last_batch_dfs_count} batches, {self._last_total_rows} rows)\n"
                    prefetch_report += f"   Processing: {window_time * 1000:.1f}ms window, {shuffle_time * 1000:.1f}ms shuffle, {tokenize_time * 1000:.1f}ms tokenize\n"
                    prefetch_report += f"   Total: {total_time * 1000:.1f}ms, {len(shuffled_cell_integer_ids)} cells, {self._last_memory_mb:.1f} MB"

                    print_prefetch(prefetch_report)

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
                # No complete cells in this chunk, continue to next iteration
                self.batch_id += 1
                continue


class AsyncPrefetcher:
    """
    Asynchronous prefetcher for Lance batch processing.

    This prefetcher runs batch processing in a background thread to minimize
    GPU idle time during training. It maintains a queue of pre-processed batches
    and provides monitoring statistics.
    """

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
        self.current_epoch = 0

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
                self.current_epoch = self.batch_processor.current_epoch
                elapsed = time.time() - (self.start_time or 0)
                rate = self.total_cells_added / elapsed if elapsed > 0 else 0

                # Print rate every 10 batches
                if batch.batch_id % 10 == 0 and batch.batch_id > self.last_rate_print:
                    avg_tokenize_ms = (
                        self.total_tokenize_time / self.tokenize_count
                    ) * 1000
                    rate_report = f"Prefetch rate: {rate:.1f} cells/sec (epoch {self.current_epoch}, total: {self.total_cells_added} cells, avg tokenize: {avg_tokenize_ms:.1f}ms)"
                    print_prefetch(rate_report)
                    self.last_rate_print = batch.batch_id

                # Put in queue
                try:
                    self.queue.put_nowait(batch)
                except queue.Full:
                    # Queue is full, wait a bit
                    time.sleep(0.1)

            except StopIteration as e:
                if "No more epochs available" in str(e):
                    print_completion(
                        f"All {self.batch_processor.n_epochs} epochs completed"
                    )
                else:
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
            "current_epoch": self.current_epoch,
            "n_epochs": self.batch_processor.n_epochs,
        }


class SLAFIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for streaming pre-tokenized SLAF data with async prefetching.

    This dataset provides efficient streaming of pre-tokenized single-cell data
    with background batch processing to minimize GPU idle time. It handles the
    complete pipeline from Lance fragments to training-ready tensors.

    Key Features:
        - Pre-tokenized sequences for maximum performance
        - Async batch processing with background prefetching
        - Device-agnostic CPU tensor output
        - Memory-efficient streaming
        - Support for multiple tokenization strategies
        - Multi-epoch training support with automatic generator reset

    Args:
        slaf_array: SLAFArray instance containing the single-cell data
        tokenizer: SLAFTokenizer instance for sequence tokenization
        batch_size: Number of cells per batch (default: 32)
        seed: Random seed for reproducible shuffling (default: 42)
        max_queue_size: Maximum size of the prefetch queue (default: 10)
        pin_memory: Whether to pin memory for faster GPU transfer (default: False)
        sampler_strategy: Sampling strategy for cells (default: "sequential")
        tokenizer_type: Type of tokenizer ("geneformer" or "scgpt", default: "geneformer")
        use_binned_expressions: Whether to use binned expressions for scGPT (default: False)
        n_epochs: Number of epochs to run. The generator will automatically reset
                 after each epoch, enabling multi-epoch training on small datasets.
                 Default: 1.

    Examples:
        >>> # Single epoch training
        >>> dataset = SLAFIterableDataset(slaf_array, tokenizer, n_epochs=1)
        >>> for batch in dataset:
        ...     # Training loop
        ...     pass

        >>> # Multi-epoch training for small datasets
        >>> dataset = SLAFIterableDataset(slaf_array, tokenizer, n_epochs=10)
        >>> for batch in dataset:
        ...     # Training loop - will automatically go through 10 epochs
        ...     pass
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
        n_epochs: int = 1,  # Add n_epochs parameter
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
        self.n_epochs = n_epochs

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
            batches_per_chunk=100,
            n_expression_bins=n_expression_bins,
            use_binned_expressions=use_binned_expressions,
            n_epochs=n_epochs,  # Pass n_epochs to batch processor
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
                print_completion(
                    f"Prefetcher ready after {time.time() - start_time:.2f}s"
                )
                return
            time.sleep(0.1)

        print_warning(f"Prefetcher not ready after {timeout}s, proceeding anyway...")

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate through batches of pre-tokenized data.

        This method yields training-ready batches containing pre-tokenized sequences,
        attention masks, and cell IDs. The data is processed asynchronously in the
        background for optimal performance.

        Yields:
            dict: Batch containing:
                - input_ids: Pre-tokenized sequences (torch.Tensor)
                - attention_mask: Boolean mask for valid tokens (torch.Tensor)
                - cell_ids: Cell integer IDs (torch.Tensor)
                - epoch: Current epoch number (int, only if n_epochs > 1)

        Note:
            All tensors are returned on CPU for device-agnostic training.
            The training loop should handle device transfer as needed.
        """
        start_time = time.time()
        batches_yielded = 0
        last_rate_time = start_time
        last_rate_batches = 0
        current_epoch = 0
        last_epoch = -1

        while True:
            # Get data from prefetcher
            data = self.prefetcher.get_batch()
            if data is None:
                # Check if prefetcher has finished all epochs
                stats = self.prefetcher.get_stats()
                if stats["current_epoch"] >= stats["n_epochs"]:
                    print_completion(
                        f"Dataset iteration complete: all {stats['n_epochs']} epochs finished"
                    )
                    break

                # Wait for more data with timeout
                wait_start = time.time()
                while not self.prefetcher.has_batch():
                    time.sleep(0.1)
                    # Timeout after 5 seconds to avoid infinite wait
                    if time.time() - wait_start > 5.0:
                        print_warning("Timeout waiting for prefetcher data")
                        break

                data = self.prefetcher.get_batch()
                if data is None:
                    # Double-check if prefetcher is done
                    stats = self.prefetcher.get_stats()
                    if stats["current_epoch"] >= stats["n_epochs"]:
                        print_completion(
                            f"Dataset iteration complete: all {stats['n_epochs']} epochs finished"
                        )
                        break
                    else:
                        print_warning("No data available from prefetcher")
                        break

            # Track epoch transitions
            current_epoch = self.batch_processor.current_epoch
            if current_epoch != last_epoch:
                print_epoch_transition(
                    f"Epoch transition detected: {last_epoch} -> {current_epoch}"
                )
                last_epoch = current_epoch

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
                    # Consolidate training batch reporting
                    training_report = f"Training batch {batches_yielded} (epoch {current_epoch}) processing:\n"
                    training_report += (
                        f"   Tensor creation: {tensor_time * 1000:.1f}ms\n"
                    )
                    training_report += (
                        "   Pre-tokenized data (no tokenization overhead)"
                    )

                    print_training(training_report)

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
                        rate_report = f"Training batch {batches_yielded} (epoch {current_epoch}): {instantaneous_rate:.1f} batches/sec (instantaneous, overall: {overall_rate:.1f})"
                        print_training(rate_report)
                    last_rate_time = current_time
                    last_rate_batches = batches_yielded

                # Prepare batch dict
                batch_dict = {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    "cell_ids": cell_ids_tensor,
                }

                # Add epoch information if using multiple epochs
                if self.n_epochs > 1:
                    batch_dict["epoch"] = current_epoch

                yield batch_dict

    def __del__(self):
        """Cleanup when dataset is destroyed"""
        self.prefetcher.stop()

    # Tokenization is now handled at the PrefetchBatchProcessor level
    # These methods are no longer needed since we consume pre-tokenized data
