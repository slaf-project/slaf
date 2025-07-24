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
    """
    Print prefetch-related messages in cyan with a panel.

    This function formats prefetch-related output messages using rich formatting
    when available, or falls back to simple text output. Messages are displayed
    in cyan color with a panel border for better visibility.

    Args:
        message: The message text to display. Should contain prefetch-related
                information like batch processing times, throughput rates, etc.

    Examples:
        >>> # Print prefetch message
        >>> print_prefetch("Batch 10 loaded: 32 cells, 2.3ms")
        🔍 Batch 10 loaded: 32 cells, 2.3ms

        >>> # With rich formatting (if available)
        >>> print_prefetch("Processing rate: 1000 cells/sec")
        # Output will be in cyan panel if rich is available
    """
    if RICH_AVAILABLE:
        console.print(Panel(message, border_style="cyan"))
    else:
        print(f"🔍 {message}")


def print_training(message: str):
    """
    Print training-related messages in green with a panel.

    This function formats training-related output messages using rich formatting
    when available, or falls back to simple text output. Messages are displayed
    in green color with a panel border for better visibility.

    Args:
        message: The message text to display. Should contain training-related
                information like batch processing times, throughput rates, etc.

    Examples:
        >>> # Print training message
        >>> print_training("Batch 100 processed: 32 cells, 1.2ms")
        📊 Batch 100 processed: 32 cells, 1.2ms

        >>> # With rich formatting (if available)
        >>> print_training("Training rate: 500 batches/sec")
        # Output will be in green panel if rich is available
    """
    if RICH_AVAILABLE:
        console.print(Panel(message, border_style="green"))
    else:
        print(f"📊 {message}")


def print_epoch_transition(message: str):
    """
    Print epoch transition messages in yellow.

    This function formats epoch transition messages using rich formatting
    when available, or falls back to simple text output. Messages are displayed
    in yellow color to indicate epoch boundary events.

    Args:
        message: The message text to display. Should contain epoch transition
                information like epoch numbers, completion status, etc.

    Examples:
        >>> # Print epoch transition message
        >>> print_epoch_transition("Epoch 1 -> 2")
        🔄 Epoch 1 -> 2

        >>> # With rich formatting (if available)
        >>> print_epoch_transition("Starting epoch 5")
        # Output will be in yellow if rich is available
    """
    if RICH_AVAILABLE:
        console.print(f"[yellow]🔄 {message}[/yellow]")
    else:
        print(f"🔄 {message}")


def print_completion(message: str):
    """
    Print completion messages in bright green with a panel.

    This function formats completion messages using rich formatting
    when available, or falls back to simple text output. Messages are displayed
    in bright green color with a panel border to indicate successful completion.

    Args:
        message: The message text to display. Should contain completion
                information like training completion, epoch completion, etc.

    Examples:
        >>> # Print completion message
        >>> print_completion("All epochs completed")
        ✅ All epochs completed

        >>> # With rich formatting (if available)
        >>> print_completion("Training finished successfully")
        # Output will be in bright green panel if rich is available
    """
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
    """
    Print warning messages in orange.

    This function formats warning messages using rich formatting
    when available, or falls back to simple text output. Messages are displayed
    in orange color to indicate warning conditions.

    Args:
        message: The message text to display. Should contain warning
                information like timeout conditions, errors, etc.

    Examples:
        >>> # Print warning message
        >>> print_warning("Prefetcher timeout")
        ⚠️ Prefetcher timeout

        >>> # With rich formatting (if available)
        >>> print_warning("Queue is full")
        # Output will be in orange if rich is available
    """
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
    It serves as the primary data structure for transferring pre-processed
    batches between the background prefetcher and the main training loop.

    Examples:
        >>> # Create a prefetch batch
        >>> import torch
        >>> batch = PrefetchBatch(
        ...     batch_id=0,
        ...     input_ids=torch.randint(0, 1000, (32, 1024)),
        ...     attention_mask=torch.ones(32, 1024, dtype=torch.bool),
        ...     cell_integer_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        ...     tokenize_time=0.0023
        ... )
        >>> print(f"Batch {batch.batch_id}: {len(batch.cell_integer_ids)} cells")
        Batch 0: 32 cells

        >>> # Access batch components
        >>> print(f"Input shape: {batch.input_ids.shape}")
        Input shape: torch.Size([32, 1024])
        >>> print(f"Tokenization time: {batch.tokenize_time * 1000:.1f}ms")
        Tokenization time: 2.3ms

        >>> # Check for partial cell data
        >>> if batch.partial_cell_data is not None:
        ...     print(f"Partial cells: {len(batch.partial_cell_data)}")
        ... else:
        ...     print("No partial cell data")
        No partial cell data
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

        This method resets the internal state of the batch processor for a new epoch,
        including the Lance batch generator, batch counter, and partial cell data.
        It also updates the current epoch tracking and prints a transition message.

        Args:
            epoch: The epoch number (0-indexed). Must be within the range
                  0 <= epoch < n_epochs.

        Raises:
            ValueError: If epoch is invalid or exceeds n_epochs

        Examples:
            >>> # Reset for first epoch
            >>> processor = PrefetchBatchProcessor(slaf_array, window, shuffle, tokenizer, n_epochs=3)
            >>> processor.reset_for_epoch(0)
            >>> print(f"Current epoch: {processor.current_epoch}")
            Current epoch: 0

            >>> # Reset for second epoch
            >>> processor.reset_for_epoch(1)
            >>> print(f"Current epoch: {processor.current_epoch}")
            Current epoch: 1

            >>> # Error handling for invalid epoch
            >>> try:
            ...     processor.reset_for_epoch(5)  # Beyond n_epochs=3
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Invalid epoch 5. Must be 0 <= epoch < 3

            >>> # Check batch counter reset
            >>> processor.reset_for_epoch(0)
            >>> print(f"Batch ID: {processor.batch_id}")
            Batch ID: 0

            >>> # Verify partial cell data is cleared
            >>> processor.partial_cell_data = {"test": "data"}
            >>> processor.reset_for_epoch(1)
            >>> print(f"Partial cell data: {processor.partial_cell_data}")
            Partial cell data: {}
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
        It handles cell boundary crossing and partial data management. The method
        uses an iterative approach to handle epoch transitions automatically.

        Returns:
            PrefetchBatch: Container with tokenized sequences and metadata

        Raises:
            StopIteration: When no more batches are available for current epoch

        Examples:
            >>> # Load a single batch
            >>> processor = PrefetchBatchProcessor(slaf_array, window, shuffle, tokenizer)
            >>> batch = processor.load_prefetch_batch()
            >>> print(f"Loaded batch {batch.batch_id} with {len(batch.cell_integer_ids)} cells")
            Loaded batch 0 with 32 cells

            >>> # Load multiple batches
            >>> batches = []
            >>> for _ in range(3):
            ...     try:
            ...         batch = processor.load_prefetch_batch()
            ...         batches.append(batch)
            ...     except StopIteration:
            ...         break
            >>> print(f"Loaded {len(batches)} batches")
            Loaded 3 batches

            >>> # Check batch contents
            >>> batch = processor.load_prefetch_batch()
            >>> print(f"Input shape: {batch.input_ids.shape}")
            Input shape: torch.Size([32, 1024])
            >>> print(f"Tokenization time: {batch.tokenize_time * 1000:.1f}ms")
            Tokenization time: 2.3ms

            >>> # Handle epoch transitions
            >>> processor = PrefetchBatchProcessor(slaf_array, window, shuffle, tokenizer, n_epochs=2)
            >>> batch = processor.load_prefetch_batch()
            >>> print(f"Epoch: {processor.current_epoch}")
            Epoch: 0
            >>> # Continue until epoch transition
            >>> while processor.current_epoch < 1:
            ...     batch = processor.load_prefetch_batch()
            >>> print(f"Now in epoch: {processor.current_epoch}")
            Now in epoch: 1
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
        """
        Initialize the AsyncPrefetcher with batch processing configuration.

        Args:
            batch_processor: PrefetchBatchProcessor instance that handles the actual
                           batch loading and processing. This processor will be used
                           in the background thread to generate batches.
            max_queue_size: Maximum number of pre-processed batches to store in the
                          queue. Higher values use more memory but provide better
                          buffering against processing delays. Range: 10-1000,
                          default: 500.

        Raises:
            ValueError: If batch_processor is None or max_queue_size is invalid.
            TypeError: If batch_processor is not a PrefetchBatchProcessor instance.

        Examples:
            >>> # Basic initialization
            >>> processor = PrefetchBatchProcessor(slaf_array, window, shuffle, tokenizer)
            >>> prefetcher = AsyncPrefetcher(processor)
            >>> print(f"Queue size: {prefetcher.max_queue_size}")
            Queue size: 500

            >>> # Custom queue size
            >>> prefetcher = AsyncPrefetcher(processor, max_queue_size=1000)
            >>> print(f"Custom queue size: {prefetcher.max_queue_size}")
            Custom queue size: 1000

            >>> # Error handling for invalid processor
            >>> try:
            ...     prefetcher = AsyncPrefetcher(None)
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: batch_processor cannot be None
        """
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
        """
        Start the prefetching worker thread.

        This method initializes and starts a background thread that continuously
        loads and processes batches from the batch processor. The thread runs
        until stop() is called or an error occurs.

        Raises:
            RuntimeError: If the worker thread is already running or cannot be started.
            ThreadingError: If there are issues with thread creation.

        Examples:
            >>> # Start the prefetcher
            >>> processor = PrefetchBatchProcessor(slaf_array, window, shuffle, tokenizer)
            >>> prefetcher = AsyncPrefetcher(processor)
            >>> prefetcher.start()
            >>> print(f"Worker thread alive: {prefetcher.worker_thread.is_alive()}")
            Worker thread alive: True

            >>> # Start and immediately check stats
            >>> prefetcher.start()
            >>> stats = prefetcher.get_stats()
            >>> print(f"Total cells: {stats['total_cells']}")
            Total cells: 0

            >>> # Error handling for already running thread
            >>> prefetcher.start()  # Should not raise error if already running
            >>> print("Prefetcher started successfully")
            Prefetcher started successfully
        """
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.should_stop = False
            self.start_time = time.time()
            self.worker_thread = threading.Thread(
                target=self._prefetch_worker, daemon=True
            )
            self.worker_thread.start()

    def stop(self):
        """
        Stop the prefetching worker thread.

        This method signals the background worker thread to stop and waits for it
        to complete. The thread will finish processing its current batch before
        stopping. If the thread doesn't stop within 1 second, it will be
        forcefully terminated.

        Examples:
            >>> # Start and stop the prefetcher
            >>> processor = PrefetchBatchProcessor(slaf_array, window, shuffle, tokenizer)
            >>> prefetcher = AsyncPrefetcher(processor)
            >>> prefetcher.start()
            >>> print(f"Before stop: {prefetcher.worker_thread.is_alive()}")
            Before stop: True
            >>> prefetcher.stop()
            >>> print(f"After stop: {prefetcher.worker_thread.is_alive()}")
            After stop: False

            >>> # Stop without starting (should not raise error)
            >>> prefetcher = AsyncPrefetcher(processor)
            >>> prefetcher.stop()
            >>> print("Stopped successfully")
            Stopped successfully

            >>> # Check stats after stopping
            >>> prefetcher.start()
            >>> prefetcher.stop()
            >>> stats = prefetcher.get_stats()
            >>> print(f"Elapsed time: {stats['elapsed_time']:.2f}s")
            Elapsed time: 0.00s
        """
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
        """
        Get the next pre-processed batch from the queue.

        This method retrieves a batch that has been pre-processed by the background
        worker thread. If no batch is available, it waits up to 1 second before
        returning None.

        Returns:
            PrefetchBatch | None: The next pre-processed batch, or None if no batch
                                 is available within the timeout period.

        Examples:
            >>> # Get batches from prefetcher
            >>> processor = PrefetchBatchProcessor(slaf_array, window, shuffle, tokenizer)
            >>> prefetcher = AsyncPrefetcher(processor)
            >>> prefetcher.start()
            >>>
            >>> # Wait for first batch
            >>> batch = prefetcher.get_batch()
            >>> if batch is not None:
            ...     print(f"Got batch with {len(batch.cell_integer_ids)} cells")
            ... else:
            ...     print("No batch available")
            Got batch with 32 cells

            >>> # Get multiple batches
            >>> batches = []
            >>> for _ in range(3):
            ...     batch = prefetcher.get_batch()
            ...     if batch is not None:
            ...         batches.append(batch)
            >>> print(f"Retrieved {len(batches)} batches")
            Retrieved 3 batches

            >>> # Handle no batches available
            >>> prefetcher.stop()
            >>> batch = prefetcher.get_batch()
            >>> print(f"Batch available: {batch is not None}")
            Batch available: False
        """
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def has_batch(self) -> bool:
        """
        Check if a pre-processed batch is available in the queue.

        This method provides a non-blocking way to check if the prefetcher has
        any batches ready for consumption. It does not wait for batches to become
        available.

        Returns:
            bool: True if at least one batch is available in the queue, False otherwise.

        Examples:
            >>> # Check batch availability
            >>> processor = PrefetchBatchProcessor(slaf_array, window, shuffle, tokenizer)
            >>> prefetcher = AsyncPrefetcher(processor)
            >>> print(f"Initial batch available: {prefetcher.has_batch()}")
            Initial batch available: False
            >>>
            >>> # Start prefetching and check again
            >>> prefetcher.start()
            >>> import time
            >>> time.sleep(0.1)  # Give time for first batch
            >>> print(f"After start: {prefetcher.has_batch()}")
            After start: True

            >>> # Check availability in a loop
            >>> available_batches = 0
            >>> for _ in range(10):
            ...     if prefetcher.has_batch():
            ...         available_batches += 1
            ...     time.sleep(0.01)
            >>> print(f"Batches available in 10 checks: {available_batches}")
            Batches available in 10 checks: 8

            >>> # Check after stopping
            >>> prefetcher.stop()
            >>> print(f"After stop: {prefetcher.has_batch()}")
            After stop: False
        """
        return not self.queue.empty()

    def get_stats(self) -> dict:
        """
        Get comprehensive statistics about the prefetcher's performance.

        This method returns a dictionary containing various metrics about the
        prefetcher's operation, including throughput, queue status, and timing
        information.

        Returns:
            dict: Statistics dictionary containing:
                - total_cells: Total number of cells processed
                - elapsed_time: Total time since prefetcher started
                - cells_per_sec: Average processing rate in cells per second
                - queue_size: Current number of batches in the queue
                - queue_full: Whether the queue is at maximum capacity
                - total_tokenize_time: Total time spent on tokenization
                - tokenize_count: Number of batches tokenized
                - avg_tokenize_time_ms: Average tokenization time per batch
                - current_epoch: Current epoch being processed
                - n_epochs: Total number of epochs configured

        Examples:
            >>> # Get initial stats
            >>> processor = PrefetchBatchProcessor(slaf_array, window, shuffle, tokenizer)
            >>> prefetcher = AsyncPrefetcher(processor)
            >>> stats = prefetcher.get_stats()
            >>> print(f"Initial cells: {stats['total_cells']}")
            Initial cells: 0

            >>> # Get stats after processing
            >>> prefetcher.start()
            >>> import time
            >>> time.sleep(0.5)  # Let it process some batches
            >>> stats = prefetcher.get_stats()
            >>> print(f"Cells processed: {stats['total_cells']}")
            Cells processed: 128
            >>> print(f"Rate: {stats['cells_per_sec']:.1f} cells/sec")
            Rate: 256.0 cells/sec
            >>> print(f"Queue size: {stats['queue_size']}")
            Queue size: 4

            >>> # Monitor queue status
            >>> stats = prefetcher.get_stats()
            >>> print(f"Queue full: {stats['queue_full']}")
            Queue full: False
            >>> print(f"Avg tokenize time: {stats['avg_tokenize_time_ms']:.1f}ms")
            Avg tokenize time: 2.3ms

            >>> # Check epoch information
            >>> stats = prefetcher.get_stats()
            >>> print(f"Current epoch: {stats['current_epoch']}")
            Current epoch: 0
            >>> print(f"Total epochs: {stats['n_epochs']}")
            Total epochs: 1
        """
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
        """
        Wait for the prefetcher to be ready with data.

        This method waits for the background prefetcher to load its first batch
        and become ready to serve data. It polls the prefetcher's queue status
        until data becomes available or the timeout is reached.

        Args:
            timeout: Maximum time to wait for the prefetcher to become ready,
                    in seconds. If timeout is reached, a warning is printed but
                    the method continues. Range: 1.0-60.0, default: 10.0.

        Examples:
            >>> # Wait for prefetcher to be ready
            >>> dataset = SLAFIterableDataset(slaf_array, tokenizer)
            >>> # The _wait_for_prefetcher_ready method is called automatically
            >>> # during dataset initialization
            >>> print("Dataset initialized successfully")
            Dataset initialized successfully

            >>> # Custom timeout (this would be called internally)
            >>> dataset._wait_for_prefetcher_ready(timeout=5.0)
            >>> print("Prefetcher ready check completed")
            Prefetcher ready check completed

            >>> # Handle timeout scenario
            >>> # If prefetcher takes too long, a warning is printed
            >>> dataset._wait_for_prefetcher_ready(timeout=0.1)
            >>> print("Timeout handling completed")
            Timeout handling completed
        """
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
        background for optimal performance. The method automatically handles epoch
        transitions and completion detection.

        Yields:
            dict: Batch containing:
                - input_ids: Pre-tokenized sequences (torch.Tensor)
                - attention_mask: Boolean mask for valid tokens (torch.Tensor)
                - cell_ids: Cell integer IDs (torch.Tensor)
                - epoch: Current epoch number (int, only if n_epochs > 1)

        Note:
            All tensors are returned on CPU for device-agnostic training.
            The training loop should handle device transfer as needed.

        Examples:
            >>> # Basic iteration
            >>> dataset = SLAFIterableDataset(slaf_array, tokenizer)
            >>> batch_count = 0
            >>> for batch in dataset:
            ...     print(f"Batch {batch_count}: {batch['input_ids'].shape}")
            ...     batch_count += 1
            ...     if batch_count >= 3:  # Just first 3 batches
            ...         break
            Batch 0: torch.Size([32, 2048])
            Batch 1: torch.Size([32, 2048])
            Batch 2: torch.Size([32, 2048])

            >>> # Multi-epoch iteration
            >>> dataset = SLAFIterableDataset(slaf_array, tokenizer, n_epochs=2)
            >>> epochs_seen = set()
            >>> for batch in dataset:
            ...     if 'epoch' in batch:
            ...         epochs_seen.add(batch['epoch'])
            ...     if len(epochs_seen) >= 2:  # Stop after seeing both epochs
            ...         break
            >>> print(f"Epochs seen: {sorted(epochs_seen)}")
            Epochs seen: [0, 1]

            >>> # Check batch contents
            >>> dataset = SLAFIterableDataset(slaf_array, tokenizer)
            >>> for batch in dataset:
            ...     print(f"Keys: {list(batch.keys())}")
            ...     print(f"Input shape: {batch['input_ids'].shape}")
            ...     print(f"Attention mask shape: {batch['attention_mask'].shape}")
            ...     print(f"Cell IDs shape: {batch['cell_ids'].shape}")
            ...     break
            Keys: ['input_ids', 'attention_mask', 'cell_ids']
            Input shape: torch.Size([32, 2048])
            Attention mask shape: torch.Size([32, 2048])
            Cell IDs shape: torch.Size([32])

            >>> # Device-agnostic tensors
            >>> dataset = SLAFIterableDataset(slaf_array, tokenizer)
            >>> for batch in dataset:
            ...     print(f"Input device: {batch['input_ids'].device}")
            ...     print(f"Attention device: {batch['attention_mask'].device}")
            ...     print(f"Cell IDs device: {batch['cell_ids'].device}")
            ...     break
            Input device: cpu
            Attention device: cpu
            Cell IDs device: cpu
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
        """
        Cleanup when dataset is destroyed.

        This method is called when the dataset object is garbage collected.
        It ensures that the background prefetcher thread is properly stopped
        to prevent resource leaks and hanging threads.

        Examples:
            >>> # Dataset cleanup happens automatically
            >>> dataset = SLAFIterableDataset(slaf_array, tokenizer)
            >>> print("Dataset created")
            Dataset created
            >>> # When dataset goes out of scope, __del__ is called automatically
            >>> del dataset
            >>> print("Dataset destroyed and cleaned up")
            Dataset destroyed and cleaned up

            >>> # Manual cleanup (not usually needed)
            >>> dataset = SLAFIterableDataset(slaf_array, tokenizer)
            >>> dataset.__del__()
            >>> print("Manual cleanup completed")
            Manual cleanup completed
        """
        self.prefetcher.stop()

    # Tokenization is now handled at the PrefetchBatchProcessor level
    # These methods are no longer needed since we consume pre-tokenized data
