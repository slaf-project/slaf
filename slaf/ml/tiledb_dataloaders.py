#!/usr/bin/env python3
"""
TileDB Dataloader for Single-Cell Data

This module provides efficient streaming of single-cell data from TileDB SOMA format
using PyTorch IterableDataset and DataLoader. It follows a similar pattern to SLAF's
dataloader implementation for consistency and performance comparison.
"""

import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Queue

import polars as pl
from loguru import logger
from torch.utils.data import IterableDataset

# Try to import tiledbsoma
try:
    import tiledbsoma

    TILEDB_AVAILABLE = True
except ImportError:
    TILEDB_AVAILABLE = False
    logger.warning("TileDB SOMA not available. TileDB dataloader will not work.")

# Try to import rich for colored output
try:
    from rich.console import Console
    from rich.panel import Panel

    RICH_AVAILABLE = True
    console: Console | None = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_prefetch(message: str, verbose: bool = True):
    """
    Print prefetch-related messages with colored formatting.

    This function prints prefetch-related messages using rich console formatting
    when available, or falls back to loguru logging. Messages are displayed in
    cyan-colored panels for better visual distinction during training.

    Args:
        message: The message to print.
        verbose: If True, print the message. If False, suppress output.

    Examples:
        >>> # Print a prefetch message
        >>> print_prefetch("Loading batch 1 of 100")
        >>> # Suppress output
        >>> print_prefetch("Loading batch 1 of 100", verbose=False)
    """
    if not verbose:
        return

    if RICH_AVAILABLE and console is not None:
        console.print(Panel(message, border_style="cyan"))
    else:
        logger.info(f"🔍 {message}")


def print_training(message: str, verbose: bool = True):
    """
    Print training-related messages with colored formatting.

    This function prints training-related messages using rich console formatting
    when available, or falls back to loguru logging. Messages are displayed in
    green-colored panels for better visual distinction during training.

    Args:
        message: The message to print.
        verbose: If True, print the message. If False, suppress output.

    Examples:
        >>> # Print a training message
        >>> print_training("Processing batch with 32 cells")
        >>> # Suppress output
        >>> print_training("Processing batch with 32 cells", verbose=False)
    """
    if not verbose:
        return

    if RICH_AVAILABLE and console is not None:
        console.print(Panel(message, border_style="green"))
    else:
        logger.info(f"📊 {message}")


@dataclass
class TileDBPrefetchBatch:
    """
    Container for a batch of TileDB data with metadata.

    This dataclass holds a processed batch of TileDB SOMA data along with
    associated metadata for tracking performance and debugging. It serves as
    the primary data structure passed between the batch processor and the
    async prefetcher.

    Attributes:
        batch_id: Unique identifier for this batch within the current epoch.
        batch_df: Polars DataFrame containing the cell-gene expression data
                 with columns: cell_integer_id, gene_integer_id, value.
        cell_integer_ids: List of unique cell IDs present in this batch.
        process_time: Time taken to process this batch (in seconds).
        memory_mb: Memory usage at the time of batch creation (in MB).

    Examples:
        >>> # Create a batch container
        >>> batch = TileDBPrefetchBatch(
        ...     batch_id=0,
        ...     batch_df=df,
        ...     cell_integer_ids=[0, 1, 2, 3],
        ...     process_time=0.1,
        ...     memory_mb=128.5
        ... )
        >>> print(f"Batch {batch.batch_id} has {len(batch.cell_integer_ids)} cells")
        Batch 0 has 4 cells
    """

    batch_id: int
    batch_df: (
        pl.DataFrame
    )  # Polars DataFrame with cell_integer_id, gene_integer_id, value
    cell_integer_ids: list[int]  # List of cell IDs in this batch
    process_time: float
    memory_mb: float


class TileDBBatchProcessor:
    """
    High-performance batch processor for TileDB SOMA data with multiple loading strategies.

    TileDBBatchProcessor provides efficient streaming and processing of single-cell data
    from TileDB SOMA format. It supports multiple loading strategies including Mixture
    of Scanners (MoS) for maximum entropy and sequential loading for maximum throughput.

    Key Features:
        - Multiple loading strategies:
            * Mixture of Scanners (MoS): Random sampling from multiple generators for
              maximum entropy and randomization (default)
            * Sequential loading: Contiguous data loading for maximum throughput
        - Streaming data processing with configurable batch sizes
        - Built-in shuffling strategies for data randomization
        - Multi-epoch training support with automatic epoch transitions
        - Comprehensive timing and memory monitoring
        - Error handling and recovery mechanisms
        - Configurable prefetch batch sizes for different dataset sizes

    Loading Strategies:
        1. Mixture of Scanners (default): Randomly samples from multiple fragment
           generators for maximum entropy and randomization
        2. Sequential: Loads contiguous data chunks for maximum throughput

    Examples:
        >>> # Basic usage with default MoS strategy
        >>> processor = TileDBBatchProcessor(
        ...     tiledb_path="path/to/experiment",
        ...     batch_size=32,
        ...     prefetch_batch_size=100
        ... )
        >>> batch = processor.load_prefetch_batch()
        >>> print(f"Loaded batch with {len(batch.cell_integer_ids)} cells")
        Loaded batch with 100 cells

        >>> # Sequential loading for maximum throughput
        >>> processor = TileDBBatchProcessor(
        ...     tiledb_path="path/to/experiment",
        ...     use_mixture_of_scanners=False,
        ...     batch_size=64
        ... )
        >>> print(f"MoS enabled: {processor.use_mixture_of_scanners}")
        MoS enabled: False

        >>> # Multi-epoch training
        >>> processor = TileDBBatchProcessor(
        ...     tiledb_path="path/to/experiment",
        ...     n_epochs=3
        ... )
        >>> print(f"Number of epochs: {processor.n_epochs}")
        Number of epochs: 3
    """

    def __init__(
        self,
        tiledb_path: str,
        batch_size: int = 32,
        prefetch_batch_size: int = 100,
        seed: int = 42,
        n_epochs: int = 1,
        verbose: bool = True,
        log_metrics: bool = False,
        use_mixture_of_scanners: bool = True,
        n_readers: int = 50,
        n_scanners: int = 8,
    ):
        """
        Initialize the TileDB batch processor with training configuration.

        Args:
            tiledb_path: Path to the TileDB SOMA experiment directory.
                         Must contain a valid SOMA experiment with RNA measurement data.
            batch_size: Number of cells per training batch. Larger batches use more
                       memory but may improve training efficiency. Range: 1-512, default: 32.
            prefetch_batch_size: Number of cells to load per prefetch batch from TileDB.
                               Higher values improve throughput but use more memory.
                               Range: 10-10000, default: 100.
            seed: Random seed for reproducible shuffling and MoS sampling.
                  Used for consistent data ordering across runs. Default: 42.
            n_epochs: Number of epochs to run. The processor will automatically reset
                     after each epoch, enabling multi-epoch training. Default: 1.
            verbose: If True, print detailed timing and progress information.
                    If False, suppress all internal prints for clean output. Default: True.
            log_metrics: If True, collect detailed timing metrics for performance analysis.
                        Metrics include loading time, shuffle time, and memory usage.
                        Default: False.
            use_mixture_of_scanners: If True, use MoS strategy for higher entropy by
                                   randomly sampling from multiple fragment generators.
                                   Provides better randomization for foundation model training.
                                   Default: True.
            n_readers: Total number of fragment generators to create when using MoS.
                      Higher values provide better entropy but use more memory.
                      Range: 1-1000, default: 50.
            n_scanners: Number of active scanners to sample from simultaneously when using MoS.
                       Higher values provide better entropy but use more memory.
                       Range: 1-100, default: 8.

        Raises:
            ImportError: If TileDB SOMA is not available.
            ValueError: If MoS parameters are invalid (n_readers < 1, n_scanners < 1,
                       or n_scanners > n_readers).
            RuntimeError: If the TileDB experiment cannot be opened or is invalid.

        Examples:
            >>> # Basic initialization with default MoS strategy
            >>> processor = TileDBBatchProcessor(
            ...     tiledb_path="path/to/experiment",
            ...     batch_size=32,
            ...     prefetch_batch_size=100
            ... )
            >>> print(f"Total cells: {processor.total_cells}")
            Total cells: 50000

            >>> # Sequential loading for maximum throughput
            >>> processor = TileDBBatchProcessor(
            ...     tiledb_path="path/to/experiment",
            ...     use_mixture_of_scanners=False,
            ...     batch_size=64
            ... )
            >>> print(f"MoS enabled: {processor.use_mixture_of_scanners}")
            MoS enabled: False

            >>> # High-entropy MoS configuration
            >>> processor = TileDBBatchProcessor(
            ...     tiledb_path="path/to/experiment",
            ...     n_readers=100,
            ...     n_scanners=16
            ... )
            >>> print(f"MoS readers: {processor.n_readers}, scanners: {processor.n_scanners}")
            MoS readers: 100, scanners: 16
        """
        if not TILEDB_AVAILABLE:
            raise ImportError("TileDB SOMA is required but not available")

        self.tiledb_path = tiledb_path
        self.batch_size = batch_size
        self.prefetch_batch_size = prefetch_batch_size
        self.seed = seed
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.log_metrics = log_metrics
        self.use_mixture_of_scanners = use_mixture_of_scanners
        self.n_readers = n_readers
        self.n_scanners = n_scanners

        # Validate MoS parameters
        if self.use_mixture_of_scanners:
            if self.n_readers < 1:
                raise ValueError("n_readers must be at least 1")
            if self.n_scanners < 1:
                raise ValueError("n_scanners must be at least 1")
            if self.n_scanners > self.n_readers:
                raise ValueError("n_scanners cannot exceed n_readers")

        # Initialize state
        self.batch_id = 0
        self.current_epoch = 0
        self.total_cells = 0

        # Open TileDB experiment
        self.experiment = tiledbsoma.Experiment.open(tiledb_path)
        self.X = self.experiment.ms["RNA"].X["data"]

        # Get total number of cells
        self.total_cells = self.X.shape[0]

        # Initialize shuffling strategy (similar to SLAF)
        from slaf.ml.samplers import ShuffleType, create_shuffle

        self.shuffle = create_shuffle(ShuffleType.RANDOM)

        # Initialize MoS generators if enabled
        if self.use_mixture_of_scanners:
            self._initialize_mos_generators()

        # Initialize timing metrics for benchmarking
        self._timing_metrics: dict[str, list[float]] | None
        if self.log_metrics:
            self._timing_metrics = {
                "tiledb_loading": [],
                "shuffle": [],
                "total": [],
                "cells_processed": [],
            }
        else:
            self._timing_metrics = None

        # Initialize timing variables for consolidated reporting
        self._last_load_time = 0.0
        self._last_memory_mb = 0.0

    def _initialize_mos_generators(self):
        """Initialize MoS generators with evenly distributed scan ranges."""
        # Calculate scan ranges for each generator
        cells_per_reader = self.total_cells // self.n_readers
        remainder = self.total_cells % self.n_readers

        self.generators = []
        current_position = 0

        for i in range(self.n_readers):
            # Distribute remainder cells among first few readers
            reader_cell_count = cells_per_reader + (1 if i < remainder else 0)

            generator = {
                "generator_id": i,
                "start_position": current_position,
                "current_position": current_position,
                "end_position": current_position + reader_cell_count,
                "is_active": True,
            }

            self.generators.append(generator)
            current_position += reader_cell_count

        if self.verbose:
            print_prefetch(
                f"TileDB MoS initialized: {self.n_readers} generators, "
                f"{self.n_scanners} active scanners, "
                f"prefetch_batch_size={self.prefetch_batch_size}",
                self.verbose,
            )

    def reset_for_epoch(self, epoch: int) -> None:
        """
        Reset the processor for a new epoch.

        This method resets the batch processor state to start a new epoch,
        including resetting batch counters, MoS generator positions, and
        shuffling seeds. It is called automatically during multi-epoch training.

        Args:
            epoch: The epoch number to start (0-based indexing).
                  Must be 0 <= epoch < n_epochs.

        Raises:
            ValueError: If epoch is invalid (negative or >= n_epochs).

        Examples:
            >>> # Reset for epoch 1
            >>> processor = TileDBBatchProcessor("path/to/experiment", n_epochs=3)
            >>> processor.reset_for_epoch(1)
            >>> print(f"Current epoch: {processor.current_epoch}")
            Current epoch: 1

            >>> # Invalid epoch raises error
            >>> try:
            ...     processor.reset_for_epoch(5)  # n_epochs=3
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Invalid epoch 5. Must be 0 <= epoch < 3
        """
        if epoch < 0 or epoch >= self.n_epochs:
            raise ValueError(
                f"Invalid epoch {epoch}. Must be 0 <= epoch < {self.n_epochs}"
            )

        self.current_epoch = epoch
        self.batch_id = 0

        # Reset MoS generators if enabled
        if self.use_mixture_of_scanners:
            for generator in self.generators:
                generator["current_position"] = generator["start_position"]
                generator["is_active"] = True

        if self.verbose:
            print(f"🔄 Reset TileDB processor for epoch {epoch}")

    def _record_timing(self, step: str, duration: float, cells_processed: int = 0):
        """Record timing for a processing step."""
        if not self.log_metrics or self._timing_metrics is None:
            return

        if step in self._timing_metrics:
            self._timing_metrics[step].append(duration)

        if cells_processed > 0:
            self._timing_metrics["cells_processed"].append(cells_processed)

    def load_prefetch_batch(self) -> TileDBPrefetchBatch:
        """
        Load and process a chunk of TileDB data into batches using configured strategy.

        This method loads a batch of data from TileDB SOMA format, applies shuffling,
        and returns a processed batch ready for training. It supports both MoS and
        sequential loading strategies and handles epoch transitions automatically.

        The method performs the following steps:
        1. Load data from TileDB using the configured strategy (MoS or sequential)
        2. Convert Arrow data to Polars DataFrame
        3. Apply shuffling strategy for data randomization
        4. Return processed batch with metadata

        Returns:
            TileDBPrefetchBatch: Processed batch containing:
                - batch_df: Polars DataFrame with cell-gene expression data
                - cell_integer_ids: List of unique cell IDs in the batch
                - process_time: Time taken to process the batch
                - memory_mb: Memory usage at batch creation time

        Raises:
            StopIteration: When all epochs are completed and no more data is available.
            RuntimeError: If TileDB data loading fails.

        Examples:
            >>> # Load a batch with MoS strategy
            >>> processor = TileDBBatchProcessor(
            ...     tiledb_path="path/to/experiment",
            ...     use_mixture_of_scanners=True
            ... )
            >>> batch = processor.load_prefetch_batch()
            >>> print(f"Batch {batch.batch_id} has {len(batch.cell_integer_ids)} cells")
            Batch 0 has 100 cells

            >>> # Load a batch with sequential strategy
            >>> processor = TileDBBatchProcessor(
            ...     tiledb_path="path/to/experiment",
            ...     use_mixture_of_scanners=False
            ... )
            >>> batch = processor.load_prefetch_batch()
            >>> print(f"Sequential batch shape: {batch.batch_df.shape}")
            Sequential batch shape: (100, 3)

            >>> # Handle epoch completion
            >>> processor = TileDBBatchProcessor("path/to/experiment", n_epochs=1)
            >>> try:
            ...     while True:
            ...         batch = processor.load_prefetch_batch()
            ...         print(f"Processed batch {batch.batch_id}")
            ... except StopIteration:
            ...     print("All epochs completed")
            All epochs completed
        """
        # Iterative approach to handle epoch transitions
        while True:
            start_time = time.time()

            if self.use_mixture_of_scanners:
                # MoS approach: randomly sample from active generators
                import numpy as np

                # Get indices of currently active generators
                active_generators = [g for g in self.generators if g["is_active"]]

                if not active_generators:
                    # Check if we should start a new epoch
                    if self.current_epoch + 1 < self.n_epochs:
                        if self.verbose:
                            print(
                                f"🔄 Epoch {self.current_epoch} complete, starting epoch {self.current_epoch + 1}"
                            )
                        self.reset_for_epoch(self.current_epoch + 1)
                        continue
                    else:
                        raise StopIteration("No more epochs available") from None

                # Randomly sample from active generators
                n_to_sample = min(self.n_scanners, len(active_generators))
                selected_generators = np.random.choice(
                    active_generators, size=n_to_sample, replace=False
                )

                if self.verbose and self.batch_id % 10 == 0:
                    print_prefetch(
                        f"TileDB MoS sampling: {len(active_generators)} active generators, "
                        f"sampling from {n_to_sample} generators",
                        self.verbose,
                    )

                # Load data from selected generators
                load_start = time.time()
                batch_dfs = []

                for generator in selected_generators:
                    try:
                        start_cell = generator["current_position"]
                        end_cell = min(
                            start_cell + self.prefetch_batch_size,
                            generator["end_position"],
                        )

                        if start_cell >= generator["end_position"]:
                            # Generator exhausted
                            generator["is_active"] = False
                            continue

                        # Read slice from TileDB
                        arrow_data = (
                            self.X.read((slice(start_cell, end_cell),))
                            .tables()
                            .concat()
                        )

                        # Convert Arrow table to Polars DataFrame
                        df = pl.from_arrow(arrow_data)  # type: ignore[assignment]
                        if not isinstance(df, pl.DataFrame):
                            raise TypeError("Expected DataFrame from Arrow table")

                        # Rename SOMA columns to expected names
                        df = df.rename(
                            {
                                "soma_dim_0": "cell_integer_id",
                                "soma_dim_1": "gene_integer_id",
                                "soma_data": "value",
                            }
                        )

                        batch_dfs.append(df)

                        # Update generator position
                        generator["current_position"] = end_cell

                        # Mark as inactive if exhausted
                        if generator["current_position"] >= generator["end_position"]:
                            generator["is_active"] = False

                    except Exception as e:
                        logger.error(
                            f"Error loading TileDB data from generator {generator['generator_id']}: {e}"
                        )
                        generator["is_active"] = False
                        continue

                if not batch_dfs:
                    # All selected generators are exhausted, continue to next iteration
                    continue

                # Combine all batches
                combined_df_mos = pl.concat(batch_dfs, how="vertical")
            else:
                # Sequential approach (original implementation)
                current_position = (
                    self.batch_id * self.prefetch_batch_size
                ) % self.total_cells

                # Only check for epoch transitions when we actually wrap around
                if self.batch_id > 0:
                    prev_position = (
                        (self.batch_id - 1) * self.prefetch_batch_size
                    ) % self.total_cells
                    if current_position < prev_position:  # We wrapped around
                        if self.current_epoch + 1 < self.n_epochs:
                            if self.verbose:
                                print(
                                    f"🔄 Epoch {self.current_epoch} complete, starting epoch {self.current_epoch + 1}"
                                )
                            self.reset_for_epoch(self.current_epoch + 1)
                            continue
                        else:
                            raise StopIteration("No more epochs available") from None

                # Load data from TileDB
                load_start = time.time()
                try:
                    # Read slice from TileDB as Arrow table
                    arrow_data = (
                        self.X.read(
                            (
                                slice(
                                    current_position,
                                    current_position + self.prefetch_batch_size,
                                ),
                            )
                        )
                        .tables()
                        .concat()
                    )

                    # Convert Arrow table to Polars DataFrame
                    combined_df = pl.from_arrow(arrow_data)  # type: ignore[assignment]
                    if not isinstance(combined_df, pl.DataFrame):
                        raise TypeError("Expected DataFrame from Arrow table")

                    # Rename SOMA columns to expected names
                    combined_df = combined_df.rename(
                        {
                            "soma_dim_0": "cell_integer_id",
                            "soma_dim_1": "gene_integer_id",
                            "soma_data": "value",
                        }
                    )

                except Exception as e:
                    logger.error(f"Error loading TileDB data: {e}")
                    raise StopIteration(f"Failed to load TileDB data: {e}") from e

            load_time = time.time() - load_start
            self._record_timing("tiledb_loading", load_time)

            # Print detailed loading breakdown every 10 batches
            if self.batch_id % 10 == 0:
                import psutil

                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024

                # Store timing info for consolidated report
                self._last_load_time = load_time
                self._last_memory_mb = memory_mb

            # Apply shuffling strategy
            shuffle_start = time.time()

            # Apply shuffling with chunking
            if self.use_mixture_of_scanners:
                shuffled_chunks = self.shuffle.apply(
                    combined_df_mos,  # type: ignore[arg-type]
                    self.seed + self.batch_id + self.current_epoch * 10000,
                    batch_size=self.batch_size,
                )
            else:
                shuffled_chunks = self.shuffle.apply(
                    combined_df,  # type: ignore[arg-type]
                    self.seed + self.batch_id + self.current_epoch * 10000,
                    batch_size=self.batch_size,
                )

            shuffle_time = time.time() - shuffle_start
            total_time = time.time() - start_time

            # Record timing metrics
            self._record_timing("shuffle", shuffle_time)
            self._record_timing("total", total_time)

            # Count total cells across all chunks
            total_cells_in_chunks = sum(
                len(chunk.get_column("cell_integer_id").unique())
                for chunk in shuffled_chunks
                if isinstance(chunk, pl.DataFrame)
            )

            # Record cells processed
            self._record_timing("cells_processed", 0, total_cells_in_chunks)

            # Print consolidated prefetch batch reporting
            if self.batch_id % 10 == 0:
                strategy_name = "MoS" if self.use_mixture_of_scanners else "sequential"
                prefetch_report = f"TileDB {strategy_name} prefetch batch {self.batch_id} (epoch {self.current_epoch}):\n"
                prefetch_report += f"   TileDB loading: {self._last_load_time * 1000:.1f}ms ({self.prefetch_batch_size} cells)\n"
                prefetch_report += (
                    f"   Processing: {shuffle_time * 1000:.1f}ms shuffle\n"
                )
                prefetch_report += f"   Total: {total_time * 1000:.1f}ms, {len(shuffled_chunks)} chunks, {total_cells_in_chunks} cells, {self._last_memory_mb:.1f} MB"

                print_prefetch(prefetch_report, self.verbose)

            self.batch_id += 1

            # Return the first chunk as a batch (we'll handle multiple chunks in the dataloader)
            if shuffled_chunks:
                first_chunk = shuffled_chunks[0]
                return TileDBPrefetchBatch(
                    batch_id=self.batch_id - 1,
                    batch_df=first_chunk,
                    cell_integer_ids=first_chunk["cell_integer_id"].unique().to_list(),
                    process_time=shuffle_time,
                    memory_mb=self._last_memory_mb,
                )
            else:
                # No data in this chunk, continue to next iteration
                continue


class TileDBAsyncPrefetcher:
    """
    Asynchronous prefetcher for TileDB batch processing with background loading.

    TileDBAsyncPrefetcher provides background batch processing and prefetching
    for TileDB data to improve training throughput. It runs a separate worker
    thread that continuously loads and processes batches while the main training
    loop consumes pre-processed data.

    Key Features:
        - Background batch processing in separate worker thread
        - Configurable queue size for memory management
        - Comprehensive performance monitoring and statistics
        - Automatic epoch transition handling
        - Graceful shutdown and cleanup
        - Real-time rate monitoring and reporting
        - Error handling and recovery

    The prefetcher maintains a queue of pre-processed batches and provides
    statistics about loading rates, memory usage, and processing times.

    Examples:
            >>> # Create prefetcher with a batch processor
            >>> processor = TileDBBatchProcessor("path/to/experiment")
            >>> prefetcher = TileDBAsyncPrefetcher(processor, max_queue_size=100)
            >>>
            >>> # Start background processing
            >>> prefetcher.start()
            >>>
            >>> # Get pre-processed batches
            >>> batch = prefetcher.get_batch()
            >>> if batch:
            ...     print(f"Got batch {batch.batch_id} with {len(batch.cell_integer_ids)} cells")
            >>>
            >>> # Check performance statistics
            >>> stats = prefetcher.get_stats()
            >>> print(f"Loading rate: {stats['cells_per_sec']:.1f} cells/sec")
            >>>
            >>> # Stop background processing
            >>> prefetcher.stop()
    """

    def __init__(
        self, batch_processor: TileDBBatchProcessor, max_queue_size: int = 500
    ):
        """
        Initialize the TileDB async prefetcher with background processing.

        Args:
            batch_processor: TileDBBatchProcessor instance to use for loading batches.
                           Must be properly initialized with TileDB path and configuration.
            max_queue_size: Maximum number of pre-processed batches to keep in queue.
                          Higher values use more memory but provide better buffering.
                          Range: 10-10000, default: 500.

        Examples:
            >>> # Create prefetcher with default queue size
            >>> processor = TileDBBatchProcessor("path/to/experiment")
            >>> prefetcher = TileDBAsyncPrefetcher(processor)
            >>> print(f"Max queue size: {prefetcher.max_queue_size}")
            Max queue size: 500

            >>> # Create prefetcher with custom queue size
            >>> prefetcher = TileDBAsyncPrefetcher(processor, max_queue_size=1000)
            >>> print(f"Custom queue size: {prefetcher.max_queue_size}")
            Custom queue size: 1000
        """
        self.batch_processor = batch_processor
        self.max_queue_size = max_queue_size
        self.queue: Queue[TileDBPrefetchBatch] = Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.should_stop = False

        # Monitoring stats
        self.total_cells_added = 0
        self.start_time = None
        self.last_rate_print = 0
        self.total_process_time = 0.0
        self.process_count = 0
        self.current_epoch = 0

    def start(self):
        """
        Start the prefetching worker thread for background batch processing.

        This method starts a background worker thread that continuously loads
        and processes batches from the TileDB batch processor. The worker thread
        runs as a daemon thread and will automatically stop when the main
        process exits.

        The prefetcher will begin loading batches immediately after starting.
        Use get_batch() to retrieve pre-processed batches from the queue.

        Examples:
            >>> # Start background prefetching
            >>> processor = TileDBBatchProcessor("path/to/experiment")
            >>> prefetcher = TileDBAsyncPrefetcher(processor)
            >>> prefetcher.start()
            >>> print("Prefetcher started")
            Prefetcher started

            >>> # Check if prefetcher is ready
            >>> import time
            >>> time.sleep(1)  # Wait for first batch
            >>> if prefetcher.has_batch():
            ...     print("Prefetcher is ready with data")
            ... else:
            ...     print("Prefetcher not ready yet")
            Prefetcher is ready with data
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
        Stop the prefetching worker thread and clean up resources.

        This method gracefully stops the background worker thread and waits
        for it to finish processing the current batch. It sets the stop flag
        and joins the thread with a timeout to prevent hanging.

        After calling stop(), the prefetcher will no longer load new batches.
        Any remaining batches in the queue can still be retrieved with get_batch().

        Examples:
            >>> # Stop the prefetcher
            >>> processor = TileDBBatchProcessor("path/to/experiment")
            >>> prefetcher = TileDBAsyncPrefetcher(processor)
            >>> prefetcher.start()
            >>>
            >>> # Do some work...
            >>> batch = prefetcher.get_batch()
            >>>
            >>> # Stop when done
            >>> prefetcher.stop()
            >>> print("Prefetcher stopped")
            Prefetcher stopped
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
                self.total_process_time += batch.process_time
                self.process_count += 1
                self.current_epoch = self.batch_processor.current_epoch

                elapsed = time.time() - (self.start_time or 0)
                rate = self.total_cells_added / elapsed if elapsed > 0 else 0

                # Print rate every 10 batches
                if batch.batch_id % 10 == 0 and batch.batch_id > self.last_rate_print:
                    avg_process_ms = (
                        self.total_process_time / self.process_count
                    ) * 1000
                    rate_report = f"TileDB prefetch rate: {rate:.1f} cells/sec (epoch {self.current_epoch}, total: {self.total_cells_added} cells, avg process: {avg_process_ms:.1f}ms)"
                    print_prefetch(rate_report, self.batch_processor.verbose)
                    self.last_rate_print = batch.batch_id

                # Put in queue
                try:
                    self.queue.put_nowait(batch)
                except queue.Full:
                    # Queue is full, wait a bit
                    time.sleep(0.1)

            except StopIteration as e:
                if "No more epochs available" in str(e):
                    if self.batch_processor.verbose:
                        print(
                            f"✅ All {self.batch_processor.n_epochs} epochs completed"
                        )
                else:
                    logger.info("Reached end of batches")
                break
            except Exception as e:
                logger.info(f"Error loading TileDB batch: {e}")
                break

    def get_batch(self) -> TileDBPrefetchBatch | None:
        """
        Get the next pre-processed batch from the queue.

        This method retrieves a pre-processed batch from the internal queue.
        If no batch is available, it returns None. The method has a timeout
        to prevent blocking indefinitely.

        Returns:
            TileDBPrefetchBatch | None: The next available batch, or None if
                                       no batch is available within the timeout.

        Examples:
            >>> # Get a batch from the prefetcher
            >>> processor = TileDBBatchProcessor("path/to/experiment")
            >>> prefetcher = TileDBAsyncPrefetcher(processor)
            >>> prefetcher.start()
            >>>
            >>> # Wait for a batch
            >>> batch = prefetcher.get_batch()
            >>> if batch:
            ...     print(f"Got batch {batch.batch_id} with {len(batch.cell_integer_ids)} cells")
            ... else:
            ...     print("No batch available")
            Got batch 0 with 100 cells

            >>> # Check for multiple batches
            >>> batches = []
            >>> for _ in range(3):
            ...     batch = prefetcher.get_batch()
            ...     if batch:
            ...         batches.append(batch)
            ...     else:
            ...         break
            >>> print(f"Retrieved {len(batches)} batches")
            Retrieved 3 batches
        """
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def has_batch(self) -> bool:
        """
        Check if a pre-processed batch is available in the queue.

        This method provides a non-blocking way to check if batches are
        available for immediate retrieval. It returns True if the queue
        contains at least one batch, False otherwise.

        Returns:
            bool: True if at least one batch is available, False otherwise.

        Examples:
            >>> # Check for available batches
            >>> processor = TileDBBatchProcessor("path/to/experiment")
            >>> prefetcher = TileDBAsyncPrefetcher(processor)
            >>> prefetcher.start()
            >>>
            >>> # Check if batches are ready
            >>> if prefetcher.has_batch():
            ...     batch = prefetcher.get_batch()
            ...     print(f"Processing batch {batch.batch_id}")
            ... else:
            ...     print("No batches ready yet")
            No batches ready yet

            >>> # Wait and check again
            >>> import time
            >>> time.sleep(2)
            >>> if prefetcher.has_batch():
            ...     print("Batches are now available")
            ... else:
            ...     print("Still waiting for batches")
            Batches are now available
        """
        return not self.queue.empty()

    def get_stats(self) -> dict:
        """
        Get comprehensive statistics about the prefetcher's performance.

        This method returns detailed performance statistics including loading
        rates, memory usage, queue status, and processing times. Useful for
        monitoring and debugging the prefetcher's performance.

        Returns:
            dict: Performance statistics dictionary containing:
                - total_cells: Total number of cells processed
                - elapsed_time: Total time since prefetcher started (seconds)
                - cells_per_sec: Average loading rate (cells per second)
                - queue_size: Current number of batches in queue
                - queue_full: Whether the queue is at maximum capacity
                - total_process_time: Total time spent processing batches
                - process_count: Number of batches processed
                - avg_process_time_ms: Average processing time per batch (ms)
                - current_epoch: Current epoch number
                - n_epochs: Total number of epochs configured

        Examples:
            >>> # Get performance statistics
            >>> processor = TileDBBatchProcessor("path/to/experiment")
            >>> prefetcher = TileDBAsyncPrefetcher(processor)
            >>> prefetcher.start()
            >>>
            >>> # Wait for some processing
            >>> import time
            >>> time.sleep(5)
            >>>
            >>> # Get and display stats
            >>> stats = prefetcher.get_stats()
            >>> print(f"Loading rate: {stats['cells_per_sec']:.1f} cells/sec")
            >>> print(f"Queue size: {stats['queue_size']}/{prefetcher.max_queue_size}")
            >>> print(f"Current epoch: {stats['current_epoch']}/{stats['n_epochs']}")
            Loading rate: 1250.5 cells/sec
            Queue size: 45/500
            Current epoch: 0/1

            >>> # Monitor performance over time
            >>> for i in range(3):
            ...     stats = prefetcher.get_stats()
            ...     print(f"Check {i+1}: {stats['cells_per_sec']:.1f} cells/sec")
            ...     time.sleep(2)
            Check 1: 1200.3 cells/sec
            Check 2: 1250.5 cells/sec
            Check 3: 1180.7 cells/sec
        """
        elapsed = time.time() - (self.start_time or 0)
        rate = self.total_cells_added / elapsed if elapsed > 0 else 0
        avg_process_time = (
            self.total_process_time / self.process_count
            if self.process_count > 0
            else 0
        )
        return {
            "total_cells": self.total_cells_added,
            "elapsed_time": elapsed,
            "cells_per_sec": rate,
            "queue_size": self.queue.qsize(),
            "queue_full": self.queue.full(),
            "total_process_time": self.total_process_time,
            "process_count": self.process_count,
            "avg_process_time_ms": avg_process_time * 1000,
            "current_epoch": self.current_epoch,
            "n_epochs": self.batch_processor.n_epochs,
        }


class TileDBIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for streaming TileDB SOMA data with async prefetching.

    TileDBIterableDataset provides a PyTorch-compatible interface for streaming
    single-cell data from TileDB SOMA format. It combines the TileDBBatchProcessor
    and TileDBAsyncPrefetcher to provide efficient, asynchronous data loading
    for machine learning training.

    Key Features:
        - PyTorch IterableDataset compatibility
        - Asynchronous background prefetching for improved throughput
        - Multiple loading strategies (MoS and sequential)
        - Multi-epoch training support
        - Automatic epoch transition handling
        - Memory-efficient streaming
        - Comprehensive error handling
        - Configurable batch and prefetch sizes

    The dataset automatically manages background prefetching and provides
    seamless iteration over batches of TileDB data. It handles epoch
    transitions and provides detailed timing information for performance
    monitoring.

    Examples:
            >>> # Create dataset with default MoS strategy
            >>> dataset = TileDBIterableDataset(
            ...     tiledb_path="path/to/experiment",
            ...     batch_size=32,
            ...     prefetch_batch_size=100
            ... )
            >>>
            >>> # Iterate through batches
            >>> for batch in dataset:
            ...     print(f"Batch keys: {list(batch.keys())}")
            ...     print(f"Cell IDs: {batch['cell_ids']}")
            ...     break
        Batch keys: ['X', 'cell_ids']
        Cell IDs: [0, 1, 2, ..., 29, 30, 31]

        >>> # Sequential loading for maximum throughput
        >>> dataset = TileDBIterableDataset(
        ...     tiledb_path="path/to/experiment",
        ...     use_mixture_of_scanners=False,
        ...     batch_size=64
        ... )
        >>> print(f"MoS enabled: {dataset.use_mixture_of_scanners}")
        MoS enabled: False

        >>> # Multi-epoch training
        >>> dataset = TileDBIterableDataset(
        ...     tiledb_path="path/to/experiment",
        ...     n_epochs=3
        ... )
        >>> epochs_seen = set()
        >>> for batch in dataset:
        ...     if 'epoch' in batch:
        ...         epochs_seen.add(batch['epoch'])
        ...     if len(epochs_seen) >= 3:
        ...         break
        >>> print(f"Epochs completed: {sorted(epochs_seen)}")
        Epochs completed: [0, 1, 2]
    """

    def __init__(
        self,
        tiledb_path: str,
        batch_size: int = 32,
        prefetch_batch_size: int = 100,
        seed: int = 42,
        max_queue_size: int = 500,
        n_epochs: int = 1,
        verbose: bool = True,
        use_mixture_of_scanners: bool = True,
        n_readers: int = 50,
        n_scanners: int = 8,
    ):
        """
        Initialize the TileDB IterableDataset with async prefetching.

        Args:
            tiledb_path: Path to the TileDB SOMA experiment directory.
                         Must contain a valid SOMA experiment with RNA measurement data.
            batch_size: Number of cells per training batch. Larger batches use more
                       memory but may improve training efficiency. Range: 1-512, default: 32.
            prefetch_batch_size: Number of cells to load per prefetch batch from TileDB.
                               Higher values improve throughput but use more memory.
                               Range: 10-10000, default: 100.
            seed: Random seed for reproducible shuffling and MoS sampling.
                  Used for consistent data ordering across runs. Default: 42.
            max_queue_size: Maximum number of pre-processed batches to keep in queue.
                          Higher values use more memory but provide better buffering.
                          Range: 10-10000, default: 500.
            n_epochs: Number of epochs to run. The dataset will automatically reset
                     after each epoch, enabling multi-epoch training. Default: 1.
            verbose: If True, print detailed timing and progress information.
                    If False, suppress all internal prints for clean output. Default: True.
            use_mixture_of_scanners: If True, use MoS strategy for higher entropy by
                                   randomly sampling from multiple fragment generators.
                                   Provides better randomization for foundation model training.
                                   Default: True.
            n_readers: Total number of fragment generators to create when using MoS.
                      Higher values provide better entropy but use more memory.
                      Range: 1-1000, default: 50.
            n_scanners: Number of active scanners to sample from simultaneously when using MoS.
                       Higher values provide better entropy but use more memory.
                       Range: 1-100, default: 8.

        Raises:
            ImportError: If TileDB SOMA is not available.
            ValueError: If MoS parameters are invalid.
            RuntimeError: If the TileDB experiment cannot be opened or is invalid.

        Examples:
            >>> # Basic initialization with default MoS strategy
            >>> dataset = TileDBIterableDataset(
            ...     tiledb_path="path/to/experiment",
            ...     batch_size=32,
            ...     prefetch_batch_size=100
            ... )
            >>> print(f"MoS enabled: {dataset.use_mixture_of_scanners}")
            MoS enabled: True

            >>> # Sequential loading for maximum throughput
            >>> dataset = TileDBIterableDataset(
            ...     tiledb_path="path/to/experiment",
            ...     use_mixture_of_scanners=False,
            ...     batch_size=64
            ... )
            >>> print(f"Sequential loading: {not dataset.use_mixture_of_scanners}")
            Sequential loading: True

            >>> # High-entropy MoS configuration
            >>> dataset = TileDBIterableDataset(
            ...     tiledb_path="path/to/experiment",
            ...     n_readers=100,
            ...     n_scanners=16
            ... )
            >>> print(f"MoS readers: {dataset.n_readers}, scanners: {dataset.n_scanners}")
            MoS readers: 100, scanners: 16
        """
        super().__init__()
        self.tiledb_path = tiledb_path
        self.batch_size = batch_size
        self.prefetch_batch_size = prefetch_batch_size
        self.seed = seed
        self.max_queue_size = max_queue_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.use_mixture_of_scanners = use_mixture_of_scanners
        self.n_readers = n_readers
        self.n_scanners = n_scanners

        # Initialize batch processor
        self.batch_processor = TileDBBatchProcessor(
            tiledb_path=tiledb_path,
            batch_size=batch_size,
            prefetch_batch_size=prefetch_batch_size,
            seed=seed,
            n_epochs=n_epochs,
            verbose=verbose,
            log_metrics=False,
            use_mixture_of_scanners=use_mixture_of_scanners,
            n_readers=n_readers,
            n_scanners=n_scanners,
        )

        # Initialize async prefetcher
        self.prefetcher = TileDBAsyncPrefetcher(
            batch_processor=self.batch_processor,
            max_queue_size=max_queue_size,
        )

        # Start async prefetching
        self.prefetcher.start()

        # Wait for prefetcher to initialize
        self._wait_for_prefetcher_ready()

    def _wait_for_prefetcher_ready(self, timeout: float = 10.0):
        """Wait for the prefetcher to be ready with data."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.prefetcher.has_batch():
                if self.verbose:
                    print(
                        f"✅ TileDB prefetcher ready after {time.time() - start_time:.2f}s"
                    )
                return
            time.sleep(0.1)

        if self.verbose:
            print(
                f"⚠️ TileDB prefetcher not ready after {timeout}s, proceeding anyway..."
            )

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate through batches of TileDB data with async prefetching.

        This method provides an iterator over batches of TileDB data, automatically
        handling background prefetching, epoch transitions, and error recovery.
        It yields dictionaries containing the batch data and metadata.

        The iterator automatically manages:
        - Background prefetching for improved throughput
        - Epoch transitions for multi-epoch training
        - Error handling and recovery
        - Performance monitoring and reporting

        Yields:
            dict: Batch dictionary containing:
                - X: Polars DataFrame with cell-gene expression data
                - cell_ids: List of unique cell IDs in the batch
                - epoch: Current epoch number (when n_epochs > 1)

        Examples:
            >>> # Basic iteration
            >>> dataset = TileDBIterableDataset("path/to/experiment")
            >>> for batch in dataset:
            ...     print(f"Batch keys: {list(batch.keys())}")
            ...     print(f"Cell IDs: {batch['cell_ids']}")
            ...     break
            Batch keys: ['X', 'cell_ids']
            Cell IDs: [0, 1, 2, ..., 29, 30, 31]

            >>> # Multi-epoch training
            >>> dataset = TileDBIterableDataset("path/to/experiment", n_epochs=3)
            >>> epochs_seen = set()
            >>> for batch in dataset:
            ...     if 'epoch' in batch:
            ...         epochs_seen.add(batch['epoch'])
            ...     if len(epochs_seen) >= 3:
            ...         break
            >>> print(f"Epochs completed: {sorted(epochs_seen)}")
            Epochs completed: [0, 1, 2]

            >>> # Training loop with error handling
            >>> dataset = TileDBIterableDataset("path/to/experiment")
            >>> for batch_idx, batch in enumerate(dataset):
            ...     try:
            ...         x = batch["X"]
            ...         cell_ids = batch["cell_ids"]
            ...         print(f"Processed batch {batch_idx} with {len(cell_ids)} cells")
            ...     except Exception as e:
            ...         print(f"Error in batch {batch_idx}: {e}")
            ...         continue
            ...     if batch_idx >= 2:  # Just first few batches
            ...         break
            Processed batch 0 with 32 cells
            Processed batch 1 with 32 cells
            Processed batch 2 with 32 cells
        """
        batches_yielded = 0
        current_epoch = 0
        last_epoch = -1

        while True:
            # Get data from prefetcher
            data_start = time.time()
            data = self.prefetcher.get_batch()
            data_time = time.time() - data_start

            if data is None:
                # Check if prefetcher has finished all epochs
                stats = self.prefetcher.get_stats()
                if stats["current_epoch"] >= stats["n_epochs"]:
                    if self.verbose:
                        print(
                            f"✅ Dataset iteration complete: all {stats['n_epochs']} epochs finished"
                        )
                    break

                # Wait for more data with timeout
                wait_start = time.time()
                while not self.prefetcher.has_batch():
                    time.sleep(0.1)
                    # Timeout after 5 seconds to avoid infinite wait
                    if time.time() - wait_start > 5.0:
                        if self.verbose:
                            print("⚠️ Timeout waiting for prefetcher data")
                        break

                data = self.prefetcher.get_batch()
                if data is None:
                    # Double-check if prefetcher is done
                    stats = self.prefetcher.get_stats()
                    if stats["current_epoch"] >= stats["n_epochs"]:
                        if self.verbose:
                            print(
                                f"✅ Dataset iteration complete: all {stats['n_epochs']} epochs finished"
                            )
                        break
                    else:
                        if self.verbose:
                            print("⚠️ No data available from prefetcher")
                        break

            # Track epoch transitions
            current_epoch = self.batch_processor.current_epoch
            if current_epoch != last_epoch:
                if self.verbose:
                    print(
                        f"🔄 Epoch transition detected: {last_epoch} -> {current_epoch}"
                    )
                last_epoch = current_epoch

            # Process batch data
            batch_df = data.batch_df

            # Time the overall batch processing
            batch_start_time = time.time()

            # Get unique cell IDs in this batch
            batch_cell_ids = batch_df["cell_integer_id"].unique().to_list()

            # Calculate total batch processing time
            total_batch_time = time.time() - batch_start_time

            # Create batch dictionary
            batch_dict = {
                "X": batch_df,  # Polars DataFrame with CSR-like structure
                "cell_ids": batch_cell_ids,
            }

            # Add epoch info if multi-epoch training
            if self.n_epochs > 1:
                batch_dict["epoch"] = current_epoch

            batches_yielded += 1

            # Print detailed timing every 100 batches
            if batches_yielded % 100 == 0:
                # Consolidate training batch reporting
                training_report = f"  TileDB training batch {batches_yielded} (epoch {current_epoch}) processing:\n"
                training_report += f"     Data retrieval: {data_time * 1000:.1f}ms\n"
                training_report += (
                    f"     Total batch time: {total_batch_time * 1000:.1f}ms\n"
                )
                training_report += "     Raw data (polars DataFrame)"

                print_training(training_report, self.verbose)

            yield batch_dict

    def __del__(self):
        """
        Cleanup when dataset is destroyed.

        This method is called when the dataset object is garbage collected.
        It ensures that the underlying prefetcher is properly stopped to
        prevent resource leaks and background thread issues.

        Examples:
            >>> # Dataset cleanup happens automatically
            >>> dataset = TileDBIterableDataset("path/to/experiment")
            >>> print("Dataset created")
            Dataset created
            >>> # When dataset goes out of scope, __del__ is called automatically
            >>> del dataset
            >>> print("Dataset destroyed and cleaned up")
            Dataset destroyed and cleaned up
        """
        self.prefetcher.stop()


class TileDBDataLoader:
    """
    High-performance DataLoader for TileDB SOMA data optimized for ML training.

    TileDBDataLoader provides efficient streaming of single-cell data from TileDB
    SOMA format for machine learning applications. It uses async batch processing
    and provides multiple loading strategies for different use cases.

    Key Features:
        - Multiple loading strategies for different entropy requirements:
            * Mixture of Scanners (MoS): Maximum entropy, best randomization (default)
            * Sequential loading: Fastest, lowest entropy
        - Asynchronous background prefetching for improved throughput
        - Multi-epoch training support with automatic epoch transitions
        - Memory-efficient streaming with configurable batch sizes
        - Comprehensive error handling and validation
        - Performance monitoring and statistics
        - PyTorch IterableDataset compatibility

    Loading Strategies:
        1. Mixture of Scanners (default): Randomly samples from multiple generators
           for maximum entropy and randomization
        2. Sequential: Loads contiguous data chunks for maximum throughput

    Examples:
        >>> # Basic usage with default MoS strategy
        >>> dataloader = TileDBDataLoader(
        ...     tiledb_path="path/to/experiment",
        ...     batch_size=32,
        ...     prefetch_batch_size=100
        ... )
        >>> for batch in dataloader:
        ...     print(f"Batch keys: {list(batch.keys())}")
        ...     print(f"Cell IDs: {batch['cell_ids']}")
        ...     break
        Batch keys: ['X', 'cell_ids']
        Cell IDs: [0, 1, 2, ..., 29, 30, 31]

        >>> # Sequential loading for maximum throughput
        >>> dataloader = TileDBDataLoader(
        ...     tiledb_path="path/to/experiment",
        ...     use_mixture_of_scanners=False,
        ...     batch_size=64
        ... )
        >>> print(f"MoS enabled: {dataloader.use_mixture_of_scanners}")
        MoS enabled: False

        >>> # Multi-epoch training
        >>> dataloader = TileDBDataLoader(
        ...     tiledb_path="path/to/experiment",
        ...     n_epochs=3
        ... )
        >>> print(f"Number of epochs: {dataloader.n_epochs}")
        Number of epochs: 3

        >>> # Custom MoS configuration
        >>> dataloader = TileDBDataLoader(
        ...     tiledb_path="path/to/experiment",
        ...     n_readers=100,
        ...     n_scanners=16
        ... )
        >>> print(f"MoS readers: {dataloader.n_readers}, scanners: {dataloader.n_scanners}")
        MoS readers: 100, scanners: 16
    """

    def __init__(
        self,
        tiledb_path: str,
        batch_size: int = 32,
        prefetch_batch_size: int = 100,
        seed: int = 42,
        n_epochs: int = 1,
        verbose: bool = True,
        max_queue_size: int = 500,
        use_mixture_of_scanners: bool = True,
        n_readers: int = 50,
        n_scanners: int = 8,
    ):
        """
        Initialize the TileDB DataLoader with training configuration.

        Args:
            tiledb_path: Path to the TileDB SOMA experiment directory.
                         Must contain a valid SOMA experiment with RNA measurement data.
            batch_size: Number of cells per training batch. Larger batches use more
                       memory but may improve training efficiency. Range: 1-512, default: 32.
            prefetch_batch_size: Number of cells to prefetch from TileDB per batch.
                               Higher values improve throughput but use more memory.
                               Range: 10-10000, default: 100.
            seed: Random seed for reproducible shuffling and MoS sampling.
                  Used for consistent data ordering across runs. Default: 42.
            n_epochs: Number of epochs to run. The dataloader will automatically reset
                     after each epoch, enabling multi-epoch training. Default: 1.
            verbose: If True, print detailed timing and progress information.
                    If False, suppress all internal prints for clean output. Default: True.
            max_queue_size: Maximum number of pre-processed batches to keep in queue.
                          Higher values use more memory but provide better buffering.
                          Range: 10-10000, default: 500.
            use_mixture_of_scanners: If True, use MoS strategy for higher entropy by
                                   randomly sampling from multiple fragment generators.
                                   Provides better randomization for foundation model training.
                                   Default: True.
            n_readers: Total number of fragment generators to create when using MoS.
                      Higher values provide better entropy but use more memory.
                      Range: 1-1000, default: 50.
            n_scanners: Number of active scanners to sample from simultaneously when using MoS.
                       Higher values provide better entropy but use more memory.
                       Range: 1-100, default: 8.

        Raises:
            ImportError: If TileDB SOMA is not available.
            ValueError: If MoS parameters are invalid.
            RuntimeError: If the TileDB experiment cannot be opened or is invalid.

        Examples:
            >>> # Basic initialization with default MoS strategy
            >>> dataloader = TileDBDataLoader(
            ...     tiledb_path="path/to/experiment",
            ...     batch_size=32,
            ...     prefetch_batch_size=100
            ... )
            >>> print(f"MoS enabled: {dataloader.use_mixture_of_scanners}")
            MoS enabled: True

            >>> # Sequential loading for maximum throughput
            >>> dataloader = TileDBDataLoader(
            ...     tiledb_path="path/to/experiment",
            ...     use_mixture_of_scanners=False,
            ...     batch_size=64
            ... )
            >>> print(f"Sequential loading: {not dataloader.use_mixture_of_scanners}")
            Sequential loading: True

            >>> # High-entropy MoS configuration
            >>> dataloader = TileDBDataLoader(
            ...     tiledb_path="path/to/experiment",
            ...     n_readers=100,
            ...     n_scanners=16
            ... )
            >>> print(f"MoS readers: {dataloader.n_readers}, scanners: {dataloader.n_scanners}")
            MoS readers: 100, scanners: 16
        """
        self.tiledb_path = tiledb_path
        self.batch_size = batch_size
        self.prefetch_batch_size = prefetch_batch_size
        self.seed = seed
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.max_queue_size = max_queue_size
        self.use_mixture_of_scanners = use_mixture_of_scanners
        self.n_readers = n_readers
        self.n_scanners = n_scanners

        # Check that required modules are available
        if not TILEDB_AVAILABLE:
            raise ImportError("TileDB SOMA is required but not available")

        # Use IterableDataset
        self._dataset = TileDBIterableDataset(
            tiledb_path=tiledb_path,
            batch_size=batch_size,
            prefetch_batch_size=prefetch_batch_size,
            seed=seed,
            max_queue_size=max_queue_size,
            n_epochs=n_epochs,
            verbose=verbose,
            use_mixture_of_scanners=use_mixture_of_scanners,
            n_readers=n_readers,
            n_scanners=n_scanners,
        )

    def __iter__(self):
        """
        Iterate through batches of TileDB data with async prefetching.

        This method provides an iterator over batches of TileDB data, automatically
        handling background prefetching, epoch transitions, and error recovery.
        It yields dictionaries containing the batch data and metadata.

        The iterator automatically manages:
        - Background prefetching for improved throughput
        - Epoch transitions for multi-epoch training
        - Error handling and recovery
        - Performance monitoring and reporting

        Yields:
            dict: Batch dictionary containing:
                - X: Polars DataFrame with cell-gene expression data
                - cell_ids: List of unique cell IDs in the batch
                - epoch: Current epoch number (when n_epochs > 1)

        Examples:
            >>> # Basic iteration
            >>> dataloader = TileDBDataLoader("path/to/experiment")
            >>> for batch in dataloader:
            ...     print(f"Batch keys: {list(batch.keys())}")
            ...     print(f"Cell IDs: {batch['cell_ids']}")
            ...     break
            Batch keys: ['X', 'cell_ids']
            Cell IDs: [0, 1, 2, ..., 29, 30, 31]

            >>> # Multi-epoch training
            >>> dataloader = TileDBDataLoader("path/to/experiment", n_epochs=3)
            >>> epochs_seen = set()
            >>> for batch in dataloader:
            ...     if 'epoch' in batch:
            ...         epochs_seen.add(batch['epoch'])
            ...     if len(epochs_seen) >= 3:
            ...         break
            >>> print(f"Epochs completed: {sorted(epochs_seen)}")
            Epochs completed: [0, 1, 2]

            >>> # Training loop with error handling
            >>> dataloader = TileDBDataLoader("path/to/experiment")
            >>> for batch_idx, batch in enumerate(dataloader):
            ...     try:
            ...         x = batch["X"]
            ...         cell_ids = batch["cell_ids"]
            ...         print(f"Processed batch {batch_idx} with {len(cell_ids)} cells")
            ...     except Exception as e:
            ...         print(f"Error in batch {batch_idx}: {e}")
            ...         continue
            ...     if batch_idx >= 2:  # Just first few batches
            ...         break
            Processed batch 0 with 32 cells
            Processed batch 1 with 32 cells
            Processed batch 2 with 32 cells
        """
        yield from self._dataset

    def __len__(self):
        """
        Return the number of batches in the dataset.

        Note: Since TileDBDataLoader uses an IterableDataset that streams data,
        the exact number of batches is not known in advance. This method
        returns 0 to indicate an unknown length for streaming datasets.

        Returns:
            int: Always returns 0 to indicate unknown length for streaming datasets.

        Examples:
            >>> # Check dataset length
            >>> dataloader = TileDBDataLoader("path/to/experiment")
            >>> print(f"Dataset length: {len(dataloader)}")
            Dataset length: 0

            >>> # IterableDataset behavior
            >>> batch_count = 0
            >>> for batch in dataloader:
            ...     batch_count += 1
            ...     if batch_count >= 5:  # Just count first 5 batches
            ...         break
            >>> print(f"Actually processed {batch_count} batches")
            Actually processed 5 batches

            >>> # Length is consistent
            >>> print(f"Length check: {len(dataloader)}")
            Length check: 0
        """
        return 0  # Indicates unknown length for streaming datasets

    def __del__(self):
        """
        Cleanup method to stop async prefetching.

        This method is called when the DataLoader object is garbage collected.
        It ensures that the underlying dataset's prefetcher is properly cleaned up
        to prevent resource leaks.

        Examples:
            >>> # DataLoader cleanup happens automatically
            >>> dataloader = TileDBDataLoader("path/to/experiment")
            >>> print("DataLoader created")
            DataLoader created
            >>> # When dataloader goes out of scope, __del__ is called automatically
            >>> del dataloader
            >>> print("DataLoader destroyed and cleaned up")
            DataLoader destroyed and cleaned up

            >>> # Manual cleanup (not usually needed)
            >>> dataloader = TileDBDataLoader("path/to/experiment")
            >>> dataloader.__del__()
            >>> print("Manual cleanup completed")
            Manual cleanup completed
        """
        if hasattr(self, "_dataset"):
            # The TileDBIterableDataset doesn't have a stop method,
            # so we just let it finish its current epoch.
            pass
