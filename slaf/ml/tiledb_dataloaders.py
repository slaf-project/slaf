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
    """Print prefetch-related messages in cyan with a panel."""
    if not verbose:
        return

    if RICH_AVAILABLE and console is not None:
        console.print(Panel(message, border_style="cyan"))
    else:
        logger.info(f"🔍 {message}")


def print_training(message: str, verbose: bool = True):
    """Print training-related messages in green with a panel."""
    if not verbose:
        return

    if RICH_AVAILABLE and console is not None:
        console.print(Panel(message, border_style="green"))
    else:
        logger.info(f"📊 {message}")


@dataclass
class TileDBPrefetchBatch:
    """Container for a batch of TileDB data."""

    batch_id: int
    batch_df: (
        pl.DataFrame
    )  # Polars DataFrame with cell_integer_id, gene_integer_id, value
    cell_integer_ids: list[int]  # List of cell IDs in this batch
    process_time: float
    memory_mb: float


class TileDBBatchProcessor:
    """
    Processes TileDB SOMA data into batches using streaming and shuffling.
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
        """Initialize the TileDB batch processor."""
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
        """Reset the processor for a new epoch."""
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
        Load and process a chunk of TileDB data into batches using MoS strategy.
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
    Asynchronous prefetcher for TileDB batch processing.
    """

    def __init__(
        self, batch_processor: TileDBBatchProcessor, max_queue_size: int = 500
    ):
        """Initialize the TileDB async prefetcher."""
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
        """Start the prefetching worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.should_stop = False
            self.start_time = time.time()
            self.worker_thread = threading.Thread(
                target=self._prefetch_worker, daemon=True
            )
            self.worker_thread.start()

    def stop(self):
        """Stop the prefetching worker thread."""
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
        """Get the next pre-processed batch from the queue."""
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def has_batch(self) -> bool:
        """Check if a pre-processed batch is available in the queue."""
        return not self.queue.empty()

    def get_stats(self) -> dict:
        """Get comprehensive statistics about the prefetcher's performance."""
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
        """Iterate through batches of TileDB data."""
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
        """Cleanup when dataset is destroyed."""
        self.prefetcher.stop()


class TileDBDataLoader:
    """
    High-performance DataLoader for TileDB SOMA data optimized for ML training.
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
            tiledb_path: Path to the TileDB SOMA experiment
            batch_size: Number of cells per batch
            prefetch_batch_size: Number of cells to prefetch from TileDB (default: 100 for 50k cell datasets)
            seed: Random seed for reproducible shuffling
            n_epochs: Number of epochs to run
            verbose: Whether to print verbose output
            max_queue_size: Maximum size of the prefetch queue
            use_mixture_of_scanners: Whether to use MoS strategy for higher entropy (default: True)
            n_readers: Total number of generators to create (default: 50)
            n_scanners: Number of active scanners per batch (default: 8)
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
        """Iterate through batches of TileDB data."""
        yield from self._dataset

    def __len__(self):
        """Return the number of batches in the dataset."""
        return 0  # Indicates unknown length for streaming datasets

    def __del__(self):
        """Cleanup method to stop async prefetching."""
        if hasattr(self, "_dataset"):
            # The TileDBIterableDataset doesn't have a stop method,
            # so we just let it finish its current epoch.
            pass
