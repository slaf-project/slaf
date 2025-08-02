#!/usr/bin/env python3
"""
scDataset Multiprocessing Benchmark

This benchmark tests scDataset performance scaling with different num_workers values.
Uses fixed block_size=4 and fetch_factor=16 as optimal parameters from the paper.
"""

import gc
import time
from dataclasses import dataclass

import anndata as ad
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from slaf.core.slaf import SLAFArray


# Define the fetch_transform callback for AnnData at module level
def fetch_transform_adata(batch):
    """Callback function to transform scDataset batches to AnnData format."""
    return batch.to_adata()


@dataclass
class MultiprocessingBenchmarkResult:
    """Results from scDataset multiprocessing benchmark."""

    num_workers: int
    throughput_cells_per_sec: float
    memory_usage_gb: float
    measurement_time: float
    total_cells: int
    batches_processed: int


class scDatasetMultiprocessingBenchmark:
    """Benchmark scDataset multiprocessing performance."""

    def __init__(self, slaf_path: str, h5ad_path: str):
        self.slaf_path = slaf_path
        self.h5ad_path = h5ad_path
        self.console = Console()

        # Load SLAF array
        self.console.print(f"Loading SLAF array from {slaf_path}...")
        self.slaf_array = SLAFArray(slaf_path)

        # Load AnnData object
        self.console.print(f"Loading AnnData from {h5ad_path}...")
        self.adata = ad.read_h5ad(h5ad_path, backed="r")
        self.console.print(
            f"Loaded {self.adata.n_obs:,} cells, {self.adata.n_vars:,} genes"
        )

        # Benchmark parameters
        self.batch_size = 64
        self.measurement_duration = 10  # seconds

        # Fixed parameters from paper
        self.block_size = 4
        self.fetch_factor = 16

        # num_workers values to test
        self.num_workers_list = [0, 1, 2, 4, 8]

    def measure_memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024 / 1024

    def track_peak_memory(self, duration_seconds: float) -> float:
        """Track peak memory usage over a duration."""
        import threading

        peak_memory = 0.0
        stop_tracking = threading.Event()

        def memory_monitor():
            nonlocal peak_memory
            while not stop_tracking.is_set():
                current_memory = self.measure_memory_usage_gb()
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.1)  # Check every 100ms

        # Start memory monitoring in background thread
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Wait for the specified duration
        time.sleep(duration_seconds)

        # Stop monitoring
        stop_tracking.set()
        monitor_thread.join(timeout=1.0)

        return peak_memory

    def benchmark_with_memory_tracking(
        self, dataloader, measurement_duration: float, desc: str
    ):
        """Run benchmark with memory tracking in parallel and progress bar."""
        import threading

        peak_memory = 0.0
        stop_tracking = threading.Event()

        def memory_monitor():
            nonlocal peak_memory
            while not stop_tracking.is_set():
                current_memory = self.measure_memory_usage_gb()
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.1)  # Check every 100ms

        # Start memory monitoring in background thread
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Run the actual benchmark with progress bar
        start_time = time.time()
        total_cells = 0
        batch_count = 0

        with tqdm(desc=desc, unit="batch") as pbar:
            for _batch in dataloader:
                batch_start = time.time()
                batch_count += 1

                # Use actual batch size from the data
                if isinstance(_batch, dict):
                    if "X" in _batch:
                        actual_batch_size = _batch["X"].shape[0]
                    elif "cell_ids" in _batch:
                        actual_batch_size = len(_batch["cell_ids"])
                    else:
                        actual_batch_size = self.batch_size
                else:
                    actual_batch_size = self.batch_size
                total_cells += actual_batch_size

                # Force actual data loading
                if hasattr(_batch, "X"):
                    _ = _batch.X  # Force data loading
                elif isinstance(_batch, dict) and "X" in _batch:
                    _ = _batch["X"]  # Force data loading

                batch_end = time.time()
                _batch_time_ms = (batch_end - batch_start) * 1000

                # Calculate current throughput
                elapsed = time.time() - start_time
                current_throughput = total_cells / elapsed if elapsed > 0 else 0

                # Update progress bar
                pbar.set_postfix(
                    {
                        "throughput": f"{current_throughput:.0f} cells/s",
                        "memory": f"{peak_memory:.1f}GB",
                    }
                )
                pbar.update(1)

                # Stop after measurement duration
                if elapsed >= measurement_duration:
                    break

        elapsed_time = time.time() - start_time

        # Stop memory monitoring
        stop_tracking.set()
        monitor_thread.join(timeout=1.0)

        return total_cells, batch_count, elapsed_time, peak_memory

    def benchmark_scdataset_multiprocessing(
        self, num_workers: int
    ) -> MultiprocessingBenchmarkResult | None:
        """Benchmark scDataset with specific num_workers value."""

        try:
            from anndata.experimental import AnnCollection
            from scdataset import scDataset
            from torch.utils.data import DataLoader
        except ImportError:
            self.console.print(
                "[yellow]Warning: scDataset not available, skipping benchmark[/yellow]"
            )
            return None

        try:
            # Setup scDataset with optimal parameters from paper
            collection = AnnCollection([self.adata])

            sc_dataset = scDataset(
                data_collection=collection,
                batch_size=self.batch_size,
                block_size=self.block_size,
                fetch_factor=self.fetch_factor,
                fetch_transform=fetch_transform_adata,  # Use module-level callback
            )

            # Create DataLoader with proper multiprocessing configuration
            dataloader = DataLoader(
                sc_dataset,
                batch_size=None,
                num_workers=num_workers,
                prefetch_factor=(
                    17 if num_workers > 0 else None
                ),  # Only use prefetch_factor when num_workers > 0
            )

            # Warm up
            warmup_batches = 3
            for i, _batch in enumerate(dataloader):
                if i >= warmup_batches:
                    break

            # Benchmark with peak memory tracking
            desc = f"num_workers={num_workers}"

            total_cells, batch_count, elapsed_time, peak_memory = (
                self.benchmark_with_memory_tracking(
                    dataloader, self.measurement_duration, desc
                )
            )

            # Clean up the dataloader to stop background processing
            del dataloader
            gc.collect()

            # Calculate metrics
            throughput_cells_per_sec = total_cells / elapsed_time

            return MultiprocessingBenchmarkResult(
                num_workers=num_workers,
                throughput_cells_per_sec=throughput_cells_per_sec,
                memory_usage_gb=peak_memory,
                measurement_time=elapsed_time,
                total_cells=total_cells,
                batches_processed=batch_count,
            )

        except Exception as e:
            self.console.print(
                f"[red]Error benchmarking num_workers={num_workers}: {e}[/red]"
            )
            return None

    def run_benchmarks(self) -> list[MultiprocessingBenchmarkResult]:
        """Run benchmarks for all num_workers values."""

        self.console.print(
            Panel.fit(
                "[bold green]scDataset Multiprocessing Benchmark[/bold green]\n"
                f"Testing performance scaling with num_workers (block_size={self.block_size}, fetch_factor={self.fetch_factor})",
                border_style="green",
            )
        )

        results = []

        # Test all num_workers values
        for num_workers in self.num_workers_list:
            self.console.print(f"\n[bold]Testing num_workers={num_workers}[/bold]")

            result = self.benchmark_scdataset_multiprocessing(num_workers)
            if result:
                results.append(result)
                self.console.print(
                    f"  Result: {result.throughput_cells_per_sec:.0f} cells/sec"
                )

        return results

    def print_results(self, results: list[MultiprocessingBenchmarkResult]):
        """Print benchmark results in formatted tables."""

        if not results:
            self.console.print("[red]No results to display[/red]")
            return

        # Create results table
        self.console.print("\n[bold]scDataset Multiprocessing Results[/bold]")

        table = Table(title="scDataset Multiprocessing Performance")
        table.add_column("Num Workers", style="cyan")
        table.add_column("Throughput (cells/sec)", style="green")
        table.add_column("Memory (GB)", style="yellow")
        table.add_column("Batches", style="blue")
        table.add_column("Time (s)", style="yellow")

        for result in results:
            # Color code the throughput
            throughput_text = f"{result.throughput_cells_per_sec:.0f}"
            if result.throughput_cells_per_sec > 3000:
                throughput_text = f"[green]{throughput_text}[/green]"
            elif result.throughput_cells_per_sec > 2000:
                throughput_text = f"[yellow]{throughput_text}[/yellow]"

            table.add_row(
                str(result.num_workers),
                throughput_text,
                f"{result.memory_usage_gb:.1f}",
                str(result.batches_processed),
                f"{result.measurement_time:.1f}",
            )

        self.console.print(table)

        # Summary statistics
        if results:
            max_throughput = max(r.throughput_cells_per_sec for r in results)
            best_config = next(
                r for r in results if r.throughput_cells_per_sec == max_throughput
            )

            self.console.print("\n[bold]Summary:[/bold]")
            self.console.print(f"Best performance: {max_throughput:.0f} cells/sec")
            self.console.print(
                f"Best configuration: num_workers={best_config.num_workers}"
            )

            # Calculate scaling factor
            single_worker_result = next(
                (r for r in results if r.num_workers == 0), None
            )
            if single_worker_result and best_config.num_workers > 0:
                scaling_factor = (
                    best_config.throughput_cells_per_sec
                    / single_worker_result.throughput_cells_per_sec
                )
                self.console.print(
                    f"Multiprocessing scaling factor: {scaling_factor:.1f}x"
                )

                if scaling_factor < 1.5:
                    self.console.print(
                        "[yellow]Note: Limited multiprocessing scaling observed - this may differ from paper results[/yellow]"
                    )
                else:
                    self.console.print(
                        "[green]Significant multiprocessing scaling observed[/green]"
                    )


def main():
    """Run the scDataset multiprocessing benchmarks."""

    # Use a smaller dataset for development
    slaf_path = "../slaf-datasets/plate1_Tahoe100M.slaf"
    h5ad_path = "../slaf-datasets/plate1_Tahoe100M.h5ad"

    benchmark = scDatasetMultiprocessingBenchmark(slaf_path, h5ad_path)
    results = benchmark.run_benchmarks()
    benchmark.print_results(results)


if __name__ == "__main__":
    main()
