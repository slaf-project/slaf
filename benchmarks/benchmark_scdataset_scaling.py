#!/usr/bin/env python3
"""
scDataset Parameter Scaling Benchmark

This benchmark tests scDataset performance scaling with different block_size and fetch_factor parameters.
Based on scDataset paper Figure 2 which shows performance scaling with these parameters.
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
class ScalingBenchmarkResult:
    """Results from scDataset scaling benchmark."""

    block_size: int
    fetch_factor: int
    throughput_cells_per_sec: float
    memory_usage_gb: float
    measurement_time: float
    total_cells: int
    batches_processed: int


class scDatasetScalingBenchmark:
    """Benchmark scDataset parameter scaling performance."""

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

        # Parameter ranges to test
        self.block_sizes = [1, 2, 4, 8, 16, 32, 64]
        self.fetch_factors = [1, 2, 4, 8, 16, 32, 64]

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

    def benchmark_scdataset_parameters(
        self, block_size: int, fetch_factor: int
    ) -> ScalingBenchmarkResult | None:
        """Benchmark scDataset with specific block_size and fetch_factor parameters."""

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
            # Setup scDataset with current parameters
            collection = AnnCollection([self.adata])

            sc_dataset = scDataset(
                data_collection=collection,
                batch_size=self.batch_size,
                block_size=block_size,
                fetch_factor=fetch_factor,
                fetch_transform=fetch_transform_adata,  # Use module-level callback
            )

            # Create DataLoader with single worker for parameter scaling test
            dataloader = DataLoader(
                sc_dataset,
                batch_size=None,
                num_workers=0,  # Single worker
                prefetch_factor=None,  # No prefetch_factor for single worker
            )

            # Warm up
            warmup_batches = 3
            for i, _batch in enumerate(dataloader):
                if i >= warmup_batches:
                    break

            # Benchmark with peak memory tracking
            desc = f"block_size={block_size}, fetch_factor={fetch_factor}"

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

            return ScalingBenchmarkResult(
                block_size=block_size,
                fetch_factor=fetch_factor,
                throughput_cells_per_sec=throughput_cells_per_sec,
                memory_usage_gb=peak_memory,
                measurement_time=elapsed_time,
                total_cells=total_cells,
                batches_processed=batch_count,
            )

        except Exception as e:
            self.console.print(
                f"[red]Error benchmarking block_size={block_size}, fetch_factor={fetch_factor}: {e}[/red]"
            )
            return None

    def run_benchmarks(self) -> list[ScalingBenchmarkResult]:
        """Run benchmarks for all parameter combinations."""

        self.console.print(
            Panel.fit(
                "[bold green]scDataset Parameter Scaling Benchmark[/bold green]\n"
                "Testing performance scaling with block_size and fetch_factor parameters",
                border_style="green",
            )
        )

        results = []

        # Test all parameter combinations
        for block_size in self.block_sizes:
            for fetch_factor in self.fetch_factors:
                self.console.print(
                    f"\n[bold]Testing block_size={block_size}, fetch_factor={fetch_factor}[/bold]"
                )

                result = self.benchmark_scdataset_parameters(block_size, fetch_factor)
                if result:
                    results.append(result)
                    self.console.print(
                        f"  Result: {result.throughput_cells_per_sec:.0f} cells/sec"
                    )

        return results

    def print_results(self, results: list[ScalingBenchmarkResult]):
        """Print benchmark results in formatted tables."""

        if not results:
            self.console.print("[red]No results to display[/red]")
            return

        # Create results table
        self.console.print("\n[bold]scDataset Parameter Scaling Results[/bold]")

        table = Table(title="scDataset Parameter Scaling Performance")
        table.add_column("Block Size", style="cyan")
        table.add_column("Fetch Factor", style="cyan")
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
                str(result.block_size),
                str(result.fetch_factor),
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
                f"Best configuration: block_size={best_config.block_size}, fetch_factor={best_config.fetch_factor}"
            )

            # Check if there's any scaling
            min_throughput = min(r.throughput_cells_per_sec for r in results)
            scaling_factor = (
                max_throughput / min_throughput if min_throughput > 0 else 1
            )
            self.console.print(f"Scaling factor: {scaling_factor:.1f}x")

            if scaling_factor < 1.5:
                self.console.print(
                    "[yellow]Note: Limited scaling observed - this may differ from paper results[/yellow]"
                )
            else:
                self.console.print("[green]Significant scaling observed[/green]")


def main():
    """Run the scDataset parameter scaling benchmarks."""

    # Use a smaller dataset for development
    slaf_path = "../slaf-datasets/plate1_Tahoe100M.slaf"
    h5ad_path = "../slaf-datasets/plate1_Tahoe100M.h5ad"

    benchmark = scDatasetScalingBenchmark(slaf_path, h5ad_path)
    results = benchmark.run_benchmarks()
    benchmark.print_results(results)


if __name__ == "__main__":
    main()
