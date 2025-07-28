#!/usr/bin/env python3
"""
SLAF Internal Dataloader Benchmarks

This benchmark measures SLAF's performance across different tokenization strategies
using a single batch size of 32 (most common in literature).

Tokenization Strategies:
1. scGPT with binning
2. scGPT without binning
3. Geneformer with percentile filtering
4. Geneformer without percentile filtering

Inspired by test_performance_breakdown_fixed.py
"""

import gc
import time
from dataclasses import dataclass

import numpy as np
import psutil
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from slaf.core.slaf import SLAFArray
from slaf.ml.dataloaders import SLAFDataLoader


@dataclass
class TokenizationConfig:
    """Configuration for different tokenization strategies."""

    name: str
    tokenizer_type: str  # 'scgpt', 'geneformer', or 'raw'
    max_genes: int = 2000
    vocab_size: int = 65000


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: TokenizationConfig
    throughput_cells_per_sec: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    avg_batch_time_ms: float
    total_tokens: int
    total_cells: int


@dataclass
class ScalingResult:
    """Results from batch size scaling test."""

    mode: str  # "raw" or "tokenized"
    batch_size: int
    throughput_cells_per_sec: float
    throughput_tokens_per_sec: float
    measurement_time: float
    total_cells: int
    total_batches: int


class InternalDataloaderBenchmark:
    """Benchmark SLAF dataloaders with different tokenization strategies."""

    def __init__(self, slaf_path: str):
        self.slaf_path = slaf_path
        self.console = Console()

        # Load SLAF array
        self.console.print(f"Loading SLAF array from {slaf_path}...")
        self.slaf_array = SLAFArray(slaf_path)

        # Define tokenization configurations
        self.configs = [
            TokenizationConfig(
                name="scGPT with binning",
                tokenizer_type="scgpt",
                max_genes=2000,
                vocab_size=8192,
            ),
            TokenizationConfig(
                name="scGPT without binning",
                tokenizer_type="scgpt",
                max_genes=2000,
                vocab_size=8192,
            ),
            TokenizationConfig(
                name="Geneformer with percentile filtering",
                tokenizer_type="geneformer",
                max_genes=2000,
                vocab_size=8192,
            ),
            TokenizationConfig(
                name="Geneformer without percentile filtering",
                tokenizer_type="geneformer",
                max_genes=2000,
                vocab_size=8192,
            ),
            TokenizationConfig(
                name="Raw mode (no tokenization)",
                tokenizer_type="raw",
                max_genes=2000,
                vocab_size=8192,
            ),
        ]

    def create_dataloader(self, config: TokenizationConfig) -> SLAFDataLoader:
        """Create a dataloader with the given tokenization configuration."""

        # Create dataloader
        dataloader = SLAFDataLoader(
            slaf_array=self.slaf_array,
            tokenizer_type=config.tokenizer_type,
            batch_size=32,
            max_genes=config.max_genes,
            vocab_size=config.vocab_size,
            n_expression_bins=10,  # Default value
            n_epochs=1000,  # Reduced from 100 to avoid unnecessary epochs
            raw_mode=(
                config.tokenizer_type == "raw"
            ),  # Enable raw mode for "raw" config
            verbose=False,  # Suppress SLAF's detailed timing prints
        )

        return dataloader

    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def create_progress_bar(self, desc: str, mode: str = "training") -> tqdm:
        """Create a beautiful progress bar for SLAF benchmarks."""

        if mode == "prefetch":
            # Prefetch progress bar with detailed timing breakdown
            bar_format = (
                "{l_bar}{bar}| {n_fmt}/{total_fmt} batches "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            )
        else:
            # Training progress bar
            bar_format = (
                "{l_bar}{bar}| {n_fmt} batches "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            )

        return tqdm(
            desc=desc, unit="batch", bar_format=bar_format, ncols=100, leave=False
        )

    def format_prefetch_postfix(self, batch_info: dict) -> str:
        """Format prefetch batch information for progress bar postfix."""

        # Extract timing information
        lance_time = batch_info.get("lance_time", 0)
        window_time = batch_info.get("window_time", 0)
        shuffle_time = batch_info.get("shuffle_time", 0)
        tokenize_time = batch_info.get("tokenize_time", 0)
        total_time = batch_info.get("total_time", 0)
        cells = batch_info.get("cells", 0)
        memory_mb = batch_info.get("memory_mb", 0)

        # Format the postfix string
        postfix = f"{total_time:.1f}ms/batch, {cells} cells, {memory_mb:.1f}MB"

        # Add timing breakdown if available
        if any([lance_time, window_time, shuffle_time, tokenize_time]):
            timing_parts = []
            if lance_time > 0:
                timing_parts.append(f"Lance: {lance_time:.1f}ms")
            if window_time > 0:
                timing_parts.append(f"Window: {window_time:.1f}ms")
            if shuffle_time > 0:
                timing_parts.append(f"Shuffle: {shuffle_time:.1f}ms")
            if tokenize_time > 0:
                timing_parts.append(f"Tokenize: {tokenize_time:.1f}ms")

            if timing_parts:
                postfix += f" | {' | '.join(timing_parts)}"

        return postfix

    def format_training_postfix(self, batch_info: dict) -> str:
        """Format training batch information for progress bar postfix."""

        # Extract timing information
        tensor_time = batch_info.get("tensor_time", 0)
        total_time = batch_info.get("total_time", 0)
        throughput = batch_info.get("throughput", 0)
        mode = batch_info.get("mode", "tokenized")

        # Format the postfix string
        postfix = f"{total_time:.1f}ms/batch, {throughput:.0f} cells/sec"

        # Add mode-specific information
        if mode == "raw":
            postfix += " | Raw data"
        else:
            postfix += " | Pre-tokenized data"

        # Add tensor creation time if available
        if tensor_time > 0:
            postfix += f" | Tensor: {tensor_time:.1f}ms"

        return postfix

    def benchmark_config(self, config: TokenizationConfig) -> BenchmarkResult:
        """Benchmark a single tokenization configuration."""

        # Disable SLAF's detailed timing prints for clean benchmark output
        # disable_slaf_printing() # This line is removed as per the edit hint

        self.console.print(f"\n[bold blue]Benchmarking: {config.name}[/bold blue]")

        # Create dataloader
        dataloader = self.create_dataloader(config)

        # Warm up
        self.console.print("Warming up...")
        warmup_batches = 3
        for i, _batch in enumerate(dataloader):
            if i >= warmup_batches:
                break

        # Measure memory before
        memory_before = self.measure_memory_usage()

        # Benchmark with progress bar
        self.console.print("Running benchmark...")
        start_time = time.time()
        measurement_duration = 10  # 10 seconds like test_performance_breakdown_fixed.py

        total_cells = 0
        total_tokens = 0
        batch_times = []
        batch_count = 0

        # Create progress bar
        with self.create_progress_bar(f"Training: {config.name}", "training") as pbar:
            for batch in dataloader:
                batch_start = time.time()
                batch_count += 1

                # Count cells and tokens
                if config.tokenizer_type == "raw":
                    # Raw mode: count cells from the batch
                    batch_size = len(batch["cell_ids"])
                    # For raw mode, we don't have tokens, so use a placeholder
                    total_tokens_in_batch = 0
                else:
                    # Tokenized mode: count cells and tokens
                    batch_size = batch["input_ids"].shape[0]
                    total_tokens_in_batch = batch["input_ids"].numel()

                total_cells += batch_size
                total_tokens += total_tokens_in_batch

                batch_end = time.time()
                batch_time_ms = (batch_end - batch_start) * 1000
                batch_times.append(batch_time_ms)

                # Calculate current throughput
                elapsed = time.time() - start_time
                current_throughput = total_cells / elapsed if elapsed > 0 else 0

                # Update progress bar
                batch_info = {
                    "total_time": batch_time_ms,
                    "throughput": current_throughput,
                    "mode": "raw" if config.tokenizer_type == "raw" else "tokenized",
                }

                postfix = self.format_training_postfix(batch_info)
                pbar.set_postfix_str(postfix)
                pbar.update(1)

                # Stop after measurement duration
                if elapsed >= measurement_duration:
                    break

        # Clean up the dataloader to stop background processing
        del dataloader
        gc.collect()

        # Measure memory after
        memory_after = self.measure_memory_usage()
        memory_usage = memory_after - memory_before

        # Calculate metrics
        elapsed_time = time.time() - start_time
        throughput_cells_per_sec = total_cells / elapsed_time
        throughput_tokens_per_sec = total_tokens / elapsed_time
        avg_batch_time_ms = np.mean(batch_times) if batch_times else 0.0

        return BenchmarkResult(
            config=config,
            throughput_cells_per_sec=throughput_cells_per_sec,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            memory_usage_mb=memory_usage,
            avg_batch_time_ms=avg_batch_time_ms,
            total_tokens=total_tokens,
            total_cells=total_cells,
        )

    def run_benchmarks(self) -> list[BenchmarkResult]:
        """Run benchmarks for all tokenization configurations."""

        self.console.print(
            Panel.fit(
                "[bold green]SLAF Internal Dataloader Benchmarks[/bold green]\n"
                "Testing 5 tokenization strategies with batch_size=32",
                border_style="green",
            )
        )

        results = []

        for config in self.configs:
            try:
                result = self.benchmark_config(config)
                results.append(result)

                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                self.console.print(f"[red]Error benchmarking {config.name}: {e}[/red]")

        return results

    def print_results(self, results: list[BenchmarkResult]):
        """Print benchmark results in a formatted table."""

        table = Table(title="SLAF Internal Dataloader Benchmark Results")

        table.add_column("Tokenization Strategy", style="cyan")
        table.add_column("Throughput (cells/sec)", style="green")
        table.add_column("Throughput (tokens/sec)", style="green")

        for result in results:
            # Handle raw mode tokenization display
            tokens_per_sec_display = (
                "N/A"
                if result.config.tokenizer_type == "raw"
                else f"{int(result.throughput_tokens_per_sec):,}"
            )

            table.add_row(
                result.config.name,
                f"{int(result.throughput_cells_per_sec):,}",
                tokens_per_sec_display,
            )

        self.console.print(table)

        # Summary statistics
        self.console.print("\n[bold]Summary:[/bold]")
        cells_per_sec_values = [r.throughput_cells_per_sec for r in results]
        # Filter out raw mode for token throughput calculations
        tokens_per_sec_values = [
            r.throughput_tokens_per_sec
            for r in results
            if r.config.tokenizer_type != "raw"
        ]

        self.console.print(
            f"Best throughput: {int(max(cells_per_sec_values)):,} cells/sec"
        )
        if tokens_per_sec_values:
            self.console.print(
                f"Best token throughput: {int(max(tokens_per_sec_values)):,} tokens/sec"
            )
        else:
            self.console.print("Best token throughput: N/A (raw mode only)")
        self.console.print(
            f"Throughput range: {int(min(cells_per_sec_values)):,} - {int(max(cells_per_sec_values)):,} cells/sec"
        )

    def benchmark_batch_size(
        self, mode: str, batch_size: int, measurement_duration: float = 10.0
    ) -> ScalingResult:
        """Benchmark a specific batch size and mode."""

        self.console.print(
            f"\n[bold blue]Testing {mode} mode with batch_size={batch_size}[/bold blue]"
        )

        # Clear memory before measurement
        gc.collect()

        # Create dataloader
        dataloader = SLAFDataLoader(
            slaf_array=self.slaf_array,
            batch_size=batch_size,
            n_epochs=1000,
            raw_mode=(mode == "raw"),
            verbose=False,  # Suppress SLAF's detailed timing prints
        )

        # Warm up
        self.console.print("  Warming up...")
        warmup_batches = 3
        for i, _batch in enumerate(dataloader):
            if i >= warmup_batches:
                break

        # Measurement phase with progress bar
        self.console.print("  Measuring...")
        start_time = time.time()
        total_cells = 0
        batch_count = 0

        with self.create_progress_bar(
            f"Training: {mode} mode (batch_size={batch_size})", "training"
        ) as pbar:
            for batch in dataloader:
                batch_start = time.time()
                batch_count += 1

                # Count cells based on actual batch size (not configured batch_size)
                if mode == "raw":
                    if "X" in batch:
                        actual_batch_size = batch["X"].shape[0]
                    elif "cell_ids" in batch:
                        actual_batch_size = len(batch["cell_ids"])
                    else:
                        actual_batch_size = batch_size  # Fallback to configured size
                else:
                    # Tokenized mode: use input_ids shape
                    actual_batch_size = batch["input_ids"].shape[0]

                total_cells += actual_batch_size

                # Force data loading for realistic measurement
                if mode == "raw":
                    if "X" in batch:
                        _ = batch["X"]  # Force data loading
                else:
                    _ = batch["input_ids"]  # Force data loading

                batch_end = time.time()
                batch_time_ms = (batch_end - batch_start) * 1000

                # Calculate current throughput
                elapsed = time.time() - start_time
                current_throughput = total_cells / elapsed if elapsed > 0 else 0

                # Update progress bar
                batch_info = {
                    "total_time": batch_time_ms,
                    "throughput": current_throughput,
                    "mode": mode,
                }

                postfix = self.format_training_postfix(batch_info)
                pbar.set_postfix_str(postfix)
                pbar.update(1)

                # Stop after measurement duration
                if elapsed >= measurement_duration:
                    break

        elapsed_time = time.time() - start_time

        # Clean up
        del dataloader
        gc.collect()

        # Calculate metrics
        throughput_cells_per_sec = total_cells / elapsed_time

        # Calculate tokens per second
        if mode == "raw":
            # Raw mode: no tokenization, so tokens/sec = 0
            throughput_tokens_per_sec = 0
        else:
            # Tokenized mode: calculate tokens per cell
            # Geneformer: max_genes tokens per cell (approximate)
            tokens_per_cell = 2048  # max_genes
            throughput_tokens_per_sec = throughput_cells_per_sec * tokens_per_cell

        result = ScalingResult(
            mode=mode,
            batch_size=batch_size,
            throughput_cells_per_sec=throughput_cells_per_sec,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            measurement_time=elapsed_time,
            total_cells=total_cells,
            total_batches=batch_count,
        )

        self.console.print(
            f"  Result: {throughput_cells_per_sec:.0f} cells/sec, {throughput_tokens_per_sec:.0f} tokens/sec"
        )

        return result

    def run_scaling_benchmarks(self) -> list[ScalingResult]:
        """Run scaling benchmarks for all batch sizes and modes."""

        self.console.print(
            Panel.fit(
                "[bold green]SLAF Batch Size Scaling Benchmarks[/bold green]\n"
                "Testing raw vs tokenized performance scaling",
                border_style="green",
            )
        )

        # Define batch sizes to test
        batch_sizes = [32, 64, 128, 256]
        results = []

        # Test raw mode scaling
        self.console.print("\n[bold]Raw Mode Scaling (cells/sec)[/bold]")
        for batch_size in batch_sizes:
            try:
                result = self.benchmark_batch_size("raw", batch_size)
                results.append(result)
            except Exception as e:
                self.console.print(
                    f"[red]Error with batch_size={batch_size}: {e}[/red]"
                )

        # Test tokenized mode scaling
        self.console.print("\n[bold]Tokenized Mode Scaling (tokens/sec)[/bold]")
        for batch_size in batch_sizes:
            try:
                result = self.benchmark_batch_size("tokenized", batch_size)
                results.append(result)
            except Exception as e:
                self.console.print(
                    f"[red]Error with batch_size={batch_size}: {e}[/red]"
                )

        return results

    def print_scaling_results(self, results: list[ScalingResult]):
        """Print scaling results in formatted tables."""

        # Separate results by mode
        raw_results = [r for r in results if r.mode == "raw"]
        tokenized_results = [r for r in results if r.mode == "tokenized"]

        # Raw Mode Results Table
        self.console.print("\n[bold blue]Raw Mode: Cells/sec Scaling[/bold blue]")
        table1 = Table(title="Raw Mode Performance Scaling")

        table1.add_column("Batch Size", style="cyan")
        table1.add_column("Throughput (cells/sec)", style="green")
        table1.add_column("Total Cells", style="blue")
        table1.add_column("Measurement Time (s)", style="yellow")

        for result in raw_results:
            # Color code the throughput
            throughput_text = f"{result.throughput_cells_per_sec:,.0f}"
            if result.throughput_cells_per_sec > 20000:
                throughput_text = f"[green]{throughput_text}[/green]"
            elif result.throughput_cells_per_sec > 15000:
                throughput_text = f"[yellow]{throughput_text}[/yellow]"

            table1.add_row(
                str(result.batch_size),
                throughput_text,
                f"{result.total_cells:,}",
                f"{result.measurement_time:.1f}",
            )

        self.console.print(table1)

        # Tokenized Mode Results Table
        self.console.print(
            "\n[bold blue]Tokenized Mode: Tokens/sec Scaling[/bold blue]"
        )
        table2 = Table(title="Tokenized Mode Performance Scaling")

        table2.add_column("Batch Size", style="cyan")
        table2.add_column("Throughput (cells/sec)", style="green")
        table2.add_column("Throughput (tokens/sec)", style="green")
        table2.add_column("Total Cells", style="blue")
        table2.add_column("Measurement Time (s)", style="yellow")

        for result in tokenized_results:
            # Color code the throughput
            cells_text = f"{result.throughput_cells_per_sec:,.0f}"
            tokens_text = f"{result.throughput_tokens_per_sec:,.0f}"

            if result.throughput_cells_per_sec > 10000:
                cells_text = f"[green]{cells_text}[/green]"
            elif result.throughput_cells_per_sec > 8000:
                cells_text = f"[yellow]{cells_text}[/yellow]"

            if result.throughput_tokens_per_sec > 20000000:
                tokens_text = f"[green]{tokens_text}[/green]"
            elif result.throughput_tokens_per_sec > 15000000:
                tokens_text = f"[yellow]{tokens_text}[/yellow]"

            table2.add_row(
                str(result.batch_size),
                cells_text,
                tokens_text,
                f"{result.total_cells:,}",
                f"{result.measurement_time:.1f}",
            )

        self.console.print(table2)

        # Analysis
        self.console.print("\n[bold]Scaling Analysis:[/bold]")

        if raw_results:
            raw_scaling = self._calculate_scaling_factor(
                raw_results, "throughput_cells_per_sec"
            )
            self.console.print(f"✓ Raw mode scaling: {raw_scaling:.1f}x improvement")

        if tokenized_results:
            tokenized_scaling = self._calculate_scaling_factor(
                tokenized_results, "throughput_tokens_per_sec"
            )
            self.console.print(
                f"✓ Tokenized mode scaling: {tokenized_scaling:.1f}x improvement"
            )

            # Key insight
            if tokenized_scaling < 1.5:
                self.console.print("\n[bold yellow]Key Insight:[/bold yellow]")
                self.console.print(
                    "Token throughput remains relatively constant across batch sizes."
                )
                self.console.print(
                    "This demonstrates why tokens/sec is the meaningful metric for training."
                )
                self.console.print(
                    "Cells/sec can be misleading as it doesn't account for tokenization overhead."
                )

    def _calculate_scaling_factor(
        self, results: list[ScalingResult], metric: str
    ) -> float:
        """Calculate scaling factor between smallest and largest batch size."""
        if len(results) < 2:
            return 1.0

        # Sort by batch size
        sorted_results = sorted(results, key=lambda r: r.batch_size)

        min_throughput = getattr(sorted_results[0], metric)
        max_throughput = getattr(sorted_results[-1], metric)

        if min_throughput > 0:
            return max_throughput / min_throughput
        return 1.0


def main():
    """Run the internal dataloader benchmarks."""

    # Use a smaller dataset for development
    slaf_path = "../slaf-datasets/plate1_Tahoe100M.slaf"

    benchmark = InternalDataloaderBenchmark(slaf_path)

    # Run tokenization strategy benchmarks
    results = benchmark.run_benchmarks()
    benchmark.print_results(results)

    # Run batch scaling benchmarks
    scaling_results = benchmark.run_scaling_benchmarks()
    benchmark.print_scaling_results(scaling_results)


if __name__ == "__main__":
    main()
