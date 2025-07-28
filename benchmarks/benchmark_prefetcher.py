#!/usr/bin/env python3
"""
SLAF Prefetcher Performance Benchmark

This benchmark measures the performance of SLAF's prefetcher pipeline
across different batches_per_chunk configurations and tokenization strategies
using direct timing metrics from PrefetchBatchProcessor.
"""

import gc
from dataclasses import dataclass

import numpy as np
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from slaf.core.slaf import SLAFArray
from slaf.ml.aggregators import GeneformerWindow, ScGPTWindow
from slaf.ml.datasets import PrefetchBatchProcessor
from slaf.ml.samplers import RandomShuffle
from slaf.ml.tokenizers import SLAFTokenizer


@dataclass
class StepTiming:
    """Timing for a single pipeline step."""

    step_name: str
    duration_ms: float
    cells_processed: int


@dataclass
class BatchTiming:
    """Complete timing breakdown for a single batch."""

    batches_per_chunk: int
    tokenizer_type: str
    total_duration_ms: float
    cells_processed: int
    steps: list[StepTiming]

    @property
    def cells_per_sec(self) -> float:
        """Calculate cells per second for this batch."""
        return (
            (self.cells_processed / self.total_duration_ms) * 1000
            if self.total_duration_ms > 0
            else 0
        )


class PrefetcherTimingAnalyzer:
    """Analyze prefetcher performance across different configurations."""

    def __init__(self, slaf_path: str, verbose: bool = True):
        self.slaf_path = slaf_path
        self.verbose = verbose
        self.console = Console()

        # Load SLAF array
        self.console.print(f"Loading SLAF array from {slaf_path}...")
        self.slaf_array = SLAFArray(slaf_path)

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def capture_batch_timing(
        self,
        tokenizer_type: str,
        batches_per_chunk: int,
        max_genes: int,
        num_batches: int = 20,  # More batches for better throughput measurement
        num_runs: int = 3,  # Multiple runs for averaging
    ) -> list[BatchTiming]:
        """Capture timing for multiple batches with a given configuration."""

        batch_timings = []

        if self.verbose:
            self.console.print(
                f"\n[bold green]Capturing timing: {tokenizer_type} (batches_per_chunk={batches_per_chunk}, max_genes={max_genes})[/bold green]"
            )

        # Clear memory before measurement
        gc.collect()

        # Create tokenizer and window
        if tokenizer_type == "scgpt":
            tokenizer = SLAFTokenizer(self.slaf_array, tokenizer_type="scgpt")
            window = ScGPTWindow()
        elif tokenizer_type == "geneformer":
            tokenizer = SLAFTokenizer(self.slaf_array, tokenizer_type="geneformer")
            window = GeneformerWindow()
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

        shuffle = RandomShuffle()

        # Multiple measurement runs
        for run_idx in range(num_runs):
            if self.verbose:
                self.console.print(f"  Run {run_idx + 1}/{num_runs}...")

            # Create processor with timing enabled
            processor = PrefetchBatchProcessor(
                slaf_array=self.slaf_array,
                window=window,
                shuffle=shuffle,
                tokenizer=tokenizer,
                seed=42 + run_idx,  # Different seed per run
                max_genes=max_genes,
                batches_per_chunk=batches_per_chunk,
                n_expression_bins=10,
                use_binned_expressions=True,
                n_epochs=1000,  # Large number to avoid epoch transitions
                raw_mode=False,
                verbose=False,  # Suppress detailed prints
                log_metrics=True,  # Enable timing metrics
            )

            # Warm up - process a few batches
            if self.verbose:
                self.console.print("    Warming up...")

            warmup_batches = 5
            for _i in range(warmup_batches):
                try:
                    batch = processor.load_prefetch_batch()
                    # Force data loading during warmup
                    if hasattr(batch, "input_ids"):
                        _ = batch.input_ids
                        _ = batch.attention_mask
                    _ = batch.cell_integer_ids
                except StopIteration:
                    break

            # Process batches for timing measurement
            if self.verbose:
                self.console.print("    Capturing timing metrics...")

            batch_count = 0
            for _batch_idx in range(num_batches):
                try:
                    batch = processor.load_prefetch_batch()

                    # Force data loading
                    if hasattr(batch, "input_ids"):
                        _ = batch.input_ids
                        _ = batch.attention_mask
                    _ = batch.cell_integer_ids

                    batch_count += 1
                except StopIteration:
                    break

            # Get timing metrics
            metrics = processor.get_timing_metrics()
            if metrics and batch_count > 0:
                # Convert to milliseconds
                lance_loading_ms = metrics.get("lance_loading", 0.0) * 1000
                window_ms = metrics.get("window", 0.0) * 1000
                shuffle_ms = metrics.get("shuffle", 0.0) * 1000
                tokenize_ms = metrics.get("tokenize", 0.0) * 1000
                total_ms = metrics.get("total", 0.0) * 1000
                cells_processed = int(metrics.get("cells_processed", 0))

                # Create step timing objects
                steps = [
                    StepTiming("Lance Loading", lance_loading_ms, cells_processed),
                    StepTiming("Window", window_ms, cells_processed),
                    StepTiming("Shuffle", shuffle_ms, cells_processed),
                    StepTiming("Tokenize", tokenize_ms, cells_processed),
                ]

                # Create batch timing object
                batch_timing = BatchTiming(
                    batches_per_chunk=batches_per_chunk,
                    tokenizer_type=tokenizer_type.title(),
                    total_duration_ms=total_ms,
                    cells_processed=cells_processed,
                    steps=steps,
                )

                batch_timings.append(batch_timing)

                if self.verbose:
                    self.console.print(
                        f"    {batch_count} batches: {total_ms:.1f}ms total, {batch_timing.cells_per_sec:.0f} cells/sec"
                    )

            # Clean up
            del processor
            gc.collect()

        return batch_timings

    def run_comprehensive_timing_analysis(self) -> dict[str, list[BatchTiming]]:
        """Run comprehensive timing analysis across all configurations."""

        self.console.print(
            Panel.fit(
                "[bold green]SLAF Prefetcher Performance Analysis[/bold green]\n"
                "Testing different batches_per_chunk configurations using direct timing metrics",
                border_style="green",
            )
        )

        # Define configurations to test
        configurations = [
            # scGPT configurations
            ("scgpt", 25, 1024),
            ("scgpt", 50, 1024),
            ("scgpt", 100, 1024),
            ("scgpt", 200, 1024),
            # Geneformer configurations
            ("geneformer", 25, 2048),
            ("geneformer", 50, 2048),
            ("geneformer", 100, 2048),
            ("geneformer", 200, 2048),
        ]

        all_results = {}

        for tokenizer_type, batches_per_chunk, max_genes in configurations:
            try:
                batch_timings = self.capture_batch_timing(
                    tokenizer_type, batches_per_chunk, max_genes
                )
                config_key = f"{tokenizer_type}_{batches_per_chunk}_{max_genes}"
                all_results[config_key] = batch_timings

            except Exception as e:
                self.console.print(
                    f"[red]Error analyzing {tokenizer_type} (batches_per_chunk={batches_per_chunk}): {e}[/red]"
                )

        return all_results

    def print_comprehensive_timing_table(
        self, all_results: dict[str, list[BatchTiming]]
    ):
        """Print comprehensive timing analysis table."""

        # Get all step names from the first batch timing
        step_names = []
        for _config_key, batch_timings in all_results.items():
            if batch_timings:
                step_names = [step.step_name for step in batch_timings[0].steps]
                break

        # Create table
        table = Table(title="SLAF Prefetcher Performance: batches_per_chunk Scaling")

        # Add columns
        table.add_column("Tokenizer", style="cyan", width=12)
        table.add_column("Batches/Chunk", style="cyan", justify="right", width=12)
        table.add_column("Total Time (ms)", style="green", justify="right")
        table.add_column("Cells/sec", style="green", justify="right")

        # Add step columns
        for step_name in step_names:
            table.add_column(f"{step_name} (ms)", style="cyan", justify="right")

        # Process each configuration - sort by tokenizer, then batches_per_chunk
        configs = []
        for config_key in all_results.keys():
            batch_timings = all_results[config_key]
            if not batch_timings:
                continue

            # Extract configuration info
            parts = config_key.split("_")
            tokenizer_type = parts[0]
            batches_per_chunk = int(parts[1])

            # Calculate averages across batches
            avg_total_time = np.mean([bt.total_duration_ms for bt in batch_timings])
            avg_cells_per_sec = np.mean([bt.cells_per_sec for bt in batch_timings])

            # Calculate average step times
            step_averages = {}
            for step_name in step_names:
                step_times = []
                for bt in batch_timings:
                    for step in bt.steps:
                        if step.step_name == step_name:
                            step_times.append(step.duration_ms)
                            break
                if step_times:
                    step_averages[step_name] = np.mean(step_times)
                else:
                    step_averages[step_name] = 0.0

            configs.append(
                {
                    "tokenizer": tokenizer_type.title(),
                    "batches_per_chunk": batches_per_chunk,
                    "total_time": avg_total_time,
                    "cells_per_sec": avg_cells_per_sec,
                    "step_averages": step_averages,
                }
            )

        # Sort by tokenizer, then batches_per_chunk
        configs.sort(key=lambda x: (x["tokenizer"], x["batches_per_chunk"]))

        # Add rows to table
        for config in configs:
            row_data = [
                config["tokenizer"],
                str(config["batches_per_chunk"]),
                f"{config['total_time']:.1f}",
                f"{config['cells_per_sec']:.0f}",
            ]

            # Add step times
            for step_name in step_names:
                step_time = config["step_averages"][step_name]
                if step_time > 0:
                    row_data.append(f"{step_time:.1f}")
                else:
                    row_data.append("0.0")

            table.add_row(*row_data)

        self.console.print(table)

        # Summary statistics
        self.console.print("\n[bold]Summary:[/bold]")
        cells_per_sec_values = [r["cells_per_sec"] for r in configs]

        if cells_per_sec_values:
            self.console.print(
                f"Best throughput: {int(max(cells_per_sec_values)):,} cells/sec"
            )
            self.console.print(
                f"Throughput range: {int(min(cells_per_sec_values)):,} - {int(max(cells_per_sec_values)):,} cells/sec"
            )


def main():
    """Run the prefetcher performance benchmark."""

    # Use a smaller dataset for development
    slaf_path = "../slaf-datasets/plate1_Tahoe100M.slaf"

    analyzer = PrefetcherTimingAnalyzer(slaf_path)

    # Run comprehensive timing analysis
    results = analyzer.run_comprehensive_timing_analysis()
    analyzer.print_comprehensive_timing_table(results)


if __name__ == "__main__":
    main()
