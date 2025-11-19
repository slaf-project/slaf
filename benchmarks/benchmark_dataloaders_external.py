#!/usr/bin/env python3
"""
SLAF External Dataloader Benchmarks

This benchmark compares SLAF against state-of-the-art dataloaders using
a standardized benchmark setup.

Two-Tier Comparison:
1. Tier 1: Raw Data Loading (cells/sec) - Compare raw data loading performance
2. Tier 2: GPU-Ready Output Comparison - Compare end-to-end pipeline performance

Competitor Systems:
- scDataset (arXiv:2506.01883)
- BioNeMo SCDL (NVIDIA)
- AnnDataLoader (scvi-tools)
- annbatch (lamin/scverse)
"""

import gc
import time
from dataclasses import dataclass

import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from slaf.core.slaf import SLAFArray
from slaf.ml.dataloaders import SLAFDataLoader


# Define the fetch_transform callback for AnnData at module level
def fetch_transform_adata(batch):
    """Callback function to transform scDataset batches to AnnData format."""
    return batch.to_adata()


@dataclass
class ExternalBenchmarkResult:
    """Results from external dataloader benchmark."""

    system_name: str
    tier: str  # "tier1" or "tier2"
    throughput_cells_per_sec: float
    memory_usage_gb: float
    output_type: str  # "raw" or "tokenized"
    processes: int
    measurement_time: float
    total_cells: int


class ExternalDataloaderBenchmark:
    """Benchmark SLAF against external dataloaders."""

    def __init__(
        self,
        slaf_path: str,
        h5ad_path: str,
        tiledb_path: str = None,
        scdl_path: str = None,
        annbatch_path: str = None,
    ):
        self.slaf_path = slaf_path
        self.h5ad_path = h5ad_path
        self.tiledb_path = tiledb_path
        self.scdl_path = scdl_path
        self.annbatch_path = annbatch_path
        self.console = Console()

        # Load SLAF array
        self.console.print(f"Loading SLAF array from {slaf_path}...")
        self.slaf_array = SLAFArray(slaf_path)

        # Load AnnData object once to avoid repeated loading
        self.console.print(f"Loading AnnData from {h5ad_path}...")
        import anndata as ad

        self.adata = ad.read_h5ad(h5ad_path, backed="r")
        self.console.print(
            f"Loaded {self.adata.n_obs:,} cells, {self.adata.n_vars:,} genes"
        )

        # Define competitor systems and their configurations
        self.batch_size = 64
        self.competitor_configs = {
            "SLAF": {
                "tier1": {"raw_mode": True, "batch_size": self.batch_size},
                "tier2": {"raw_mode": False, "batch_size": self.batch_size},
                "processes": 1,
            },
            "scDataset": {
                "tier1": {"raw_mode": True, "batch_size": self.batch_size},
                "tier2": {
                    "raw_mode": True,
                    "batch_size": self.batch_size,
                },  # No tokenization
                "processes": 0,  # Multiprocessing with anndata-loaded objects
                "claimed_performance": 4000,  # cells/sec from paper
            },
            "AnnDataLoader": {
                "tier1": {"raw_mode": True, "batch_size": self.batch_size},
                "tier2": {
                    "raw_mode": True,
                    "batch_size": self.batch_size,
                },  # No tokenization
                "processes": 1,
                "estimated_performance": 500,  # cells/sec estimate
            },
            "AnnLoader": {
                "tier1": {"raw_mode": True, "batch_size": self.batch_size},
                "tier2": {
                    "raw_mode": True,
                    "batch_size": self.batch_size,
                },  # No tokenization
                "processes": 1,
                "estimated_performance": 300,  # cells/sec estimate
            },
            "TileDB DataLoader": {
                "tier1": {"raw_mode": True, "batch_size": self.batch_size},
                "tier2": {
                    "raw_mode": True,
                    "batch_size": self.batch_size,
                },  # No tokenization
                "processes": 1,
                "estimated_performance": 800,  # cells/sec estimate
            },
            "BioNeMo SCDL": {
                "tier1": {"raw_mode": True, "batch_size": self.batch_size},
                "tier2": {
                    "raw_mode": True,
                    "batch_size": self.batch_size,
                },  # No tokenization
                "processes": 1,
                "estimated_performance": 2000,  # cells/sec estimate from NVIDIA docs
            },
            "annbatch": {
                "tier1": {"raw_mode": True, "batch_size": self.batch_size},
                "tier2": {
                    "raw_mode": True,
                    "batch_size": self.batch_size,
                },  # No tokenization
                "processes": 1,
                "estimated_performance": 1000000,  # cells/sec from annbatch docs
            },
        }

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

    def benchmark_with_memory_tracking(
        self, dataloader, total_duration: float, system_name: str = "Unknown"
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
        measurement_started = False
        warmup_duration = 10  # Skip first 10 seconds for MoS stabilization

        with self.create_progress_bar(f"Training: {system_name}", "training") as pbar:
            for _batch in dataloader:
                batch_start = time.time()
                batch_count += 1

                # Use actual batch size for SLAF, configured batch size for external dataloaders
                if system_name.startswith("SLAF"):
                    # SLAF dataloaders: use actual batch size from the data
                    if isinstance(_batch, dict):
                        if "X" in _batch:
                            actual_batch_size = _batch["X"].shape[0]
                        elif "cell_ids" in _batch:
                            actual_batch_size = len(_batch["cell_ids"])
                        else:
                            actual_batch_size = self.batch_size  # Fallback
                    else:
                        actual_batch_size = self.batch_size  # Fallback
                elif system_name == "annbatch":
                    # annbatch: get batch size from AnnData object
                    if hasattr(_batch, "to_adata"):
                        adata = _batch.to_adata()
                        actual_batch_size = adata.n_obs
                        # Force data loading while we have the adata object
                        _ = adata.X
                    else:
                        actual_batch_size = self.batch_size  # Fallback
                else:
                    # External dataloaders: use configured batch size
                    # Note: External dataloaders may have different batch formats
                    # and we use configured batch_size for consistency
                    actual_batch_size = self.batch_size

                # Force actual data loading by accessing the data
                # This is crucial for getting realistic throughput measurements
                if system_name != "annbatch":  # annbatch already handled above
                    if hasattr(_batch, "X"):
                        # scDataset returns AnnCollectionView objects
                        _ = _batch.X  # Force data loading
                    elif isinstance(_batch, dict) and "X" in _batch:
                        # AnnDataLoader returns dict with 'X' key
                        _ = _batch["X"]  # Force data loading

                batch_end = time.time()
                batch_time_ms = (batch_end - batch_start) * 1000

                # Calculate elapsed time
                elapsed = time.time() - start_time

                # Skip warmup period for measurement (for all dataloaders)
                if elapsed >= warmup_duration and not measurement_started:
                    measurement_started = True
                    self.console.print(
                        f"  Starting measurement after {warmup_duration}s warmup..."
                    )

                # Only count cells after warmup period
                if measurement_started:
                    total_cells += actual_batch_size

                # Calculate current throughput (only for display)
                current_throughput = (
                    total_cells / (elapsed - warmup_duration)
                    if elapsed > warmup_duration
                    else 0
                )

                # Update progress bar
                batch_info = {
                    "total_time": batch_time_ms,
                    "throughput": current_throughput,
                    "mode": "raw",  # External dataloaders are all raw mode
                }

                postfix = self.format_training_postfix(batch_info)
                pbar.set_postfix_str(postfix)
                pbar.update(1)

                # Stop after total duration
                if elapsed >= total_duration:
                    break

        elapsed_time = time.time() - start_time

        # Stop memory monitoring
        stop_tracking.set()
        monitor_thread.join(timeout=1.0)

        return total_cells, batch_count, elapsed_time, peak_memory

    def benchmark_slaf_tier1(self) -> ExternalBenchmarkResult:
        """Benchmark SLAF in Tier 1 (raw data loading)."""

        # Disable SLAF's detailed timing prints for clean benchmark output
        # disable_slaf_printing() # This line is removed as per the edit hint

        self.console.print(
            "\n[bold blue]Benchmarking SLAF - Tier 1 (Raw Data Loading)[/bold blue]"
        )

        # Create SLAF dataloader in raw mode (MoS with same config as internal benchmarks)
        dataloader = SLAFDataLoader(
            slaf_array=self.slaf_array,
            tokenizer_type="raw",  # Explicitly set for raw mode
            batch_size=self.batch_size,
            max_genes=2000,  # Match internal benchmark
            vocab_size=65000,  # Match internal benchmark
            n_expression_bins=10,  # Match internal benchmark
            n_epochs=1000,
            raw_mode=True,
            verbose=False,  # Added verbose=False to suppress SLAF's detailed timing prints
            # MoS with same configuration as internal benchmarks
            use_mixture_of_scanners=True,
            by_fragment=False,
            batches_per_chunk=1,
            n_scanners=16,
            prefetch_batch_size=1048576,  # 1M rows
        )

        # Warm up (increased for MoS stability)
        self.console.print("Warming up...")
        warmup_batches = 15  # Increased from 3 for MoS warmup
        for i, _batch in enumerate(dataloader):
            if i >= warmup_batches:
                break

        # Benchmark with peak memory tracking
        self.console.print("Running benchmark...")
        warmup_duration = 10  # Skip first 10 seconds for MoS stabilization
        measurement_duration = 30  # Then measure for 30 seconds

        total_cells, batch_count, elapsed_time, peak_memory = (
            self.benchmark_with_memory_tracking(
                dataloader, warmup_duration + measurement_duration, "SLAF Tier 1"
            )
        )

        # Clean up the dataloader to stop background processing
        del dataloader
        gc.collect()

        # Calculate metrics (using only measurement time, not warmup)
        measurement_time = elapsed_time - warmup_duration
        throughput_cells_per_sec = (
            total_cells / measurement_time if measurement_time > 0 else 0
        )

        # Debug output
        self.console.print(
            f"  Debug: {total_cells} cells, {measurement_time:.2f}s measurement, {throughput_cells_per_sec:.0f} cells/sec"
        )

        return ExternalBenchmarkResult(
            system_name="SLAF",
            tier="tier1",
            throughput_cells_per_sec=throughput_cells_per_sec,
            memory_usage_gb=peak_memory,
            output_type="raw",
            processes=1,
            measurement_time=measurement_time,
            total_cells=total_cells,
        )

    def benchmark_slaf_tier2(self) -> ExternalBenchmarkResult:
        """Benchmark SLAF in Tier 2 (GPU-ready output)."""

        # Disable SLAF's detailed timing prints for clean benchmark output
        # disable_slaf_printing() # This line is removed as per the edit hint

        self.console.print(
            "\n[bold blue]Benchmarking SLAF - Tier 2 (GPU-Ready Output)[/bold blue]"
        )

        # Create SLAF dataloader in tokenized mode (MoS with same config as internal benchmarks)
        dataloader = SLAFDataLoader(
            slaf_array=self.slaf_array,
            tokenizer_type="geneformer",  # Use Geneformer to match internal benchmark
            batch_size=self.batch_size,
            max_genes=2000,  # Match internal benchmark
            vocab_size=65000,  # Match internal benchmark
            n_expression_bins=10,  # Match internal benchmark
            n_epochs=1000,
            raw_mode=False,  # Tokenized mode for Tier 2
            verbose=False,  # Added verbose=False to suppress SLAF's detailed timing prints
            # MoS with same configuration as internal benchmarks
            use_mixture_of_scanners=True,
            by_fragment=False,
            batches_per_chunk=1,
            n_scanners=16,
            prefetch_batch_size=1048576,  # 1M rows
        )

        # Warm up (increased for MoS stability)
        self.console.print("Warming up...")
        warmup_batches = 15  # Increased from 3 for MoS warmup
        for i, _batch in enumerate(dataloader):
            if i >= warmup_batches:
                break

        # Benchmark with peak memory tracking
        self.console.print("Running benchmark...")
        warmup_duration = 10  # Skip first 10 seconds for MoS stabilization
        measurement_duration = 30  # Then measure for 30 seconds

        total_cells, batch_count, elapsed_time, peak_memory = (
            self.benchmark_with_memory_tracking(
                dataloader, warmup_duration + measurement_duration, "SLAF Tier 2"
            )
        )

        # Clean up the dataloader to stop background processing
        del dataloader
        gc.collect()

        # Calculate metrics (using only measurement time, not warmup)
        measurement_time = elapsed_time - warmup_duration
        throughput_cells_per_sec = (
            total_cells / measurement_time if measurement_time > 0 else 0
        )

        # Debug output
        self.console.print(
            f"  Debug: {total_cells} cells, {measurement_time:.2f}s measurement, {throughput_cells_per_sec:.0f} cells/sec"
        )

        return ExternalBenchmarkResult(
            system_name="SLAF",
            tier="tier2",
            throughput_cells_per_sec=throughput_cells_per_sec,
            memory_usage_gb=peak_memory,
            output_type="tokenized",
            processes=1,
            measurement_time=measurement_time,
            total_cells=total_cells,
        )

    def benchmark_scdataset_tier1(self) -> ExternalBenchmarkResult | None:
        """Benchmark scDataset in Tier 1 (raw data loading)."""

        try:
            from anndata.experimental import AnnCollection
            from scdataset import scDataset
            from torch.utils.data import DataLoader
        except ImportError:
            self.console.print(
                "[yellow]Warning: scDataset not available, skipping scDataset benchmark[/yellow]"
            )
            return None

        self.console.print(
            "\n[bold blue]Benchmarking scDataset - Tier 1 (Raw Data Loading)[/bold blue]"
        )

        # Convert SLAF to AnnData for scDataset
        try:
            from anndata.experimental import AnnCollection
            from scdataset import BlockShuffling
            from torch.utils.data import DataLoader

            # Use the shared AnnData object
            collection = AnnCollection([self.adata])

            # Create scDataset with optimal parameters from the paper
            # Using BlockShuffling strategy (v0.2.0+ API)
            strategy = BlockShuffling(block_size=8)
            sc_dataset = scDataset(
                collection,  # First positional arg is data
                strategy,  # Second positional arg is strategy
                batch_size=self.batch_size,
                fetch_factor=64,
                fetch_transform=fetch_transform_adata,  # Use module-level callback
            )

            # Create DataLoader with single worker
            dataloader = DataLoader(
                sc_dataset,
                batch_size=None,
                num_workers=0,
                prefetch_factor=None,
            )
        except Exception as e:
            self.console.print(f"[red]Error setting up scDataset: {e}[/red]")
            return None

        # Warm up (increased for fair comparison)
        self.console.print("Warming up...")
        warmup_batches = 15  # Increased from 3 for fair comparison
        for i, _batch in enumerate(dataloader):
            if i >= warmup_batches:
                break

        # Benchmark with peak memory tracking
        self.console.print("Running benchmark...")
        warmup_duration = 10  # Skip first 10 seconds for fair comparison
        measurement_duration = 30  # Then measure for 30 seconds

        total_cells, batch_count, elapsed_time, peak_memory = (
            self.benchmark_with_memory_tracking(
                dataloader, warmup_duration + measurement_duration, "scDataset"
            )
        )

        # Clean up the dataloader to stop background processing
        del dataloader
        gc.collect()

        # Calculate metrics (using only measurement time, not warmup)
        measurement_time = elapsed_time - warmup_duration
        throughput_cells_per_sec = (
            total_cells / measurement_time if measurement_time > 0 else 0
        )

        # Debug output
        self.console.print(
            f"  Debug: {total_cells} cells, {measurement_time:.2f}s measurement, {throughput_cells_per_sec:.0f} cells/sec"
        )

        return ExternalBenchmarkResult(
            system_name="scDataset",
            tier="tier1",
            throughput_cells_per_sec=throughput_cells_per_sec,
            memory_usage_gb=peak_memory,
            output_type="raw",
            processes=0,  # Single worker
            measurement_time=measurement_time,
            total_cells=total_cells,
        )

    def benchmark_scdataset_tier2(self) -> ExternalBenchmarkResult | None:
        """Benchmark scDataset in Tier 2 (GPU-ready output)."""

        # scDataset only provides raw data, not GPU-ready output
        # So we don't report it in tier 2
        return None

    def benchmark_anndataloader_tier1(self) -> ExternalBenchmarkResult | None:
        """Benchmark AnnDataLoader in Tier 1 (raw data loading)."""

        try:
            from scvi.data import AnnDataManager
            from scvi.data.fields import LayerField
            from scvi.dataloaders import AnnDataLoader
        except ImportError:
            self.console.print(
                "[yellow]Warning: scvi-tools not available, skipping AnnDataLoader benchmark[/yellow]"
            )
            return None

        self.console.print(
            "\n[bold blue]Benchmarking AnnDataLoader - Tier 1 (Raw Data Loading)[/bold blue]"
        )

        # Convert SLAF to AnnData for AnnDataLoader
        try:
            from scvi.data import AnnDataManager
            from scvi.data.fields import LayerField
            from scvi.dataloaders import AnnDataLoader

            # Use the shared AnnData object
            anndata_fields = [
                LayerField(registry_key="X", layer=None, is_count_data=True)
            ]
            adata_manager = AnnDataManager(fields=anndata_fields)
            adata_manager.register_fields(self.adata)

            dataloader = AnnDataLoader(
                adata_manager,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                load_sparse_tensor=True,
            )
        except Exception as e:
            self.console.print(f"[red]Error setting up AnnDataLoader: {e}[/red]")
            return None

        # Warm up (increased for fair comparison)
        self.console.print("Warming up...")
        warmup_batches = 15  # Increased from 3 for fair comparison
        for i, _batch in enumerate(dataloader):
            if i >= warmup_batches:
                break

        # Benchmark with peak memory tracking
        self.console.print("Running benchmark...")
        warmup_duration = 10  # Skip first 10 seconds for fair comparison
        measurement_duration = 30  # Then measure for 30 seconds

        total_cells, batch_count, elapsed_time, peak_memory = (
            self.benchmark_with_memory_tracking(
                dataloader, warmup_duration + measurement_duration, "AnnDataLoader"
            )
        )

        # Clean up the dataloader to stop background processing
        del dataloader
        gc.collect()

        # Calculate metrics (using only measurement time, not warmup)
        measurement_time = elapsed_time - warmup_duration
        throughput_cells_per_sec = (
            total_cells / measurement_time if measurement_time > 0 else 0
        )

        # Debug output
        self.console.print(
            f"  Debug: {total_cells} cells, {measurement_time:.2f}s measurement, {throughput_cells_per_sec:.0f} cells/sec"
        )

        return ExternalBenchmarkResult(
            system_name="AnnDataLoader",
            tier="tier1",
            throughput_cells_per_sec=throughput_cells_per_sec,
            memory_usage_gb=peak_memory,
            output_type="raw",
            processes=1,  # We're running single process
            measurement_time=measurement_time,
            total_cells=total_cells,
        )

    def benchmark_anndataloader_tier2(self) -> ExternalBenchmarkResult | None:
        """Benchmark AnnDataLoader in Tier 2 (GPU-ready output)."""

        # AnnDataLoader only provides raw data, not GPU-ready output
        # So we don't report it in tier 2
        return None

    def benchmark_annloader_tier1(self) -> ExternalBenchmarkResult | None:
        """Benchmark AnnLoader in Tier 1 (raw data loading)."""

        try:
            from anndata.experimental import AnnLoader
        except ImportError:
            self.console.print(
                "[yellow]Warning: AnnLoader not available (requires anndata experimental features), skipping AnnLoader benchmark[/yellow]"
            )
            return None

        self.console.print(
            "\n[bold blue]Benchmarking AnnLoader - Tier 1 (Raw Data Loading)[/bold blue]"
        )

        # Setup AnnLoader
        try:
            from anndata.experimental import AnnLoader

            dataloader = AnnLoader(
                adatas=[self.adata],
                batch_size=self.batch_size,
                shuffle=True,
                use_default_converter=True,
            )
        except Exception as e:
            self.console.print(f"[red]Error setting up AnnLoader: {e}[/red]")
            return None

        # Warm up (increased for fair comparison)
        self.console.print("Warming up...")
        warmup_batches = 15  # Increased from 3 for fair comparison
        for i, _batch in enumerate(dataloader):
            if i >= warmup_batches:
                break

        # Benchmark with peak memory tracking
        self.console.print("Running benchmark...")
        warmup_duration = 10  # Skip first 10 seconds for fair comparison
        measurement_duration = 30  # Then measure for 30 seconds

        total_cells, batch_count, elapsed_time, peak_memory = (
            self.benchmark_with_memory_tracking(
                dataloader, warmup_duration + measurement_duration, "AnnLoader"
            )
        )

        # Clean up the dataloader to stop background processing
        del dataloader
        gc.collect()

        # Calculate metrics (using only measurement time, not warmup)
        measurement_time = elapsed_time - warmup_duration
        throughput_cells_per_sec = (
            total_cells / measurement_time if measurement_time > 0 else 0
        )

        # Debug output
        self.console.print(
            f"  Debug: {total_cells} cells, {measurement_time:.2f}s measurement, {throughput_cells_per_sec:.0f} cells/sec"
        )

        return ExternalBenchmarkResult(
            system_name="AnnLoader",
            tier="tier1",
            throughput_cells_per_sec=throughput_cells_per_sec,
            memory_usage_gb=peak_memory,
            output_type="raw",
            processes=1,
            measurement_time=measurement_time,
            total_cells=total_cells,
        )

    def benchmark_annloader_tier2(self) -> ExternalBenchmarkResult | None:
        """Benchmark AnnLoader in Tier 2 (GPU-ready output)."""

        # AnnLoader only provides raw data, not GPU-ready output
        # So we don't report it in tier 2
        return None

    def benchmark_tiledbloader_tier1(self) -> ExternalBenchmarkResult | None:
        """Benchmark TileDB DataLoader in Tier 1 (raw data loading)."""

        try:
            from slaf.ml.tiledb_dataloaders import TileDBDataLoader
        except ImportError:
            self.console.print(
                "[yellow]Warning: TileDB dataloader not available, skipping TileDB benchmark[/yellow]"
            )
            return None

        self.console.print(
            "\n[bold blue]Benchmarking TileDB DataLoader - Tier 1 (Raw Data Loading with MoS)[/bold blue]"
        )

        # Setup TileDB DataLoader with MoS enabled
        try:
            if not self.tiledb_path:
                self.console.print(
                    "[yellow]Warning: TileDB path not provided, skipping TileDB benchmark[/yellow]"
                )
                return None

            dataloader = TileDBDataLoader(
                tiledb_path=self.tiledb_path,
                batch_size=self.batch_size,
                prefetch_batch_size=100,  # Updated for 50k cell dataset
                seed=42,
                n_epochs=1000,  # Match SLAF's n_epochs for continuous streaming
                verbose=False,  # Suppress verbose output for clean benchmark
                max_queue_size=500,
                use_mixture_of_scanners=True,  # Enable MoS for fair comparison
                n_readers=50,  # Default for 50k cells
                n_scanners=8,  # Default
            )
        except Exception as e:
            self.console.print(f"[red]Error setting up TileDB DataLoader: {e}[/red]")
            return None

        # Warm up (increased for fair comparison)
        self.console.print("Warming up...")
        warmup_batches = 15  # Increased from 3 for fair comparison
        for i, _batch in enumerate(dataloader):
            if i >= warmup_batches:
                break

        # Benchmark with peak memory tracking
        self.console.print("Running benchmark...")
        warmup_duration = 10  # Skip first 10 seconds for fair comparison
        measurement_duration = 30  # Then measure for 30 seconds

        total_cells, batch_count, elapsed_time, peak_memory = (
            self.benchmark_with_memory_tracking(
                dataloader, warmup_duration + measurement_duration, "TileDB DataLoader"
            )
        )

        # Clean up the dataloader to stop background processing
        del dataloader
        gc.collect()

        # Calculate metrics (using only measurement time, not warmup)
        measurement_time = elapsed_time - warmup_duration
        throughput_cells_per_sec = (
            total_cells / measurement_time if measurement_time > 0 else 0
        )

        # Debug output
        self.console.print(
            f"  Debug: {total_cells} cells, {measurement_time:.2f}s measurement, {throughput_cells_per_sec:.0f} cells/sec"
        )

        return ExternalBenchmarkResult(
            system_name="TileDB DataLoader (MoS)",
            tier="tier1",
            throughput_cells_per_sec=throughput_cells_per_sec,
            memory_usage_gb=peak_memory,
            output_type="raw",
            processes=1,  # Single process
            measurement_time=measurement_time,
            total_cells=total_cells,
        )

    def benchmark_tiledbloader_tier2(self) -> ExternalBenchmarkResult | None:
        """Benchmark TileDB DataLoader in Tier 2 (GPU-ready output)."""

        # TileDB DataLoader only provides raw data, not GPU-ready output
        # So we don't report it in tier 2
        return None

    def benchmark_scdl_tier1(self) -> ExternalBenchmarkResult | None:
        """Benchmark BioNeMo SCDL in Tier 1 (raw data loading)."""

        try:
            from bionemo.scdl.io.single_cell_memmap_dataset import (
                SingleCellMemMapDataset,
            )
            from bionemo.scdl.util.torch_dataloader_utils import (
                collate_sparse_matrix_batch,
            )
            from torch.utils.data import DataLoader
        except ImportError:
            self.console.print(
                "[yellow]Warning: BioNeMo SCDL not available, skipping SCDL benchmark[/yellow]"
            )
            return None

        self.console.print(
            "\n[bold blue]Benchmarking BioNeMo SCDL - Tier 1 (Raw Data Loading)[/bold blue]"
        )

        # Setup SCDL DataLoader
        try:
            if not self.scdl_path:
                self.console.print(
                    "[yellow]Warning: SCDL path not provided, skipping SCDL benchmark[/yellow]"
                )
                return None

            # Load SCDL dataset
            scdl_dataset = SingleCellMemMapDataset(self.scdl_path)

            # Create DataLoader with collation function for sparse matrices
            # Note: SCDL doesn't support infinite epochs, so we'll create a wrapper
            dataloader = DataLoader(
                scdl_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_sparse_matrix_batch,
                num_workers=0,  # Single worker for fair comparison
            )
        except Exception as e:
            self.console.print(f"[red]Error setting up BioNeMo SCDL: {e}[/red]")
            return None

        # Create a wrapper to make SCDL dataloader run continuously
        def infinite_scdl_dataloader(dl):
            """Create an infinite iterator over SCDL dataloader."""
            while True:
                yield from dl

        # Warm up (increased for fair comparison)
        self.console.print("Warming up...")
        warmup_batches = 15  # Increased from 3 for fair comparison
        infinite_dataloader = infinite_scdl_dataloader(dataloader)
        for i, _batch in enumerate(infinite_dataloader):
            if i >= warmup_batches:
                break

        # Benchmark with peak memory tracking
        self.console.print("Running benchmark...")
        warmup_duration = 10  # Skip first 10 seconds for fair comparison
        measurement_duration = 30  # Then measure for 30 seconds

        total_cells, batch_count, elapsed_time, peak_memory = (
            self.benchmark_with_memory_tracking(
                infinite_dataloader,
                warmup_duration + measurement_duration,
                "BioNeMo SCDL",
            )
        )

        # Clean up the dataloader to stop background processing
        del dataloader
        gc.collect()

        # Calculate metrics (using only measurement time, not warmup)
        measurement_time = elapsed_time - warmup_duration
        throughput_cells_per_sec = (
            total_cells / measurement_time if measurement_time > 0 else 0
        )

        # Debug output
        self.console.print(
            f"  Debug: {total_cells} cells, {measurement_time:.2f}s measurement, {throughput_cells_per_sec:.0f} cells/sec"
        )

        return ExternalBenchmarkResult(
            system_name="BioNeMo SCDL",
            tier="tier1",
            throughput_cells_per_sec=throughput_cells_per_sec,
            memory_usage_gb=peak_memory,
            output_type="raw",
            processes=1,  # Single worker
            measurement_time=measurement_time,
            total_cells=total_cells,
        )

    def benchmark_scdl_tier2(self) -> ExternalBenchmarkResult | None:
        """Benchmark BioNeMo SCDL in Tier 2 (GPU-ready output)."""

        # BioNeMo SCDL only provides raw data, not GPU-ready output
        # So we don't report it in tier 2
        return None

    def benchmark_annbatch_tier1(self) -> ExternalBenchmarkResult | None:
        """Benchmark annbatch in Tier 1 (raw data loading)."""

        try:
            from pathlib import Path

            import anndata as ad
            import zarr
            from annbatch import ZarrSparseDataset
        except ImportError:
            self.console.print(
                "[yellow]Warning: annbatch not available, skipping annbatch benchmark[/yellow]"
            )
            return None

        self.console.print(
            "\n[bold blue]Benchmarking annbatch - Tier 1 (Raw Data Loading)[/bold blue]"
        )

        # Setup annbatch dataloader
        try:
            if not self.annbatch_path:
                self.console.print(
                    "[yellow]Warning: annbatch path not provided, skipping annbatch benchmark[/yellow]"
                )
                return None

            # Configure zarr to use zarrs-python for performance
            # Import zarrs first to register the codec pipeline
            try:
                import zarrs  # noqa: F401

                zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
            except (ImportError, Exception):
                # Continue without zarrs if not available
                self.console.print(
                    "[yellow]Warning: zarrs-python not available. Performance may be reduced.[/yellow]"
                )
                pass

            # Find all zarr files in the collection directory
            collection_path = Path(self.annbatch_path)
            zarr_files = list(collection_path.glob("dataset_*.zarr"))

            if not zarr_files:
                self.console.print(
                    f"[yellow]Warning: No zarr files found in {self.annbatch_path}, skipping annbatch benchmark[/yellow]"
                )
                return None

            # Create AnnData objects from zarr files
            anndatas = [
                ad.AnnData(
                    X=ad.io.sparse_dataset(zarr.open(str(p))["X"]),
                    obs=ad.io.read_elem(zarr.open(str(p))["obs"]),
                )
                for p in zarr_files
            ]

            # Create annbatch dataloader
            # Using recommended parameters from annbatch docs
            dataloader = ZarrSparseDataset(
                batch_size=self.batch_size,
                chunk_size=32,  # Number of rows to load sequentially
                preload_nchunks=256,  # Number of contiguous chunks to preload
                preload_to_gpu=False,
                to_torch=True,
            ).add_anndatas(
                anndatas,
                obs_keys=None,  # No specific obs keys needed for benchmark
            )
        except Exception as e:
            self.console.print(f"[red]Error setting up annbatch: {e}[/red]")
            import traceback

            traceback.print_exc()
            return None

        # Warm up (increased for fair comparison)
        self.console.print("Warming up...")
        warmup_batches = 15  # Increased from 3 for fair comparison
        for i, _batch in enumerate(dataloader):
            if i >= warmup_batches:
                break

        # Benchmark with peak memory tracking
        self.console.print("Running benchmark...")
        warmup_duration = 10  # Skip first 10 seconds for fair comparison
        measurement_duration = 30  # Then measure for 30 seconds

        total_cells, batch_count, elapsed_time, peak_memory = (
            self.benchmark_with_memory_tracking(
                dataloader, warmup_duration + measurement_duration, "annbatch"
            )
        )

        # Clean up the dataloader to stop background processing
        del dataloader
        gc.collect()

        # Calculate metrics (using only measurement time, not warmup)
        measurement_time = elapsed_time - warmup_duration
        throughput_cells_per_sec = (
            total_cells / measurement_time if measurement_time > 0 else 0
        )

        # Debug output
        self.console.print(
            f"  Debug: {total_cells} cells, {measurement_time:.2f}s measurement, {throughput_cells_per_sec:.0f} cells/sec"
        )

        return ExternalBenchmarkResult(
            system_name="annbatch",
            tier="tier1",
            throughput_cells_per_sec=throughput_cells_per_sec,
            memory_usage_gb=peak_memory,
            output_type="raw",
            processes=1,  # Single process
            measurement_time=measurement_time,
            total_cells=total_cells,
        )

    def benchmark_annbatch_tier2(self) -> ExternalBenchmarkResult | None:
        """Benchmark annbatch in Tier 2 (GPU-ready output)."""

        # annbatch only provides raw data, not GPU-ready output
        # So we don't report it in tier 2
        return None

    def run_benchmarks(self) -> list[ExternalBenchmarkResult]:
        """Run benchmarks for all external dataloaders."""

        self.console.print(
            Panel.fit(
                "[bold green]SLAF External Dataloader Benchmarks[/bold green]\n"
                "Comparing SLAF against state-of-the-art dataloaders with MoS strategy\n"
                "(15 warmup batches + 10s warmup + 30s measurement)",
                border_style="green",
            )
        )

        results = []

        # Run SLAF benchmarks
        try:
            slaf_tier1 = self.benchmark_slaf_tier1()
            if slaf_tier1:
                results.append(slaf_tier1)

            slaf_tier2 = self.benchmark_slaf_tier2()
            if slaf_tier2:
                results.append(slaf_tier2)
        except Exception as e:
            self.console.print(f"[red]Error benchmarking SLAF: {e}[/red]")

        # Run scDataset benchmarks
        try:
            scdataset_tier1 = self.benchmark_scdataset_tier1()
            if scdataset_tier1:
                results.append(scdataset_tier1)

            scdataset_tier2 = self.benchmark_scdataset_tier2()
            if scdataset_tier2:
                results.append(scdataset_tier2)
        except Exception as e:
            self.console.print(f"[red]Error benchmarking scDataset: {e}[/red]")

        # Run AnnDataLoader benchmarks
        try:
            anndataloader_tier1 = self.benchmark_anndataloader_tier1()
            if anndataloader_tier1:
                results.append(anndataloader_tier1)

            anndataloader_tier2 = self.benchmark_anndataloader_tier2()
            if anndataloader_tier2:
                results.append(anndataloader_tier2)
        except Exception as e:
            self.console.print(f"[red]Error benchmarking AnnDataLoader: {e}[/red]")

        # Run AnnLoader benchmarks
        try:
            annloader_tier1 = self.benchmark_annloader_tier1()
            if annloader_tier1:
                results.append(annloader_tier1)

            annloader_tier2 = self.benchmark_annloader_tier2()
            if annloader_tier2:
                results.append(annloader_tier2)
        except Exception as e:
            self.console.print(f"[red]Error benchmarking AnnLoader: {e}[/red]")

        # Run TileDB DataLoader benchmarks (MoS mode)
        try:
            tiledbloader_tier1 = self.benchmark_tiledbloader_tier1()
            if tiledbloader_tier1:
                results.append(tiledbloader_tier1)

            tiledbloader_tier2 = self.benchmark_tiledbloader_tier2()
            if tiledbloader_tier2:
                results.append(tiledbloader_tier2)
        except Exception as e:
            self.console.print(f"[red]Error benchmarking TileDB DataLoader: {e}[/red]")

        # Run BioNeMo SCDL benchmarks
        try:
            scdl_tier1 = self.benchmark_scdl_tier1()
            if scdl_tier1:
                results.append(scdl_tier1)

            scdl_tier2 = self.benchmark_scdl_tier2()
            if scdl_tier2:
                results.append(scdl_tier2)
        except Exception as e:
            self.console.print(f"[red]Error benchmarking BioNeMo SCDL: {e}[/red]")

        # Run annbatch benchmarks
        try:
            annbatch_tier1 = self.benchmark_annbatch_tier1()
            if annbatch_tier1:
                results.append(annbatch_tier1)

            annbatch_tier2 = self.benchmark_annbatch_tier2()
            if annbatch_tier2:
                results.append(annbatch_tier2)
        except Exception as e:
            self.console.print(f"[red]Error benchmarking annbatch: {e}[/red]")

        return results

    def print_results(self, results: list[ExternalBenchmarkResult]):
        """Print benchmark results in formatted tables."""

        # Separate results by tier
        tier1_results = [r for r in results if r.tier == "tier1"]
        tier2_results = [r for r in results if r.tier == "tier2"]

        # Tier 1 Results Table
        self.console.print(
            "\n[bold blue]Tier 1: Raw Data Loading Comparison[/bold blue]"
        )
        table1 = Table(title="Raw Data Loading Performance (cells/sec)")

        table1.add_column("System", style="cyan")
        table1.add_column("Throughput (cells/sec)", style="green")

        for result in tier1_results:
            # Color code the throughput
            throughput_text = f"{result.throughput_cells_per_sec:,.0f}"
            if result.throughput_cells_per_sec > 10000:
                throughput_text = f"[green]{throughput_text}[/green]"
            elif result.throughput_cells_per_sec > 5000:
                throughput_text = f"[yellow]{throughput_text}[/yellow]"

            table1.add_row(
                result.system_name,
                throughput_text,
            )

        self.console.print(table1)

        # Tier 2 Results Table
        self.console.print(
            "\n[bold blue]Tier 2: GPU-Ready Output Comparison[/bold blue]"
        )
        table2 = Table(title="GPU-Ready Output Performance")

        table2.add_column("System", style="cyan")
        table2.add_column("Throughput (cells/sec)", style="green")
        table2.add_column("Throughput (tokens/sec)", style="green")

        for result in tier2_results:
            # Color code the throughput
            throughput_text = f"{result.throughput_cells_per_sec:,.0f}"
            if result.throughput_cells_per_sec > 10000:
                throughput_text = f"[green]{throughput_text}[/green]"
            elif result.throughput_cells_per_sec > 5000:
                throughput_text = f"[yellow]{throughput_text}[/yellow]"

            # Calculate tokens per second for SLAF (approximate)
            if result.system_name == "SLAF" and result.output_type == "tokenized":
                # Geneformer: ~2048 tokens per cell (max_genes)
                tokens_per_cell = 2048
                tokens_per_sec = result.throughput_cells_per_sec * tokens_per_cell
                tokens_text = f"{tokens_per_sec:,.0f}"
            else:
                tokens_text = "N/A"

            table2.add_row(
                result.system_name,
                throughput_text,
                tokens_text,
            )

        self.console.print(table2)

        # Summary statistics
        self.console.print("\n[bold]Summary:[/bold]")

        # Find SLAF results
        slaf_tier1 = next((r for r in tier1_results if r.system_name == "SLAF"), None)
        slaf_tier2 = next((r for r in tier2_results if r.system_name == "SLAF"), None)

        if slaf_tier1 and slaf_tier2:
            self.console.print(
                f"✓ SLAF Tier 1 (Raw): {slaf_tier1.throughput_cells_per_sec:,.0f} cells/sec"
            )
            self.console.print(
                f"✓ SLAF Tier 2 (Tokenized): {slaf_tier2.throughput_cells_per_sec:,.0f} cells/sec"
            )

            # Calculate and show tokens/sec for SLAF
            tokens_per_cell = 2048  # Geneformer tokens per cell
            slaf_tokens_per_sec = slaf_tier2.throughput_cells_per_sec * tokens_per_cell
            self.console.print(
                f"✓ SLAF Token Throughput: {slaf_tokens_per_sec:,.0f} tokens/sec"
            )

            # Compare with scDataset
            scdataset_tier1 = next(
                (r for r in tier1_results if r.system_name == "scDataset"), None
            )
            if scdataset_tier1:
                advantage = (
                    slaf_tier1.throughput_cells_per_sec
                    / scdataset_tier1.throughput_cells_per_sec
                    - 1
                ) * 100
                self.console.print(
                    f"✓ SLAF vs scDataset Tier 1: {advantage:+.1f}% advantage"
                )


def main():
    """Run the external dataloader benchmarks."""

    # Centralized dataset paths
    slaf_path = "../slaf-datasets/plate1_Tahoe100M_v21.slaf"
    h5ad_path = "../slaf-datasets/plate1_Tahoe100M.h5ad"
    tiledb_path = "../slaf-datasets/synthetic_50k_processed.tiledb"
    scdl_path = "../slaf-datasets/plate1_Tahoe100M.scdl"
    annbatch_path = "../slaf-datasets/plate1_Tahoe100M_annbatch"

    benchmark = ExternalDataloaderBenchmark(
        slaf_path, h5ad_path, tiledb_path, scdl_path, annbatch_path
    )
    results = benchmark.run_benchmarks()
    benchmark.print_results(results)


if __name__ == "__main__":
    main()
