import argparse
import gc
import multiprocessing as mp
import time

import numpy as np
from benchmark_utils import get_object_memory_usage, get_slaf_memory_usage
from rich.console import Console
from rich.table import Table

from slaf.core.slaf import SLAFArray
from slaf.ml.dataloaders import SLAFDataLoader
from slaf.ml.tokenizers import SLAFTokenizer


# Global worker functions for multiprocessing
def worker_h5ad(worker_id, h5ad_path, scenario, result_queue):
    """Worker function for h5ad multi-process benchmark"""
    import numpy as np
    import scanpy as sc
    from benchmark_utils import get_object_memory_usage

    # Each process loads the full h5ad file
    adata = sc.read_h5ad(h5ad_path)

    # Simulate processing the scenario
    batch_size = scenario["batch_size"]
    max_genes = scenario["max_genes"]

    # Process one batch per worker (sufficient for benchmarking)
    n_cells = len(adata)
    worker_start = (worker_id * batch_size) % n_cells
    worker_end = min(worker_start + batch_size, n_cells)

    # Start timing AFTER data loading
    start_time = time.time()

    try:
        # Get expression data for this batch
        if hasattr(adata, "X") and adata.X is not None:
            # Convert to dense array if it's sparse
            try:
                batch_data = adata.X[worker_start:worker_end, :].toarray()
            except AttributeError:
                batch_data = np.asarray(adata.X[worker_start:worker_end, :])

            # Implement the same tokenization logic as SLAF
            # Define special tokens (same as SLAF)
            special_tokens = {
                "CLS": 0,  # Start of sequence
                "SEP": 1,  # End of sequence
                "PAD": 2,  # Padding token
                "UNK": 3,  # Unknown gene
            }

            # Expression bin tokens start after special tokens
            n_expression_bins = 10
            expr_bin_start = len(special_tokens)  # 4

            # Build gene vocabulary (same as SLAF)
            gene_ids = adata.var.index.tolist()
            vocab_size = min(50000, len(gene_ids))

            gene_vocab = {}
            for i, gene_id in enumerate(gene_ids[:vocab_size]):
                # Token IDs start after special tokens and expression bins
                gene_vocab[gene_id] = i + n_expression_bins + len(special_tokens)

            # Expression binning function (same as SLAF)
            def expression_to_bin(expression_value):
                if expression_value <= 0:
                    return special_tokens["PAD"]

                # Log transform and bin
                log_expr = np.log1p(expression_value)

                # Simple binning: assume log expression ranges from 0 to ~10
                bin_id = min(
                    n_expression_bins - 1, int(log_expr * n_expression_bins / 10)
                )

                return expr_bin_start + bin_id

            # Tokenize using the same logic as SLAF geneformer
            token_sequences = []
            for cell_expr in batch_data:
                # Get non-zero genes sorted by expression (descending)
                non_zero_mask = cell_expr > 0
                non_zero_indices = np.where(non_zero_mask)[0]
                non_zero_expr = cell_expr[non_zero_mask]

                if len(non_zero_indices) == 0:
                    # No expression - all PAD tokens
                    tokens = [special_tokens["PAD"]] * max_genes
                else:
                    # Sort by expression (descending)
                    sorted_indices = non_zero_indices[np.argsort(-non_zero_expr)]

                    # Limit to max_genes
                    if len(sorted_indices) > max_genes:
                        sorted_indices = sorted_indices[:max_genes]

                    # Create tokens using the same mapping as SLAF
                    tokens = []
                    for gene_idx in sorted_indices:
                        gene_id = adata.var.index[gene_idx]
                        gene_token = gene_vocab.get(gene_id, special_tokens["UNK"])
                        tokens.append(gene_token)

                    # Pad to max_genes
                    if len(tokens) < max_genes:
                        tokens.extend(
                            [special_tokens["PAD"]] * (max_genes - len(tokens))
                        )
                    else:
                        tokens = tokens[:max_genes]

                token_sequences.append(tokens)

            # Convert to numpy array (same as SLAF dataloader output)
            token_array = np.array(token_sequences, dtype=np.int64)

        else:
            # Fallback if X is not available
            _ = adata.obs.iloc[worker_start:worker_end]
            token_array = np.array([], dtype=np.int64)
    except Exception:
        # If any processing fails, just continue
        token_array = np.array([], dtype=np.int64)

    # End timing BEFORE memory measurement
    end_time = time.time()

    # Measure memory AFTER timing (separate from performance measurement)
    # Each h5ad process loads the full dataset
    full_dataset_memory = (
        get_object_memory_usage(adata.X)
        + get_object_memory_usage(adata.obs)
        + get_object_memory_usage(adata.var)
    )

    # Estimate token memory based on batch size and max_genes
    estimated_token_memory = (
        batch_size * max_genes * 8 / (1024 * 1024)
    )  # 8 bytes per int64
    total_memory = full_dataset_memory + estimated_token_memory

    # Just return basic info about the token array to avoid expensive queue transfer
    token_info = {
        "shape": token_array.shape,
        "dtype": str(token_array.dtype),
        "sample_values": (
            token_array.flatten()[:5].tolist() if token_array.size > 0 else []
        ),
    }

    result_queue.put((end_time - start_time, total_memory, token_info))


def worker_slaf(worker_id, slaf_path, scenario, result_queue):
    """Worker function for SLAF multi-process benchmark"""
    from benchmark_utils import get_object_memory_usage

    from slaf.core.slaf import SLAFArray
    from slaf.ml.dataloaders import SLAFDataLoader

    # Each process opens SLAF independently (concurrent disk reads)
    slaf = SLAFArray(slaf_path)

    # Create dataloader for this scenario
    dataloader = SLAFDataLoader(
        slaf_array=slaf,
        tokenizer_type=scenario["tokenizer_type"],
        batch_size=scenario["batch_size"],
        max_genes=scenario["max_genes"],
    )

    # Process one batch per worker (sufficient for benchmarking)
    # Start timing AFTER data loading
    start_time = time.time()

    # Get the first batch
    batch = next(iter(dataloader))

    # Extract token sequences (SLAF dataloader already returns numpy arrays when torch not available)
    if "input_ids" in batch:
        token_array = batch["input_ids"]  # Already numpy array
    else:
        token_array = np.array([], dtype=np.int64)

    # End timing BEFORE memory measurement
    end_time = time.time()

    # Measure memory AFTER timing (separate from performance measurement)
    # Metadata is shared across processes, but each process has its own token data
    metadata_memory = get_object_memory_usage(slaf.obs) + get_object_memory_usage(
        slaf.var
    )

    # Estimate token memory based on batch size and max_genes
    estimated_token_memory = (
        scenario["batch_size"] * scenario["max_genes"] * 8 / (1024 * 1024)
    )  # 8 bytes per int64

    # Total memory should include metadata + estimated token representation
    total_memory = metadata_memory + estimated_token_memory

    # Just return basic info about the token array to avoid expensive queue transfer
    token_info = {
        "shape": token_array.shape,
        "dtype": str(token_array.dtype),
        "sample_values": (
            token_array.flatten()[:5].tolist() if token_array.size > 0 else []
        ),
    }

    result_queue.put((end_time - start_time, total_memory, token_info))


def benchmark_multi_process_scaling(
    h5ad_path: str, slaf_path: str, max_processes: int = 8, verbose: bool = True
) -> list:
    """Benchmark multi-process scaling comparing h5ad vs SLAF

    This demonstrates SLAF's advantage in concurrent disk reads vs h5ad's memory duplication.

    Note: This benchmark measures the average processing time per process (excluding
    process startup overhead), which reflects real-world distributed training scenarios
    where each node/process trains independently and synchronizes weights periodically.
    This provides a fair comparison with single-process tokenizer benchmarks.

    Args:
        h5ad_path: Path to h5ad file
        slaf_path: Path to SLAF file
        max_processes: Maximum number of processes to test
        verbose: Enable verbose output

    Returns:
        List of benchmark results in the expected format
    """
    console = Console()

    if verbose:
        console.print("[bold blue]Multi-Process Scaling Benchmark[/bold blue]")
        console.print("Testing 1, 4, and 8 processes\n")

    # Test scenarios
    scenarios = [
        {
            "name": "Small batch",
            "batch_size": 32,
            "max_genes": 512,
            "tokenizer_type": "geneformer",
            "description": "Small batch (32 cells, 512 genes)",
        },
        {
            "name": "Medium batch",
            "batch_size": 128,
            "max_genes": 1024,
            "tokenizer_type": "geneformer",
            "description": "Medium batch (128 cells, 1024 genes)",
        },
        {
            "name": "Large batch",
            "batch_size": 512,
            "max_genes": 2048,
            "tokenizer_type": "geneformer",
            "description": "Large batch (512 cells, 2048 genes)",
        },
    ]

    results = []

    # Test specific process counts for faster debugging
    process_counts = [1, 4, 8] if max_processes >= 8 else [1, 2, 4]

    for scenario in scenarios:
        scenario_name = scenario["name"]
        if verbose:
            console.print(f"Scenario: {scenario['description']}")

        for n_proc in process_counts:
            if verbose:
                console.print(f"  Testing {n_proc} process(es)...")

            try:
                # Benchmark h5ad
                h5ad_time, h5ad_total_mem, h5ad_token_info = (
                    _benchmark_h5ad_multi_process(h5ad_path, scenario, n_proc)
                )

                # Benchmark SLAF
                slaf_time, slaf_total_mem, slaf_token_info = (
                    _benchmark_slaf_multi_process(slaf_path, scenario, n_proc)
                )

                # Verify that both workers produce identical token sequences
                if verbose and h5ad_token_info["shape"] == slaf_token_info["shape"]:
                    try:
                        # Compare content (allow for small differences in gene ranking due to ties)
                        # For geneformer tokenization, we care about the top genes being similar
                        h5ad_top_genes = h5ad_token_info["sample_values"][
                            : min(10, len(h5ad_token_info["sample_values"]))
                        ]
                        slaf_top_genes = slaf_token_info["sample_values"][
                            : min(10, len(slaf_token_info["sample_values"]))
                        ]

                        # Check if at least 70% of top genes match (accounting for ties)
                        matches = 0
                        total_genes = len(h5ad_top_genes)
                        for i in range(total_genes):
                            if h5ad_top_genes[i] in slaf_top_genes:
                                matches += 1

                        match_ratio = matches / total_genes if total_genes > 0 else 1.0
                        if match_ratio < 0.7:
                            print(
                                f"WARNING: Token sequences differ significantly (match ratio: {match_ratio:.2f})"
                            )
                            print(f"  h5ad top genes: {h5ad_top_genes[:5]}")
                            print(f"  SLAF top genes: {slaf_top_genes[:5]}")
                        else:
                            print(
                                f"✓ Token sequences match well (match ratio: {match_ratio:.2f})"
                            )

                    except Exception as e:
                        print(f"WARNING: Could not verify token sequences: {e}")

                # Calculate speedup and memory efficiency
                speedup = h5ad_time / slaf_time if slaf_time > 0 else 0
                mem_efficiency = (
                    h5ad_total_mem / slaf_total_mem if slaf_total_mem > 0 else 0
                )

                if verbose:
                    console.print(
                        f"    h5ad: {h5ad_time:.1f}ms, {h5ad_total_mem:.1f}MB total | "
                        f"SLAF: {slaf_time:.1f}ms, {slaf_total_mem:.1f}MB total | "
                        f"Speedup: {speedup:.1f}x, Mem: {mem_efficiency:.1f}x"
                    )

                # Store result in expected format
                results.append(
                    {
                        "scenario": f"{scenario_name} ({n_proc} processes)",
                        "h5ad_time_ms": h5ad_time,
                        "slaf_time_ms": slaf_time,
                        "total_speedup": speedup,
                        "h5ad_total_memory_mb": h5ad_total_mem,  # Total across all processes
                        "slaf_total_memory_mb": slaf_total_mem,  # Total across all processes
                        "memory_efficiency": mem_efficiency,
                        "processes": n_proc,
                        "batch_size": scenario["batch_size"],
                        "max_genes": scenario["max_genes"],
                    }
                )

            except Exception as e:
                if verbose:
                    console.print(f"    ❌ Failed: {e}")
                continue

    if verbose and results:
        print_multi_process_results_table_simple(results)

    return results


def _benchmark_h5ad_multi_process(
    h5ad_path: str, scenario: dict, n_processes: int
) -> tuple:
    """Benchmark h5ad with multiple processes"""
    # Run multiple processes
    result_queue: mp.Queue = mp.Queue()
    processes = []

    for i in range(n_processes):
        p = mp.Process(target=worker_h5ad, args=(i, h5ad_path, scenario, result_queue))
        processes.append(p)
        p.start()

    # Wait for all processes
    for p in processes:
        p.join()

    # Collect results
    times = []
    memories = []
    token_infos = []
    while not result_queue.empty():
        t, m, token_info = result_queue.get()
        times.append(t)
        memories.append(m)
        token_infos.append(token_info)

    # Return average processing time (excluding process startup) and total memory
    # Convert to ms for consistency with other benchmarks
    avg_time_ms = (
        np.mean(times) * 1000
    )  # Average time per process (realistic for distributed training)
    total_memory = np.sum(memories)  # Total memory across all processes

    return avg_time_ms, total_memory, token_infos[0]


def _benchmark_slaf_multi_process(
    slaf_path: str, scenario: dict, n_processes: int
) -> tuple:
    """Benchmark SLAF with multiple processes"""
    # Run multiple processes
    result_queue: mp.Queue = mp.Queue()
    processes = []

    for i in range(n_processes):
        p = mp.Process(target=worker_slaf, args=(i, slaf_path, scenario, result_queue))
        processes.append(p)
        p.start()

    # Wait for all processes
    for p in processes:
        p.join()

    # Collect results
    times = []
    memories = []
    token_infos = []
    while not result_queue.empty():
        t, m, token_info = result_queue.get()
        times.append(t)
        memories.append(m)
        token_infos.append(token_info)

    # Return average processing time (excluding process startup) and total memory
    # Convert to ms for consistency with other benchmarks
    avg_time_ms = (
        np.mean(times) * 1000
    )  # Average time per process (realistic for distributed training)
    total_memory = np.sum(memories)  # Total memory across all processes

    return avg_time_ms, total_memory, token_infos[0]


def print_multi_process_results_table_simple(results):
    """Print simplified multi-process benchmark results table"""
    console = Console()

    table = Table(
        title="MULTI-PROCESS SCALING BENCHMARK RESULTS",
        show_header=True,
        header_style="bold magenta",
        title_style="bold blue",
    )

    table.add_column("Scenario", style="cyan", width=35)
    table.add_column("h5ad Time (ms)", justify="right", style="red", width=12)
    table.add_column("SLAF Time (ms)", justify="right", style="green", width=12)
    table.add_column("Speedup", justify="right", style="yellow", width=8)
    table.add_column("h5ad Total Mem (MB)", justify="right", style="red", width=14)
    table.add_column("SLAF Total Mem (MB)", justify="right", style="green", width=14)
    table.add_column("Mem Efficiency", justify="right", style="blue", width=12)

    for result in results:
        # Color code the speedup
        speedup_text = f"{result['total_speedup']:.1f}x"
        if result["total_speedup"] > 1.5:
            speedup_text = f"[green]{speedup_text}[/green]"
        elif result["total_speedup"] > 1.0:
            speedup_text = f"[yellow]{speedup_text}[/yellow]"
        else:
            speedup_text = f"[red]{speedup_text}[/red]"

        # Color code memory efficiency
        mem_text = f"{result['memory_efficiency']:.1f}x"
        if result["memory_efficiency"] > 2.0:
            mem_text = f"[green]{mem_text}[/green]"
        elif result["memory_efficiency"] > 1.5:
            mem_text = f"[yellow]{mem_text}[/yellow]"
        else:
            mem_text = f"[red]{mem_text}[/red]"

        table.add_row(
            result["scenario"],
            f"{result['h5ad_time_ms']:.1f}",
            f"{result['slaf_time_ms']:.1f}",
            speedup_text,
            f"{result['h5ad_total_memory_mb']:.1f}",
            f"{result['slaf_total_memory_mb']:.1f}",
            mem_text,
        )

    console.print(table)

    # Summary statistics
    speedups = [r["total_speedup"] for r in results if r["total_speedup"] > 0]
    mem_efficiencies = [
        r["memory_efficiency"] for r in results if r["memory_efficiency"] > 0
    ]

    if speedups:
        avg_speedup = np.mean(speedups)
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"Average speedup: {avg_speedup:.1f}x")

        if avg_speedup > 1.5:
            console.print(
                "[bold green]✓ SLAF shows significant speedup in multi-process scenarios[/bold green]"
            )
        elif avg_speedup > 1.0:
            console.print(
                "[bold yellow]⚠ SLAF shows moderate speedup in multi-process scenarios[/bold yellow]"
            )
        else:
            console.print(
                "[bold red]✗ SLAF doesn't show speedup in multi-process scenarios[/bold red]"
            )

    if mem_efficiencies:
        avg_mem_eff = np.mean(mem_efficiencies)
        console.print(f"Average memory efficiency: {avg_mem_eff:.1f}x")

        if avg_mem_eff > 2.0:
            console.print(
                "[bold green]✓ SLAF shows significant memory efficiency[/bold green]"
            )
        elif avg_mem_eff > 1.5:
            console.print(
                "[bold yellow]⚠ SLAF shows moderate memory efficiency[/bold yellow]"
            )
        else:
            console.print("[bold red]✗ SLAF doesn't show memory efficiency[/bold red]")


def benchmark_dataloaders(
    h5ad_path: str,
    slaf_path: str,
    include_memory: bool = True,
    verbose: bool = False,
    print_table: bool = True,
) -> list:
    """Benchmark dataloader overhead vs direct tokenizer calls

    Args:
        h5ad_path: Path to h5ad file (not used, but required for interface)
        slaf_path: Path to SLAF file
        include_memory: Whether to include memory measurements
        verbose: Enable verbose output
        print_table: Whether to print the results table

    Returns:
        List of benchmark results in the expected format
    """

    console = Console()

    # Load SLAF dataset
    slaf = SLAFArray(slaf_path)
    n_cells, n_genes = slaf.shape

    # Test scenarios
    scenarios = [
        {
            "name": "Geneformer small batch",
            "batch_size": 32,
            "max_genes": 512,
            "tokenizer_type": "geneformer",
            "description": "Geneformer small batch (32 cells, 512 genes)",
        },
        {
            "name": "Geneformer medium batch",
            "batch_size": 128,
            "max_genes": 1024,
            "tokenizer_type": "geneformer",
            "description": "Geneformer medium batch (128 cells, 1024 genes)",
        },
        {
            "name": "Geneformer large batch",
            "batch_size": 512,
            "max_genes": 2048,
            "tokenizer_type": "geneformer",
            "description": "Geneformer large batch (512 cells, 2048 genes)",
        },
        {
            "name": "scGPT small batch",
            "batch_size": 32,
            "max_genes": 512,
            "tokenizer_type": "scgpt",
            "description": "scGPT small batch (32 cells, 512 genes)",
        },
        {
            "name": "scGPT medium batch",
            "batch_size": 128,
            "max_genes": 1024,
            "tokenizer_type": "scgpt",
            "description": "scGPT medium batch (128 cells, 1024 genes)",
        },
        {
            "name": "scGPT large batch",
            "batch_size": 512,
            "max_genes": 1024,
            "tokenizer_type": "scgpt",
            "description": "scGPT large batch (512 cells, 1024 genes)",
        },
    ]

    results = []

    # Initialize tokenizers once to avoid overhead
    tokenizer_geneformer = SLAFTokenizer(slaf, vocab_size=1000, chunk_size=256)
    tokenizer_scgpt = SLAFTokenizer(slaf, vocab_size=1000, chunk_size=256)

    for i, scenario in enumerate(scenarios):
        if verbose:
            console.print(f"\n[bold blue]Testing: {scenario['name']}[/bold blue]")

        # Clear caches at the start of each scenario
        from benchmark_utils import clear_caches

        clear_caches()

        # For the first scenario, do a burn-in run to eliminate cold start effects
        if i == 0:
            if verbose:
                console.print("  Running burn-in for first scenario...")

            # Create a temporary SLAF instance for burn-in
            temp_slaf = SLAFArray(slaf_path)

            # Use centralized warm-up system
            from benchmark_utils import warm_up_slaf_database

            warm_up_slaf_database(temp_slaf, verbose=verbose)

            # Clear caches again after burn-in
            clear_caches()

            # Clean up temporary instances
            del temp_slaf

        # Use the appropriate tokenizer
        tokenizer = (
            tokenizer_geneformer
            if scenario["tokenizer_type"] == "geneformer"
            else tokenizer_scgpt
        )

        # Create dataloader with the same tokenizer instance to ensure fair comparison
        dataloader = SLAFDataLoader(
            slaf,
            tokenizer_type=scenario["tokenizer_type"],
            batch_size=scenario["batch_size"],
            max_genes=scenario["max_genes"],
            vocab_size=1000,
            chunk_size=256,
        )

        # Replace the dataloader's tokenizer with our pre-created one to ensure identical behavior
        dataloader.tokenizer = tokenizer

        # Test 1: Dataloader iteration (tokenization + tensor conversion + attention mask) - measure one batch only
        gc.collect()
        start_time = time.time()

        # Get just the first batch
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        total_tokens_dataloader = input_ids.shape[0] * input_ids.shape[1]

        dataloader_time = time.time() - start_time

        # Clear caches to ensure fair comparison
        clear_caches()

        # Test 2: Direct tokenizer calls (just tokenization) - measure one batch only
        gc.collect()

        # Get just the first batch range
        first_batch_range = dataloader.cell_integer_ranges[0]

        # Use the same approach as dataloader (timer inside the branch)
        if scenario["tokenizer_type"] == "geneformer":
            start_time = time.time()
            tokens = tokenizer.tokenize_geneformer(
                cell_integer_id_range=first_batch_range,
                max_genes=int(scenario["max_genes"]),
            )
            direct_time = time.time() - start_time
        else:  # scgpt
            start_time = time.time()
            tokens = tokenizer.tokenize_scgpt(
                cell_integer_id_range=first_batch_range,
                max_genes=int(scenario["max_genes"]),
            )
            direct_time = time.time() - start_time

        total_tokens_direct = sum(len(seq) for seq in tokens)

        # Calculate overhead (dataloader includes more work, so this is fair)
        overhead_ratio = (dataloader_time - direct_time) / direct_time * 100
        speedup = direct_time / dataloader_time if dataloader_time > 0 else 0

        # Memory measurements
        if include_memory:
            gc.collect()
            start_memory = get_slaf_memory_usage(slaf)

            # Iterate through dataloader and measure peak memory
            peak_memory = start_memory
            for batch in dataloader:
                batch_memory = get_object_memory_usage(batch)
                peak_memory = max(peak_memory, start_memory + batch_memory)

            memory_overhead = peak_memory - start_memory
        else:
            memory_overhead = 0

        # Format result to match expected interface
        result = {
            "scenario": f"S{i + 1}",
            "scenario_description": scenario["description"],
            "h5ad_time_ms": direct_time * 1000,  # Direct tokenizer time
            "slaf_time_ms": dataloader_time * 1000,  # Dataloader iteration time
            "total_speedup": speedup,
            "h5ad_total_memory_mb": 0,  # Not applicable
            "slaf_total_memory_mb": memory_overhead,
            "memory_efficiency": 0,  # Not applicable (no h5ad comparison)
            "overhead_percent": overhead_ratio,
            "tokens_per_second": (
                total_tokens_dataloader / dataloader_time if dataloader_time > 0 else 0
            ),
            # Add fields expected by comprehensive benchmark system
            "query_speedup": speedup,  # Same as total_speedup for this benchmark
            "load_speedup": 0,  # Not applicable
        }

        results.append(result)

        if verbose:
            console.print(f"  Direct tokenizer: {direct_time * 1000:.1f}ms")
            console.print(f"  Dataloader: {dataloader_time * 1000:.1f}ms")
            console.print(f"  Overhead: {overhead_ratio:.1f}%")
            console.print(f"  Memory overhead: {memory_overhead:.1f}MB")
            console.print(f"  Tokens/sec: {result['tokens_per_second']:,.0f}")
            console.print(
                f"  Direct tokens: {total_tokens_direct:,}, Dataloader tokens: {total_tokens_dataloader:,}"
            )
            console.print(f"  First batch range: {first_batch_range}")

    # Print table if requested
    if print_table:
        print_dataloader_results_table(results)

    return results


def benchmark_dataloader_overhead(slaf_path: str, verbose: bool = True):
    """Legacy function for standalone dataloader benchmarking"""
    return benchmark_dataloaders("", slaf_path, include_memory=True, verbose=verbose)


def print_dataloader_results_table(results):
    """Print dataloader benchmark results table"""
    console = Console()

    table = Table(
        title="DATALOADER OVERHEAD BENCHMARK RESULTS",
        show_header=True,
        header_style="bold magenta",
        title_style="bold blue",
    )

    table.add_column("Scenario", style="cyan", width=35)
    table.add_column("Direct Tokenizer (ms)", justify="right", style="red", width=16)
    table.add_column("Dataloader (ms)", justify="right", style="green", width=14)
    table.add_column("Speedup", justify="right", style="yellow", width=8)
    table.add_column("Overhead (%)", justify="right", style="blue", width=12)
    table.add_column("Description", style="cyan", width=40)

    for result in results:
        # Color code the speedup
        speedup_text = f"{result['total_speedup']:.1f}x"
        if result["total_speedup"] > 1.5:
            speedup_text = f"[green]{speedup_text}[/green]"
        elif result["total_speedup"] > 1.0:
            speedup_text = f"[yellow]{speedup_text}[/yellow]"
        else:
            speedup_text = f"[red]{speedup_text}[/red]"

        # Color code overhead percentage
        overhead_text = f"{result['overhead_percent']:.1f}%"
        if result["overhead_percent"] < 5:
            overhead_text = f"[green]{overhead_text}[/green]"
        elif result["overhead_percent"] < 15:
            overhead_text = f"[yellow]{overhead_text}[/yellow]"
        else:
            overhead_text = f"[red]{overhead_text}[/red]"

        table.add_row(
            result["scenario"],
            f"{result['h5ad_time_ms']:.1f}",
            f"{result['slaf_time_ms']:.1f}",
            speedup_text,
            overhead_text,
            result["scenario_description"],
        )

    console.print(table)

    # Summary statistics
    avg_overhead = np.mean([r["overhead_percent"] for r in results])
    avg_speedup = np.mean([r["total_speedup"] for r in results])

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"Average overhead: {avg_overhead:.1f}%")
    console.print(f"Average speedup: {avg_speedup:.2f}x")

    if avg_overhead < 5:
        console.print(
            "[bold green]✓ Dataloader overhead is minimal (< 5%)[/bold green]"
        )
    elif avg_overhead < 15:
        console.print(
            "[bold yellow]⚠ Dataloader overhead is moderate (5-15%)[/bold yellow]"
        )
    else:
        console.print(
            "[bold red]✗ Dataloader overhead is significant (> 15%)[/bold red]"
        )


def benchmark_data_vs_tokenization_timing(
    h5ad_path: str, slaf_path: str, verbose: bool = True
) -> dict:
    """Benchmark data loading vs tokenization timing separately

    This helps identify whether performance differences come from:
    1. Data loading/querying efficiency
    2. Tokenization algorithm efficiency

    Args:
        h5ad_path: Path to h5ad file
        slaf_path: Path to SLAF file
        verbose: Enable verbose output

    Returns:
        Dictionary with timing breakdowns
    """
    console = Console()

    if verbose:
        console.print(
            "[bold blue]Data Loading vs Tokenization Timing Analysis[/bold blue]\n"
        )

    # Test scenario
    scenario = {
        "batch_size": 128,
        "max_genes": 1024,
        "tokenizer_type": "geneformer",
    }

    results = {}

    # Test h5ad timing breakdown
    if verbose:
        console.print("Testing h5ad timing breakdown...")

    h5ad_results = _benchmark_h5ad_timing_breakdown(h5ad_path, scenario)
    results["h5ad"] = h5ad_results

    # Test SLAF timing breakdown
    if verbose:
        console.print("Testing SLAF timing breakdown...")

    slaf_results = _benchmark_slaf_timing_breakdown(slaf_path, scenario)
    results["slaf"] = slaf_results

    # Print comparison
    if verbose:
        console.print("\n[bold]Timing Breakdown Comparison:[/bold]")
        console.print(
            f"{'Component':<20} {'h5ad (ms)':<12} {'SLAF (ms)':<12} {'Ratio (h5ad/SLAF)':<15}"
        )
        console.print("-" * 60)

        for component in ["data_loading", "tokenization", "total"]:
            h5ad_time = h5ad_results[f"{component}_time_ms"]
            slaf_time = slaf_results[f"{component}_time_ms"]
            ratio = h5ad_time / slaf_time if slaf_time > 0 else 0
            console.print(
                f"{component:<20} {h5ad_time:<12.1f} {slaf_time:<12.1f} {ratio:<15.1f}"
            )

        # Analysis
        data_ratio = (
            h5ad_results["data_loading_time_ms"] / slaf_results["data_loading_time_ms"]
        )
        token_ratio = (
            h5ad_results["tokenization_time_ms"] / slaf_results["tokenization_time_ms"]
        )

        console.print("\n[bold]Analysis:[/bold]")
        if data_ratio < 0.7:
            console.print(
                f"🔍 h5ad data loading is {1 / data_ratio:.1f}x faster - SLAF query optimization opportunity"
            )
        elif data_ratio > 1.5:
            console.print(
                f"🔍 SLAF data loading is {data_ratio:.1f}x faster - SLAF query advantage"
            )
        else:
            console.print(f"🔍 Data loading performance is similar ({data_ratio:.1f}x)")

        if token_ratio < 0.7:
            console.print(
                f"🔍 h5ad tokenization is {1 / token_ratio:.1f}x faster - SLAF tokenizer optimization opportunity"
            )
        elif token_ratio > 1.5:
            console.print(
                f"🔍 SLAF tokenization is {token_ratio:.1f}x faster - SLAF tokenizer advantage"
            )
        else:
            console.print(
                f"🔍 Tokenization performance is similar ({token_ratio:.1f}x)"
            )

    return results


def _benchmark_h5ad_timing_breakdown(h5ad_path: str, scenario: dict) -> dict:
    """Break down h5ad timing into data loading vs tokenization"""
    import time

    import numpy as np
    import scanpy as sc

    # Load data (this is the data loading phase)
    start_data = time.time()
    adata = sc.read_h5ad(h5ad_path)
    end_data = time.time()

    # Get expression matrix - handle different types robustly
    try:
        # Try to convert to dense array if it's sparse
        X = adata.X.toarray()  # type: ignore
    except AttributeError:
        try:
            # Try todense if toarray doesn't exist
            X = adata.X.todense()  # type: ignore
        except AttributeError:
            # Otherwise use as-is
            X = adata.X

    # Ensure X is a numpy array
    X = np.asarray(X)

    # Tokenization phase
    start_token = time.time()

    batch_size = scenario["batch_size"]
    max_genes = scenario["max_genes"]

    # Process first batch
    start_idx = 0
    end_idx = min(batch_size, X.shape[0])

    batch_tokens = []
    for i in range(start_idx, end_idx):
        # Get expression for this cell
        cell_expr = X[i, :]

        # Rank genes by expression (descending)
        gene_ranks = np.argsort(cell_expr)[::-1]

        # Take top max_genes
        top_genes = gene_ranks[:max_genes]

        # Convert to token IDs (simple mapping: gene index -> token ID)
        tokens = top_genes.tolist()

        # Pad if needed
        if len(tokens) < max_genes:
            tokens.extend([0] * (max_genes - len(tokens)))  # 0 as PAD token

        batch_tokens.append(tokens)

    end_token = time.time()

    return {
        "data_loading_time_ms": (end_data - start_data) * 1000,
        "tokenization_time_ms": (end_token - start_token) * 1000,
        "total_time_ms": (end_token - start_data) * 1000,
        "batch_tokens_shape": np.array(batch_tokens).shape,
    }


def _benchmark_slaf_timing_breakdown(slaf_path: str, scenario: dict) -> dict:
    """Break down SLAF timing into data loading vs tokenization"""
    import time

    import numpy as np

    from slaf.core.slaf import SLAFArray
    from slaf.ml.tokenizers import SLAFTokenizer

    # Data loading phase (SLAF initialization and query)
    start_data = time.time()
    slaf = SLAFArray(slaf_path)

    # Query expression data for the batch
    batch_size = scenario["batch_size"]
    max_genes = scenario["max_genes"]

    # Query expression data sorted by expression
    sql = f"""
    WITH ranked_genes AS (
        SELECT
            cell_id,
            gene_id,
            value,
            ROW_NUMBER() OVER (
                PARTITION BY cell_id
                ORDER BY value DESC
            ) as gene_rank
        FROM expression
        WHERE cell_integer_id < {batch_size}
    ),
    limited_genes AS (
        SELECT cell_id, gene_id, value
        FROM ranked_genes
        WHERE gene_rank <= {max_genes}
    )
    SELECT
        cell_id,
        array_agg(gene_id ORDER BY value DESC) as gene_sequence,
        array_agg(value ORDER BY value DESC) as expr_sequence
    FROM limited_genes
    GROUP BY cell_id
    ORDER BY cell_id
    """

    batch_data = slaf.query(sql)
    end_data = time.time()

    # Tokenization phase (convert to token sequences)
    start_token = time.time()

    # Initialize tokenizer
    tokenizer = SLAFTokenizer(
        slaf_array=slaf,
        vocab_size=50000,
        n_expression_bins=10,
        chunk_size=1024,
    )

    # Convert to token sequences
    batch_tokens = []
    for _, row in batch_data.iterrows():
        # Map gene IDs to vocabulary tokens
        tokens = [
            tokenizer.gene_vocab.get(gene_id, tokenizer.special_tokens["UNK"])
            for gene_id in row["gene_sequence"]
        ]

        # Pad or truncate to max_genes
        if len(tokens) < max_genes:
            tokens.extend([tokenizer.special_tokens["PAD"]] * (max_genes - len(tokens)))
        else:
            tokens = tokens[:max_genes]

        batch_tokens.append(tokens)

    end_token = time.time()

    return {
        "data_loading_time_ms": (end_data - start_data) * 1000,
        "tokenization_time_ms": (end_token - start_token) * 1000,
        "total_time_ms": (end_token - start_data) * 1000,
        "batch_tokens_shape": np.array(batch_tokens).shape,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SLAF dataloaders")
    parser.add_argument("--h5ad", required=True, help="Path to h5ad file")
    parser.add_argument("--slaf", required=True, help="Path to SLAF file")
    parser.add_argument(
        "--max-processes", type=int, default=8, help="Max processes for scaling test"
    )
    parser.add_argument(
        "--include-memory", action="store_true", help="Include memory profiling"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--timing-breakdown", action="store_true", help="Run timing breakdown analysis"
    )

    args = parser.parse_args()

    if args.timing_breakdown:
        # Run timing breakdown analysis
        benchmark_data_vs_tokenization_timing(
            args.h5ad, args.slaf, verbose=args.verbose
        )
    else:
        # Run standard benchmarks
        console = Console()
        console.print("[bold blue]SLAF Dataloader Benchmarks[/bold blue]\n")

        # Run multi-process scaling benchmark
        results = benchmark_multi_process_scaling(
            args.h5ad, args.slaf, max_processes=args.max_processes, verbose=args.verbose
        )

        if args.verbose:
            console.print("\n[bold green]Benchmark completed![/bold green]")
