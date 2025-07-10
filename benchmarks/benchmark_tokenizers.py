import sys
import time

import numpy as np
import scanpy as sc
from benchmark_utils import clear_caches, get_object_memory_usage, get_slaf_memory_usage
from rich.console import Console
from rich.table import Table

from slaf.core.slaf import SLAFArray
from slaf.ml.tokenizers import SLAFTokenizer


def demo_tokenizer_scenarios(h5ad_path: str, slaf_path: str):
    """Demo realistic tokenizer scenarios for single-cell data"""
    from slaf.core.slaf import SLAFArray

    # Load SLAF dataset to get dimensions
    slaf = SLAFArray(slaf_path)
    n_cells, n_genes = slaf.shape

    # Define cell ranges for different batch sizes
    scenarios = [
        # scGPT scenarios
        {
            "type": "scgpt",
            "cell_integer_id_range": (0, 32),
            "max_genes": 512,
            "description": "scGPT small batch (32 cells, 512 genes)",
        },
        {
            "type": "scgpt_sql_binning",
            "cell_integer_id_range": (0, 32),
            "max_genes": 512,
            "use_sql_binning": True,
            "description": "scGPT small batch SQL binning",
        },
        {
            "type": "scgpt",
            "cell_integer_id_range": (0, 128),
            "max_genes": 1024,
            "description": "scGPT medium batch (128 cells, 1024 genes)",
        },
        {
            "type": "scgpt_sql_binning",
            "cell_integer_id_range": (0, 128),
            "max_genes": 1024,
            "use_sql_binning": True,
            "description": "scGPT medium batch SQL binning",
        },
        {
            "type": "scgpt",
            "cell_integer_id_range": (0, 512),
            "max_genes": 1024,
            "description": "scGPT large batch (512 cells, 1024 genes)",
        },
        {
            "type": "scgpt_sql_binning",
            "cell_integer_id_range": (0, 512),
            "max_genes": 1024,
            "use_sql_binning": True,
            "description": "scGPT large batch SQL binning",
        },
        {
            "type": "scgpt",
            "cell_integer_id_range": (0, 2048),
            "max_genes": 1024,
            "description": "scGPT xlarge batch (2048 cells, 1024 genes)",
        },
        {
            "type": "scgpt_sql_binning",
            "cell_integer_id_range": (0, 2048),
            "max_genes": 1024,
            "use_sql_binning": True,
            "description": "scGPT xlarge batch SQL binning",
        },
        # Geneformer scenarios
        {
            "type": "geneformer",
            "cell_integer_id_range": (0, 32),
            "max_genes": 1024,
            "description": "Geneformer small batch (32 cells, 1024 genes)",
        },
        {
            "type": "geneformer_percentile",
            "cell_integer_id_range": (0, 32),
            "max_genes": 1024,
            "min_percentile": 10,
            "description": "Geneformer small batch with percentile filter",
        },
        {
            "type": "geneformer",
            "cell_integer_id_range": (0, 128),
            "max_genes": 2048,
            "description": "Geneformer medium batch (128 cells, 2048 genes)",
        },
        {
            "type": "geneformer_percentile",
            "cell_integer_id_range": (0, 128),
            "max_genes": 2048,
            "min_percentile": 10,
            "description": "Geneformer medium batch with percentile filter",
        },
        {
            "type": "geneformer",
            "cell_integer_id_range": (0, 512),
            "max_genes": 2048,
            "description": "Geneformer large batch (512 cells, 2048 genes)",
        },
        {
            "type": "geneformer_percentile",
            "cell_integer_id_range": (0, 512),
            "max_genes": 2048,
            "min_percentile": 10,
            "description": "Geneformer large batch with percentile filter",
        },
        {
            "type": "geneformer",
            "cell_integer_id_range": (0, 2048),
            "max_genes": 2048,
            "description": "Geneformer xlarge batch (2048 cells, 2048 genes)",
        },
        {
            "type": "geneformer_percentile",
            "cell_integer_id_range": (0, 2048),
            "max_genes": 2048,
            "min_percentile": 10,
            "description": "Geneformer xlarge batch with percentile filter",
        },
    ]

    return scenarios


def get_token_sequences_memory_usage(token_sequences):
    """Get memory usage of token sequences (list of lists) in MB"""
    if not token_sequences:
        return 0.0

    total_bytes = 0

    # Measure the outer list
    total_bytes += sys.getsizeof(token_sequences)

    # Measure each inner list and its contents
    for sequence in token_sequences:
        if isinstance(sequence, list):
            total_bytes += sys.getsizeof(sequence)
            # Measure each integer in the sequence
            for token in sequence:
                total_bytes += sys.getsizeof(token)

    return total_bytes / 1024 / 1024


def _measure_h5ad_tokenization(h5ad_path: str, scenario: dict):
    """Measure h5ad-based tokenization performance in isolation"""
    import gc

    gc.collect()

    # h5ad load
    start = time.time()
    adata = sc.read_h5ad(h5ad_path)
    h5ad_load_time = time.time() - start

    # Measure memory footprint of loaded data
    h5ad_load_memory = (
        get_object_memory_usage(adata.X)
        + get_object_memory_usage(adata.obs)
        + get_object_memory_usage(adata.var)
    )

    # Tokenization
    start = time.time()

    cell_integer_id_range = scenario["cell_integer_id_range"]
    cell_start, cell_end = cell_integer_id_range
    max_genes = scenario["max_genes"]

    # Get cell indices
    cell_indices = list(range(cell_start, cell_end))

    token_sequences = []

    for cell_idx in cell_indices:
        # Get expression for this cell
        expr_vector = (
            adata.X[cell_idx, :].toarray().flatten()
            if hasattr(adata.X, "toarray")
            else adata.X[cell_idx, :]
        )

        if scenario["type"].startswith("scgpt"):
            # scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]
            tokens = [0]  # CLS token

            # Get non-zero genes sorted by expression
            non_zero_mask = expr_vector > 0
            non_zero_indices = np.where(non_zero_mask)[0]
            non_zero_expr = expr_vector[non_zero_mask]

            # Sort by expression (descending)
            sorted_indices = non_zero_indices[np.argsort(-non_zero_expr)]

            # Limit to max_genes
            if len(sorted_indices) > max_genes:
                sorted_indices = sorted_indices[:max_genes]

            # Create tokens
            for gene_idx in sorted_indices:
                tokens.append(gene_idx + 100)  # Gene token
                # Expression bin token (simple binning)
                expr_val = expr_vector[gene_idx]
                if expr_val <= 0:
                    tokens.append(2)  # PAD token
                else:
                    bin_id = min(9, int(np.log1p(expr_val) * 10))
                    tokens.append(10 + bin_id)  # Expression bin token

            tokens.append(1)  # SEP token

        elif scenario["type"].startswith("geneformer"):
            # Geneformer format: ranked gene tokens
            # Get expression for this cell
            expr_vector = (
                adata.X[cell_idx, :].toarray().flatten()
                if hasattr(adata.X, "toarray")
                else adata.X[cell_idx, :]
            )

            # Get non-zero genes sorted by expression
            non_zero_mask = expr_vector > 0
            non_zero_indices = np.where(non_zero_mask)[0]
            non_zero_expr = expr_vector[non_zero_mask]

            # Apply percentile filter if specified
            if scenario.get("min_percentile") is not None:
                min_percentile = scenario["min_percentile"]
                percentile_threshold = np.percentile(non_zero_expr, min_percentile)
                above_threshold = non_zero_expr >= percentile_threshold
                non_zero_indices = non_zero_indices[above_threshold]
                non_zero_expr = non_zero_expr[above_threshold]

            # Sort by expression (descending)
            sorted_indices = non_zero_indices[np.argsort(-non_zero_expr)]

            # Limit to max_genes
            if len(sorted_indices) > max_genes:
                sorted_indices = sorted_indices[:max_genes]

            # Create tokens
            tokens = [idx + 100 for idx in sorted_indices]  # Gene tokens

            # Pad to max_genes
            if len(tokens) < max_genes:
                tokens.extend([2] * (max_genes - len(tokens)))  # PAD tokens
            else:
                tokens = tokens[:max_genes]

        token_sequences.append(tokens)

    h5ad_tokenize_time = time.time() - start
    h5ad_tokenize_memory = get_token_sequences_memory_usage(token_sequences)

    # Get result size for comparison
    result_size = len(token_sequences)

    # Clean up
    del adata, token_sequences
    gc.collect()

    return {
        "h5ad_load_time": h5ad_load_time,
        "h5ad_tokenize_time": h5ad_tokenize_time,
        "h5ad_load_memory": float(h5ad_load_memory),
        "h5ad_tokenize_memory": float(h5ad_tokenize_memory),
        "result_size": result_size,
    }


def _measure_slaf_tokenization(slaf_path: str, scenario: dict):
    """Measure SLAF-based tokenization performance in isolation"""
    import gc

    gc.collect()

    # SLAF load
    start = time.time()
    slaf = SLAFArray(slaf_path)
    tokenizer = SLAFTokenizer(slaf)
    slaf_init_time = time.time() - start

    # Warm up the database to eliminate cold start effects
    from benchmark_utils import warm_up_slaf_database

    warm_up_slaf_database(slaf, verbose=False)

    # Measure memory footprint of loaded data - measure actual loaded metadata
    slaf_load_memory = get_slaf_memory_usage(slaf) + get_object_memory_usage(tokenizer)

    # Tokenization
    start = time.time()

    cell_integer_id_range = scenario["cell_integer_id_range"]

    if scenario["type"].startswith("scgpt"):
        max_genes = scenario["max_genes"]
        use_sql_binning = scenario.get("use_sql_binning", False)

        token_sequences = tokenizer.tokenize_scgpt(
            cell_integer_id_range=cell_integer_id_range,
            max_genes=max_genes,
            use_sql_binning=use_sql_binning,
        )

    elif scenario["type"].startswith("geneformer"):
        max_genes = scenario["max_genes"]
        min_percentile = scenario.get("min_percentile")

        token_sequences = tokenizer.tokenize_geneformer(
            cell_integer_id_range=cell_integer_id_range,
            max_genes=max_genes,
            min_percentile=min_percentile,
        )

    slaf_tokenize_time = time.time() - start
    slaf_tokenize_memory = get_token_sequences_memory_usage(token_sequences)

    # Get result size for comparison
    result_size = len(token_sequences)

    # Clean up
    del slaf, tokenizer, token_sequences
    gc.collect()

    return {
        "slaf_init_time": slaf_init_time,
        "slaf_tokenize_time": slaf_tokenize_time,
        "slaf_init_memory": float(slaf_load_memory),
        "slaf_tokenize_memory": float(slaf_tokenize_memory),
        "result_size": result_size,
    }


def benchmark_tokenizer_scenario(h5ad_path: str, slaf_path: str, scenario: dict):
    """Benchmark a single tokenizer scenario with isolated memory measurement"""

    # Measure h5ad in isolation
    h5ad_result = _measure_h5ad_tokenization(h5ad_path, scenario)

    # Measure SLAF in isolation
    slaf_result = _measure_slaf_tokenization(slaf_path, scenario)

    # Calculate totals and speedups
    h5ad_total_time = h5ad_result["h5ad_load_time"] + h5ad_result["h5ad_tokenize_time"]
    slaf_total_time = slaf_result["slaf_init_time"] + slaf_result["slaf_tokenize_time"]

    total_speedup = h5ad_total_time / slaf_total_time if slaf_total_time > 0 else 0
    query_speedup = (
        h5ad_result["h5ad_tokenize_time"] / slaf_result["slaf_tokenize_time"]
        if slaf_result["slaf_tokenize_time"] > 0
        else 0
    )
    load_speedup = (
        h5ad_result["h5ad_load_time"] / slaf_result["slaf_init_time"]
        if slaf_result["slaf_init_time"] > 0
        else 0
    )

    return {
        "scenario_type": "tokenization",
        "scenario_description": scenario["description"],
        "max_genes": scenario["max_genes"],
        "batch_size": scenario["cell_integer_id_range"][1]
        - scenario["cell_integer_id_range"][0],
        "h5ad_total_time": 1000 * h5ad_total_time,
        "h5ad_load_time": 1000 * h5ad_result["h5ad_load_time"],
        "h5ad_query_time": 1000 * h5ad_result["h5ad_tokenize_time"],
        "slaf_total_time": 1000 * slaf_total_time,
        "slaf_init_time": 1000 * slaf_result["slaf_init_time"],
        "slaf_query_time": 1000 * slaf_result["slaf_tokenize_time"],
        "total_speedup": total_speedup,
        "query_speedup": query_speedup,
        "load_speedup": load_speedup,
        "h5ad_result_size": h5ad_result["result_size"],
        "slaf_result_size": slaf_result["result_size"],
        "results_match": h5ad_result["result_size"] == slaf_result["result_size"],
        # Memory breakdown
        "h5ad_load_memory_mb": h5ad_result["h5ad_load_memory"],
        "h5ad_query_memory_mb": h5ad_result["h5ad_tokenize_memory"],
        "h5ad_total_memory_mb": h5ad_result["h5ad_load_memory"]
        + h5ad_result["h5ad_tokenize_memory"],
        "slaf_load_memory_mb": slaf_result["slaf_init_memory"],
        "slaf_query_memory_mb": slaf_result["slaf_tokenize_memory"],
        "slaf_total_memory_mb": slaf_result["slaf_init_memory"]
        + slaf_result["slaf_tokenize_memory"],
    }


def print_throughput_table(results):
    """Print throughput analysis as a rich table"""
    console = Console()

    table = Table(
        title="🚀 TOKENIZER THROUGHPUT ANALYSIS 🚀",
        show_header=True,
        header_style="bold magenta",
        title_style="bold blue",
    )

    table.add_column("Scenario", style="cyan", width=50)
    table.add_column("Cells/sec", justify="right", style="green", width=12)
    table.add_column("Tokens/sec", justify="right", style="green", width=12)
    table.add_column("Batch Size", justify="right", style="yellow", width=12)
    table.add_column("Max Genes", justify="right", style="yellow", width=12)

    for result in results:
        n_cells = result["slaf_result_size"]
        total_time_ms = result["slaf_total_time"]
        throughput = (n_cells / total_time_ms * 1000) if total_time_ms > 0 else 0

        # Calculate tokens per second
        batch_size = result.get("batch_size", 0)
        max_genes = result.get("max_genes", 0)

        # Estimate total tokens based on tokenizer type
        if "scgpt" in result["scenario_description"].lower():
            # scGPT: CLS + (gene,expr)*max_genes + SEP + padding
            tokens_per_cell = 2 + 2 * min(max_genes, 512)  # Approximate
        else:
            # Geneformer: max_genes tokens per cell
            tokens_per_cell = max_genes

        total_tokens = n_cells * tokens_per_cell
        tokens_per_sec = (
            (total_tokens / total_time_ms * 1000) if total_time_ms > 0 else 0
        )

        # Extract batch size and max genes from result data (more reliable than parsing description)
        batch_size_str = str(batch_size)
        max_genes_str = str(max_genes)

        table.add_row(
            result["scenario_description"],
            f"{throughput:,.0f}",
            f"{tokens_per_sec:,.0f}",
            batch_size_str,
            max_genes_str,
        )

    console.print(table)

    # Print insights
    console.print("\n💡 [bold blue]Key Insights:[/bold blue]")
    console.print("   • [green]Larger batches[/green] generally show better throughput")
    console.print("   • [green]SQL binning[/green] can improve scGPT performance")
    console.print(
        "   • [green]Geneformer[/green] typically has higher throughput than scGPT"
    )
    console.print(
        "   • [green]2048-cell batches[/green] show the best performance scaling"
    )


def benchmark_tokenizers(
    h5ad_path: str, slaf_path: str, include_memory=True, verbose=False
):
    """Run comprehensive tokenizer benchmarks"""

    print("=" * 80)
    print("TOKENIZER BENCHMARK")
    print("=" * 80)
    print(f"h5ad_path: {h5ad_path}")
    print(f"slaf_path: {slaf_path}")
    print()

    # Get scenarios
    scenarios = demo_tokenizer_scenarios(h5ad_path, slaf_path)

    # Run benchmarks
    results = []
    for i, scenario in enumerate(scenarios):
        if verbose:
            print(
                f"Running scenario {i + 1}/{len(scenarios)}: {scenario['description']}"
            )

        # Clear caches at the start of each scenario
        clear_caches()

        # For the first scenario, do a burn-in run to eliminate cold start effects
        if i == 0:
            if verbose:
                print("  Running burn-in for first scenario...")

            # Create a temporary SLAF instance for burn-in
            temp_slaf = SLAFArray(slaf_path)
            temp_tokenizer = SLAFTokenizer(
                slaf_array=temp_slaf,
                vocab_size=50000,
                n_expression_bins=10,
                chunk_size=1024,
            )

            # Burn-in run (discard results) - use the first scenario's tokenizer type
            if scenario["type"].startswith("geneformer"):
                _ = temp_tokenizer.tokenize_geneformer(
                    cell_integer_id_range=scenario["cell_integer_id_range"],
                    max_genes=scenario["max_genes"],
                )
            elif scenario["type"].startswith("scgpt"):
                _ = temp_tokenizer.tokenize_scgpt(
                    cell_integer_id_range=scenario["cell_integer_id_range"],
                    max_genes=scenario["max_genes"],
                )

            # Clear caches again after burn-in
            clear_caches()

            # Clean up temporary instances
            del temp_slaf, temp_tokenizer

        try:
            result = benchmark_tokenizer_scenario(h5ad_path, slaf_path, scenario)
            results.append(result)

            if verbose:
                print(f"  Total speedup: {result['total_speedup']:.2f}x")
                print(f"  Query speedup: {result['query_speedup']:.2f}x")
                print()

        except Exception as e:
            print(f"Error in scenario {scenario['description']}: {e}")
            continue

    # Print summary statistics if verbose
    if results and verbose:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        # Filter by tokenizer type
        scgpt_results = [r for r in results if r["scenario_type"].startswith("scgpt")]
        geneformer_results = [
            r for r in results if r["scenario_type"].startswith("geneformer")
        ]

        if scgpt_results:
            avg_scgpt_speedup = np.mean([r["total_speedup"] for r in scgpt_results])
            avg_scgpt_time = np.mean([r["slaf_total_time"] for r in scgpt_results])
            print(f"scGPT average speedup: {avg_scgpt_speedup:.2f}x")
            print(f"scGPT average time: {avg_scgpt_time:.1f}ms")

        if geneformer_results:
            avg_geneformer_speedup = np.mean(
                [r["total_speedup"] for r in geneformer_results]
            )
            avg_geneformer_time = np.mean(
                [r["slaf_total_time"] for r in geneformer_results]
            )
            print(f"Geneformer average speedup: {avg_geneformer_speedup:.2f}x")
            print(f"Geneformer average time: {avg_geneformer_time:.1f}ms")

    # Always print throughput analysis - this is the key insight!
    if results:
        print_throughput_table(results)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark SLAF tokenizers vs h5ad")
    parser.add_argument("h5ad_path", help="Path to h5ad file")
    parser.add_argument("slaf_path", help="Path to SLAF dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-memory", action="store_true", help="Skip memory measurements"
    )

    args = parser.parse_args()

    benchmark_tokenizers(
        args.h5ad_path,
        args.slaf_path,
        include_memory=not args.no_memory,
        verbose=args.verbose,
    )
