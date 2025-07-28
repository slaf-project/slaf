import time

import polars as pl
import scanpy as sc
from benchmark_utils import clear_caches, get_object_memory_usage, get_slaf_memory_usage

from slaf.core.slaf import SLAFArray


def demo_realistic_gene_queries():
    """Demo realistic gene filtering scenarios for PBMC data"""
    scenarios = [
        # Expression-based filtering (very common)
        {
            "name": "min_cells_10",
            "description": "Genes expressed in >=10 cells",
            "h5ad_code": lambda adata: adata.var[adata.var.n_cells_by_counts >= 10],
            "slaf_code": lambda slaf: slaf.filter_genes(n_cells_by_counts=">=10"),
        },
        {
            "name": "min_total_counts_100",
            "description": "Genes with >=100 total counts",
            "h5ad_code": lambda adata: adata.var[adata.var.total_counts >= 100],
            "slaf_code": lambda slaf: slaf.filter_genes(total_counts=">=100"),
        },
        {
            "name": "min_mean_counts_0.1",
            "description": "Genes with mean expression >=0.1",
            "h5ad_code": lambda adata: adata.var[adata.var.mean_counts >= 0.1],
            "slaf_code": lambda slaf: slaf.filter_genes(mean_counts=">=0.1"),
        },
        {
            "name": "exclude_mt",
            "description": "Exclude mitochondrial genes",
            "h5ad_code": lambda adata: adata.var[~adata.var.mt],
            "slaf_code": lambda slaf: slaf.filter_genes(mt=False),
        },
        # Highly variable gene filtering (post-analysis)
        {
            "name": "highly_variable",
            "description": "Highly variable genes",
            "h5ad_code": lambda adata: adata.var[adata.var.highly_variable],
            "slaf_code": lambda slaf: slaf.filter_genes(highly_variable=True),
        },
        {
            "name": "non_highly_variable",
            "description": "Non-highly variable genes",
            "h5ad_code": lambda adata: adata.var[~adata.var.highly_variable],
            "slaf_code": lambda slaf: slaf.filter_genes(highly_variable=False),
        },
        # Combined filtering (most realistic)
        {
            "name": "min_cells_50_total_counts_500",
            "description": "Genes in >=50 cells with >=500 total counts",
            "h5ad_code": lambda adata: adata.var[
                (adata.var.n_cells_by_counts >= 50) & (adata.var.total_counts >= 500)
            ],
            "slaf_code": lambda slaf: slaf.filter_genes(
                n_cells_by_counts=">=50", total_counts=">=500"
            ),
        },
        # Range queries
        {
            "name": "total_counts_100_10000",
            "description": "Genes with 100-10000 total counts",
            "h5ad_code": lambda adata: adata.var[
                (adata.var.total_counts >= 100) & (adata.var.total_counts <= 10000)
            ],
            "slaf_code": lambda slaf: slaf.filter_genes(total_counts=">=100").filter(
                pl.col("total_counts") <= 10000
            ),
        },
        {
            "name": "cells_5_1000",
            "description": "Genes in 5-1000 cells",
            "h5ad_code": lambda adata: adata.var[
                (adata.var.n_cells_by_counts >= 5)
                & (adata.var.n_cells_by_counts <= 1000)
            ],
            "slaf_code": lambda slaf: slaf.filter_genes(n_cells_by_counts=">=5").filter(
                pl.col("n_cells_by_counts") <= 1000
            ),
        },
    ]
    return scenarios


def _measure_h5ad_gene_filtering(h5ad_path: str, scenario: dict):
    """Measure h5ad gene filtering performance"""
    import gc

    gc.collect()

    # Load h5ad
    start = time.time()
    adata = sc.read_h5ad(h5ad_path, backed="r")
    h5ad_load_time = time.time() - start

    # Measure memory footprint
    h5ad_load_memory = (
        get_object_memory_usage(adata.X)
        + get_object_memory_usage(adata.obs)
        + get_object_memory_usage(adata.var)
    )

    # Execute the filtering operation
    start = time.time()
    try:
        result = scenario["h5ad_code"](adata)
        h5ad_query_time = time.time() - start
        h5ad_query_memory = get_object_memory_usage(result)
        h5ad_count = len(result)
    except Exception as e:
        print(f"h5ad filtering failed: {e}")
        h5ad_query_time = 0
        h5ad_query_memory = 0
        h5ad_count = 0

    # Clean up
    del adata
    gc.collect()

    return {
        "h5ad_load_time": h5ad_load_time,
        "h5ad_query_time": h5ad_query_time,
        "h5ad_load_memory": float(h5ad_load_memory),
        "h5ad_query_memory": float(h5ad_query_memory),
        "h5ad_count": int(h5ad_count),
    }


def _measure_slaf_gene_filtering(slaf_path: str, scenario: dict):
    """Measure SLAF gene filtering performance"""
    import gc

    gc.collect()

    # Load SLAF
    start = time.time()
    slaf = SLAFArray(slaf_path)
    slaf_init_time = time.time() - start

    # Measure memory footprint
    slaf_load_memory = get_slaf_memory_usage(slaf)

    # Execute the filtering operation
    start = time.time()
    try:
        result = scenario["slaf_code"](slaf)
        slaf_query_time = time.time() - start
        slaf_query_memory = get_object_memory_usage(result)
        slaf_count = len(result)
    except Exception as e:
        print(f"SLAF filtering failed: {e}")
        slaf_query_time = 0
        slaf_query_memory = 0
        slaf_count = 0

    # Clean up
    del slaf
    gc.collect()

    return {
        "slaf_init_time": slaf_init_time,
        "slaf_query_time": slaf_query_time,
        "slaf_init_memory": float(slaf_load_memory),
        "slaf_query_memory": float(slaf_query_memory),
        "slaf_count": int(slaf_count),
    }


def benchmark_gene_filtering_scenario(
    h5ad_path: str,
    slaf_path: str,
    scenario: dict,
):
    """Benchmark a single gene filtering scenario with isolated memory measurement"""

    # Measure h5ad in isolation
    h5ad_result = _measure_h5ad_gene_filtering(h5ad_path, scenario)

    # Measure SLAF in isolation
    slaf_result = _measure_slaf_gene_filtering(slaf_path, scenario)

    # Calculate totals and speedups
    h5ad_total_time = h5ad_result["h5ad_load_time"] + h5ad_result["h5ad_query_time"]
    slaf_total_time = slaf_result["slaf_init_time"] + slaf_result["slaf_query_time"]

    total_speedup = h5ad_total_time / slaf_total_time if slaf_total_time > 0 else 0
    query_speedup = (
        h5ad_result["h5ad_query_time"] / slaf_result["slaf_query_time"]
        if slaf_result["slaf_query_time"] > 0
        else 0
    )
    load_speedup = (
        h5ad_result["h5ad_load_time"] / slaf_result["slaf_init_time"]
        if slaf_result["slaf_init_time"] > 0
        else 0
    )

    return {
        "scenario_type": "gene_filtering",
        "scenario_description": scenario["description"],
        "h5ad_total_time": 1000 * h5ad_total_time,
        "h5ad_load_time": 1000 * h5ad_result["h5ad_load_time"],
        "h5ad_query_time": 1000 * h5ad_result["h5ad_query_time"],
        "slaf_total_time": 1000 * slaf_total_time,
        "slaf_init_time": 1000 * slaf_result["slaf_init_time"],
        "slaf_query_time": 1000 * slaf_result["slaf_query_time"],
        "total_speedup": total_speedup,
        "query_speedup": query_speedup,
        "load_speedup": load_speedup,
        "h5ad_count": h5ad_result["h5ad_count"],
        "slaf_count": slaf_result["slaf_count"],
        "results_match": h5ad_result["h5ad_count"] == slaf_result["slaf_count"],
        # Memory breakdown
        "h5ad_load_memory_mb": h5ad_result["h5ad_load_memory"],
        "h5ad_query_memory_mb": h5ad_result["h5ad_query_memory"],
        "h5ad_total_memory_mb": h5ad_result["h5ad_load_memory"]
        + h5ad_result["h5ad_query_memory"],
        "slaf_load_memory_mb": slaf_result["slaf_init_memory"],
        "slaf_query_memory_mb": slaf_result["slaf_query_memory"],
        "slaf_total_memory_mb": slaf_result["slaf_init_memory"]
        + slaf_result["slaf_query_memory"],
    }


def benchmark_gene_filtering(
    h5ad_path: str, slaf_path: str, include_memory=True, verbose=False
):
    """Benchmark realistic gene filtering scenarios"""
    scenarios = demo_realistic_gene_queries()

    if verbose:
        print("Benchmarking gene filtering scenarios")
        print("=" * 50)

    results = []
    for i, scenario in enumerate(scenarios):
        if verbose:
            print(f"Running scenario {i + 1}/{len(scenarios)}: {scenario['name']}")

        # Clear caches at the start of each scenario
        clear_caches()

        # For the first scenario, do a burn-in run to eliminate cold start effects
        if i == 0:
            if verbose:
                print("  Running burn-in for first scenario...")

            # Create a temporary SLAF instance for burn-in
            temp_slaf = SLAFArray(slaf_path)

            # Use centralized warm-up system
            from benchmark_utils import warm_up_slaf_database

            warm_up_slaf_database(temp_slaf, verbose=verbose)

            # Clear caches again after burn-in
            clear_caches()

            # Clean up temporary instance
            del temp_slaf

        try:
            # Always include loading time for each scenario
            result = benchmark_gene_filtering_scenario(h5ad_path, slaf_path, scenario)

            results.append(result)

            if verbose:
                print(f"  ✓ Completed: {result['total_speedup']:.1f}x speedup")

        except Exception as e:
            if verbose:
                print(f"  ✗ Error: {e}")
            continue

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark gene filtering")
    parser.add_argument("h5ad_path", help="Path to h5ad file")
    parser.add_argument("slaf_path", help="Path to SLAF dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    results = benchmark_gene_filtering(
        args.h5ad_path, args.slaf_path, verbose=args.verbose
    )

    # Print results table
    from benchmark_utils import print_benchmark_table

    print_benchmark_table(results, scenario_type="Gene Filtering")
