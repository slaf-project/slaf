import time

import scanpy as sc

# Import shared utilities
from benchmark_utils import (
    clear_caches,
    get_object_memory_usage,
    get_slaf_memory_usage,
)

from slaf.core.slaf import SLAFArray


def demo_realistic_cell_queries():
    """Demo realistic cell filtering scenarios for PBMC data"""
    scenarios = [
        # QC-based filtering (very common)
        {
            "name": "min_genes_500",
            "description": "Cells with >=500 genes",
            "h5ad_code": lambda adata: adata.obs[adata.obs.n_genes_by_counts >= 500],
            "slaf_code": lambda slaf: slaf.filter_cells(n_genes_by_counts=">=500"),
        },
        {
            "name": "max_pct_mt_15",
            "description": "Cells with <=15% mitochondrial genes",
            "h5ad_code": lambda adata: adata.obs[adata.obs.pct_counts_mt <= 15],
            "slaf_code": lambda slaf: slaf.filter_cells(pct_counts_mt="<=15"),
        },
        {
            "name": "low_mito",
            "description": "Cells with low mitochondrial content",
            "h5ad_code": lambda adata: adata.obs[~adata.obs.high_mito],
            "slaf_code": lambda slaf: slaf.filter_cells(high_mito=False),
        },
        # Cluster-based filtering (common after clustering)
        {
            "name": "clusters_0_1_2",
            "description": "Cells in clusters 0,1,2",
            "h5ad_code": lambda adata: adata.obs[
                adata.obs.leiden.isin(["0", "1", "2"])
            ],
            "slaf_code": lambda slaf: slaf.filter_cells(leiden=["0", "1", "2"]),
        },
        {
            "name": "cluster_0",
            "description": "Cells in largest cluster (0)",
            "h5ad_code": lambda adata: adata.obs[adata.obs.leiden == "0"],
            "slaf_code": lambda slaf: slaf.filter_cells(leiden="0"),
        },
        # Batch filtering (very common)
        {
            "name": "batch_1",
            "description": "Cells from batch_1",
            "h5ad_code": lambda adata: adata.obs[adata.obs.batch == "batch_1"],
            "slaf_code": lambda slaf: slaf.filter_cells(batch="batch_1"),
        },
        # Combined filtering (most realistic)
        {
            "name": "clusters_0_1_batch_1",
            "description": "Cells in clusters 0,1 from batch_1",
            "h5ad_code": lambda adata: adata.obs[
                (adata.obs.leiden.isin(["0", "1"])) & (adata.obs.batch == "batch_1")
            ],
            "slaf_code": lambda slaf: slaf.filter_cells(
                leiden=["0", "1"], batch="batch_1"
            ),
        },
        {
            "name": "high_quality",
            "description": "High-quality cells (>=1000 genes, <=10% mt)",
            "h5ad_code": lambda adata: adata.obs[
                (adata.obs.n_genes_by_counts >= 1000) & (adata.obs.pct_counts_mt <= 10)
            ],
            "slaf_code": lambda slaf: slaf.filter_cells(
                n_genes_by_counts=">=1000", pct_counts_mt="<=10"
            ),
        },
        # Additional range queries with new operators
        {
            "name": "total_counts_800_2000",
            "description": "Cells with 800-2000 total counts",
            "h5ad_code": lambda adata: adata.obs[
                (adata.obs.total_counts >= 800) & (adata.obs.total_counts <= 2000)
            ],
            "slaf_code": lambda slaf: slaf.filter_cells(total_counts=">=800").query(
                "total_counts <= 2000"
            ),
        },
        {
            "name": "genes_200_1500",
            "description": "Cells with 200-1500 genes",
            "h5ad_code": lambda adata: adata.obs[
                (adata.obs.n_genes_by_counts >= 200)
                & (adata.obs.n_genes_by_counts <= 1500)
            ],
            "slaf_code": lambda slaf: slaf.filter_cells(
                n_genes_by_counts=">=200"
            ).query("n_genes_by_counts <= 1500"),
        },
    ]
    return scenarios


def _measure_h5ad_cell_filtering(h5ad_path: str, scenario: dict):
    """Measure h5ad cell filtering performance"""
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


def _measure_slaf_cell_filtering(slaf_path: str, scenario: dict):
    """Measure SLAF cell filtering performance"""
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


def benchmark_cell_filtering_scenario(
    h5ad_path: str,
    slaf_path: str,
    scenario: dict,
):
    """Benchmark a single cell filtering scenario with isolated memory measurement"""

    # Measure h5ad in isolation
    h5ad_result = _measure_h5ad_cell_filtering(h5ad_path, scenario)

    # Measure SLAF in isolation
    slaf_result = _measure_slaf_cell_filtering(slaf_path, scenario)

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
        "scenario_type": "cell_filtering",
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


def benchmark_cell_filtering(
    h5ad_path: str, slaf_path: str, include_memory=True, verbose=False
):
    """Benchmark realistic cell filtering scenarios"""
    scenarios = demo_realistic_cell_queries()

    if verbose:
        print("Benchmarking cell filtering scenarios")
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
            result = benchmark_cell_filtering_scenario(h5ad_path, slaf_path, scenario)

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

    parser = argparse.ArgumentParser(description="Benchmark cell filtering")
    parser.add_argument("h5ad_path", help="Path to h5ad file")
    parser.add_argument("slaf_path", help="Path to SLAF dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    results = benchmark_cell_filtering(
        args.h5ad_path, args.slaf_path, verbose=args.verbose
    )

    # Print results table
    from benchmark_utils import print_benchmark_table

    print_benchmark_table(results, scenario_type="Cell Filtering")
