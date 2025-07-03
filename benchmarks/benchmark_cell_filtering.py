import pandas as pd
import scanpy as sc
import time
from slaf.core.slaf import SLAFArray

# Import shared utilities
from benchmark_utils import (
    get_object_memory_usage,
    parse_filter_for_h5ad,
    parse_filter_for_slaf,
    clear_caches,
)


def demo_realistic_cell_queries():
    """Demo realistic cell filtering scenarios for PBMC data"""
    scenarios = [
        # QC-based filtering (very common)
        {
            "type": "filtering",
            "operation": "filter_cells",
            "filters": {"n_genes_by_counts": ">=500"},
            "description": "Cells with >=500 genes",
        },
        {
            "type": "filtering",
            "operation": "filter_cells",
            "filters": {"pct_counts_mt": "<=15"},
            "description": "Cells with <=15% mitochondrial genes",
        },
        {
            "type": "filtering",
            "operation": "filter_cells",
            "filters": {"high_mito": False},
            "description": "Cells with low mitochondrial content",
        },
        # Cluster-based filtering (common after clustering)
        {
            "type": "filtering",
            "operation": "filter_cells",
            "filters": {"leiden": ["0", "1", "2"]},
            "description": "Cells in clusters 0,1,2",
        },
        {
            "type": "filtering",
            "operation": "filter_cells",
            "filters": {"leiden": "0"},
            "description": "Cells in largest cluster (0)",
        },
        # Batch filtering (very common)
        {
            "type": "filtering",
            "operation": "filter_cells",
            "filters": {"batch": "batch_1"},
            "description": "Cells from batch_1",
        },
        # Combined filtering (most realistic)
        {
            "type": "filtering",
            "operation": "filter_cells",
            "filters": {"leiden": ["0", "1"], "batch": "batch_1"},
            "description": "Cells in clusters 0,1 from batch_1",
        },
        {
            "type": "filtering",
            "operation": "filter_cells",
            "filters": {"n_genes_by_counts": ">=1000", "pct_counts_mt": "<=10"},
            "description": "High-quality cells (>=1000 genes, <=10% mt)",
        },
        # Additional range queries with new operators
        {
            "type": "filtering",
            "operation": "filter_cells",
            "filters": {"total_counts": ">=800", "total_counts": "<=2000"},
            "description": "Cells with 800-2000 total counts",
        },
        {
            "type": "filtering",
            "operation": "filter_cells",
            "filters": {"n_genes_by_counts": ">=200", "n_genes_by_counts": "<=1500"},
            "description": "Cells with 200-1500 genes",
        },
    ]
    return scenarios


def _measure_h5ad_cell_filtering(h5ad_path: str, scenario: dict):
    """Measure h5ad cell filtering performance in isolation"""
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
        + (
            get_object_memory_usage(adata.raw)
            if hasattr(adata, "raw") and adata.raw is not None
            else 0
        )
        + (get_object_memory_usage(adata.uns) if hasattr(adata, "uns") else 0)
    )

    # Parse filter for h5ad
    h5ad_filter = parse_filter_for_h5ad(scenario["filters"])

    # h5ad query
    start = time.time()
    mask = pd.Series([True] * adata.n_obs, index=adata.obs.index)
    for column, (op, value) in h5ad_filter.items():
        if column in adata.obs.columns:
            if op == "eq":
                if isinstance(value, list):
                    mask &= adata.obs[column].isin(value)
                else:
                    mask &= adata.obs[column] == value
            elif op == "gt":
                mask &= adata.obs[column] > value
            elif op == "lt":
                mask &= adata.obs[column] < value
            elif op == "ge":
                mask &= adata.obs[column] >= value
            elif op == "le":
                mask &= adata.obs[column] <= value
        else:
            # Handle missing columns gracefully
            mask = pd.Series([False] * adata.n_obs, index=adata.obs.index)
            break

    filtered_cells = adata.obs[mask]
    h5ad_query_time = time.time() - start
    h5ad_query_memory = get_object_memory_usage(filtered_cells)
    h5ad_count = len(filtered_cells)

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
    """Measure SLAF cell filtering performance in isolation"""
    import gc

    gc.collect()

    # slaf load
    start = time.time()
    slaf = SLAFArray(slaf_path)
    slaf_init_time = time.time() - start

    # Measure memory footprint of loaded metadata
    slaf_load_memory = get_object_memory_usage(slaf)

    # Parse filter for SLAF
    slaf_filter = parse_filter_for_slaf(scenario["filters"])

    # slaf query using filter_cells method
    start = time.time()
    filtered_cells_slaf = slaf.filter_cells(**slaf_filter)
    slaf_query_time = time.time() - start
    slaf_query_memory = get_object_memory_usage(filtered_cells_slaf)
    slaf_count = len(filtered_cells_slaf)

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
            print(f"Running scenario {i+1}/{len(scenarios)}: {scenario['filters']}")

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
