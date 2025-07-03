import pandas as pd
import numpy as np
import scanpy as sc
import time
import sys
from slaf.core.slaf import SLAFArray


def demo_realistic_gene_queries():
    """Demo realistic gene filtering scenarios for PBMC data"""
    scenarios = [
        # Expression-based filtering (very common)
        {
            "type": "filtering",
            "operation": "filter_genes",
            "filters": {"n_cells_by_counts": ">=10"},
            "description": "Genes expressed in >=10 cells",
        },
        {
            "type": "filtering",
            "operation": "filter_genes",
            "filters": {"total_counts": ">=100"},
            "description": "Genes with >=100 total counts",
        },
        {
            "type": "filtering",
            "operation": "filter_genes",
            "filters": {"mean_counts": ">=0.1"},
            "description": "Genes with mean expression >=0.1",
        },
        {
            "type": "filtering",
            "operation": "filter_genes",
            "filters": {"mt": False},
            "description": "Exclude mitochondrial genes",
        },
        # Highly variable gene filtering (post-analysis)
        {
            "type": "filtering",
            "operation": "filter_genes",
            "filters": {"highly_variable": True},
            "description": "Highly variable genes",
        },
        {
            "type": "filtering",
            "operation": "filter_genes",
            "filters": {"highly_variable": False},
            "description": "Non-highly variable genes",
        },
        # Combined filtering (most realistic)
        {
            "type": "filtering",
            "operation": "filter_genes",
            "filters": {"n_cells_by_counts": ">=50", "total_counts": ">=500"},
            "description": "Genes in >=50 cells with >=500 total counts",
        },
        # Range queries
        {
            "type": "filtering",
            "operation": "filter_genes",
            "filters": {"total_counts": ">=100", "total_counts": "<=10000"},
            "description": "Genes with 100-10000 total counts",
        },
        {
            "type": "filtering",
            "operation": "filter_genes",
            "filters": {"n_cells_by_counts": ">=5", "n_cells_by_counts": "<=1000"},
            "description": "Genes in 5-1000 cells",
        },
    ]
    return scenarios


def parse_filter_for_h5ad(filter_dict):
    """Convert filter dict to h5ad-compatible format"""
    parsed = {}
    for key, value in filter_dict.items():
        if isinstance(value, str) and value.startswith(">="):
            parsed[key] = ("ge", float(value[2:]))
        elif isinstance(value, str) and value.startswith("<="):
            parsed[key] = ("le", float(value[2:]))
        elif isinstance(value, str) and value.startswith(">"):
            parsed[key] = ("gt", float(value[1:]))
        elif isinstance(value, str) and value.startswith("<"):
            parsed[key] = ("lt", float(value[1:]))
        else:
            parsed[key] = ("eq", value)
    return parsed


def parse_filter_for_slaf(filter_dict):
    """Convert filter dict to SLAF SQL-compatible format"""
    parsed = {}
    for key, value in filter_dict.items():
        if isinstance(value, str) and value.startswith(">="):
            parsed[key] = f">= {value[2:]}"
        elif isinstance(value, str) and value.startswith("<="):
            parsed[key] = f"<= {value[2:]}"
        elif isinstance(value, str) and value.startswith(">"):
            parsed[key] = f"> {value[1:]}"
        elif isinstance(value, str) and value.startswith("<"):
            parsed[key] = f"< {value[1:]}"
        else:
            parsed[key] = value
    return parsed


def get_object_memory_usage(obj):
    """Get memory usage of a Python object in MB"""
    # For pandas objects with memory_usage method
    if hasattr(obj, "memory_usage"):
        memory_usage = obj.memory_usage(deep=True)
        if hasattr(memory_usage, "sum"):  # If it's a Series, sum it
            return memory_usage.sum() / 1024 / 1024
        else:  # If it's already a scalar
            return memory_usage / 1024 / 1024
    # For numpy arrays and similar
    elif hasattr(obj, "nbytes"):
        return obj.nbytes / 1024 / 1024
    # For sparse matrices, use comprehensive size calculation
    elif hasattr(obj, "getnnz"):
        total_bytes = get_sparse_matrix_size(obj)
        return total_bytes / 1024 / 1024
    else:
        # Fallback to sys.getsizeof
        return sys.getsizeof(obj) / 1024 / 1024


def get_sparse_matrix_size(sparse_matrix):
    """Get the total memory size of a sparse matrix in bytes"""
    total = 0
    for attr in ["data", "indices", "indptr", "row", "col", "offsets"]:
        if hasattr(sparse_matrix, attr):
            attr_data = getattr(sparse_matrix, attr)
            if attr_data is not None:
                total += attr_data.nbytes
    return total


def _measure_h5ad_gene_filtering(h5ad_path: str, scenario: dict):
    """Measure h5ad gene filtering performance in isolation"""
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

    # h5ad query - filter genes based on var metadata
    start = time.time()
    mask = pd.Series([True] * adata.n_vars, index=adata.var.index)
    for column, (op, value) in h5ad_filter.items():
        if column in adata.var.columns:
            if op == "eq":
                if isinstance(value, list):
                    mask &= adata.var[column].isin(value)
                else:
                    mask &= adata.var[column] == value
            elif op == "gt":
                mask &= adata.var[column] > value
            elif op == "lt":
                mask &= adata.var[column] < value
            elif op == "ge":
                mask &= adata.var[column] >= value
            elif op == "le":
                mask &= adata.var[column] <= value
        else:
            # Handle computed metrics that might not be in var
            if column == "n_cells_by_counts":
                # Count non-zero cells per gene
                try:
                    # Try to convert to numpy array first
                    X_array = np.array(adata.X)
                    n_cells_per_gene = (X_array > 0).sum(axis=0)
                except:
                    # Fallback to sparse matrix operations
                    n_cells_per_gene = np.array((adata.X > 0).sum(axis=0)).flatten()
                if op == "ge":
                    mask &= n_cells_per_gene >= value
                elif op == "le":
                    mask &= n_cells_per_gene <= value
            elif column == "total_counts":
                # Sum counts per gene
                try:
                    # Try to convert to numpy array first
                    X_array = np.array(adata.X)
                    total_counts_per_gene = X_array.sum(axis=0)
                except:
                    # Fallback to sparse matrix operations
                    total_counts_per_gene = np.array(adata.X.sum(axis=0)).flatten()
                if op == "ge":
                    mask &= total_counts_per_gene >= value
                elif op == "le":
                    mask &= total_counts_per_gene <= value
            elif column == "mean_counts":
                # Mean counts per gene
                try:
                    # Try to convert to numpy array first
                    X_array = np.array(adata.X)
                    mean_counts_per_gene = X_array.mean(axis=0)
                except:
                    # Fallback to sparse matrix operations
                    mean_counts_per_gene = np.array(adata.X.mean(axis=0)).flatten()
                if op == "ge":
                    mask &= mean_counts_per_gene >= value
                elif op == "le":
                    mask &= mean_counts_per_gene <= value

    filtered_genes = adata.var[mask]
    h5ad_query_time = time.time() - start
    h5ad_query_memory = get_object_memory_usage(filtered_genes)
    h5ad_count = len(filtered_genes)

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
    """Measure SLAF gene filtering performance in isolation"""
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

    # slaf query using filter_genes method
    start = time.time()
    filtered_genes_slaf = slaf.filter_genes(**slaf_filter)
    slaf_query_time = time.time() - start
    slaf_query_memory = get_object_memory_usage(filtered_genes_slaf)
    slaf_count = len(filtered_genes_slaf)

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
            print(f"Running scenario {i+1}/{len(scenarios)}: {scenario['filters']}")

        # Clear caches at the start of each scenario
        from benchmark_utils import clear_caches

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
