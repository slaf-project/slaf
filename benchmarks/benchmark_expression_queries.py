import sys
import time

import scanpy as sc
from benchmark_utils import get_slaf_memory_usage

from slaf.core.slaf import SLAFArray


def demo_realistic_expression_queries(h5ad_path: str, slaf_path: str):
    """Demo realistic expression query scenarios for PBMC data"""

    from slaf.core.slaf import SLAFArray

    # Load SLAF dataset
    slaf = SLAFArray(slaf_path)

    # Query cells and genes tables for IDs
    sample_cell_ids = slaf.obs.index[:5].tolist()
    sample_gene_ids = slaf.var.index[:5].tolist()

    # Get dataset dimensions for submatrix scenarios
    n_cells, n_genes = slaf.shape

    scenarios = [
        # Single cell expression queries
        {
            "type": "single_cell",
            "cell_id": sample_cell_ids[0],
            "description": "Single cell expression",
        },
        {
            "type": "single_cell",
            "cell_id": sample_cell_ids[1],
            "description": "Another single cell",
        },
        # Multiple cells expression queries
        {
            "type": "multiple_cells",
            "cell_ids": sample_cell_ids[:2],
            "description": "Two cells",
        },
        {
            "type": "multiple_cells",
            "cell_ids": sample_cell_ids[:3],
            "description": "Three cells",
        },
        # Single gene expression queries
        {
            "type": "single_gene",
            "gene_id": sample_gene_ids[0],
            "description": "Single gene across all cells",
        },
        {
            "type": "single_gene",
            "gene_id": sample_gene_ids[1],
            "description": "Another single gene",
        },
        # Multiple genes expression queries
        {
            "type": "multiple_genes",
            "gene_ids": sample_gene_ids[:2],
            "description": "Two genes",
        },
        {
            "type": "multiple_genes",
            "gene_ids": sample_gene_ids[:3],
            "description": "Three genes",
        },
        # Submatrix queries (cells x genes) - use safe ranges
        {
            "type": "submatrix",
            "cell_range": (0, min(100, n_cells)),
            "gene_range": (0, min(50, n_genes)),
            "description": "100x50 submatrix",
        },
        {
            "type": "submatrix",
            "cell_range": (0, min(500, n_cells)),
            "gene_range": (0, min(100, n_genes)),
            "description": "500x100 submatrix",
        },
        {
            "type": "submatrix",
            "cell_range": (min(1000, n_cells // 2), min(1500, n_cells)),
            "gene_range": (min(500, n_genes // 2), min(1000, n_genes)),
            "description": "500x500 submatrix",
        },
        # Additional aggregation operations
        {
            "type": "aggregation",
            "operation": "count",
            "axis": 0,
            "description": "Number of cells expressing each gene",
        },
        {
            "type": "aggregation",
            "operation": "count",
            "axis": 1,
            "description": "Number of genes expressed in each cell",
        },
    ]
    return scenarios


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


def _measure_h5ad_expression_query(h5ad_path: str, scenario: dict):
    """Measure h5ad expression query performance in isolation"""
    import gc

    gc.collect()

    # h5ad load
    start = time.time()
    adata = sc.read_h5ad(h5ad_path, backed="r")
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

    # h5ad expression query
    start = time.time()

    if scenario["type"] == "single_cell":
        cell_id = scenario["cell_id"]
        cell_idx = adata.obs.index.get_loc(cell_id)
        result = adata.X[cell_idx, :]

    elif scenario["type"] == "multiple_cells":
        cell_ids = scenario["cell_ids"]
        cell_indices = [adata.obs.index.get_loc(cid) for cid in cell_ids]
        result = adata.X[cell_indices, :]

    elif scenario["type"] == "single_gene":
        gene_id = scenario["gene_id"]
        gene_idx = adata.var.index.get_loc(gene_id)
        result = adata.X[:, gene_idx]

    elif scenario["type"] == "multiple_genes":
        gene_ids = scenario["gene_ids"]
        gene_indices = [adata.var.index.get_loc(gid) for gid in gene_ids]
        result = adata.X[:, gene_indices]

    elif scenario["type"] == "submatrix":
        cell_range = scenario["cell_range"]
        gene_range = scenario["gene_range"]
        cell_start, cell_end = cell_range
        gene_start, gene_end = gene_range
        result = adata.X[cell_start:cell_end, gene_start:gene_end]

    elif scenario["type"] == "aggregation":
        operation = scenario["operation"]
        axis = scenario["axis"]
        if operation == "mean":
            result = adata.X.mean(axis=axis)
        elif operation == "sum":
            result = adata.X.sum(axis=axis)
        elif operation == "count":
            # Count non-zero elements (expressed genes/cells)
            result = (adata.X != 0).sum(axis=axis)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    else:
        raise ValueError(f"Unknown scenario type: {scenario['type']}")

    h5ad_query_time = time.time() - start
    h5ad_query_memory = get_object_memory_usage(result)

    # Get result size for comparison
    if hasattr(result, "shape"):
        result_size = result.shape
    elif hasattr(result, "__len__"):
        result_size = len(result)
    else:
        result_size = 1

    # Clean up
    del adata, result
    gc.collect()

    return {
        "h5ad_load_time": h5ad_load_time,
        "h5ad_query_time": h5ad_query_time,
        "h5ad_load_memory": float(h5ad_load_memory),
        "h5ad_query_memory": float(h5ad_query_memory),
        "result_size": result_size,
    }


def _measure_slaf_expression_query(slaf_path: str, scenario: dict):
    """Measure SLAF expression query performance in isolation"""
    import gc

    gc.collect()

    # slaf load
    start = time.time()
    slaf = SLAFArray(slaf_path)
    slaf_init_time = time.time() - start

    # Measure memory footprint of loaded metadata
    slaf_load_memory = get_slaf_memory_usage(slaf)

    # slaf expression query
    start = time.time()

    if scenario["type"] == "single_cell":
        cell_id = scenario["cell_id"]
        result = slaf.get_cell_expression(cell_id)

    elif scenario["type"] == "multiple_cells":
        cell_ids = scenario["cell_ids"]
        result = slaf.get_cell_expression(cell_ids)

    elif scenario["type"] == "single_gene":
        gene_id = scenario["gene_id"]
        result = slaf.get_gene_expression(gene_id)

    elif scenario["type"] == "multiple_genes":
        gene_ids = scenario["gene_ids"]
        result = slaf.get_gene_expression(gene_ids)

    elif scenario["type"] == "submatrix":
        cell_range = scenario["cell_range"]
        gene_range = scenario["gene_range"]
        cell_start, cell_end = cell_range
        gene_start, gene_end = gene_range
        result = slaf.get_submatrix(
            cell_selector=slice(cell_start, cell_end),
            gene_selector=slice(gene_start, gene_end),
        )
    else:
        raise ValueError(f"Unknown scenario type: {scenario['type']}")

    slaf_query_time = time.time() - start
    slaf_query_memory = get_object_memory_usage(result)

    # Get result size for comparison
    if hasattr(result, "shape"):
        result_size = result.shape
    elif hasattr(result, "__len__"):
        result_size = len(result)
    else:
        result_size = 1

    # Clean up
    del slaf, result
    gc.collect()

    return {
        "slaf_init_time": slaf_init_time,
        "slaf_query_time": slaf_query_time,
        "slaf_init_memory": float(slaf_load_memory),
        "slaf_query_memory": float(slaf_query_memory),
        "result_size": result_size,
    }


def benchmark_expression_query_scenario(
    h5ad_path: str,
    slaf_path: str,
    scenario: dict,
):
    """Benchmark a single expression query scenario with isolated memory measurement"""

    # Measure h5ad in isolation
    h5ad_result = _measure_h5ad_expression_query(h5ad_path, scenario)

    # Measure SLAF in isolation
    slaf_result = _measure_slaf_expression_query(slaf_path, scenario)

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
        "scenario_type": "expression_query",
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
        "h5ad_result_size": h5ad_result["result_size"],
        "slaf_result_size": slaf_result["result_size"],
        "results_match": h5ad_result["result_size"] == slaf_result["result_size"],
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


def benchmark_expression_queries(
    h5ad_path: str, slaf_path: str, include_memory=True, verbose=False
):
    """Benchmark realistic expression query scenarios"""
    scenarios = demo_realistic_expression_queries(h5ad_path, slaf_path)

    if verbose:
        print("Benchmarking expression query scenarios")
        print("=" * 50)

    results = []
    for i, scenario in enumerate(scenarios):
        if verbose:
            print(
                f"Running scenario {i + 1}/{len(scenarios)}: {scenario['description']}"
            )

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
            result = benchmark_expression_query_scenario(h5ad_path, slaf_path, scenario)

            results.append(result)

            if verbose:
                print(f"  ✓ Completed: {result['total_speedup']:.1f}x speedup")

        except Exception as e:
            if verbose:
                print(f"  ✗ Error: {e}")
            continue

    return results
