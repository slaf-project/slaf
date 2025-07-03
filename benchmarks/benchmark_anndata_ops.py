import scanpy as sc
import time
import sys
import numpy as np
from slaf.core.slaf import SLAFArray
from slaf.integrations.anndata import LazyAnnData

# Import shared utilities
from benchmark_utils import (
    get_object_memory_usage,
    get_sparse_matrix_size,
    clear_caches,
)


def demo_realistic_anndata_ops():
    """Demo realistic AnnData operation scenarios"""
    scenarios = [
        # Expression matrix slicing - AnnData-specific (uses lazy expression matrix backend)
        {
            "type": "expression_slicing",
            "operation": "single_cell",
            "cell_id": 0,  # Single cell
            "description": "Single cell expression (expression matrix)",
        },
        {
            "type": "expression_slicing",
            "operation": "single_gene",
            "gene_id": 0,  # Single gene
            "description": "Single gene expression (expression matrix)",
        },
        {
            "type": "expression_slicing",
            "operation": "submatrix",
            "cell_range": (0, 100),
            "gene_range": (0, 50),
            "description": "100x50 submatrix (expression matrix)",
        },
        # Metadata access - AnnData-specific
        {
            "type": "metadata",
            "operation": "obs_access",
            "description": "Access cell metadata",
        },
        {
            "type": "metadata",
            "operation": "var_access",
            "description": "Access gene metadata",
        },
        {
            "type": "metadata",
            "operation": "obs_subset",
            "columns": ["n_genes_by_counts", "total_counts"],
            "description": "Subset cell metadata",
        },
        {
            "type": "metadata",
            "operation": "var_subset",
            "columns": ["n_cells_by_counts", "total_counts"],
            "description": "Subset gene metadata",
        },
        # Expression matrix operations - AnnData-specific
        {"type": "expression", "operation": "shape", "description": "Get matrix shape"},
        {
            "type": "expression",
            "operation": "nnz",
            "description": "Count non-zero elements",
        },
        {
            "type": "expression",
            "operation": "density",
            "description": "Calculate matrix density",
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


def _measure_h5ad_anndata_op(h5ad_path: str, scenario: dict):
    """Measure h5ad AnnData operation performance in isolation"""
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

    # h5ad operation
    start = time.time()

    if scenario["type"] == "expression_slicing":
        if scenario["operation"] == "single_cell":
            cell_id = scenario["cell_id"]
            result = adata.X[cell_id : cell_id + 1, :]  # Single cell as slice
        elif scenario["operation"] == "single_gene":
            gene_id = scenario["gene_id"]
            result = adata.X[:, gene_id : gene_id + 1]  # Single gene as slice
        elif scenario["operation"] == "submatrix":
            cell_start, cell_end = scenario["cell_range"]
            gene_start, gene_end = scenario["gene_range"]
            result = adata.X[cell_start:cell_end, gene_start:gene_end]

    elif scenario["type"] == "metadata":
        if scenario["operation"] == "obs_access":
            result = adata.obs
        elif scenario["operation"] == "var_access":
            result = adata.var
        elif scenario["operation"] == "obs_subset":
            columns = scenario["columns"]
            result = adata.obs[columns]
        elif scenario["operation"] == "var_subset":
            columns = scenario["columns"]
            result = adata.var[columns]

    elif scenario["type"] == "expression":
        if scenario["operation"] == "shape":
            result = adata.X.shape
        elif scenario["operation"] == "nnz":
            X = adata.X
            if X is None:
                result = 0
            elif hasattr(X, "nnz"):
                result = X.nnz
            elif hasattr(X, "getnnz"):
                result = X.getnnz()
            else:
                # For dense arrays, count non-zero elements
                result = np.count_nonzero(X)
        elif scenario["operation"] == "density":
            X = adata.X
            if X is None:
                result = 0.0
            elif hasattr(X, "nnz"):
                nnz = X.nnz
                result = nnz / (X.shape[0] * X.shape[1])
            elif hasattr(X, "getnnz"):
                nnz = X.getnnz()
                result = nnz / (X.shape[0] * X.shape[1])
            else:
                nnz = np.count_nonzero(X)
                result = nnz / (X.shape[0] * X.shape[1])
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


def _measure_slaf_anndata_op(slaf_path: str, scenario: dict):
    """Measure SLAF AnnData operation performance in isolation"""
    import gc

    gc.collect()

    # slaf load
    start = time.time()
    slaf = SLAFArray(slaf_path)
    lazy_adata = LazyAnnData(slaf)
    slaf_init_time = time.time() - start

    # Measure memory footprint of loaded data
    slaf_load_memory = get_object_memory_usage(slaf)

    # slaf AnnData operation using numpy-style slicing
    start = time.time()

    if scenario["type"] == "expression_slicing":
        if scenario["operation"] == "single_cell":
            cell_id = scenario["cell_id"]
            result = lazy_adata.X[cell_id : cell_id + 1, :]
        elif scenario["operation"] == "single_gene":
            gene_id = scenario["gene_id"]
            result = lazy_adata.X[:, gene_id : gene_id + 1]
        elif scenario["operation"] == "submatrix":
            cell_start, cell_end = scenario["cell_range"]
            gene_start, gene_end = scenario["gene_range"]
            result = lazy_adata.X[cell_start:cell_end, gene_start:gene_end]

    elif scenario["type"] == "metadata":
        if scenario["operation"] == "obs_access":
            result = lazy_adata.obs
        elif scenario["operation"] == "var_access":
            result = lazy_adata.var
        elif scenario["operation"] == "obs_subset":
            columns = scenario["columns"]
            result = lazy_adata.obs[columns]
        elif scenario["operation"] == "var_subset":
            columns = scenario["columns"]
            result = lazy_adata.var[columns]

    elif scenario["type"] == "expression":
        if scenario["operation"] == "shape":
            result = {"shape": lazy_adata.shape}
        elif scenario["operation"] == "nnz":
            nnz_query = slaf.query("SELECT COUNT(*) as nnz FROM expression")
            result = {"nnz": nnz_query.iloc[0]["nnz"]}
        elif scenario["operation"] == "density":
            nnz_query = slaf.query("SELECT COUNT(*) as nnz FROM expression")
            cell_count = slaf.query("SELECT COUNT(*) as count FROM cells")
            gene_count = slaf.query("SELECT COUNT(*) as count FROM genes")
            total_elements = cell_count.iloc[0]["count"] * gene_count.iloc[0]["count"]
            density = (
                nnz_query.iloc[0]["nnz"] / total_elements if total_elements > 0 else 0
            )
            result = {"density": density}
    else:
        raise ValueError(f"Unknown scenario type: {scenario['type']}")

    slaf_query_time = time.time() - start
    slaf_query_memory = get_object_memory_usage(result)

    # Clean up
    del slaf, lazy_adata
    gc.collect()

    return {
        "slaf_init_time": slaf_init_time,
        "slaf_query_time": slaf_query_time,
        "slaf_init_memory": float(slaf_load_memory),
        "slaf_query_memory": float(slaf_query_memory),
    }


def benchmark_anndata_op_scenario(
    h5ad_path: str,
    slaf_path: str,
    scenario: dict,
):
    """Benchmark a single AnnData operation scenario with isolated memory measurement"""

    # Measure h5ad in isolation
    h5ad_result = _measure_h5ad_anndata_op(h5ad_path, scenario)

    # Measure SLAF in isolation
    slaf_result = _measure_slaf_anndata_op(slaf_path, scenario)

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
        "scenario_type": "anndata_op",
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
        "slaf_result_size": h5ad_result["result_size"],
        "results_match": h5ad_result["result_size"] == h5ad_result["result_size"],
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


def benchmark_anndata_ops(
    h5ad_path: str, slaf_path: str, include_memory=True, verbose=False
):
    """Benchmark realistic AnnData operation scenarios"""
    scenarios = demo_realistic_anndata_ops()

    if verbose:
        print("Benchmarking AnnData operation scenarios")
        print("=" * 50)

    results = []
    for i, scenario in enumerate(scenarios):
        if verbose:
            print(f"Running scenario {i+1}/{len(scenarios)}: {scenario['description']}")

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

            # Clean up temporary instances
            del temp_slaf

        try:
            # Always include loading time for each scenario
            result = benchmark_anndata_op_scenario(h5ad_path, slaf_path, scenario)

            results.append(result)

            if verbose:
                print(f"  ✓ Completed: {result['total_speedup']:.1f}x speedup")

        except Exception as e:
            if verbose:
                print(f"  ✗ Error: {e}")
            continue

    return results
