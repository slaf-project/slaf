import time

import numpy as np
import scanpy as sc

# Import shared utilities
from benchmark_utils import (
    clear_caches,
    get_object_memory_usage,
    get_slaf_memory_usage,
)

from slaf.core.slaf import SLAFArray


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


def _measure_h5ad_anndata_op(h5ad_path: str, scenario: dict):
    """Measure h5ad AnnData operation performance in isolation"""
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
            result = adata.X.shape if adata.X is not None else (0, 0)
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
                shape = X.shape
                result = nnz / (shape[0] * shape[1])
            elif hasattr(X, "getnnz"):
                nnz = X.getnnz()
                shape = X.shape
                result = nnz / (shape[0] * shape[1])
            else:
                nnz = np.count_nonzero(X)
                shape = X.shape
                result = nnz / (shape[0] * shape[1])
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
        result_size = None

    # Clean up
    del adata
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
    slaf_init_time = time.time() - start

    # Measure memory footprint of loaded data
    slaf_load_memory = get_slaf_memory_usage(slaf)

    # slaf operation
    start = time.time()

    if scenario["type"] == "expression_slicing":
        if scenario["operation"] == "single_cell":
            cell_id = scenario["cell_id"]
            # Convert integer index to cell ID string
            cell_ids = slaf.obs["cell_id"].to_list()[cell_id : cell_id + 1]
            result = slaf.get_cell_expression(cell_ids[0])
        elif scenario["operation"] == "single_gene":
            gene_id = scenario["gene_id"]
            # Convert integer index to gene ID string
            gene_ids = slaf.var["gene_id"].to_list()[gene_id : gene_id + 1]
            result = slaf.get_gene_expression(gene_ids[0])
        elif scenario["operation"] == "submatrix":
            cell_start, cell_end = scenario["cell_range"]
            gene_start, gene_end = scenario["gene_range"]
            result = slaf.get_submatrix(
                cell_selector=slice(cell_start, cell_end),
                gene_selector=slice(gene_start, gene_end),
            )

    elif scenario["type"] == "metadata":
        if scenario["operation"] == "obs_access":
            result = slaf.obs
        elif scenario["operation"] == "var_access":
            result = slaf.var
        elif scenario["operation"] == "obs_subset":
            columns = scenario["columns"]
            result = slaf.obs[columns]
        elif scenario["operation"] == "var_subset":
            columns = scenario["columns"]
            result = slaf.var[columns]

    elif scenario["type"] == "expression":
        if scenario["operation"] == "shape":
            result = slaf.shape
        elif scenario["operation"] == "nnz":
            # Count non-zero elements using SQL query
            nnz_result = slaf.query(
                "SELECT COUNT(*) as nnz FROM expression WHERE value > 0"
            )
            result = nnz_result.item(0, "nnz")
        elif scenario["operation"] == "density":
            # Count non-zero elements and calculate density
            nnz_result = slaf.query(
                "SELECT COUNT(*) as nnz FROM expression WHERE value > 0"
            )
            nnz = nnz_result.item(0, "nnz")
            shape = slaf.shape
            result = nnz / (shape[0] * shape[1])
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
        result_size = None

    # Clean up
    del slaf
    gc.collect()

    return {
        "slaf_init_time": slaf_init_time,
        "slaf_query_time": slaf_query_time,
        "slaf_init_memory": float(slaf_load_memory),
        "slaf_query_memory": float(slaf_query_memory),
        "result_size": result_size,
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
        "scenario_type": "anndata_ops",
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
        # Memory breakdown
        "h5ad_load_memory_mb": h5ad_result["h5ad_load_memory"],
        "h5ad_query_memory_mb": h5ad_result["h5ad_query_memory"],
        "h5ad_total_memory_mb": h5ad_result["h5ad_load_memory"]
        + h5ad_result["h5ad_query_memory"],
        "slaf_load_memory_mb": slaf_result["slaf_init_memory"],
        "slaf_query_memory_mb": slaf_result["slaf_query_memory"],
        "slaf_total_memory_mb": slaf_result["slaf_init_memory"]
        + slaf_result["slaf_query_memory"],
        # Result comparison
        "h5ad_result_size": h5ad_result["result_size"],
        "slaf_result_size": slaf_result["result_size"],
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
            print(
                f"Running scenario {i + 1}/{len(scenarios)}: {scenario['description']}"
            )

        # Clear caches at the start of each scenario
        clear_caches()

        # For the first scenario, do a burn-in run to eliminate cold start effects
        if i == 0:
            # Run the scenario once without timing to warm up
            try:
                _measure_h5ad_anndata_op(h5ad_path, scenario)
                _measure_slaf_anndata_op(slaf_path, scenario)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Burn-in failed: {e}")

        # Run the actual benchmark
        try:
            result = benchmark_anndata_op_scenario(h5ad_path, slaf_path, scenario)
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed: {e}")
            continue

    return results
