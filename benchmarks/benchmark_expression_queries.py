import time

import scanpy as sc
from benchmark_utils import (
    get_object_memory_usage,
    get_slaf_memory_usage,
    get_tiledb_memory_usage,
)

from slaf.core.slaf import SLAFArray

# Try to import tiledbsoma
try:
    import tiledbsoma

    TILEDB_AVAILABLE = True
except ImportError:
    TILEDB_AVAILABLE = False


def demo_realistic_expression_queries(
    h5ad_path: str, slaf_path: str, tiledb_path: str = None
):
    """Demo realistic expression query scenarios for PBMC data"""

    from slaf.core.slaf import SLAFArray

    # Load SLAF dataset
    slaf = SLAFArray(slaf_path)

    # Query cells and genes tables for IDs
    sample_cell_ids = slaf.obs["cell_id"].to_list()[:5]
    sample_gene_ids = slaf.var["gene_id"].to_list()[:5]

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
    ]
    return scenarios


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
        # Materialize the submatrix for backed mode
        result = adata.X[cell_start:cell_end, gene_start:gene_end][:]
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


def _measure_tiledb_expression_query(tiledb_path: str, scenario: dict):
    """Measure TileDB expression query performance in isolation"""
    import gc

    import polars as pl

    if not TILEDB_AVAILABLE:
        raise ImportError("TileDB SOMA is required but not available")

    gc.collect()

    # TileDB load
    start = time.time()
    experiment = tiledbsoma.Experiment.open(tiledb_path)
    X = experiment.ms["RNA"].X["data"]
    tiledb_load_time = time.time() - start
    # Measure memory footprint of the experiment object (minimal overhead)
    tiledb_load_memory = get_tiledb_memory_usage(experiment)

    # Read metadata for ID mapping
    obs_table = experiment.obs.read().concat()
    obs_df = pl.from_arrow(obs_table)
    var_table = experiment.ms["RNA"]["var"].read().concat()
    var_df = pl.from_arrow(var_table)

    # Execute the query based on scenario type
    start = time.time()
    try:
        if scenario["type"] == "single_cell":
            cell_id = scenario["cell_id"]
            # Find cell index using soma_joinid
            cell_row = obs_df.filter(pl.col("obs_id") == cell_id)
            if cell_row.height == 0:
                raise ValueError(f"Cell ID {cell_id} not found in TileDB dataset")
            cell_idx = cell_row["soma_joinid"][0]
            result = X.read(([cell_idx], slice(None))).coos().concat().to_scipy()

        elif scenario["type"] == "multiple_cells":
            cell_ids = scenario["cell_ids"]
            # Find cell indices using soma_joinid
            cell_indices = []
            for cell_id in cell_ids:
                cell_row = obs_df.filter(pl.col("obs_id") == cell_id)
                if cell_row.height == 0:
                    raise ValueError(f"Cell ID {cell_id} not found in TileDB dataset")
                cell_indices.append(cell_row["soma_joinid"][0])
            result = X.read((cell_indices, slice(None))).coos().concat().to_scipy()

        elif scenario["type"] == "single_gene":
            gene_id = scenario["gene_id"]
            # Find gene index using soma_joinid
            gene_row = var_df.filter(pl.col("var_id") == gene_id)
            if gene_row.height == 0:
                raise ValueError(f"Gene ID {gene_id} not found in TileDB dataset")
            gene_idx = gene_row["soma_joinid"][0]
            result = X.read((slice(None), [gene_idx])).coos().concat().to_scipy()

        elif scenario["type"] == "multiple_genes":
            gene_ids = scenario["gene_ids"]
            # Find gene indices using soma_joinid
            gene_indices = []
            for gene_id in gene_ids:
                gene_row = var_df.filter(pl.col("var_id") == gene_id)
                if gene_row.height == 0:
                    raise ValueError(f"Gene ID {gene_id} not found in TileDB dataset")
                gene_indices.append(gene_row["soma_joinid"][0])
            result = X.read((slice(None), gene_indices)).coos().concat().to_scipy()

        elif scenario["type"] == "submatrix":
            cell_range = scenario["cell_range"]
            gene_range = scenario["gene_range"]
            result = (
                X.read(
                    (
                        slice(cell_range[0], cell_range[1]),
                        slice(gene_range[0], gene_range[1]),
                    )
                )
                .coos()
                .concat()
                .to_scipy()
            )

        else:
            raise ValueError(f"Unknown scenario type: {scenario['type']}")

        tiledb_query_time = time.time() - start
        # Measure memory of the result (experiment + result)
        tiledb_query_memory = get_tiledb_memory_usage(experiment, result)
        result_size = (
            result.shape[0] * result.shape[1] if hasattr(result, "shape") else 0
        )

    except Exception as e:
        print(f"TileDB query failed: {e}")
        tiledb_query_time = 0
        tiledb_query_memory = 0
        result_size = 0

    # Clean up
    del experiment, obs_df, var_df
    gc.collect()

    return {
        "tiledb_load_time": tiledb_load_time,
        "tiledb_query_time": tiledb_query_time,
        "tiledb_load_memory": float(tiledb_load_memory),
        "tiledb_query_memory": float(tiledb_query_memory),
        "result_size": result_size,
    }


def benchmark_expression_query_scenario(
    h5ad_path: str,
    slaf_path: str,
    tiledb_path: str,
    scenario: dict,
):
    """Benchmark a single expression query scenario with isolated memory measurement"""

    # Measure h5ad in isolation
    h5ad_result = _measure_h5ad_expression_query(h5ad_path, scenario)

    # Measure SLAF in isolation
    slaf_result = _measure_slaf_expression_query(slaf_path, scenario)

    # Measure TileDB in isolation (if available)
    tiledb_result = None
    if TILEDB_AVAILABLE and tiledb_path:
        try:
            tiledb_result = _measure_tiledb_expression_query(tiledb_path, scenario)
        except Exception as e:
            print(f"Warning: TileDB measurement failed: {e}")

    # Calculate totals and speedups
    h5ad_total_time = h5ad_result["h5ad_load_time"] + h5ad_result["h5ad_query_time"]
    slaf_total_time = slaf_result["slaf_init_time"] + slaf_result["slaf_query_time"]

    # SLAF vs h5ad speedups
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

    # Initialize result dictionary
    result = {
        "scenario_type": "expression_query",
        "scenario_description": scenario["description"],
        "h5ad_total_time": 1000 * h5ad_total_time,
        "h5ad_load_time": 1000 * h5ad_result["h5ad_load_time"],
        "h5ad_query_time": 1000 * h5ad_result["h5ad_query_time"],
        "slaf_total_time": 1000 * slaf_total_time,
        "slaf_init_time": 1000 * slaf_result["slaf_init_time"],
        "slaf_query_time": 1000 * slaf_result["slaf_query_time"],
        "slaf_vs_h5ad_speedup": total_speedup,
        "slaf_vs_h5ad_query_speedup": query_speedup,
        "slaf_vs_h5ad_load_speedup": load_speedup,
        "h5ad_result_size": h5ad_result["result_size"],
        "slaf_result_size": slaf_result["result_size"],
        "h5ad_slaf_results_match": h5ad_result["result_size"]
        == slaf_result["result_size"],
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

    # Add TileDB results if available
    if tiledb_result:
        tiledb_total_time = (
            tiledb_result["tiledb_load_time"] + tiledb_result["tiledb_query_time"]
        )

        # SLAF vs TileDB speedups
        slaf_vs_tiledb_total_speedup = (
            tiledb_total_time / slaf_total_time if slaf_total_time > 0 else 0
        )
        slaf_vs_tiledb_query_speedup = (
            tiledb_result["tiledb_query_time"] / slaf_result["slaf_query_time"]
            if slaf_result["slaf_query_time"] > 0
            else 0
        )
        slaf_vs_tiledb_load_speedup = (
            tiledb_result["tiledb_load_time"] / slaf_result["slaf_init_time"]
            if slaf_result["slaf_init_time"] > 0
            else 0
        )

        result.update(
            {
                "tiledb_total_time": 1000 * tiledb_total_time,
                "tiledb_load_time": 1000 * tiledb_result["tiledb_load_time"],
                "tiledb_query_time": 1000 * tiledb_result["tiledb_query_time"],
                "slaf_vs_tiledb_speedup": slaf_vs_tiledb_total_speedup,
                "slaf_vs_tiledb_query_speedup": slaf_vs_tiledb_query_speedup,
                "slaf_vs_tiledb_load_speedup": slaf_vs_tiledb_load_speedup,
                "tiledb_result_size": tiledb_result["result_size"],
                "slaf_tiledb_results_match": slaf_result["result_size"]
                == tiledb_result["result_size"],
                # TileDB memory breakdown
                "tiledb_load_memory_mb": tiledb_result["tiledb_load_memory"],
                "tiledb_query_memory_mb": tiledb_result["tiledb_query_memory"],
                "tiledb_total_memory_mb": tiledb_result["tiledb_load_memory"]
                + tiledb_result["tiledb_query_memory"],
            }
        )

    return result


def benchmark_expression_queries(
    h5ad_path: str,
    slaf_path: str,
    tiledb_path: str = None,
    include_memory=True,
    verbose=False,
):
    """Benchmark realistic expression query scenarios"""
    scenarios = demo_realistic_expression_queries(h5ad_path, slaf_path, tiledb_path)

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

            # Create temporary instances for burn-in
            temp_slaf = SLAFArray(slaf_path)

            # Warm up both SLAF and TileDB for fair comparison
            from benchmark_utils import warm_up_slaf_database, warm_up_tiledb_database

            # Warm up SLAF
            warm_up_slaf_database(temp_slaf, verbose=verbose)

            # Warm up TileDB if available
            if TILEDB_AVAILABLE and tiledb_path:
                try:
                    temp_experiment = tiledbsoma.Experiment.open(tiledb_path)
                    warm_up_tiledb_database(temp_experiment, verbose=verbose)
                    temp_experiment.close()
                except Exception as e:
                    if verbose:
                        print(f"    Warning: TileDB warm-up failed: {e}")

            # Clear caches again after burn-in
            clear_caches()

            # Clean up temporary instances
            del temp_slaf

        try:
            # Always include loading time for each scenario
            result = benchmark_expression_query_scenario(
                h5ad_path, slaf_path, tiledb_path, scenario
            )

            results.append(result)

            if verbose:
                print(f"  ✓ Completed: {result['slaf_vs_h5ad_speedup']:.1f}x speedup")

        except Exception as e:
            if verbose:
                print(f"  ✗ Error: {e}")
            continue

    return results


def main():
    """CLI interface for expression queries benchmark"""
    import argparse

    from benchmark_utils import print_benchmark_table

    parser = argparse.ArgumentParser(
        description="Benchmark expression queries across h5ad, SLAF, and TileDB"
    )
    parser.add_argument("h5ad_path", help="Path to h5ad file")
    parser.add_argument("slaf_path", help="Path to SLAF file")
    parser.add_argument("tiledb_path", help="Path to TileDB file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run the benchmark
    results = benchmark_expression_queries(
        h5ad_path=args.h5ad_path,
        slaf_path=args.slaf_path,
        tiledb_path=args.tiledb_path,
        verbose=args.verbose,
    )

    # Print results table
    if results:
        print_benchmark_table(results)
    else:
        print("No benchmark results to display")


if __name__ == "__main__":
    main()
