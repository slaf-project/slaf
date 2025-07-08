import time

import numpy as np
import scanpy as sc

# Import shared utilities
from benchmark_utils import (
    clear_caches,
    get_object_memory_usage,
)

from slaf.core.slaf import SLAFArray
from slaf.integrations.anndata import LazyAnnData
from slaf.integrations.scanpy import pp


def demo_realistic_scanpy_preprocessing():
    """Demo realistic Scanpy preprocessing scenarios"""
    scenarios = [
        # QC metrics calculation
        {
            "type": "qc_metrics",
            "operation": "calculate_qc_metrics",
            "description": "Calculate QC metrics",
        },
        # Cell filtering
        {
            "type": "filtering",
            "operation": "filter_cells",
            "min_counts": 500,
            "min_genes": 200,
            "description": "Filter cells (min_counts=500, min_genes=200)",
        },
        {
            "type": "filtering",
            "operation": "filter_cells",
            "min_counts": 1000,
            "min_genes": 500,
            "description": "Filter cells (min_counts=1000, min_genes=500)",
        },
        {
            "type": "filtering",
            "operation": "filter_cells",
            "max_counts": 5000,
            "max_genes": 2500,
            "description": "Filter cells (max_counts=5000, max_genes=2500)",
        },
        # Gene filtering
        {
            "type": "filtering",
            "operation": "filter_genes",
            "min_counts": 10,
            "min_cells": 5,
            "description": "Filter genes (min_counts=10, min_cells=5)",
        },
        {
            "type": "filtering",
            "operation": "filter_genes",
            "min_counts": 50,
            "min_cells": 10,
            "description": "Filter genes (min_counts=50, min_cells=10)",
        },
        # Normalization
        {
            "type": "normalization",
            "operation": "normalize_total",
            "target_sum": 1e4,
            "description": "Normalize total (target_sum=1e4)",
        },
        {
            "type": "normalization",
            "operation": "normalize_total",
            "target_sum": 1e6,
            "description": "Normalize total (target_sum=1e6)",
        },
        # Log transformation
        {
            "type": "transformation",
            "operation": "log1p",
            "description": "Log1p transformation",
        },
        # Highly variable genes
        {
            "type": "hvg",
            "operation": "highly_variable_genes",
            "min_mean": 0.0125,
            "max_mean": 3,
            "min_disp": 0.5,
            "description": "Find highly variable genes",
        },
        {
            "type": "hvg",
            "operation": "highly_variable_genes",
            "n_top_genes": 2000,
            "description": "Find top 2000 highly variable genes",
        },
        # Combined workflows
        {
            "type": "workflow",
            "operation": "qc_and_filter",
            "description": "QC metrics + cell filtering + gene filtering",
        },
        # Transformed data operations (lazy evaluation scenarios)
        {
            "type": "transformed_ops",
            "operation": "normalize_then_slice",
            "slice_size": (100, 50),
            "description": "Normalize total + slice 100x50 submatrix (lazy)",
        },
        {
            "type": "transformed_ops",
            "operation": "log1p_then_slice",
            "slice_size": (200, 100),
            "description": "Log1p + slice 200x100 submatrix (lazy)",
        },
        {
            "type": "transformed_ops",
            "operation": "normalize_log1p_then_slice",
            "slice_size": (500, 250),
            "description": "Normalize + Log1p + slice 500x250 submatrix (lazy)",
        },
        {
            "type": "transformed_ops",
            "operation": "transformed_aggregation",
            "description": "Normalize + Log1p + mean per gene (lazy)",
        },
        {
            "type": "transformed_ops",
            "operation": "transformed_statistics",
            "description": "Normalize + Log1p + variance per cell (lazy)",
        },
    ]
    return scenarios


def _measure_h5ad_scanpy_preprocessing(h5ad_path: str, scenario: dict):
    """Measure h5ad scanpy preprocessing performance in isolation"""
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

    # h5ad scanpy preprocessing
    start = time.time()

    if scenario["type"] == "qc_metrics":
        if scenario["operation"] == "calculate_qc_metrics":
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            result = adata

    elif scenario["type"] == "filtering":
        if scenario["operation"] == "filter_cells":
            min_counts = scenario.get("min_counts")
            min_genes = scenario.get("min_genes")
            max_counts = scenario.get("max_counts")
            max_genes = scenario.get("max_genes")

            # scanpy only accepts one parameter at a time, so we need to chain them
            if min_counts is not None:
                sc.pp.filter_cells(adata, min_counts=min_counts, inplace=True)
            if min_genes is not None:
                sc.pp.filter_cells(adata, min_genes=min_genes, inplace=True)
            if max_counts is not None:
                sc.pp.filter_cells(adata, max_counts=max_counts, inplace=True)
            if max_genes is not None:
                sc.pp.filter_cells(adata, max_genes=max_genes, inplace=True)
            result = adata

        elif scenario["operation"] == "filter_genes":
            min_counts = scenario.get("min_counts")
            min_cells = scenario.get("min_cells")

            # scanpy only accepts one parameter at a time, so we need to chain them
            if min_counts is not None:
                sc.pp.filter_genes(adata, min_counts=min_counts, inplace=True)
            if min_cells is not None:
                sc.pp.filter_genes(adata, min_cells=min_cells, inplace=True)
            result = adata

    elif scenario["type"] == "normalization":
        if scenario["operation"] == "normalize_total":
            target_sum = scenario.get("target_sum", 1e4)
            sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
            result = adata

    elif scenario["type"] == "transformation":
        if scenario["operation"] == "log1p":
            # scanpy log1p doesn't have inplace parameter, it returns a new object
            result = sc.pp.log1p(adata)

    elif scenario["type"] == "hvg":
        if scenario["operation"] == "highly_variable_genes":
            min_mean = scenario.get("min_mean", 0.0125)
            max_mean = scenario.get("max_mean", 3)
            min_disp = scenario.get("min_disp", 0.5)
            n_top_genes = scenario.get("n_top_genes")
            if n_top_genes:
                sc.pp.highly_variable_genes(
                    adata, n_top_genes=n_top_genes, inplace=True
                )
            else:
                sc.pp.highly_variable_genes(
                    adata,
                    min_mean=min_mean,
                    max_mean=max_mean,
                    min_disp=min_disp,
                    inplace=True,
                )
            result = adata

    elif scenario["type"] == "workflow":
        if scenario["operation"] == "qc_and_filter":
            # Calculate QC metrics
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            # Filter cells (chain the filters)
            sc.pp.filter_cells(adata, min_counts=500, inplace=True)
            sc.pp.filter_cells(adata, min_genes=200, inplace=True)
            # Filter genes (chain the filters)
            sc.pp.filter_genes(adata, min_counts=10, inplace=True)
            sc.pp.filter_genes(adata, min_cells=5, inplace=True)
            result = adata

    elif scenario["type"] == "transformed_ops":
        if scenario["operation"] == "normalize_then_slice":
            # Apply normalization
            sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
            # Slice the transformed data
            slice_size = scenario["slice_size"]
            result = adata[0 : slice_size[0], 0 : slice_size[1]].X

        elif scenario["operation"] == "log1p_then_slice":
            # Apply log1p transformation
            transformed_adata = sc.pp.log1p(adata)
            # Check if log1p returned a valid object
            if transformed_adata is None:
                transformed_adata = adata
            # Slice the transformed data
            slice_size = scenario["slice_size"]
            result = transformed_adata[0 : slice_size[0], 0 : slice_size[1]].X

        elif scenario["operation"] == "normalize_log1p_then_slice":
            # Apply both transformations
            sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
            transformed_adata = sc.pp.log1p(adata)
            # Check if log1p returned a valid object
            if transformed_adata is None:
                transformed_adata = adata
            # Slice the transformed data
            slice_size = scenario["slice_size"]
            result = transformed_adata[0 : slice_size[0], 0 : slice_size[1]].X

        elif scenario["operation"] == "transformed_aggregation":
            # Apply transformations
            sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
            transformed_adata = sc.pp.log1p(adata)
            # Check if log1p returned a valid object
            if transformed_adata is None:
                transformed_adata = adata
            # Perform aggregation on transformed data
            try:
                if hasattr(transformed_adata.X, "mean"):
                    result = transformed_adata.X.mean(axis=0)
                else:
                    # Fallback for non-numpy arrays
                    result = np.array(transformed_adata.X).mean(axis=0)
            except Exception:
                result = np.array([0])  # Fallback

        elif scenario["operation"] == "transformed_statistics":
            # Apply transformations
            sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
            transformed_adata = sc.pp.log1p(adata)
            # Check if log1p returned a valid object
            if transformed_adata is None:
                transformed_adata = adata
            # Perform statistics on transformed data
            try:
                if hasattr(transformed_adata.X, "var"):
                    result = transformed_adata.X.var(axis=1)
                else:
                    # Fallback for non-numpy arrays
                    result = np.array(transformed_adata.X).var(axis=1)
            except Exception:
                result = np.array([0])  # Fallback

    else:
        raise ValueError(f"Unknown scenario type: {scenario['type']}")

    h5ad_query_time = time.time() - start
    h5ad_query_memory = get_object_memory_usage(result)

    # Calculate result size for comparison
    if result is None:
        result_size = None
    elif hasattr(result, "shape"):
        result_size = result.shape
    elif hasattr(result, "__len__"):
        try:
            result_size = len(result)
        except (TypeError, AttributeError):
            result_size = 1
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


def _measure_slaf_scanpy_preprocessing(slaf_path: str, scenario: dict):
    """Measure SLAF scanpy preprocessing performance in isolation"""
    import gc

    gc.collect()

    # slaf load
    start = time.time()
    slaf = SLAFArray(slaf_path)
    lazy_adata = LazyAnnData(slaf, backend="scipy")
    slaf_init_time = time.time() - start

    # Measure memory footprint of loaded data
    slaf_load_memory = get_object_memory_usage(slaf) + get_object_memory_usage(
        lazy_adata
    )

    # slaf scanpy preprocessing using the new integration
    start = time.time()

    try:
        if scenario["type"] == "qc_metrics":
            if scenario["operation"] == "calculate_qc_metrics":
                result = pp.calculate_qc_metrics(lazy_adata, inplace=False)

        elif scenario["type"] == "filtering":
            if scenario["operation"] == "filter_cells":
                min_counts = scenario.get("min_counts")
                min_genes = scenario.get("min_genes")
                max_counts = scenario.get("max_counts")
                max_genes = scenario.get("max_genes")

                result = pp.filter_cells(
                    lazy_adata,
                    min_counts=min_counts,
                    min_genes=min_genes,
                    max_counts=max_counts,
                    max_genes=max_genes,
                    inplace=False,
                )

            elif scenario["operation"] == "filter_genes":
                min_counts = scenario.get("min_counts")
                min_cells = scenario.get("min_cells")

                result = pp.filter_genes(
                    lazy_adata,
                    min_counts=min_counts,
                    min_cells=min_cells,
                    inplace=False,
                )

        elif scenario["type"] == "normalization":
            if scenario["operation"] == "normalize_total":
                target_sum = scenario.get("target_sum", 1e4)
                result = pp.normalize_total(
                    lazy_adata, target_sum=target_sum, inplace=False
                )

        elif scenario["type"] == "transformation":
            if scenario["operation"] == "log1p":
                result = pp.log1p(lazy_adata, inplace=False)

        elif scenario["type"] == "hvg":
            if scenario["operation"] == "highly_variable_genes":
                min_mean = scenario.get("min_mean", 0.0125)
                max_mean = scenario.get("max_mean", 3)
                min_disp = scenario.get("min_disp", 0.5)
                n_top_genes = scenario.get("n_top_genes")

                if n_top_genes:
                    result = pp.highly_variable_genes(
                        lazy_adata, n_top_genes=n_top_genes, inplace=False
                    )
                else:
                    result = pp.highly_variable_genes(
                        lazy_adata,
                        min_mean=min_mean,
                        max_mean=max_mean,
                        min_disp=min_disp,
                        inplace=False,
                    )

        elif scenario["type"] == "workflow":
            if scenario["operation"] == "qc_and_filter":
                # Combined workflow - use inplace=False to get results
                pp.calculate_qc_metrics(lazy_adata, inplace=True)
                # For operations that return None or tuples, use the original lazy_adata
                pp.filter_cells(lazy_adata, min_counts=500, min_genes=200, inplace=True)
                result = pp.filter_genes(
                    lazy_adata, min_counts=10, min_cells=5, inplace=False
                )

        elif scenario["type"] == "transformed_ops":
            if scenario["operation"] == "normalize_then_slice":
                # Apply normalization
                pp.normalize_total(lazy_adata, target_sum=1e4, inplace=True)
                # Slice the transformed data
                slice_size = scenario["slice_size"]
                result = lazy_adata.X[0 : slice_size[0], 0 : slice_size[1]]

            elif scenario["operation"] == "log1p_then_slice":
                # Apply log1p transformation
                pp.log1p(lazy_adata, inplace=True)
                # Slice the transformed data
                slice_size = scenario["slice_size"]
                result = lazy_adata.X[0 : slice_size[0], 0 : slice_size[1]]

            elif scenario["operation"] == "normalize_log1p_then_slice":
                # Apply both transformations
                pp.normalize_total(lazy_adata, target_sum=1e4, inplace=True)
                pp.log1p(lazy_adata, inplace=True)
                # Slice the transformed data
                slice_size = scenario["slice_size"]
                result = lazy_adata.X[0 : slice_size[0], 0 : slice_size[1]]

            elif scenario["operation"] == "transformed_aggregation":
                # Apply transformations
                pp.normalize_total(lazy_adata, target_sum=1e4, inplace=True)
                pp.log1p(lazy_adata, inplace=True)
                # Perform aggregation on transformed data
                result = lazy_adata.X.mean(axis=0)

            elif scenario["operation"] == "transformed_statistics":
                # Apply transformations
                pp.normalize_total(lazy_adata, target_sum=1e4, inplace=True)
                pp.log1p(lazy_adata, inplace=True)
                # Perform statistics on transformed data
                result = lazy_adata.X.var(axis=1)

        else:
            # Default operation
            result = lazy_adata

        # Calculate result size for comparison
        if result is None:
            result_size = None
        elif hasattr(result, "shape"):
            result_size = result.shape
        elif hasattr(result, "__len__"):
            try:
                result_size = len(result)
            except (TypeError, AttributeError):
                result_size = 1
        else:
            result_size = 1

    except Exception as e:
        # If any operation fails, return None to indicate failure
        print(f"  SLAF operation failed for {scenario['description']}: {e}")
        result = None
        result_size = None

    slaf_query_time = time.time() - start
    slaf_query_memory = get_object_memory_usage(result) if result is not None else 0

    # Clean up
    del slaf, lazy_adata
    if result is not None:
        del result
    gc.collect()

    return {
        "slaf_init_time": slaf_init_time,
        "slaf_query_time": slaf_query_time,
        "slaf_init_memory": float(slaf_load_memory),
        "slaf_query_memory": float(slaf_query_memory),
        "result_size": result_size,
    }


def benchmark_scanpy_preprocessing_scenario(
    h5ad_path: str,
    slaf_path: str,
    scenario: dict,
):
    """Benchmark a single scanpy preprocessing scenario with isolated memory measurement"""

    # Measure h5ad in isolation
    try:
        h5ad_result = _measure_h5ad_scanpy_preprocessing(h5ad_path, scenario)
    except Exception as e:
        print(f"  h5ad operation failed for {scenario['description']}: {e}")
        return None

    # Measure SLAF in isolation
    try:
        slaf_result = _measure_slaf_scanpy_preprocessing(slaf_path, scenario)
    except Exception as e:
        print(f"  SLAF operation failed for {scenario['description']}: {e}")
        return None

    # Check if either operation failed
    if h5ad_result is None or slaf_result is None:
        return None

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
        "scenario_type": "scanpy_preprocessing",
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
        "results_match": (
            h5ad_result["result_size"] == slaf_result["result_size"]
            if h5ad_result["result_size"] is not None
            and slaf_result["result_size"] is not None
            else False
        ),
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


def benchmark_scanpy_preprocessing(
    h5ad_path: str, slaf_path: str, include_memory=True, verbose=False
):
    """Benchmark realistic scanpy preprocessing scenarios"""
    scenarios = demo_realistic_scanpy_preprocessing()

    if verbose:
        print("Benchmarking scanpy preprocessing scenarios")
        print("=" * 50)

    results = []
    failed_scenarios = []

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
            result = benchmark_scanpy_preprocessing_scenario(
                h5ad_path, slaf_path, scenario
            )

            if result is not None:
                results.append(result)
                if verbose:
                    print(f"  ✓ Completed: {result['total_speedup']:.1f}x speedup")
            else:
                failed_scenarios.append(scenario["description"])
                if verbose:
                    print(f"  ✗ Failed: {scenario['description']}")

        except Exception as e:
            failed_scenarios.append(scenario["description"])
            if verbose:
                print(f"  ✗ Error: {e}")
            continue

    # Print summary of failures
    if failed_scenarios:
        print(f"\n⚠️  {len(failed_scenarios)} scenarios failed:")
        for failed in failed_scenarios:
            print(f"   - {failed}")

    print(f"\n✅ {len(results)} scenarios completed successfully")

    return results
