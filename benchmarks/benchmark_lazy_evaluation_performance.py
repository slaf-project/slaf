#!/usr/bin/env python3
"""
Phase 0.4: Lazy Evaluation Performance Benchmark

This benchmark validates the performance and memory benefits of lazy evaluation
using real datasets. It compares the old query() approach vs the new
lazy_query(...).compute() approach across various scenarios.

Usage:
    python benchmark.py run --types lazy_evaluation_performance
"""

import gc
import time
from pathlib import Path

import psutil
from benchmark_utils import clear_caches

from slaf import SLAFArray
from slaf.integrations.anndata import LazyAnnData
from slaf.integrations.scanpy import pp


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def demo_lazy_evaluation_scenarios(h5ad_path: str, slaf_path: str):
    """Define realistic lazy evaluation performance scenarios"""

    # Load SLAF dataset to get dimensions and available columns
    slaf = SLAFArray(slaf_path)

    # Get available columns for queries
    cell_columns = (
        list(slaf.obs.columns) if hasattr(slaf, "obs") and slaf.obs is not None else []
    )
    gene_columns = (
        list(slaf.var.columns) if hasattr(slaf, "var") and slaf.var is not None else []
    )

    # Find a good column for grouping (prefer batch, cell_type, etc.)
    group_column = None
    for col in ["batch", "cell_type", "leiden"]:
        if col in cell_columns:
            group_column = col
            break
    if not group_column and cell_columns:
        group_column = cell_columns[0]

    # Find a good gene column for grouping
    gene_group_column = None
    for col in ["highly_variable", "gene_type"]:
        if col in gene_columns:
            gene_group_column = col
            break
    if not gene_group_column and gene_columns:
        gene_group_column = gene_columns[0]

    scenarios = [
        # Basic query scenarios
        {
            "type": "basic_query",
            "description": "Count cells by group",
            "query": (
                f"SELECT {group_column}, COUNT(*) as count FROM cells GROUP BY {group_column} ORDER BY count DESC"
                if group_column
                else "SELECT COUNT(*) as count FROM cells"
            ),
        },
        {
            "type": "basic_query",
            "description": "Count genes by group",
            "query": (
                f"SELECT {gene_group_column}, COUNT(*) as count FROM genes GROUP BY {gene_group_column} ORDER BY count DESC"
                if gene_group_column
                else "SELECT COUNT(*) as count FROM genes"
            ),
        },
        {
            "type": "basic_query",
            "description": "Count expression records",
            "query": "SELECT COUNT(*) as count FROM expression",
        },
        # Complex aggregation scenarios
        {
            "type": "complex_aggregation",
            "description": "Cell-gene expression analysis",
            "query": (
                f"""
                SELECT
                    c.{group_column},
                    g.{gene_group_column},
                    COUNT(DISTINCT e.cell_id) as cells_expressed,
                    AVG(e.value) as avg_expression,
                    MAX(e.value) as max_expression,
                    SUM(e.value) as total_expression
                FROM cells c
                JOIN expression e ON c.cell_integer_id = e.cell_integer_id
                JOIN genes g ON e.gene_integer_id = g.gene_integer_id
                WHERE c.total_counts > 1000
                GROUP BY c.{group_column}, g.{gene_group_column}
                HAVING COUNT(DISTINCT e.cell_id) > 10
                ORDER BY total_expression DESC
                LIMIT 100
            """
                if group_column and gene_group_column
                else """
                SELECT
                    COUNT(DISTINCT e.cell_id) as cells_expressed,
                    AVG(e.value) as avg_expression,
                    MAX(e.value) as max_expression,
                    SUM(e.value) as total_expression
                FROM expression e
                WHERE e.value > 0
            """
            ),
        },
        # NEW: Lazy query composition scenarios
        {
            "type": "lazy_composition",
            "description": "Basic lazy query composition",
            "composition_steps": [
                "SELECT * FROM cells",
                "filter: total_counts > 1000",
                "select: cell_id, batch, total_counts",
                "group_by: batch",
                "select: batch, COUNT(*) as count, AVG(total_counts) as avg_counts",
                "order_by: avg_counts DESC",
            ],
        },
        {
            "type": "lazy_composition",
            "description": "Complex lazy query composition",
            "composition_steps": [
                "SELECT * FROM expression",
                "filter: value > 0",
                "select: cell_id, gene_id, value",
                "group_by: cell_id",
                "select: cell_id, COUNT(*) as genes_expressed, AVG(value) as avg_expression",
                "order_by: avg_expression DESC",
                "limit: 100",
            ],
        },
        {
            "type": "lazy_composition",
            "description": "Multi-step lazy composition",
            "composition_steps": [
                "SELECT * FROM cells",
                "filter: total_counts > 500",
                "select: cell_id, batch, total_counts, n_genes_by_counts",
                "filter: n_genes_by_counts > 200",
                "group_by: batch",
                "select: batch, COUNT(*) as cell_count, AVG(total_counts) as avg_counts, AVG(n_genes_by_counts) as avg_genes",
                "order_by: avg_counts DESC",
                "limit: 50",
            ],
        },
        # NEW: Query building scenarios
        {
            "type": "query_building",
            "description": "Dynamic query building",
            "building_steps": [
                "base: SELECT * FROM cells",
                "add_filter: total_counts > 1000",
                "add_filter: n_genes_by_counts > 200",
                "add_select: cell_id, batch, total_counts",
                "add_group_by: batch",
                "add_select: batch, COUNT(*) as count, AVG(total_counts) as avg_counts",
                "add_order_by: avg_counts DESC",
            ],
        },
        {
            "type": "query_building",
            "description": "Conditional query building",
            "building_steps": [
                "base: SELECT * FROM expression",
                "add_filter: value > 0",
                "add_select: cell_id, gene_id, value",
                "add_group_by: cell_id",
                "add_select: cell_id, COUNT(*) as genes_expressed, SUM(value) as total_expression",
                "add_order_by: total_expression DESC",
                "add_limit: 100",
            ],
        },
        # Scanpy preprocessing scenarios
        {
            "type": "scanpy_pipeline",
            "description": "QC metrics calculation",
            "pipeline_steps": ["qc_metrics"],
        },
        {
            "type": "scanpy_pipeline",
            "description": "Cell filtering",
            "pipeline_steps": ["filter_cells"],
        },
        {
            "type": "scanpy_pipeline",
            "description": "Gene filtering",
            "pipeline_steps": ["filter_genes"],
        },
        {
            "type": "scanpy_pipeline",
            "description": "Normalization",
            "pipeline_steps": ["normalize_total"],
        },
        {
            "type": "scanpy_pipeline",
            "description": "Log transform",
            "pipeline_steps": ["log1p"],
        },
        {
            "type": "scanpy_pipeline",
            "description": "Full preprocessing pipeline",
            "pipeline_steps": [
                "qc_metrics",
                "filter_cells",
                "filter_genes",
                "normalize_total",
                "log1p",
            ],
        },
        # Memory efficiency scenarios
        {
            "type": "memory_efficiency",
            "description": "Repeated basic queries",
            "queries": [
                "SELECT COUNT(*) as count FROM cells",
                "SELECT COUNT(*) as count FROM genes",
                "SELECT COUNT(*) as count FROM expression",
                (
                    f"SELECT {group_column}, COUNT(*) as count FROM cells GROUP BY {group_column}"
                    if group_column
                    else "SELECT COUNT(*) as count FROM cells"
                ),
                (
                    f"SELECT {gene_group_column}, COUNT(*) as count FROM genes GROUP BY {gene_group_column}"
                    if gene_group_column
                    else "SELECT COUNT(*) as count FROM genes"
                ),
            ],
        },
        # NEW: Memory efficiency with lazy composition
        {
            "type": "memory_efficiency_lazy",
            "description": "Repeated lazy composition queries",
            "composition_queries": [
                [
                    "SELECT * FROM cells",
                    "filter: total_counts > 1000",
                    "select: cell_id, batch",
                ],
                [
                    "SELECT * FROM genes",
                    "filter: highly_variable = true",
                    "select: gene_id",
                ],
                [
                    "SELECT * FROM expression",
                    "filter: value > 0",
                    "select: cell_id, gene_id, value",
                ],
            ],
        },
        # Large dataset scenarios
        {
            "type": "large_dataset",
            "description": "Large expression query",
            "query": "SELECT * FROM expression LIMIT 100000",
        },
        {
            "type": "large_dataset",
            "description": "Complex join query",
            "query": """
                SELECT c.cell_id, g.gene_id, e.value
                FROM cells c
                JOIN expression e ON c.cell_integer_id = e.cell_integer_id
                JOIN genes g ON e.gene_integer_id = g.gene_integer_id
                LIMIT 50000
            """,
        },
        # NEW: Large dataset with lazy composition
        {
            "type": "large_dataset_lazy",
            "description": "Large lazy composition query",
            "composition_steps": [
                "SELECT * FROM expression",
                "filter: value > 0",
                "select: cell_id, gene_id, value",
                "group_by: cell_id",
                "select: cell_id, COUNT(*) as genes_expressed, AVG(value) as avg_expression",
                "order_by: avg_expression DESC",
                "limit: 10000",
            ],
        },
    ]

    return scenarios


def _measure_old_approach(slaf_path: str, scenario: dict):
    """Measure performance using the old query() approach"""

    gc.collect()

    # Load SLAF
    start_time = time.time()
    slaf = SLAFArray(slaf_path)
    load_time = time.time() - start_time

    # Measure memory after loading
    load_memory = get_memory_usage()

    # Execute query
    start_time = time.time()

    if scenario["type"] == "basic_query":
        result = slaf.query(scenario["query"])
    elif scenario["type"] == "complex_aggregation":
        result = slaf.query(scenario["query"])
    elif scenario["type"] == "large_dataset":
        result = slaf.query(scenario["query"])
    else:
        raise ValueError(f"Unknown scenario type: {scenario['type']}")

    query_time = time.time() - start_time

    # Measure memory after query
    query_memory = get_memory_usage()

    # Get result size
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
        "load_time": load_time,
        "query_time": query_time,
        "load_memory": load_memory,
        "query_memory": query_memory,
        "result_size": result_size,
    }


def _measure_new_approach(slaf_path: str, scenario: dict):
    """Measure performance using the new lazy_query(...).compute() approach"""

    gc.collect()

    # Load SLAF
    start_time = time.time()
    slaf = SLAFArray(slaf_path)
    load_time = time.time() - start_time

    # Measure memory after loading
    load_memory = get_memory_usage()

    # Execute lazy query
    start_time = time.time()

    if scenario["type"] == "basic_query":
        result = slaf.lazy_query(scenario["query"]).compute()
    elif scenario["type"] == "complex_aggregation":
        result = slaf.lazy_query(scenario["query"]).compute()
    elif scenario["type"] == "large_dataset":
        result = slaf.lazy_query(scenario["query"]).compute()
    elif scenario["type"] == "lazy_composition":
        # Build lazy query step by step
        composition_steps = scenario["composition_steps"]
        lazy_query = None

        for step in composition_steps:
            if step.startswith("SELECT"):
                lazy_query = slaf.lazy_query(step)
            elif step.startswith("filter:") and lazy_query is not None:
                condition = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.filter(condition)
            elif step.startswith("select:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.select(columns)
            elif step.startswith("group_by:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.group_by(columns)
            elif step.startswith("order_by:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.order_by(columns)
            elif step.startswith("limit:") and lazy_query is not None:
                limit = int(step.split(":", 1)[1].strip())
                lazy_query = lazy_query.limit(limit)

        if lazy_query is not None:
            result = lazy_query.compute()
        else:
            raise ValueError("Failed to build lazy query")
    elif scenario["type"] == "query_building":
        # Build query dynamically
        building_steps = scenario["building_steps"]
        lazy_query = None

        for step in building_steps:
            if step.startswith("base:"):
                sql = step.split(":", 1)[1].strip()
                lazy_query = slaf.lazy_query(sql)
            elif step.startswith("add_filter:") and lazy_query is not None:
                condition = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.filter(condition)
            elif step.startswith("add_select:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.select(columns)
            elif step.startswith("add_group_by:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.group_by(columns)
            elif step.startswith("add_order_by:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.order_by(columns)
            elif step.startswith("add_limit:") and lazy_query is not None:
                limit = int(step.split(":", 1)[1].strip())
                lazy_query = lazy_query.limit(limit)

        if lazy_query is not None:
            result = lazy_query.compute()
        else:
            raise ValueError("Failed to build lazy query")
    elif scenario["type"] == "large_dataset_lazy":
        # Build large lazy composition query
        composition_steps = scenario["composition_steps"]
        lazy_query = None

        for step in composition_steps:
            if step.startswith("SELECT"):
                lazy_query = slaf.lazy_query(step)
            elif step.startswith("filter:") and lazy_query is not None:
                condition = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.filter(condition)
            elif step.startswith("select:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.select(columns)
            elif step.startswith("group_by:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.group_by(columns)
            elif step.startswith("order_by:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.order_by(columns)
            elif step.startswith("limit:") and lazy_query is not None:
                limit = int(step.split(":", 1)[1].strip())
                lazy_query = lazy_query.limit(limit)

        if lazy_query is not None:
            result = lazy_query.compute()
        else:
            raise ValueError("Failed to build lazy query")
    else:
        raise ValueError(f"Unknown scenario type: {scenario['type']}")

    query_time = time.time() - start_time

    # Measure memory after query
    query_memory = get_memory_usage()

    # Get result size
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
        "load_time": load_time,
        "query_time": query_time,
        "load_memory": load_memory,
        "query_memory": query_memory,
        "result_size": result_size,
    }


def _measure_scanpy_pipeline(slaf_path: str, scenario: dict):
    """Measure scanpy preprocessing pipeline performance"""

    gc.collect()

    # Load SLAF and create LazyAnnData
    start_time = time.time()
    slaf = SLAFArray(slaf_path)
    adata = LazyAnnData(slaf)
    load_time = time.time() - start_time

    # Measure memory after loading
    load_memory = get_memory_usage()

    # Execute pipeline steps
    start_time = time.time()

    pipeline_steps = scenario["pipeline_steps"]

    for step in pipeline_steps:
        if step == "qc_metrics":
            pp.calculate_qc_metrics(adata)
        elif step == "filter_cells":
            pp.filter_cells(adata, min_counts=100, min_genes=50)
        elif step == "filter_genes":
            pp.filter_genes(adata, min_counts=10, min_cells=5)
        elif step == "normalize_total":
            pp.normalize_total(adata, target_sum=1e4)
        elif step == "log1p":
            pp.log1p(adata)

    # Final data access
    subset = adata[:1000, :5000]  # First 1000 cells, first 5000 genes
    X = subset.X.compute()

    query_time = time.time() - start_time

    # Measure memory after pipeline
    query_memory = get_memory_usage()

    # Get result size
    result_size = X.shape

    # Clean up
    del slaf, adata, subset, X
    gc.collect()

    return {
        "load_time": load_time,
        "query_time": query_time,
        "load_memory": load_memory,
        "query_memory": query_memory,
        "result_size": result_size,
    }


def _measure_memory_efficiency(slaf_path: str, scenario: dict):
    """Measure memory efficiency during repeated operations"""

    gc.collect()

    # Load SLAF
    start_time = time.time()
    slaf = SLAFArray(slaf_path)
    load_time = time.time() - start_time

    # Measure memory after loading
    initial_memory = get_memory_usage()
    load_memory = initial_memory

    # Execute repeated queries
    start_time = time.time()

    queries = scenario["queries"]
    memory_readings = [initial_memory]

    for query in queries:
        result = slaf.lazy_query(query).compute()
        current_memory = get_memory_usage()
        memory_readings.append(current_memory)
        del result

    query_time = time.time() - start_time

    # Measure final memory
    final_memory = get_memory_usage()
    query_memory = final_memory

    # Calculate memory increase
    total_increase = final_memory - initial_memory

    # Clean up
    del slaf
    gc.collect()

    return {
        "load_time": load_time,
        "query_time": query_time,
        "load_memory": load_memory,
        "query_memory": query_memory,
        "result_size": len(queries),
        "memory_increase": total_increase,
        "memory_readings": memory_readings,
    }


def _measure_memory_efficiency_lazy(slaf_path: str, scenario: dict):
    """Measure memory efficiency during repeated lazy composition operations"""

    gc.collect()

    # Load SLAF
    start_time = time.time()
    slaf = SLAFArray(slaf_path)
    load_time = time.time() - start_time

    # Measure memory after loading
    initial_memory = get_memory_usage()
    load_memory = initial_memory

    # Execute repeated lazy composition queries
    start_time = time.time()

    composition_queries = scenario["composition_queries"]
    memory_readings = [initial_memory]

    for composition_steps in composition_queries:
        # Build lazy query step by step
        lazy_query = None

        for step in composition_steps:
            if step.startswith("SELECT"):
                lazy_query = slaf.lazy_query(step)
            elif step.startswith("filter:") and lazy_query is not None:
                condition = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.filter(condition)
            elif step.startswith("select:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.select(columns)
            elif step.startswith("group_by:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.group_by(columns)
            elif step.startswith("order_by:") and lazy_query is not None:
                columns = step.split(":", 1)[1].strip()
                lazy_query = lazy_query.order_by(columns)
            elif step.startswith("limit:") and lazy_query is not None:
                limit = int(step.split(":", 1)[1].strip())
                lazy_query = lazy_query.limit(limit)

        if lazy_query is not None:
            result = lazy_query.compute()
            current_memory = get_memory_usage()
            memory_readings.append(current_memory)
            del result
        else:
            raise ValueError("Failed to build lazy query")

    query_time = time.time() - start_time

    # Measure final memory
    final_memory = get_memory_usage()
    query_memory = final_memory

    # Calculate memory increase
    total_increase = final_memory - initial_memory

    # Clean up
    del slaf
    gc.collect()

    return {
        "load_time": load_time,
        "query_time": query_time,
        "load_memory": load_memory,
        "query_memory": query_memory,
        "result_size": len(composition_queries),
        "memory_increase": total_increase,
        "memory_readings": memory_readings,
    }


def benchmark_lazy_evaluation_scenario(
    h5ad_path: str,
    slaf_path: str,
    scenario: dict,
):
    """Benchmark a single lazy evaluation scenario"""

    # Handle different scenario types
    if scenario["type"] in ["basic_query", "complex_aggregation", "large_dataset"]:
        # Compare old vs new approach
        old_result = _measure_old_approach(slaf_path, scenario)
        new_result = _measure_new_approach(slaf_path, scenario)

        # Calculate speedups
        old_total_time = old_result["load_time"] + old_result["query_time"]
        new_total_time = new_result["load_time"] + new_result["query_time"]

        total_speedup = old_total_time / new_total_time if new_total_time > 0 else 0
        query_speedup = (
            old_result["query_time"] / new_result["query_time"]
            if new_result["query_time"] > 0
            else 0
        )
        load_speedup = (
            old_result["load_time"] / new_result["load_time"]
            if new_result["load_time"] > 0
            else 0
        )

        # Calculate memory efficiency
        old_total_memory = old_result["load_memory"] + old_result["query_memory"]
        new_total_memory = new_result["load_memory"] + new_result["query_memory"]
        memory_efficiency = (
            old_total_memory / new_total_memory if new_total_memory > 0 else 0
        )

        return {
            "scenario_type": "lazy_evaluation_comparison",
            "scenario_description": scenario["description"],
            "h5ad_total_time": 1000 * old_total_time,  # Convert to ms
            "h5ad_load_time": 1000 * old_result["load_time"],
            "h5ad_query_time": 1000 * old_result["query_time"],
            "slaf_total_time": 1000 * new_total_time,
            "slaf_load_time": 1000 * new_result["load_time"],
            "slaf_query_time": 1000 * new_result["query_time"],
            "total_speedup": total_speedup,
            "query_speedup": query_speedup,
            "load_speedup": load_speedup,
            "h5ad_total_memory_mb": old_total_memory,
            "slaf_total_memory_mb": new_total_memory,
            "memory_efficiency": memory_efficiency,
            "results_match": old_result["result_size"] == new_result["result_size"],
        }

    elif scenario["type"] in [
        "lazy_composition",
        "query_building",
        "large_dataset_lazy",
    ]:
        # Measure new lazy composition approach only
        result = _measure_new_approach(slaf_path, scenario)

        return {
            "scenario_type": "lazy_composition_only",
            "scenario_description": scenario["description"],
            "slaf_total_time": 1000 * (result["load_time"] + result["query_time"]),
            "slaf_load_time": 1000 * result["load_time"],
            "slaf_query_time": 1000 * result["query_time"],
            "slaf_total_memory_mb": result["query_memory"],
            "composition_steps": len(
                scenario.get("composition_steps", scenario.get("building_steps", []))
            ),
            "result_size": result["result_size"],
        }

    elif scenario["type"] == "scanpy_pipeline":
        # Measure scanpy pipeline performance
        result = _measure_scanpy_pipeline(slaf_path, scenario)

        return {
            "scenario_type": "scanpy_pipeline",
            "scenario_description": scenario["description"],
            "slaf_total_time": 1000 * (result["load_time"] + result["query_time"]),
            "slaf_load_time": 1000 * result["load_time"],
            "slaf_query_time": 1000 * result["query_time"],
            "slaf_total_memory_mb": result["query_memory"],
            "pipeline_steps": len(scenario["pipeline_steps"]),
            "result_shape": result["result_size"],
        }

    elif scenario["type"] == "memory_efficiency":
        # Measure memory efficiency
        result = _measure_memory_efficiency(slaf_path, scenario)

        return {
            "scenario_type": "memory_efficiency",
            "scenario_description": scenario["description"],
            "slaf_total_time": 1000 * result["query_time"],
            "slaf_total_memory_mb": result["query_memory"],
            "memory_increase_mb": result["memory_increase"],
            "queries_executed": result["result_size"],
        }

    elif scenario["type"] == "memory_efficiency_lazy":
        # Measure memory efficiency with lazy composition
        result = _measure_memory_efficiency_lazy(slaf_path, scenario)

        return {
            "scenario_type": "memory_efficiency_lazy",
            "scenario_description": scenario["description"],
            "slaf_total_time": 1000 * result["query_time"],
            "slaf_total_memory_mb": result["query_memory"],
            "memory_increase_mb": result["memory_increase"],
            "queries_executed": result["result_size"],
        }

    else:
        raise ValueError(f"Unknown scenario type: {scenario['type']}")


def benchmark_lazy_evaluation_performance(
    h5ad_path: str, slaf_path: str, include_memory=True, verbose=False
):
    """Benchmark lazy evaluation performance across various scenarios"""

    scenarios = demo_lazy_evaluation_scenarios(h5ad_path, slaf_path)

    if verbose:
        print("Benchmarking lazy evaluation performance")
        print("=" * 60)
        print(f"Dataset: {Path(slaf_path).name}")
        print(f"Scenarios: {len(scenarios)}")
        print()

    results = []

    for i, scenario in enumerate(scenarios):
        if verbose:
            print(
                f"Running scenario {i + 1}/{len(scenarios)}: {scenario['description']}"
            )

        # Clear caches at the start of each scenario
        clear_caches()

        # For the first scenario, do a burn-in run
        if i == 0:
            if verbose:
                print("  Running burn-in for first scenario...")

            # Create a temporary SLAF instance for burn-in
            temp_slaf = SLAFArray(slaf_path)

            # Warm up the database
            from benchmark_utils import warm_up_slaf_database

            warm_up_slaf_database(temp_slaf, verbose=verbose)

            # Clear caches again after burn-in
            clear_caches()

            # Clean up temporary instance
            del temp_slaf

        try:
            result = benchmark_lazy_evaluation_scenario(h5ad_path, slaf_path, scenario)
            results.append(result)

            if verbose:
                if (
                    "total_speedup" in result
                    and result.get("total_speedup") is not None
                ):
                    print(
                        f"  ✓ Completed: {result['total_speedup']:.2f}x speedup, "
                        f"{result.get('memory_efficiency', 0):.2f}x memory efficiency"
                    )
                elif result.get("scenario_type") == "scanpy_pipeline":
                    print(
                        f"  ✓ Completed: {result['slaf_total_time']:.1f}ms, "
                        f"{result.get('slaf_total_memory_mb', 0):.1f}MB used"
                    )
                elif result.get("scenario_type") == "lazy_composition_only":
                    print(
                        f"  ✓ Completed: {result['slaf_total_time']:.1f}ms, "
                        f"{result.get('slaf_total_memory_mb', 0):.1f}MB used, "
                        f"{result.get('composition_steps', 0)} steps"
                    )
                elif result.get("scenario_type") == "memory_efficiency":
                    print(
                        f"  ✓ Completed: {result['slaf_total_time']:.1f}ms, "
                        f"+{result.get('memory_increase_mb', 0):.1f}MB increase"
                    )
                elif result.get("scenario_type") == "memory_efficiency_lazy":
                    print(
                        f"  ✓ Completed: {result['slaf_total_time']:.1f}ms, "
                        f"+{result.get('memory_increase_mb', 0):.1f}MB increase"
                    )
                else:
                    print("  ✓ Completed.")

        except Exception as e:
            if verbose:
                print(f"  ✗ Error: {e}")
            continue

    if verbose:
        print(f"\n✅ Completed {len(results)} scenarios")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark lazy evaluation performance"
    )
    parser.add_argument("--h5ad", required=True, help="Path to h5ad file")
    parser.add_argument("--slaf", required=True, help="Path to SLAF file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run the benchmark
    results = benchmark_lazy_evaluation_performance(
        args.h5ad, args.slaf, verbose=args.verbose
    )

    print(f"\n✅ Benchmark completed with {len(results)} scenarios")
