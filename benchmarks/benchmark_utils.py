import sys

import numpy as np
from rich.console import Console
from rich.table import Table


def get_object_memory_usage(obj):
    """Get memory usage of a Python object in MB"""
    # For pandas objects with memory_usage method
    if hasattr(obj, "memory_usage"):
        memory_usage = obj.memory_usage(deep=True)
        if hasattr(memory_usage, "sum"):  # If it's a Series, sum it
            return memory_usage.sum() / 1024 / 1024
        else:  # If it's already a scalar
            return memory_usage / 1024 / 1024
    # For polars DataFrames - estimate memory usage based on data types and shape
    elif hasattr(obj, "estimated_size") and hasattr(obj, "shape"):
        # Use polars' built-in size estimation
        return obj.estimated_size() / 1024 / 1024
    # For polars Series - estimate based on data type and length
    elif hasattr(obj, "estimated_size") and hasattr(obj, "len"):
        return obj.estimated_size() / 1024 / 1024
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


def get_slaf_memory_usage(slaf_obj):
    """Get comprehensive memory usage of a SLAF object including all data attributes"""
    total_memory = 0.0

    # Measure the SLAF object itself
    total_memory += get_object_memory_usage(slaf_obj)

    # Measure cell metadata (obs)
    if hasattr(slaf_obj, "obs") and slaf_obj.obs is not None:
        total_memory += get_object_memory_usage(slaf_obj.obs)

    # Measure gene metadata (var)
    if hasattr(slaf_obj, "var") and slaf_obj.var is not None:
        total_memory += get_object_memory_usage(slaf_obj.var)

    # Measure Lance datasets (these are file-based but may have some in-memory components)
    # Note: Lance datasets are primarily file-based, but we measure any in-memory components
    for attr_name in ["expression", "cells", "genes"]:
        if hasattr(slaf_obj, attr_name) and getattr(slaf_obj, attr_name) is not None:
            total_memory += get_object_memory_usage(getattr(slaf_obj, attr_name))

    # Measure LanceDB connection and tables
    if hasattr(slaf_obj, "lancedb_conn") and slaf_obj.lancedb_conn is not None:
        total_memory += get_object_memory_usage(slaf_obj.lancedb_conn)

    for attr_name in ["expression_table", "cells_table", "genes_table"]:
        if hasattr(slaf_obj, attr_name) and getattr(slaf_obj, attr_name) is not None:
            total_memory += get_object_memory_usage(getattr(slaf_obj, attr_name))

    # Measure config dictionary
    if hasattr(slaf_obj, "config") and slaf_obj.config is not None:
        total_memory += get_object_memory_usage(slaf_obj.config)

    return total_memory


def get_sparse_matrix_size(sparse_matrix):
    """Get the total memory size of a sparse matrix in bytes"""
    total = 0
    for attr in ["data", "indices", "indptr", "row", "col", "offsets"]:
        if hasattr(sparse_matrix, attr):
            attr_data = getattr(sparse_matrix, attr)
            if attr_data is not None:
                total += attr_data.nbytes
    return total


def print_benchmark_table(
    results: list[dict], dataset_name: str = "", scenario_type: str = ""
):
    """Print a comprehensive benchmark results table

    Note: All timing measurements include loading time for each scenario,
    providing true end-to-end performance comparisons.
    """

    if not results:
        print("No benchmark results to display")
        return

    # Check if this is a three-way comparison (includes TileDB)
    has_tiledb = any("tiledb_total_time" in r for r in results)

    if has_tiledb:
        # Three-way comparison: SLAF vs h5ad and SLAF vs TileDB
        slaf_vs_h5ad_speedups = [
            r["slaf_vs_h5ad_speedup"] for r in results if r["slaf_vs_h5ad_speedup"] > 0
        ]
        slaf_vs_tiledb_speedups = [
            r["slaf_vs_tiledb_speedup"]
            for r in results
            if r["slaf_vs_tiledb_speedup"] > 0
        ]

        avg_slaf_vs_h5ad_speedup = (
            np.mean(slaf_vs_h5ad_speedups) if slaf_vs_h5ad_speedups else 0
        )
        avg_slaf_vs_tiledb_speedup = (
            np.mean(slaf_vs_tiledb_speedups) if slaf_vs_tiledb_speedups else 0
        )

        # Memory statistics for three-way comparison
        memory_results = [r for r in results if "h5ad_total_memory_mb" in r]
        if memory_results:
            avg_h5ad_total_memory = np.mean(
                [r["h5ad_total_memory_mb"] for r in memory_results]
            )
            avg_slaf_total_memory = np.mean(
                [r["slaf_total_memory_mb"] for r in memory_results]
            )
            avg_tiledb_total_memory = np.mean(
                [r["tiledb_total_memory_mb"] for r in memory_results]
            )
            memory_efficiency = (
                avg_h5ad_total_memory / avg_slaf_total_memory
                if avg_slaf_total_memory > 0
                else 0
            )
        else:
            avg_h5ad_total_memory = avg_slaf_total_memory = avg_tiledb_total_memory = (
                memory_efficiency
            ) = 0

        _print_rich_table_three_way(
            results,
            dataset_name,
            scenario_type,
            avg_slaf_vs_h5ad_speedup,
            avg_slaf_vs_tiledb_speedup,
            memory_efficiency,
            avg_h5ad_total_memory,
            avg_slaf_total_memory,
            avg_tiledb_total_memory,
        )
    else:
        # Original two-way comparison (SLAF vs h5ad only)
        total_speedups = [r["total_speedup"] for r in results if r["total_speedup"] > 0]
        query_speedups = [r["query_speedup"] for r in results if r["query_speedup"] > 0]
        load_speedups = [
            r["load_speedup"] for r in results if r.get("load_speedup", 0) > 0
        ]

        avg_total_speedup = np.mean(total_speedups) if total_speedups else 0
        avg_query_speedup = np.mean(query_speedups) if query_speedups else 0
        avg_load_speedup = np.mean(load_speedups) if load_speedups else 0

        # Memory statistics
        memory_results = [r for r in results if "h5ad_total_memory_mb" in r]
        if memory_results:
            # Calculate memory efficiency for each result using the same logic as comprehensive summary
            memory_efficiencies = []
            for r in memory_results:
                h5ad_mem = r.get("h5ad_total_memory_mb", 0)
                slaf_mem = r.get("slaf_total_memory_mb", 0)

                if slaf_mem > 0.1 and h5ad_mem > 0.1:
                    # Both used significant memory
                    mem_eff = h5ad_mem / slaf_mem
                    memory_efficiencies.append(mem_eff)
                elif h5ad_mem > 0.1 and slaf_mem <= 0.1:
                    # h5ad used memory but SLAF didn't - use actual ratio
                    mem_eff = h5ad_mem / 0.01  # Assume SLAF used ~0.01 MB
                    memory_efficiencies.append(mem_eff)
                elif h5ad_mem <= 0.1 and slaf_mem <= 0.1:
                    # Both used negligible memory - treat as 1x
                    memory_efficiencies.append(1.0)

            if memory_efficiencies:
                avg_h5ad_total_memory = np.mean(
                    [r["h5ad_total_memory_mb"] for r in memory_results]
                )
                avg_slaf_total_memory = np.mean(
                    [r["slaf_total_memory_mb"] for r in memory_results]
                )
                memory_efficiency = np.mean(memory_efficiencies)
            else:
                avg_h5ad_total_memory = avg_slaf_total_memory = memory_efficiency = 0
        else:
            avg_h5ad_total_memory = avg_slaf_total_memory = memory_efficiency = 0

        _print_rich_table(
            results,
            dataset_name,
            scenario_type,
            avg_total_speedup,
            avg_query_speedup,
            avg_load_speedup,
            memory_efficiency,
            avg_h5ad_total_memory,
            avg_slaf_total_memory,
        )


def _print_rich_table(
    results,
    dataset_name,
    scenario_type,
    avg_total_speedup,
    avg_query_speedup,
    avg_load_speedup,
    memory_efficiency,
    avg_h5ad_total_memory,
    avg_slaf_total_memory,
):
    """Print table using rich library for better formatting"""
    console = Console()

    # Create main table
    title = f"BENCHMARK RESULTS: {dataset_name.upper() if dataset_name else 'ALL SCENARIOS'}"
    if scenario_type:
        title += f" - {scenario_type.upper()}"

    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
        title_style="bold blue",
    )

    # Add columns with 2-row header structure
    table.add_column("Scenario", style="cyan", width=8)
    table.add_column("h5ad\nLoad (ms)", justify="right", style="red", width=6)
    table.add_column("h5ad\nQuery (ms)", justify="right", style="red", width=6)
    table.add_column("h5ad\nTotal (ms)", justify="right", style="red", width=6)
    table.add_column("SLAF\nLoad (ms)", justify="right", style="green", width=6)
    table.add_column("SLAF\nQuery (ms)", justify="right", style="green", width=6)
    table.add_column("SLAF\nTotal (ms)", justify="right", style="green", width=6)
    table.add_column("Total\nSpeedup", justify="right", style="yellow", width=8)
    table.add_column("Query\nSpeedup", justify="right", style="yellow", width=8)
    table.add_column("h5ad\nLoad (MB)", justify="right", style="red", width=6)
    table.add_column("h5ad\nQuery (MB)", justify="right", style="red", width=6)
    table.add_column("h5ad\nTotal (MB)", justify="right", style="red", width=6)
    table.add_column("SLAF\nLoad (MB)", justify="right", style="green", width=6)
    table.add_column("SLAF\nQuery (MB)", justify="right", style="green", width=6)
    table.add_column("SLAF\nTotal (MB)", justify="right", style="green", width=6)
    table.add_column("Memory\nEfficiency", justify="right", style="blue", width=12)

    # Add description column if available
    has_description = any("scenario_description" in r for r in results)
    if has_description:
        table.add_column("Description", style="white", width=20)

    # Add data rows
    for i, result in enumerate(results):
        scenario_name = f"S{i + 1}"

        # Timing breakdown (already in milliseconds)
        h5ad_load = result.get("h5ad_load_time", 0)
        h5ad_query = result.get("h5ad_query_time", 0)
        h5ad_total = result.get("h5ad_total_time", 0)

        slaf_load = result.get("slaf_init_time", 0)
        slaf_query = result.get("slaf_query_time", 0)
        slaf_total = result.get("slaf_total_time", 0)

        total_speedup = result.get("total_speedup", 0)
        query_speedup = result.get("query_speedup", 0)

        # Memory info
        h5ad_load_memory = result.get("h5ad_load_memory_mb", 0)
        h5ad_query_memory = result.get("h5ad_query_memory_mb", 0)
        h5ad_total_memory = result.get("h5ad_total_memory_mb", 0)

        slaf_load_memory = result.get("slaf_load_memory_mb", 0)
        slaf_query_memory = result.get("slaf_query_memory_mb", 0)
        slaf_total_memory = result.get("slaf_total_memory_mb", 0)

        # Calculate memory efficiency (h5ad total / SLAF total)
        if (
            slaf_total_memory > 0.01
        ):  # Lower threshold to catch small but meaningful differences
            mem_efficiency = h5ad_total_memory / slaf_total_memory
            mem_efficiency_text = f"{mem_efficiency:.1f}x"
            # Color code memory efficiency
            if mem_efficiency > 1:
                mem_efficiency_text = f"[green]{mem_efficiency_text}[/green]"
            else:
                mem_efficiency_text = f"[red]{mem_efficiency_text}[/red]"
        elif h5ad_total_memory > 0.01:  # h5ad used memory but SLAF didn't
            # Calculate a more realistic efficiency ratio
            mem_efficiency = h5ad_total_memory / 0.01  # Assume SLAF used ~0.01 MB
            if mem_efficiency > 100:
                mem_efficiency_text = (
                    "[green]>100x[/green]"  # SLAF is much more efficient
                )
            else:
                mem_efficiency_text = f"[green]{mem_efficiency:.1f}x[/green]"
        else:  # Both used negligible memory
            mem_efficiency_text = "~1x"

        # Format speedup with color coding
        total_speedup_text = f"{total_speedup:.1f}x" if total_speedup > 0 else "N/A"
        query_speedup_text = f"{query_speedup:.1f}x" if query_speedup > 0 else "N/A"

        # Color code speedups
        if total_speedup > 1:
            total_speedup_text = f"[green]{total_speedup_text}[/green]"
        elif total_speedup > 0:
            total_speedup_text = f"[red]{total_speedup_text}[/red]"

        if query_speedup > 1:
            query_speedup_text = f"[green]{query_speedup_text}[/green]"
        elif query_speedup > 0:
            query_speedup_text = f"[red]{query_speedup_text}[/red]"

        # Build row data
        row_data = [
            scenario_name,
            f"{h5ad_load:.1f}",
            f"{h5ad_query:.1f}",
            f"{h5ad_total:.1f}",
            f"{slaf_load:.1f}",
            f"{slaf_query:.1f}",
            f"{slaf_total:.1f}",
            total_speedup_text,
            query_speedup_text,
            f"{h5ad_load_memory:.1f}",
            f"{h5ad_query_memory:.1f}",
            f"{h5ad_total_memory:.1f}",
            f"{slaf_load_memory:.1f}",
            f"{slaf_query_memory:.1f}",
            f"{slaf_total_memory:.1f}",
            mem_efficiency_text,
        ]

        # Add description if available
        if has_description:
            description = result.get("scenario_description", "")
            row_data.append(description)

        table.add_row(*row_data)

    # Print the table
    console.print(table)

    # Print summary
    summary_table = Table(title="SUMMARY", show_header=False, box=None)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Average Total Speedup", f"{avg_total_speedup:.1f}x")
    summary_table.add_row("Average Query Speedup", f"{avg_query_speedup:.1f}x")
    if avg_load_speedup > 0:
        summary_table.add_row("Average Load Speedup", f"{avg_load_speedup:.1f}x")
    if memory_efficiency > 0:
        summary_table.add_row("Memory Efficiency", f"{memory_efficiency:.1f}x")
        summary_table.add_row("Avg h5ad Memory (MB)", f"{avg_h5ad_total_memory:.1f}")
        summary_table.add_row("Avg SLAF Memory (MB)", f"{avg_slaf_total_memory:.1f}")

    console.print(summary_table)


def clear_caches():
    """Clear all potential caches that could affect benchmark timing"""
    import gc

    gc.collect()

    # Clear DuckDB query cache if possible
    try:
        import duckdb

        duckdb.query("PRAGMA clear_cache;")
    except Exception:
        pass  # Ignore if DuckDB cache clearing fails

    # Clear any Lance dataset caches
    try:
        pass

        # Note: This would need to be called on specific SLAF instances
        # We'll handle this in the burn_in function
    except Exception:
        pass  # Ignore if Lance cache clearing fails


def warm_up_slaf_database(slaf_instance, verbose=False):
    """Comprehensive warm-up for SLAF database to eliminate cold start effects

    This function performs multiple warm-up operations:
    1. Clears all caches
    2. Forces dataset reload
    3. Runs representative queries to prime the database
    4. Warms up common query patterns

    Args:
        slaf_instance: SLAF instance to warm up
        verbose: Whether to print warm-up status
    """
    if verbose:
        console = Console()
        console.print("  🔥 Warming up SLAF database...")

    # Clear all caches first
    clear_caches()

    # Clear Lance dataset caches if SLAF instance provided
    if slaf_instance is not None:
        try:
            # Force reload of Lance datasets to clear any internal caches
            slaf_instance._setup_datasets()
        except Exception:
            pass  # Ignore if Lance cache clearing fails

    # Run representative warm-up queries to prime the database
    warmup_queries = [
        # Basic metadata queries
        "SELECT COUNT(*) FROM cells",
        "SELECT COUNT(*) FROM genes",
        "SELECT COUNT(*) FROM expression",
        # Common filtering patterns
        "SELECT cell_integer_id FROM cells LIMIT 10",
        "SELECT gene_integer_id FROM genes LIMIT 10",
        # Expression queries
        "SELECT cell_integer_id, gene_integer_id, value FROM expression WHERE cell_integer_id = 0 LIMIT 10",
        "SELECT cell_integer_id, gene_integer_id, value FROM expression WHERE gene_integer_id = 0 LIMIT 10",
        # Aggregation queries
        "SELECT MIN(value) AS min_value, MAX(value) AS max_value FROM expression",
        "SELECT COUNT(*) FROM expression WHERE value > 0",
        # Window function queries (for tokenizer scenarios) - simplified without ROW_NUMBER
        "SELECT cell_integer_id, gene_integer_id, value FROM expression WHERE cell_integer_id < 10 ORDER BY value DESC LIMIT 10",
        # Complex queries with joins - simplified to avoid column name conflicts
        "SELECT e.cell_integer_id, e.gene_integer_id, e.value FROM expression e LIMIT 10",
    ]

    failed_queries = []
    for i, query in enumerate(warmup_queries):
        if verbose and i % 3 == 0:  # Print progress every 3 queries
            console.print(f"    Warming up query {i + 1}/{len(warmup_queries)}...")

        try:
            # Execute warm-up query (discard results)
            _ = slaf_instance.query(query)
        except Exception as e:
            failed_queries.append((i + 1, str(e)))
            if verbose:
                console.print(f"    ❌ Query {i + 1} failed: {e}")

    if failed_queries and verbose:
        console.print(f"    ⚠️  {len(failed_queries)} warm-up queries failed:")
        for query_num, error in failed_queries:
            console.print(f"      Query {query_num}: {error}")
        # Continue anyway - partial warm-up is better than none

    if verbose:
        console.print("  ✅ SLAF database warm-up completed")


def warm_up_tiledb_database(experiment, verbose=False):
    """Comprehensive warm-up for TileDB database to eliminate cold start effects

    This function performs equivalent warm-up operations to SLAF:
    1. Clears all caches
    2. Runs equivalent queries to prime the database
    3. Warms up common query patterns

    Args:
        experiment: TileDB SOMA experiment to warm up
        verbose: Whether to print warm-up status
    """
    if verbose:
        console = Console()
        console.print("  🔥 Warming up TileDB database...")

    # Clear all caches first
    clear_caches()

    # Define equivalent warm-up operations for TileDB
    # Use correct SOMA API structure based on documentation and tiledb_dataloaders.py
    warmup_operations = [
        # Basic metadata queries (equivalent to SLAF COUNT queries)
        lambda: experiment.obs.read().concat(),
        lambda: experiment.ms["RNA"]["var"].read().concat(),
        lambda: experiment.ms["RNA"]["X"]["data"]
        .read((slice(0, 10),))
        .tables()
        .concat(),
        # Common filtering patterns (equivalent to SLAF LIMIT queries)
        lambda: experiment.obs.read().concat(),
        lambda: experiment.ms["RNA"]["var"].read().concat(),
        # Expression queries (equivalent to SLAF WHERE queries)
        lambda: experiment.ms["RNA"]["X"]["data"]
        .read((slice(0, 1),))
        .tables()
        .concat(),
        lambda: experiment.ms["RNA"]["X"]["data"]
        .read((slice(None), slice(0, 1)))
        .tables()
        .concat(),
        # Aggregation queries (equivalent to SLAF MIN/MAX/COUNT queries)
        lambda: experiment.ms["RNA"]["X"]["data"]
        .read((slice(0, 100),))
        .tables()
        .concat(),
        lambda: experiment.ms["RNA"]["X"]["data"]
        .read((slice(0, 50),))
        .tables()
        .concat(),
        # Simple filtering (equivalent to SLAF WHERE value > 0)
        lambda: experiment.ms["RNA"]["X"]["data"]
        .read((slice(0, 20),))
        .tables()
        .concat(),
        # Metadata reading (equivalent to SLAF JOIN queries)
        lambda: experiment.obs.read().concat(),
    ]

    failed_operations = []
    for i, operation in enumerate(warmup_operations):
        if verbose and i % 3 == 0:  # Print progress every 3 operations
            console.print(
                f"    Warming up operation {i + 1}/{len(warmup_operations)}..."
            )

        try:
            # Execute warm-up operation (discard results)
            _ = operation()
        except Exception as e:
            failed_operations.append((i + 1, str(e)))
            if verbose:
                console.print(f"    ❌ Operation {i + 1} failed: {e}")

    if failed_operations and verbose:
        console.print(f"    ⚠️  {len(failed_operations)} warm-up operations failed:")
        for op_num, error in failed_operations:
            console.print(f"      Operation {op_num}: {error}")
        # Continue anyway - partial warm-up is better than none

    if verbose:
        console.print("  ✅ TileDB database warm-up completed")


def burn_in_first_scenario(slaf_instance=None, verbose=False):
    """Perform burn-in for the first scenario to eliminate cold start effects

    Args:
        slaf_instance: Optional SLAF instance to clear caches on
        verbose: Whether to print burn-in status

    Returns:
        None
    """
    if verbose:
        console = Console()
        console.print("  Running burn-in for first scenario...")

    # Use the enhanced warm-up function
    warm_up_slaf_database(slaf_instance, verbose)


def run_with_burn_in(
    scenarios, benchmark_func, slaf_instance=None, verbose=False, **kwargs
):
    """Run benchmarks with automatic burn-in for the first scenario

    Args:
        scenarios: List of scenario dictionaries
        benchmark_func: Function to run for each scenario
        slaf_instance: Optional SLAF instance for cache clearing
        verbose: Whether to print burn-in status
        **kwargs: Additional arguments to pass to benchmark_func

    Returns:
        List of benchmark results
    """
    results = []

    for i, scenario in enumerate(scenarios):
        if verbose:
            console = Console()
            console.print(
                f"\n[bold blue]Testing: {scenario.get('name', f'Scenario {i + 1}')}[/bold blue]"
            )

        # Clear caches at the start of each scenario
        clear_caches()

        # For the first scenario, do a comprehensive warm-up run
        if i == 0:
            burn_in_first_scenario(slaf_instance, verbose)

        # Run the actual benchmark
        try:
            result = benchmark_func(scenario, **kwargs)
            results.append(result)
        except Exception as e:
            if verbose:
                console.print(f"  ❌ Failed: {e}")
            continue

    return results


def _print_rich_table_three_way(
    results,
    dataset_name,
    scenario_type,
    avg_slaf_vs_h5ad_speedup,
    avg_slaf_vs_tiledb_speedup,
    memory_efficiency,
    avg_h5ad_total_memory,
    avg_slaf_total_memory,
    avg_tiledb_total_memory,
):
    """Print three-way comparison table using rich library for better formatting"""
    console = Console()

    # Create main table
    title = f"BENCHMARK RESULTS: {dataset_name.upper() if dataset_name else 'ALL SCENARIOS'}"
    if scenario_type:
        title += f" - {scenario_type.upper()}"
    title += " (SLAF vs h5ad vs TileDB)"

    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
        title_style="bold blue",
    )

    # Add columns for three-way comparison
    table.add_column("Scenario", style="cyan", width=8)
    table.add_column("h5ad\nTotal (ms)", justify="right", style="red", width=8)
    table.add_column("SLAF\nTotal (ms)", justify="right", style="green", width=8)
    table.add_column("TileDB\nTotal (ms)", justify="right", style="blue", width=8)
    table.add_column("SLAF vs\nh5ad", justify="right", style="yellow", width=8)
    table.add_column("SLAF vs\nTileDB", justify="right", style="yellow", width=8)
    table.add_column("h5ad\nMemory (MB)", justify="right", style="red", width=8)
    table.add_column("SLAF\nMemory (MB)", justify="right", style="green", width=8)
    table.add_column("TileDB\nMemory (MB)", justify="right", style="blue", width=8)

    # Add description column if available
    has_description = any("scenario_description" in r for r in results)
    if has_description:
        table.add_column("Description", style="white", width=20)

    # Add data rows
    for i, result in enumerate(results):
        scenario_name = f"S{i + 1}"

        # Timing breakdown (already in milliseconds)
        h5ad_total = result.get("h5ad_total_time", 0)
        slaf_total = result.get("slaf_total_time", 0)
        tiledb_total = result.get("tiledb_total_time", 0)

        slaf_vs_h5ad_speedup = result.get("slaf_vs_h5ad_speedup", 0)
        slaf_vs_tiledb_speedup = result.get("slaf_vs_tiledb_speedup", 0)

        # Memory info
        h5ad_total_memory = result.get("h5ad_total_memory_mb", 0)
        slaf_total_memory = result.get("slaf_total_memory_mb", 0)
        tiledb_total_memory = result.get("tiledb_total_memory_mb", 0)

        # Format speedups with color coding
        slaf_vs_h5ad_text = (
            f"{slaf_vs_h5ad_speedup:.1f}x" if slaf_vs_h5ad_speedup > 0 else "N/A"
        )
        slaf_vs_tiledb_text = (
            f"{slaf_vs_tiledb_speedup:.1f}x" if slaf_vs_tiledb_speedup > 0 else "N/A"
        )

        # Color code speedups
        if slaf_vs_h5ad_speedup > 1:
            slaf_vs_h5ad_text = f"[green]{slaf_vs_h5ad_text}[/green]"
        elif slaf_vs_h5ad_speedup > 0:
            slaf_vs_h5ad_text = f"[red]{slaf_vs_h5ad_text}[/red]"

        if slaf_vs_tiledb_speedup > 1:
            slaf_vs_tiledb_text = f"[green]{slaf_vs_tiledb_text}[/green]"
        elif slaf_vs_tiledb_speedup > 0:
            slaf_vs_tiledb_text = f"[red]{slaf_vs_tiledb_text}[/red]"

        # Build row data
        row_data = [
            scenario_name,
            f"{h5ad_total:.1f}",
            f"{slaf_total:.1f}",
            f"{tiledb_total:.1f}",
            slaf_vs_h5ad_text,
            slaf_vs_tiledb_text,
            f"{h5ad_total_memory:.1f}",
            f"{slaf_total_memory:.1f}",
            f"{tiledb_total_memory:.1f}",
        ]

        # Add description if available
        if has_description:
            description = result.get("scenario_description", "")
            row_data.append(description)

        table.add_row(*row_data)

    # Print the table
    console.print(table)

    # Print summary
    summary_table = Table(title="SUMMARY", show_header=False, box=None)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row(
        "Average SLAF vs h5ad Speedup", f"{avg_slaf_vs_h5ad_speedup:.1f}x"
    )
    summary_table.add_row(
        "Average SLAF vs TileDB Speedup", f"{avg_slaf_vs_tiledb_speedup:.1f}x"
    )
    summary_table.add_row("Average h5ad Memory (MB)", f"{avg_h5ad_total_memory:.1f}")
    summary_table.add_row("Average SLAF Memory (MB)", f"{avg_slaf_total_memory:.1f}")
    summary_table.add_row(
        "Average TileDB Memory (MB)", f"{avg_tiledb_total_memory:.1f}"
    )
    summary_table.add_row("Memory Efficiency (h5ad/SLAF)", f"{memory_efficiency:.1f}x")

    console.print(summary_table)

    # Print interpretation
    console.print("\n[bold]Interpretation:[/bold]")
    console.print(
        f"• SLAF vs h5ad: {avg_slaf_vs_h5ad_speedup:.1f}x faster than traditional h5ad"
    )
    console.print(
        f"• SLAF vs TileDB: {avg_slaf_vs_tiledb_speedup:.1f}x faster than TileDB SOMA"
    )
    console.print(
        f"• Memory efficiency: SLAF uses {memory_efficiency:.1f}x less memory than h5ad"
    )
