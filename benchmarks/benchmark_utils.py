import numpy as np
import sys
from typing import Dict, List
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


def print_benchmark_table(
    results: List[Dict], dataset_name: str = "", scenario_type: str = ""
):
    """Print a comprehensive benchmark results table

    Note: All timing measurements include loading time for each scenario,
    providing true end-to-end performance comparisons.
    """

    if not results:
        print("No benchmark results to display")
        return

    # Calculate summary statistics
    total_speedups = [r["total_speedup"] for r in results if r["total_speedup"] > 0]
    query_speedups = [r["query_speedup"] for r in results if r["query_speedup"] > 0]
    load_speedups = [r["load_speedup"] for r in results if r.get("load_speedup", 0) > 0]

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
        scenario_name = f"S{i+1}"

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
        if slaf_total_memory > 0.1:  # Avoid division by very small numbers
            mem_efficiency = h5ad_total_memory / slaf_total_memory
            mem_efficiency_text = f"{mem_efficiency:.1f}x"
            # Color code memory efficiency
            if mem_efficiency > 1:
                mem_efficiency_text = f"[green]{mem_efficiency_text}[/green]"
            else:
                mem_efficiency_text = f"[red]{mem_efficiency_text}[/red]"
        elif h5ad_total_memory > 0.1:  # h5ad used memory but SLAF didn't
            mem_efficiency_text = "[green]>100x[/green]"  # SLAF is much more efficient
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
    except:
        pass  # Ignore if DuckDB cache clearing fails

    # Clear any Lance dataset caches
    try:
        import lance

        # Note: This would need to be called on specific SLAF instances
        # We'll handle this in the burn_in function
    except:
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
            import lance

            # Force reload of Lance datasets to clear any internal caches
            slaf_instance._setup_datasets()
        except:
            pass  # Ignore if Lance cache clearing fails

    # Run representative warm-up queries to prime the database
    warmup_queries = [
        # Basic metadata queries
        "SELECT COUNT(*) FROM cells",
        "SELECT COUNT(*) FROM genes",
        "SELECT COUNT(*) FROM expression",
        # Common filtering patterns
        "SELECT cell_id FROM cells LIMIT 10",
        "SELECT gene_id FROM genes LIMIT 10",
        # Expression queries
        "SELECT cell_id, gene_id, value FROM expression WHERE cell_integer_id = 0 LIMIT 10",
        "SELECT cell_id, gene_id, value FROM expression WHERE gene_integer_id = 0 LIMIT 10",
        # Aggregation queries
        "SELECT MIN(value), MAX(value) FROM expression",
        "SELECT COUNT(*) FROM expression WHERE value > 0",
        # Window function queries (for tokenizer scenarios)
        "SELECT cell_id, gene_id, value, ROW_NUMBER() OVER (PARTITION BY cell_id ORDER BY value DESC) as rank FROM expression WHERE cell_integer_id < 10",
        # Complex queries with joins
        "SELECT c.cell_id, g.gene_id, e.value FROM expression e JOIN cells c ON e.cell_id = c.cell_id JOIN genes g ON e.gene_id = g.gene_id LIMIT 10",
    ]

    try:
        for i, query in enumerate(warmup_queries):
            if verbose and i % 3 == 0:  # Print progress every 3 queries
                console.print(f"    Warming up query {i+1}/{len(warmup_queries)}...")

            # Execute warm-up query (discard results)
            _ = slaf_instance.query(query)

    except Exception as e:
        if verbose:
            console.print(f"    Warning: Some warm-up queries failed: {e}")
        # Continue anyway - partial warm-up is better than none

    if verbose:
        console.print("  ✅ SLAF database warm-up completed")


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
                f"\n[bold blue]Testing: {scenario.get('name', f'Scenario {i+1}')}[/bold blue]"
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
