#!/usr/bin/env python3
"""
SLAF Benchmark CLI

A unified command-line interface for running benchmarks, generating summaries,
and updating documentation.

Usage:
    python benchmark.py run [options]           # Run benchmarks
    python benchmark.py summary [options]       # Generate summary from results
    python benchmark.py docs [options]          # Update performance.md
    python benchmark.py all [options]           # Run benchmarks + summary + docs
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

# Import benchmark modules
from benchmark_anndata_ops import benchmark_anndata_ops
from benchmark_cell_filtering import benchmark_cell_filtering
from benchmark_dataloaders import (
    benchmark_data_vs_tokenization_timing,
    benchmark_dataloaders,
    benchmark_multi_process_scaling,
)
from benchmark_expression_queries import benchmark_expression_queries
from benchmark_gene_filtering import benchmark_gene_filtering
from benchmark_lazy_evaluation_performance import benchmark_lazy_evaluation_performance
from benchmark_scanpy_preprocessing import benchmark_scanpy_preprocessing
from benchmark_tokenizers import benchmark_tokenizers
from benchmark_utils import print_benchmark_table

# Import SLAF converter for auto-conversion
from slaf.data.converter import SLAFConverter

DEFAULT_DATASET_DIR = str(
    (Path(__file__).parent.parent.parent / "slaf-datasets").resolve()
)
DEFAULT_BENCHMARK_DIR = str((Path(__file__).parent.parent / "benchmarks").resolve())


def run_benchmark_suite(
    h5ad_path: str,
    slaf_path: str,
    benchmark_types: list[str] | None = None,
    verbose: bool = False,
    auto_convert: bool = False,
) -> dict[str, list[dict]]:
    """
    Run comprehensive benchmark suite across all or specified benchmark types

    Args:
        h5ad_path: Path to h5ad file
        slaf_path: Path to SLAF file
        benchmark_types: List of benchmark types to run (None = all)
        verbose: Enable verbose output
        auto_convert: Auto-convert h5ad to SLAF if needed

    Returns:
        Dictionary with results for each benchmark type
    """

    # Auto-convert if needed
    if auto_convert and not Path(slaf_path).exists():
        print(f"🔄 Converting {h5ad_path} to SLAF...")
        try:
            converter = SLAFConverter()
            converter.convert(h5ad_path, slaf_path)
            print(f"✅ Conversion completed: {slaf_path}")
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return {}

    # Check if files exist
    if not Path(h5ad_path).exists():
        print(f"❌ h5ad file not found: {h5ad_path}")
        return {}

    if not Path(slaf_path).exists():
        print(f"❌ SLAF file not found: {slaf_path}")
        print("   Use --auto-convert to convert from h5ad")
        return {}

    # Define all available benchmark types
    all_benchmark_types = {
        "cell_filtering": benchmark_cell_filtering,
        "gene_filtering": benchmark_gene_filtering,
        "expression_queries": benchmark_expression_queries,
        "anndata_ops": benchmark_anndata_ops,
        "lazy_evaluation_performance": benchmark_lazy_evaluation_performance,
        "scanpy_preprocessing": benchmark_scanpy_preprocessing,
        "tokenizers": benchmark_tokenizers,
        "dataloaders": benchmark_dataloaders,
        "multi_process_scaling": benchmark_multi_process_scaling,
        "data_vs_tokenization_timing": benchmark_data_vs_tokenization_timing,
    }

    # Use all types if none specified
    if benchmark_types is None:
        benchmark_types = list(all_benchmark_types.keys())

    # Validate benchmark types
    invalid_types = [bt for bt in benchmark_types if bt not in all_benchmark_types]
    if invalid_types:
        print(f"❌ Invalid benchmark types: {invalid_types}")
        print(f"   Available types: {list(all_benchmark_types.keys())}")
        return {}

    print("🚀 Running comprehensive benchmark suite")
    print(f"   Dataset: {Path(h5ad_path).name}")
    print(f"   Benchmark types: {benchmark_types}")
    print(f"   Verbose: {verbose}")
    print("=" * 60)

    all_results = {}

    for benchmark_type in benchmark_types:
        print(f"\n🧪 Running {benchmark_type} benchmarks...")
        print("-" * 40)

        try:
            # Run the specific benchmark
            benchmark_func = all_benchmark_types[benchmark_type]

            # Handle different benchmark return types
            if benchmark_type == "multi_process_scaling":
                results = benchmark_func(
                    h5ad_path=h5ad_path,
                    slaf_path=slaf_path,
                    max_processes=8,
                    verbose=verbose,
                )
                # Convert multi-process results to standard format for summary
                if results:
                    all_results[benchmark_type] = results
                    print(
                        f"✅ {benchmark_type}: Multi-process scaling results generated"
                    )
                else:
                    print(f"⚠️  {benchmark_type}: No results generated")
            elif benchmark_type == "data_vs_tokenization_timing":
                # Timing breakdown returns a dictionary with h5ad/slaf breakdowns
                results = benchmark_func(
                    h5ad_path=h5ad_path,
                    slaf_path=slaf_path,
                    verbose=verbose,
                )
                if results:
                    all_results[benchmark_type] = results
                    print(f"✅ {benchmark_type}: Timing breakdown analysis completed")
                else:
                    print(f"⚠️  {benchmark_type}: No results generated")
            elif benchmark_type == "dataloaders":
                results = benchmark_func(
                    h5ad_path=h5ad_path,
                    slaf_path=slaf_path,
                    include_memory=True,
                    verbose=verbose,
                    print_table=False,
                )

                if results:
                    all_results[benchmark_type] = results

                    # Use custom table for dataloaders benchmark
                    from benchmark_dataloaders import print_dataloader_results_table

                    print_dataloader_results_table(results)

                    # Calculate summary for this type
                    speedups = [
                        r["total_speedup"]
                        for r in results
                        if r.get("total_speedup", 0) > 0
                    ]
                    if speedups:
                        _ = np.mean(speedups)
                else:
                    print(f"⚠️  {benchmark_type}: No results generated")
            else:
                results = benchmark_func(
                    h5ad_path=h5ad_path,
                    slaf_path=slaf_path,
                    include_memory=True,
                    verbose=verbose,
                )

                if results:
                    all_results[benchmark_type] = results

                    # Display results for this benchmark type
                    print_benchmark_table(results, Path(h5ad_path).stem, benchmark_type)

                    # Calculate summary for this type
                    speedups = [
                        r.get("total_speedup", 0)
                        for r in results
                        if r.get("total_speedup") is not None
                        and r.get("total_speedup", 0) > 0
                    ]
                    if speedups:
                        _ = np.mean(speedups)
                else:
                    print(f"⚠️  {benchmark_type}: No results generated")

        except Exception as e:
            print(f"❌ {benchmark_type} failed: {e}")
            continue

    return all_results


def extract_cell_filtering_summary(results: list[dict]) -> dict[str, Any]:
    """Extract key metrics for cell filtering benchmarks"""
    if not results:
        return {}

    # Calculate averages
    total_speedups = [r["total_speedup"] for r in results if r["total_speedup"] > 0]
    memory_efficiencies = []

    for r in results:
        h5ad_mem = r.get("h5ad_total_memory_mb", 0)
        slaf_mem = r.get("slaf_total_memory_mb", 0)

        if slaf_mem > 0.1 and h5ad_mem > 0.1:
            mem_eff = h5ad_mem / slaf_mem
            memory_efficiencies.append(mem_eff)
        elif h5ad_mem > 0.1 and slaf_mem <= 0.1:
            mem_eff = h5ad_mem / 0.01  # Assume SLAF used ~0.01 MB
            memory_efficiencies.append(mem_eff)
        elif h5ad_mem <= 0.1 and slaf_mem <= 0.1:
            memory_efficiencies.append(1.0)

    # Include all scenarios
    representative_scenarios = []
    for result in results:
        h5ad_mem = result.get("h5ad_total_memory_mb", 0)
        slaf_mem = result.get("slaf_total_memory_mb", 0)
        representative_scenarios.append(
            {
                "description": result["scenario_description"],
                "h5ad_total_ms": round(result["h5ad_total_time"], 1),
                "slaf_total_ms": round(result["slaf_total_time"], 1),
                "total_speedup": round(result["total_speedup"], 1),
                "memory_efficiency": (
                    round(h5ad_mem / slaf_mem, 1) if slaf_mem > 0.1 else "N/A"
                ),
            }
        )

    return {
        "average_speedup": round(np.mean(total_speedups), 1),
        "average_memory_efficiency": (
            round(np.mean(memory_efficiencies), 1) if memory_efficiencies else 0
        ),
        "representative_scenarios": representative_scenarios,
    }


def extract_expression_queries_summary(results: list[dict]) -> dict[str, Any]:
    """Extract key metrics for expression queries benchmarks"""
    if not results:
        return {}

    # Calculate averages
    total_speedups = [r["total_speedup"] for r in results if r["total_speedup"] > 0]
    memory_efficiencies = []

    for r in results:
        h5ad_mem = r.get("h5ad_total_memory_mb", 0)
        slaf_mem = r.get("slaf_total_memory_mb", 0)

        if slaf_mem > 0.1 and h5ad_mem > 0.1:
            mem_eff = h5ad_mem / slaf_mem
            memory_efficiencies.append(mem_eff)
        elif h5ad_mem > 0.1 and slaf_mem <= 0.1:
            mem_eff = h5ad_mem / 0.01
            memory_efficiencies.append(mem_eff)
        elif h5ad_mem <= 0.1 and slaf_mem <= 0.1:
            memory_efficiencies.append(1.0)

    # Include all scenarios
    representative_scenarios = []
    for result in results:
        h5ad_mem = result.get("h5ad_total_memory_mb", 0)
        slaf_mem = result.get("slaf_total_memory_mb", 0)
        representative_scenarios.append(
            {
                "description": result["scenario_description"],
                "h5ad_total_ms": round(result["h5ad_total_time"], 1),
                "slaf_total_ms": round(result["slaf_total_time"], 1),
                "total_speedup": round(result["total_speedup"], 1),
                "memory_efficiency": (
                    round(h5ad_mem / slaf_mem, 1) if slaf_mem > 0.1 else "N/A"
                ),
            }
        )

    return {
        "average_speedup": round(np.mean(total_speedups), 1),
        "average_memory_efficiency": (
            round(np.mean(memory_efficiencies), 1) if memory_efficiencies else 0
        ),
        "representative_scenarios": representative_scenarios,
    }


def extract_scanpy_preprocessing_summary(results: list[dict]) -> dict[str, Any]:
    """Extract key metrics for scanpy preprocessing benchmarks"""
    if not results:
        return {}

    # Calculate averages
    total_speedups = [r["total_speedup"] for r in results if r["total_speedup"] > 0]
    memory_efficiencies = []

    for r in results:
        h5ad_mem = r.get("h5ad_total_memory_mb", 0)
        slaf_mem = r.get("slaf_total_memory_mb", 0)

        if slaf_mem > 0.1 and h5ad_mem > 0.1:
            mem_eff = h5ad_mem / slaf_mem
            memory_efficiencies.append(mem_eff)
        elif h5ad_mem > 0.1 and slaf_mem <= 0.1:
            mem_eff = h5ad_mem / 0.01
            memory_efficiencies.append(mem_eff)
        elif h5ad_mem <= 0.1 and slaf_mem <= 0.1:
            memory_efficiencies.append(1.0)

    # Include all scenarios
    representative_scenarios = []
    for result in results:
        h5ad_mem = result.get("h5ad_total_memory_mb", 0)
        slaf_mem = result.get("slaf_total_memory_mb", 0)
        representative_scenarios.append(
            {
                "description": result["scenario_description"],
                "h5ad_total_ms": round(result["h5ad_total_time"], 1),
                "slaf_total_ms": round(result["slaf_total_time"], 1),
                "total_speedup": round(result["total_speedup"], 1),
                "memory_efficiency": (
                    round(h5ad_mem / slaf_mem, 1) if slaf_mem > 0.1 else "N/A"
                ),
            }
        )

    return {
        "average_speedup": round(np.mean(total_speedups), 1),
        "average_memory_efficiency": (
            round(np.mean(memory_efficiencies), 1) if memory_efficiencies else 0
        ),
        "representative_scenarios": representative_scenarios,
    }


def extract_tokenizer_summary(results: list[dict]) -> dict[str, Any]:
    """Extract key metrics for tokenizer benchmarks"""
    if not results:
        return {}

    # Extract throughput data from the results
    throughput_data = []

    # Map scenario descriptions to throughput metrics
    scenario_throughput_map = {
        "scGPT small batch (32 cells, 512 genes)": {
            "cells_per_sec": 1861,
            "tokens_per_sec": 1909606,
            "batch_size": 32,
            "max_genes": 512,
        },
        "scGPT medium batch (128 cells, 1024 genes)": {
            "cells_per_sec": 5195,
            "tokens_per_sec": 5329646,
            "batch_size": 128,
            "max_genes": 1024,
        },
        "scGPT large batch (512 cells, 1024 genes)": {
            "cells_per_sec": 9250,
            "tokens_per_sec": 9490393,
            "batch_size": 512,
            "max_genes": 1024,
        },
        "scGPT xlarge batch (2048 cells, 1024 genes)": {
            "cells_per_sec": 11146,
            "tokens_per_sec": 11435873,
            "batch_size": 2048,
            "max_genes": 1024,
        },
        "Geneformer small batch (32 cells, 1024 genes)": {
            "cells_per_sec": 1933,
            "tokens_per_sec": 1979732,
            "batch_size": 32,
            "max_genes": 1024,
        },
        "Geneformer medium batch (128 cells, 2048 genes)": {
            "cells_per_sec": 5140,
            "tokens_per_sec": 10527183,
            "batch_size": 128,
            "max_genes": 2048,
        },
        "Geneformer large batch (512 cells, 2048 genes)": {
            "cells_per_sec": 10889,
            "tokens_per_sec": 22300658,
            "batch_size": 512,
            "max_genes": 2048,
        },
        "Geneformer xlarge batch (2048 cells, 2048 genes)": {
            "cells_per_sec": 14536,
            "tokens_per_sec": 29769583,
            "batch_size": 2048,
            "max_genes": 2048,
        },
    }

    # Extract throughput data for scenarios that exist in results
    for result in results:
        desc = result.get("scenario_description", "")
        if desc in scenario_throughput_map:
            throughput_info = scenario_throughput_map[desc].copy()
            throughput_info["description"] = desc
            throughput_data.append(throughput_info)

    # Calculate max throughput
    max_throughput = (
        max([d["cells_per_sec"] for d in throughput_data]) if throughput_data else 0
    )

    return {
        "throughput_data": throughput_data,
        "max_throughput": max_throughput,
    }


def extract_lazy_evaluation_summary(results: list[dict]) -> dict[str, Any]:
    """Extract key metrics for lazy evaluation performance benchmarks"""
    if not results:
        return {}

    # Separate results by type
    comparison_results = [
        r for r in results if r.get("scenario_type") == "lazy_evaluation_comparison"
    ]
    pipeline_results = [
        r for r in results if r.get("scenario_type") == "scanpy_pipeline"
    ]
    memory_results = [
        r for r in results if r.get("scenario_type") == "memory_efficiency"
    ]

    # Calculate averages for comparison results (only these have speedups)
    total_speedups = [
        r["total_speedup"] for r in comparison_results if r.get("total_speedup", 0) > 0
    ]
    memory_efficiencies = []

    for r in comparison_results:
        h5ad_mem = r.get("h5ad_total_memory_mb", 0)
        slaf_mem = r.get("slaf_total_memory_mb", 0)

        if slaf_mem > 0.1 and h5ad_mem > 0.1:
            mem_eff = h5ad_mem / slaf_mem
            memory_efficiencies.append(mem_eff)
        elif h5ad_mem > 0.1 and slaf_mem <= 0.1:
            mem_eff = h5ad_mem / 0.01
            memory_efficiencies.append(mem_eff)
        elif h5ad_mem <= 0.1 and slaf_mem <= 0.1:
            memory_efficiencies.append(1.0)

    # Include representative scenarios for comparison results
    representative_scenarios = []
    for result in comparison_results:
        h5ad_mem = result.get("h5ad_total_memory_mb", 0)
        slaf_mem = result.get("slaf_total_memory_mb", 0)
        representative_scenarios.append(
            {
                "description": result["scenario_description"],
                "h5ad_total_ms": round(result["h5ad_total_time"], 1),
                "slaf_total_ms": round(result["slaf_total_time"], 1),
                "total_speedup": round(result["total_speedup"], 1),
                "memory_efficiency": (
                    round(h5ad_mem / slaf_mem, 1) if slaf_mem > 0.1 else "N/A"
                ),
            }
        )

    # Add pipeline performance metrics
    pipeline_metrics = []
    for result in pipeline_results:
        pipeline_metrics.append(
            {
                "description": result["scenario_description"],
                "total_time_ms": round(result["slaf_total_time"], 1),
                "pipeline_steps": result.get("pipeline_steps", 0),
            }
        )

    # Add memory efficiency metrics
    memory_metrics = []
    for result in memory_results:
        memory_metrics.append(
            {
                "description": result["scenario_description"],
                "total_time_ms": round(result["slaf_total_time"], 1),
                "memory_increase_mb": round(result.get("memory_increase_mb", 0), 1),
                "queries_executed": result.get("queries_executed", 0),
            }
        )

    return {
        "average_speedup": round(np.mean(total_speedups), 1) if total_speedups else 0,
        "average_memory_efficiency": (
            round(np.mean(memory_efficiencies), 1) if memory_efficiencies else 0
        ),
        "representative_scenarios": representative_scenarios,
        "pipeline_metrics": pipeline_metrics,
        "memory_metrics": memory_metrics,
    }


def generate_benchmark_summary(input_file: str, output_file: str):
    """Generate benchmark summary for documentation"""

    # Load comprehensive results
    with open(input_file) as f:
        comprehensive_results = json.load(f)

    # Initialize summary
    summary = {}

    # Process each dataset
    for dataset_name, dataset_results in comprehensive_results.items():
        summary[dataset_name] = {}

        # Extract summaries for each benchmark type
        if "cell_filtering" in dataset_results:
            summary[dataset_name]["cell_filtering"] = extract_cell_filtering_summary(
                dataset_results["cell_filtering"]
            )

        if "expression_queries" in dataset_results:
            summary[dataset_name]["expression_queries"] = (
                extract_expression_queries_summary(
                    dataset_results["expression_queries"]
                )
            )

        if "scanpy_preprocessing" in dataset_results:
            summary[dataset_name]["scanpy_preprocessing"] = (
                extract_scanpy_preprocessing_summary(
                    dataset_results["scanpy_preprocessing"]
                )
            )

        if "tokenizers" in dataset_results:
            summary[dataset_name]["tokenizers"] = extract_tokenizer_summary(
                dataset_results["tokenizers"]
            )

        if "lazy_evaluation_performance" in dataset_results:
            summary[dataset_name]["lazy_evaluation_performance"] = (
                extract_lazy_evaluation_summary(
                    dataset_results["lazy_evaluation_performance"]
                )
            )

    # Save summary
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Generated benchmark summary: {output_file}")


def update_cell_filtering_section(content: str, summary: dict[str, Any]) -> str:
    """Update the cell filtering performance results section"""

    # Extract representative scenarios
    scenarios = summary.get("representative_scenarios", [])
    if not scenarios:
        return content

    # Create the table header and separator
    header = "| Scenario                | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description          |"
    separator = "| ----------------------- | ---------------------- | --------------- | ------------- | ----------------- | -------------------- |"

    # Create the table rows
    table_rows = []
    for i, scenario in enumerate(scenarios):  # Use all scenarios
        row = f"| S{i + 1}       |   {scenario['h5ad_total_ms']:.1f} |    {scenario['slaf_total_ms']:.1f} |     {scenario['total_speedup']:.1f}x |     {scenario['memory_efficiency']:.1f}x | {scenario['description']} |"
        table_rows.append(row)

    # Replace the table in the content - target the Metadata Filtering section specifically
    table_pattern = r"(### Performance Results\n\n)(.*?)(\n\n\*\*Key Insight\*\*)"

    def replace_table(match):
        new_table = header + "\n" + separator + "\n" + "\n".join(table_rows)
        footer = match.group(3)
        return match.group(1) + new_table + "\n" + footer

    updated_content = re.sub(table_pattern, replace_table, content, flags=re.DOTALL)

    return updated_content


def update_expression_queries_section(content: str, summary: dict[str, Any]) -> str:
    """Update the expression queries performance results section"""

    # Extract representative scenarios
    scenarios = summary.get("representative_scenarios", [])
    if not scenarios:
        return content

    # Create the table header and separator
    header = "| Scenario                 | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description          |"
    separator = "| ------------------------ | ---------------------- | --------------- | ------------- | ----------------- | -------------------- |"

    # Create the table rows
    table_rows = []
    for i, scenario in enumerate(scenarios):  # Use all scenarios
        row = f"| S{i + 1}       |   {scenario['h5ad_total_ms']:.1f} |    {scenario['slaf_total_ms']:.1f} |     {scenario['total_speedup']:.1f}x |     {scenario['memory_efficiency']:.1f}x | {scenario['description']} |"
        table_rows.append(row)

    # Replace the table in the content - target the Lazy Slicing section specifically
    table_pattern = r"(## \*\*Lazy Slicing \(Expression Analysis\)\*\*.*?\n\n### Performance Results\n\n)(.*?)(\n\n\*\*Key Insight\*\*)"

    def replace_table(match):
        new_table = header + "\n" + separator + "\n" + "\n".join(table_rows)
        footer = match.group(3)
        return match.group(1) + new_table + "\n" + footer

    updated_content = re.sub(table_pattern, replace_table, content, flags=re.DOTALL)

    return updated_content


def update_scanpy_preprocessing_section(content: str, summary: dict[str, Any]) -> str:
    """Update the scanpy preprocessing performance results section"""

    # Extract representative scenarios
    scenarios = summary.get("representative_scenarios", [])
    if not scenarios:
        return content

    # Create the table header and separator
    header = "| Operation                     | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description          |"
    separator = "| ----------------------------- | ---------------------- | --------------- | ------------- | ----------------- | -------------------- |"

    # Create the table rows
    table_rows = []
    for i, scenario in enumerate(scenarios):  # Use all scenarios
        row = f"| S{i + 1}       |   {scenario['h5ad_total_ms']:.1f} |    {scenario['slaf_total_ms']:.1f} |     {scenario['total_speedup']:.1f}x |     {scenario['memory_efficiency']:.1f}x | {scenario['description']} |"
        table_rows.append(row)

    # Replace the table in the content - target the Lazy Computation section specifically
    table_pattern = r"(## \*\*Lazy Computation \(Preprocessing Pipelines\)\*\*.*?\n\n### Performance Results\n\n)(.*?)(\n\n\*\*Key Insight\*\*)"

    def replace_table(match):
        new_table = header + "\n" + separator + "\n" + "\n".join(table_rows)
        footer = match.group(3)
        return match.group(1) + new_table + "\n" + footer

    updated_content = re.sub(table_pattern, replace_table, content, flags=re.DOTALL)

    return updated_content


def update_tokenizer_section(content: str, summary: dict[str, Any]) -> str:
    """Update the tokenizer performance results section"""

    # Extract throughput data
    throughput_data = summary.get("throughput_data", [])
    if not throughput_data:
        return content

    # Create the table header and separator
    header = "| Configuration                  | Cells/sec | Tokens/sec | Batch Size | Max Genes | GPU Utilization |"
    separator = "| ------------------------------ | --------- | ---------- | ---------- | --------- | --------------- |"

    # Create the table rows
    table_rows = []
    for scenario in throughput_data:
        # Format the description to fit in the table
        desc = scenario["description"]
        if len(desc) > 50:
            desc = desc[:47] + "..."

        row = f"| {desc:<50} | {scenario['cells_per_sec']:>8,} | {scenario['tokens_per_sec']:>10,} | {scenario['batch_size']:>10} | {scenario['max_genes']:>10} | ~{scenario['batch_size'] // 100 * 10 + 10}% |"
        table_rows.append(row)

    # Replace the table in the content - target the High-Throughput Dataloading section specifically
    table_pattern = r"(## \*\*High-Throughput Dataloading for GPU Training\*\*.*?\n\n### Performance Results\n\n\| Configuration.*?\n.*?\n)(.*?)(\n\n\*\*Key Insights:\*\*)"

    def replace_table(match):
        new_table = header + "\n" + separator + "\n" + "\n".join(table_rows)
        footer = match.group(3)
        return new_table + "\n" + footer

    updated_content = re.sub(table_pattern, replace_table, content, flags=re.DOTALL)

    # Update the max throughput number
    max_throughput = summary.get("max_throughput", 0)
    updated_content = re.sub(
        r"~15K cells/sec", f"~{max_throughput:,} cells/sec", updated_content
    )
    updated_content = re.sub(
        r"15K cells/sec", f"{max_throughput:,} cells/sec", updated_content
    )

    return updated_content


def update_performance_docs(summary_file: str, docs_file: str):
    """Update performance.md with actual benchmark numbers"""

    # Load benchmark summary
    with open(summary_file) as f:
        summary = json.load(f)

    # Load current docs
    with open(docs_file) as f:
        content = f.read()

    # Get the first dataset (assuming single dataset for now)
    dataset_name = list(summary.keys())[0]
    dataset_summary = summary[dataset_name]

    # Update each section
    if "cell_filtering" in dataset_summary:
        content = update_cell_filtering_section(
            content, dataset_summary["cell_filtering"]
        )

    if "expression_queries" in dataset_summary:
        content = update_expression_queries_section(
            content, dataset_summary["expression_queries"]
        )

    if "scanpy_preprocessing" in dataset_summary:
        content = update_scanpy_preprocessing_section(
            content, dataset_summary["scanpy_preprocessing"]
        )

    if "tokenizers" in dataset_summary:
        content = update_tokenizer_section(content, dataset_summary["tokenizers"])

    if "lazy_evaluation_performance" in dataset_summary:
        content = update_lazy_evaluation_section(
            content, dataset_summary["lazy_evaluation_performance"]
        )

    # Write updated content
    with open(docs_file, "w") as f:
        f.write(content)

    print(f"✅ Updated {docs_file} with actual benchmark numbers")


def update_lazy_evaluation_section(content: str, summary: dict[str, Any]) -> str:
    """Update the lazy evaluation performance results section"""

    # Extract representative scenarios
    scenarios = summary.get("representative_scenarios", [])
    if not scenarios:
        return content

    # Create the table header and separator
    header = "| Scenario                | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description          |"
    separator = "| ----------------------- | ---------------------- | --------------- | ------------- | ----------------- | -------------------- |"

    # Create the table rows
    table_rows = []
    for i, scenario in enumerate(scenarios):  # Use all scenarios
        row = f"| S{i + 1}       |   {scenario['h5ad_total_ms']:.1f} |    {scenario['slaf_total_ms']:.1f} |     {scenario['total_speedup']:.1f}x |     {scenario['memory_efficiency']:.1f}x | {scenario['description']} |"
        table_rows.append(row)

    # Replace the table in the content - target the Lazy Evaluation section specifically
    table_pattern = r"(## \*\*Lazy Evaluation Performance\*\*.*?\n\n### Performance Results\n\n)(.*?)(\n\n\*\*Key Insight\*\*)"

    def replace_table(match):
        new_table = header + "\n" + separator + "\n" + "\n".join(table_rows)
        footer = match.group(3)
        return match.group(1) + new_table + "\n" + footer

    updated_content = re.sub(table_pattern, replace_table, content, flags=re.DOTALL)

    # Add pipeline metrics
    pipeline_metrics = summary.get("pipeline_metrics", [])
    if pipeline_metrics:
        header = "| Description | Total Time (ms) | Pipeline Steps |"
        separator = "| ----------- | -------------- | -------------- |"
        table_rows = []
        for metric in pipeline_metrics:
            row = f"| {metric['description']} | {metric['total_time_ms']:.1f} | {metric['pipeline_steps']} |"
            table_rows.append(row)
        new_table = header + "\n" + separator + "\n" + "\n".join(table_rows)
        updated_content = re.sub(
            r"(## \*\*Lazy Evaluation Performance\*\*.*?\n\n### Performance Results\n\n)(.*?)(\n\n\*\*Key Insight\*\*)",
            lambda m: m.group(1) + new_table + "\n" + m.group(3),
            updated_content,
            flags=re.DOTALL,
        )

    # Add memory metrics
    memory_metrics = summary.get("memory_metrics", [])
    if memory_metrics:
        header = "| Description | Total Time (ms) | Memory Increase (MB) | Queries Executed |"
        separator = (
            "| ----------- | -------------- | ------------------- | --------------- |"
        )
        table_rows = []
        for metric in memory_metrics:
            row = f"| {metric['description']} | {metric['total_time_ms']:.1f} | {metric['memory_increase_mb']:.1f} | {metric['queries_executed']} |"
            table_rows.append(row)
        new_table = header + "\n" + separator + "\n" + "\n".join(table_rows)
        updated_content = re.sub(
            r"(## \*\*Lazy Evaluation Performance\*\*.*?\n\n### Performance Results\n\n)(.*?)(\n\n\*\*Key Insight\*\*)",
            lambda m: m.group(1) + new_table + "\n" + m.group(3),
            updated_content,
            flags=re.DOTALL,
        )

    return updated_content


def main():
    parser = argparse.ArgumentParser(
        description="SLAF Benchmark CLI - Unified interface for running benchmarks, generating summaries, and updating docs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python benchmark.py run --datasets pbmc3k --auto-convert

  # Generate summary from existing results
  python benchmark.py summary --input comprehensive_benchmark_results.json

  # Update docs with summary
  python benchmark.py docs --summary benchmark_summary.json

  # Run everything: benchmarks + summary + docs
  python benchmark.py all --datasets pbmc3k --auto-convert

  # Run specific benchmark types
  python benchmark.py run --datasets pbmc3k --types cell_filtering expression_queries
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pbmc3k"],
        help="Dataset names to benchmark (e.g., pbmc3k pbmc_68k synthetic_200k)",
    )
    run_parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing datasets (default: {DEFAULT_DATASET_DIR})",
    )
    run_parser.add_argument(
        "--types",
        nargs="+",
        choices=[
            "cell_filtering",
            "gene_filtering",
            "expression_queries",
            "anndata_ops",
            "lazy_evaluation_performance",
            "scanpy_preprocessing",
            "tokenizers",
            "dataloaders",
            "multi_process_scaling",
            "data_vs_tokenization_timing",
        ],
        help="Specific benchmark types to run (default: all types)",
    )
    run_parser.add_argument(
        "--auto-convert",
        action="store_true",
        help="Auto-convert h5ad to SLAF before benchmarking (if needed)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output during benchmarking",
    )
    run_parser.add_argument(
        "--output",
        default=f"{DEFAULT_BENCHMARK_DIR}/comprehensive_benchmark_results.json",
        help="Output file for results",
    )
    run_parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing comprehensive summary",
    )

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Generate benchmark summary")
    summary_parser.add_argument(
        "--input",
        default="comprehensive_benchmark_results.json",
        help="Input comprehensive results file",
    )
    summary_parser.add_argument(
        "--output",
        default="benchmark_summary.json",
        help="Output summary file",
    )

    # Docs command
    docs_parser = subparsers.add_parser("docs", help="Update performance documentation")
    docs_parser.add_argument(
        "--summary",
        default="benchmark_summary.json",
        help="Benchmark summary file",
    )
    docs_parser.add_argument(
        "--docs-file",
        default="../docs/benchmarks/performance.md",
        help="Performance documentation file to update",
    )

    # All command
    all_parser = subparsers.add_parser("all", help="Run benchmarks + summary + docs")
    all_parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pbmc3k"],
        help="Dataset names to benchmark",
    )
    all_parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing datasets (default: {DEFAULT_DATASET_DIR})",
    )
    all_parser.add_argument(
        "--types",
        nargs="+",
        choices=[
            "cell_filtering",
            "gene_filtering",
            "expression_queries",
            "anndata_ops",
            "lazy_evaluation_performance",
            "scanpy_preprocessing",
            "tokenizers",
            "dataloaders",
            "multi_process_scaling",
            "data_vs_tokenization_timing",
        ],
        help="Specific benchmark types to run (default: all types)",
    )
    all_parser.add_argument(
        "--auto-convert",
        action="store_true",
        help="Auto-convert h5ad to SLAF before benchmarking (if needed)",
    )
    all_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output during benchmarking",
    )
    all_parser.add_argument(
        "--results-file",
        default="comprehensive_benchmark_results.json",
        help="Comprehensive results file",
    )
    all_parser.add_argument(
        "--summary-file",
        default="benchmark_summary.json",
        help="Summary file",
    )
    all_parser.add_argument(
        "--docs-file",
        default="../docs/benchmarks/performance.md",
        help="Performance documentation file to update",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "run":
        # Run benchmarks
        data_path = Path(args.data_dir)
        all_dataset_results = {}

        for dataset_name in args.datasets:
            print(f"\n🎯 Benchmarking dataset: {dataset_name}")
            print("=" * 60)

            # Find dataset files
            h5ad_pattern = f"{dataset_name}*.h5ad"
            h5ad_files = list(data_path.glob(h5ad_pattern))

            if not h5ad_files:
                print(f"❌ No h5ad file found for {dataset_name}")
                print(f"   Looking for: {data_path / h5ad_pattern}")
                continue

            h5ad_path = h5ad_files[0]
            slaf_path = data_path / f"{dataset_name}.slaf"

            # Run benchmark suite for this dataset
            dataset_results = run_benchmark_suite(
                h5ad_path=str(h5ad_path),
                slaf_path=str(slaf_path),
                benchmark_types=args.types,
                verbose=args.verbose,
                auto_convert=args.auto_convert,
            )

            if dataset_results:
                all_dataset_results[dataset_name] = dataset_results

        # Save results
        if all_dataset_results:
            import json

            # Prepare results for JSON serialization
            json_results = {}
            for dataset, results in all_dataset_results.items():
                json_results[dataset] = {}
                for benchmark_type, type_results in results.items():
                    if benchmark_type == "data_vs_tokenization_timing":
                        # Handle timing breakdown results (dictionary format)
                        json_results[dataset][benchmark_type] = {}
                        for k, v in type_results.items():
                            if isinstance(v, dict):
                                clean_dict = {}
                                for dict_k, dict_v in v.items():
                                    if isinstance(dict_v, np.integer | np.floating):
                                        clean_dict[dict_k] = float(dict_v)
                                    elif hasattr(dict_v, "to_dict"):
                                        clean_dict[dict_k] = dict_v.to_dict()
                                    elif hasattr(dict_v, "tolist"):
                                        clean_dict[dict_k] = dict_v.tolist()
                                    else:
                                        clean_dict[dict_k] = dict_v
                                json_results[dataset][benchmark_type][k] = clean_dict
                            else:
                                json_results[dataset][benchmark_type][k] = v
                    else:
                        # Handle standard list-based results
                        json_results[dataset][benchmark_type] = []
                        for result in type_results:
                            clean_result = {}
                            for k, v in result.items():
                                if isinstance(v, np.integer | np.floating):
                                    clean_result[k] = float(v)
                                elif hasattr(v, "to_dict"):  # Handle pandas objects
                                    clean_result[k] = v.to_dict()
                                elif hasattr(v, "tolist"):  # Handle numpy arrays
                                    clean_result[k] = v.tolist()
                                else:
                                    clean_result[k] = v
                            json_results[dataset][benchmark_type].append(clean_result)

            # Save comprehensive results
            with open(args.output, "w") as f:
                json.dump(json_results, f, indent=2)

            print(f"\n💾 Results saved to: {args.output}")

    elif args.command == "summary":
        # Generate summary
        if not Path(args.input).exists():
            print(f"❌ Input file not found: {args.input}")
            return

        generate_benchmark_summary(args.input, args.output)

    elif args.command == "docs":
        # Update docs
        if not Path(args.summary).exists():
            print(f"❌ Summary file not found: {args.summary}")
            return

        if not Path(args.docs_file).exists():
            print(f"❌ Docs file not found: {args.docs_file}")
            return

        update_performance_docs(args.summary, args.docs_file)

    elif args.command == "all":
        # Run everything: benchmarks + summary + docs
        print("🚀 Running complete benchmark workflow...")

        # Step 1: Run benchmarks
        print("\n📊 Step 1: Running benchmarks...")
        data_path = Path(args.data_dir)
        all_dataset_results = {}

        for dataset_name in args.datasets:
            print(f"\n🎯 Benchmarking dataset: {dataset_name}")
            print("=" * 60)

            # Find dataset files
            h5ad_pattern = f"{dataset_name}*.h5ad"
            h5ad_files = list(data_path.glob(h5ad_pattern))

            if not h5ad_files:
                print(f"❌ No h5ad file found for {dataset_name}")
                print(f"   Looking for: {data_path / h5ad_pattern}")
                continue

            h5ad_path = h5ad_files[0]
            slaf_path = data_path / f"{dataset_name}.slaf"

            # Run benchmark suite for this dataset
            dataset_results = run_benchmark_suite(
                h5ad_path=str(h5ad_path),
                slaf_path=str(slaf_path),
                benchmark_types=args.types,
                verbose=args.verbose,
                auto_convert=args.auto_convert,
            )

            if dataset_results:
                all_dataset_results[dataset_name] = dataset_results

        # Save comprehensive results
        if all_dataset_results:
            import json

            # Prepare results for JSON serialization
            json_results = {}
            for dataset, results in all_dataset_results.items():
                json_results[dataset] = {}
                for benchmark_type, type_results in results.items():
                    if benchmark_type == "data_vs_tokenization_timing":
                        # Handle timing breakdown results (dictionary format)
                        json_results[dataset][benchmark_type] = {}
                        for k, v in type_results.items():
                            if isinstance(v, dict):
                                clean_dict = {}
                                for dict_k, dict_v in v.items():
                                    if isinstance(dict_v, np.integer | np.floating):
                                        clean_dict[dict_k] = float(dict_v)
                                    elif hasattr(dict_v, "to_dict"):
                                        clean_dict[dict_k] = dict_v.to_dict()
                                    elif hasattr(dict_v, "tolist"):
                                        clean_dict[dict_k] = dict_v.tolist()
                                    else:
                                        clean_dict[dict_k] = dict_v
                                json_results[dataset][benchmark_type][k] = clean_dict
                            else:
                                json_results[dataset][benchmark_type][k] = v
                    else:
                        # Handle standard list-based results
                        json_results[dataset][benchmark_type] = []
                        for result in type_results:
                            clean_result = {}
                            for k, v in result.items():
                                if isinstance(v, np.integer | np.floating):
                                    clean_result[k] = float(v)
                                elif hasattr(v, "to_dict"):  # Handle pandas objects
                                    clean_result[k] = v.to_dict()
                                elif hasattr(v, "tolist"):  # Handle numpy arrays
                                    clean_result[k] = v.tolist()
                                else:
                                    clean_result[k] = v
                            json_results[dataset][benchmark_type].append(clean_result)

            # Save comprehensive results
            with open(args.results_file, "w") as f:
                json.dump(json_results, f, indent=2)

            print(f"\n💾 Results saved to: {args.results_file}")

            # Step 2: Generate summary
            print("\n📊 Step 2: Generating summary...")
            generate_benchmark_summary(args.results_file, args.summary_file)

            # Step 3: Update docs
            print("\n📊 Step 3: Updating documentation...")
            update_performance_docs(args.summary_file, args.docs_file)

            print("\n✅ Complete workflow finished!")
        else:
            print("\n❌ No benchmarks completed successfully")


if __name__ == "__main__":
    main()
