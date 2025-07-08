#!/usr/bin/env python3
"""
Comprehensive SLAF Benchmark Suite

This script runs a complete set of benchmarks comparing SLAF vs h5ad across multiple domains:
- Cell filtering
- Gene filtering
- Expression queries
- AnnData operations
- Scanpy preprocessing

Each benchmark type is run in isolation with proper memory management and timing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from benchmark_anndata_ops import benchmark_anndata_ops

# Import individual benchmark modules
from benchmark_cell_filtering import benchmark_cell_filtering
from benchmark_dataloaders import (
    benchmark_data_vs_tokenization_timing,
    benchmark_dataloaders,
    benchmark_multi_process_scaling,
)
from benchmark_expression_queries import benchmark_expression_queries
from benchmark_gene_filtering import benchmark_gene_filtering
from benchmark_scanpy_preprocessing import benchmark_scanpy_preprocessing
from benchmark_tokenizers import benchmark_tokenizers

# Import utilities
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
                    _ = np.mean(
                        [r["total_speedup"] for r in results if r["total_speedup"] > 0]
                    )
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
                    _ = np.mean(
                        [r["total_speedup"] for r in results if r["total_speedup"] > 0]
                    )
                else:
                    print(f"⚠️  {benchmark_type}: No results generated")

        except Exception as e:
            print(f"❌ {benchmark_type} failed: {e}")
            continue

    return all_results


def print_comprehensive_summary(all_results: dict[str, list[dict]], dataset_name: str):
    """Print comprehensive summary across all benchmark types"""

    if not all_results:
        print("❌ No benchmark results to summarize")
        return

    print("\n" + "=" * 80)
    print("🏁 COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Benchmark types: {len(all_results)}")
    print("\n📊 Memory Efficiency Calculation:")
    print("   • Memory efficiency = h5ad memory / SLAF memory")
    print("   • Higher values indicate SLAF uses less memory")
    print(
        "   • Cases where SLAF uses <0.1MB are estimated as 0.01MB for realistic ratios"
    )
    print("   • Both using <0.1MB are treated as 1x efficiency")

    # Calculate overall statistics
    all_speedups = []
    all_memory_efficiencies = []
    type_summaries = []

    for _benchmark_type, type_results in all_results.items():
        if not type_results:
            continue

        # Handle different result formats
        if _benchmark_type == "data_vs_tokenization_timing":
            # Timing breakdown returns a dictionary with h5ad/slaf breakdowns
            # Skip this in the standard summary since it has its own analysis
            continue
        elif _benchmark_type == "multi_process_scaling":
            # Handle multi-process scaling results (different format)
            continue
        elif _benchmark_type == "dataloaders":
            # Dataloaders measures overhead, not speedup vs h5ad
            # Skip this in the standard summary since it has its own section
            continue
        elif isinstance(type_results, dict):
            # Handle other dictionary-based results
            continue
        else:
            # Standard list-based results
            # Calculate statistics for this type
            speedups = [
                r["total_speedup"] for r in type_results if r["total_speedup"] > 0
            ]
            memory_efficiencies = []

            # Calculate memory efficiency for each result
            for r in type_results:
                h5ad_mem = r.get("h5ad_total_memory_mb", 0)
                slaf_mem = r.get("slaf_total_memory_mb", 0)

                if slaf_mem > 0.1 and h5ad_mem > 0.1:
                    # Both used significant memory
                    mem_eff = h5ad_mem / slaf_mem
                    memory_efficiencies.append(mem_eff)
                elif h5ad_mem > 0.1 and slaf_mem <= 0.1:
                    # h5ad used memory but SLAF didn't - use actual ratio
                    # For very small SLAF memory, use a reasonable estimate
                    mem_eff = h5ad_mem / 0.01  # Assume SLAF used ~0.01 MB
                    memory_efficiencies.append(mem_eff)
                elif h5ad_mem <= 0.1 and slaf_mem <= 0.1:
                    # Both used negligible memory - treat as 1x
                    memory_efficiencies.append(1.0)

            if speedups:
                avg_speedup = np.mean(speedups)
                min_speedup = np.min(speedups)
                max_speedup = np.max(speedups)
                all_speedups.extend(speedups)

                # Memory efficiency statistics
                if memory_efficiencies:
                    avg_mem_eff = np.mean(memory_efficiencies)
                    min_mem_eff = np.min(memory_efficiencies)
                    max_mem_eff = np.max(memory_efficiencies)
                    all_memory_efficiencies.extend(memory_efficiencies)
                else:
                    avg_mem_eff = min_mem_eff = max_mem_eff = 0

                type_summaries.append(
                    {
                        "type": _benchmark_type,
                        "scenarios": len(type_results),
                        "avg_speedup": avg_speedup,
                        "min_speedup": min_speedup,
                        "max_speedup": max_speedup,
                        "avg_mem_eff": avg_mem_eff,
                        "min_mem_eff": min_mem_eff,
                        "max_mem_eff": max_mem_eff,
                    }
                )

    # Print type-by-type summary
    print("\n📊 BENCHMARK TYPE SUMMARY")
    print("-" * 80)

    # Use rich table for better formatting
    from rich.console import Console
    from rich.table import Table

    console = Console()
    summary_table = Table(
        title="BENCHMARK TYPE SUMMARY",
        show_header=True,
        header_style="bold magenta",
        title_style="bold blue",
    )

    summary_table.add_column("Type", style="cyan", width=20)
    summary_table.add_column("Scenarios", justify="right", style="yellow", width=10)
    summary_table.add_column("Avg Speedup", justify="right", style="green", width=12)
    summary_table.add_column("Min Speedup", justify="right", style="yellow", width=12)
    summary_table.add_column("Max Speedup", justify="right", style="green", width=12)
    summary_table.add_column("Avg Mem Eff", justify="right", style="cyan", width=12)
    summary_table.add_column("Min Mem Eff", justify="right", style="yellow", width=12)
    summary_table.add_column("Max Mem Eff", justify="right", style="cyan", width=12)

    for summary in type_summaries:
        # Color code speedups
        avg_speedup_text = f"{summary['avg_speedup']:.1f}x"
        min_speedup_text = f"{summary['min_speedup']:.1f}x"
        max_speedup_text = f"{summary['max_speedup']:.1f}x"

        if summary["avg_speedup"] > 2:
            avg_speedup_text = f"[green]{avg_speedup_text}[/green]"
        elif summary["avg_speedup"] > 1:
            avg_speedup_text = f"[yellow]{avg_speedup_text}[/yellow]"
        else:
            avg_speedup_text = f"[red]{avg_speedup_text}[/red]"

        # Color code memory efficiency
        avg_mem_eff_text = f"{summary['avg_mem_eff']:.1f}x"
        min_mem_eff_text = f"{summary['min_mem_eff']:.1f}x"
        max_mem_eff_text = f"{summary['max_mem_eff']:.1f}x"

        if summary["avg_mem_eff"] > 5:
            avg_mem_eff_text = f"[green]{avg_mem_eff_text}[/green]"
        elif summary["avg_mem_eff"] > 2:
            avg_mem_eff_text = f"[yellow]{avg_mem_eff_text}[/yellow]"
        else:
            avg_mem_eff_text = f"[red]{avg_mem_eff_text}[/red]"

        summary_table.add_row(
            summary["type"],
            str(summary["scenarios"]),
            avg_speedup_text,
            min_speedup_text,
            max_speedup_text,
            avg_mem_eff_text,
            min_mem_eff_text,
            max_mem_eff_text,
        )

    console.print(summary_table)

    # Handle special benchmark types
    special_benchmarks = []
    if "dataloaders" in all_results:
        dataloader_results = all_results["dataloaders"]
        if dataloader_results:
            overheads = [r.get("overhead_percent", 0) for r in dataloader_results]
            avg_overhead = np.mean(overheads)
            overhead_status = (
                "✅ Good"
                if avg_overhead < 15
                else "⚠️ High"
                if avg_overhead < 50
                else "❌ Critical"
            )
            special_benchmarks.append(
                f"📦 Dataloader overhead: {avg_overhead:.1f}% ({overhead_status})"
            )

    if "multi_process_scaling" in all_results:
        scaling_results = all_results["multi_process_scaling"]
        if scaling_results:
            # Extract scaling efficiency from multi-process results
            speedups = [
                r.get("total_speedup", 0)
                for r in scaling_results
                if r.get("total_speedup", 0) > 0
            ]
            if speedups:
                avg_scaling_speedup = np.mean(speedups)
                scaling_status = "✅ Good" if avg_scaling_speedup > 1.0 else "❌ Poor"
                special_benchmarks.append(
                    f"🔄 Multi-process scaling: {avg_scaling_speedup:.1f}x average ({scaling_status})"
                )
            else:
                special_benchmarks.append(
                    "🔄 Multi-process scaling: See detailed results"
                )

    if special_benchmarks:
        print("\n🔧 SPECIAL BENCHMARK TYPES")
        print("-" * 40)
        for benchmark in special_benchmarks:
            print(f"   {benchmark}")

            # Print overall summary
        if all_speedups:
            overall_avg = np.mean(all_speedups)
            overall_median = np.median(all_speedups)
            overall_min = np.min(all_speedups)
            overall_max = np.max(all_speedups)

            # Calculate percentiles for better understanding of distribution
            speedup_25th = np.percentile(all_speedups, 25)
            speedup_75th = np.percentile(all_speedups, 75)

        print("\n🎯 OVERALL SUMMARY")
        print("-" * 40)
        print(f"Total scenarios: {len(all_speedups)}")
        print(f"Average speedup: {overall_avg:.1f}x")
        print(f"Median speedup: {overall_median:.1f}x")
        print(f"Speedup range: {overall_min:.1f}x - {overall_max:.1f}x")
        print(
            f"Speedup (25th-75th percentile): {speedup_25th:.1f}x - {speedup_75th:.1f}x"
        )

        # Add memory efficiency to overall summary
        if all_memory_efficiencies:
            overall_mem_avg = np.mean(all_memory_efficiencies)
            overall_mem_median = np.median(all_memory_efficiencies)
            overall_mem_min = np.min(all_memory_efficiencies)
            overall_mem_max = np.max(all_memory_efficiencies)

            # Calculate percentiles for better understanding of distribution
            mem_25th = np.percentile(all_memory_efficiencies, 25)
            mem_75th = np.percentile(all_memory_efficiencies, 75)

            print(f"Average memory efficiency: {overall_mem_avg:.1f}x")
            print(f"Median memory efficiency: {overall_mem_median:.1f}x")
            print(
                f"Memory efficiency range: {overall_mem_min:.1f}x - {overall_mem_max:.1f}x"
            )
            print(
                f"Memory efficiency (25th-75th percentile): {mem_25th:.1f}x - {mem_75th:.1f}x"
            )

        # Performance insights
        print("\n💡 PERFORMANCE INSIGHTS")
        print("-" * 40)

        # Speedup insights
        if overall_avg > 3:
            print("🚀 SLAF shows exceptional performance improvements!")
        elif overall_avg > 2:
            print("🚀 SLAF shows excellent performance improvements!")
        elif overall_avg > 1.5:
            print("✅ SLAF shows good performance improvements")
        elif overall_avg > 1:
            print("👍 SLAF shows modest performance improvements")
        else:
            print("⚠️  SLAF performance needs investigation")

        # Memory efficiency insights
        if all_memory_efficiencies:
            if overall_mem_avg > 100:
                print("💾 SLAF shows exceptional memory efficiency!")
            elif overall_mem_avg > 50:
                print("💾 SLAF shows excellent memory efficiency!")
            elif overall_mem_avg > 20:
                print("💾 SLAF shows very good memory efficiency")
            elif overall_mem_avg > 10:
                print("💾 SLAF shows good memory efficiency")
            elif overall_mem_avg > 5:
                print("💾 SLAF shows moderate memory efficiency")
            else:
                print("⚠️  SLAF memory efficiency needs investigation")

        # Identify best and worst performing types
        if type_summaries:
            best_type = max(type_summaries, key=lambda x: x["avg_speedup"])
            worst_type = min(type_summaries, key=lambda x: x["avg_speedup"])

            print(
                f"Best performing: {best_type['type']} ({best_type['avg_speedup']:.1f}x)"
            )
            print(
                f"Needs attention: {worst_type['type']} ({worst_type['avg_speedup']:.1f}x)"
            )

        # Optimization priorities
        print("\n🎯 OPTIMIZATION PRIORITIES")
        print("-" * 40)

        # Find areas needing attention
        needs_attention = [s for s in type_summaries if s["avg_speedup"] < 1.0]
        if needs_attention:
            print("🔴 Critical issues (speedup < 1.0x):")
            for issue in needs_attention:
                print(f"   • {issue['type']}: {issue['avg_speedup']:.1f}x")

        # Add multi-process scaling if it's critically poor
        if "multi_process_scaling" in all_results:
            scaling_results = all_results["multi_process_scaling"]
            if scaling_results:
                speedups = [
                    r.get("total_speedup", 0)
                    for r in scaling_results
                    if r.get("total_speedup", 0) > 0
                ]
                if speedups:
                    avg_scaling_speedup = np.mean(speedups)
                    if avg_scaling_speedup < 1.0:
                        print(f"   • multi_process_scaling: {avg_scaling_speedup:.1f}x")

        # Find speedup wins
        speedup_wins = [s for s in type_summaries if s["avg_speedup"] > 2.0]
        if speedup_wins:
            print("🚀 Speedup wins (>2.0x):")
            for win in speedup_wins:
                speedup_level = (
                    "exceptional"
                    if win["avg_speedup"] > 3.0
                    else "excellent"
                    if win["avg_speedup"] > 2.5
                    else "very good"
                )
                print(
                    f"   • {win['type']}: {win['avg_speedup']:.1f}x speedup ({speedup_level})"
                )

        # Find memory efficiency wins
        memory_wins = [s for s in type_summaries if s["avg_mem_eff"] > 10]
        if memory_wins:
            print("💚 Memory efficiency wins (>10x):")
            for win in memory_wins:
                efficiency_level = (
                    "exceptional"
                    if win["avg_mem_eff"] > 50
                    else "excellent"
                    if win["avg_mem_eff"] > 20
                    else "very good"
                )
                print(
                    f"   • {win['type']}: {win['avg_mem_eff']:.1f}x memory efficiency ({efficiency_level})"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive SLAF performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks on pbmc3k
  python run_comprehensive_benchmarks.py --datasets pbmc3k --auto-convert

  # Run specific benchmark types
  python run_comprehensive_benchmarks.py --datasets pbmc3k --types cell_filtering expression_queries

  # Run with verbose output
  python run_comprehensive_benchmarks.py --datasets pbmc3k --verbose --auto-convert

  # Run on multiple datasets
  python run_comprehensive_benchmarks.py --datasets pbmc3k pbmc_68k --auto-convert
        """,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pbmc3k"],
        help="Dataset names to benchmark (e.g., pbmc3k pbmc_68k synthetic_200k)",
    )

    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing datasets (default: {DEFAULT_DATASET_DIR})",
    )

    parser.add_argument(
        "--types",
        nargs="+",
        choices=[
            "cell_filtering",
            "gene_filtering",
            "expression_queries",
            "anndata_ops",
            "scanpy_preprocessing",
            "tokenizers",
            "dataloaders",
            "multi_process_scaling",
            "data_vs_tokenization_timing",
        ],
        help="Specific benchmark types to run (default: all types)",
    )

    parser.add_argument(
        "--auto-convert",
        action="store_true",
        help="Auto-convert h5ad to SLAF before benchmarking (if needed)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output during benchmarking",
    )

    parser.add_argument(
        "--output",
        default=f"{DEFAULT_BENCHMARK_DIR}/comprehensive_benchmark_results.json",
        help="Output file for results (default: benchmarks/comprehensive_benchmark_results.json)",
    )

    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing comprehensive summary",
    )

    parser.add_argument(
        "--output-stdout",
        help="Capture stdout to file (e.g., --output-stdout benchmark_output.txt)",
    )

    args = parser.parse_args()

    # Set up stdout capture if requested
    original_stdout = None
    if args.output_stdout:
        original_stdout = sys.stdout
        sys.stdout = open(args.output_stdout, "w")
        print(f"# SLAF Benchmark Output - {args.output_stdout}")
        print(f"# Generated on: {Path(__file__).parent}")
        print("#" * 80)

    print("⚡ SLAF Comprehensive Performance Benchmark Suite")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Datasets: {args.datasets}")
    print(f"Benchmark types: {args.types if args.types else 'all'}")
    print(f"Auto-convert: {args.auto_convert}")
    print(f"Verbose: {args.verbose}")
    print(f"Output: {args.output}")
    print("\n" + "=" * 60)

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

            # Print comprehensive summary for this dataset
            if not args.no_summary:
                print_comprehensive_summary(dataset_results, dataset_name)
        else:
            print(f"❌ No benchmarks completed for {dataset_name}")

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

        # Save to file
        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"\n💾 Results saved to: {args.output}")

        # Print final summary
        if not args.no_summary and len(all_dataset_results) > 1:
            print("\n" + "=" * 80)
            print("🏁 FINAL SUMMARY ACROSS ALL DATASETS")
            print("=" * 80)

            for dataset_name, results in all_dataset_results.items():
                all_speedups = []
                for _benchmark_type, type_results in results.items():
                    speedups = [
                        r["total_speedup"]
                        for r in type_results
                        if r["total_speedup"] > 0
                    ]
                    all_speedups.extend(speedups)

                if all_speedups:
                    avg_speedup = np.mean(all_speedups)
                    print(f"✅ {dataset_name}: {avg_speedup:.1f}x average speedup")

    else:
        print("\n❌ No benchmarks completed successfully")
        print("   Check dataset availability and paths")

    print("\n" + "=" * 60)
    print("🏁 BENCHMARK SUITE COMPLETED")
    print("=" * 60)

    # Restore stdout if it was redirected
    if original_stdout:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"✅ Benchmark output saved to: {args.output_stdout}")


if __name__ == "__main__":
    main()
