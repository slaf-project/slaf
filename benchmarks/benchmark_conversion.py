#!/usr/bin/env python3
"""
Conversion Performance Benchmark

Compares conversion performance between SLAF and TileDB SOMA formats.
Tests both CLI-based conversion (SLAF) and programmatic conversion (TileDB).
"""

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import psutil
import tiledbsoma.io as soma_io

from slaf.data.converter import SLAFConverter


def monitor_memory(process, duration, interval=0.1):
    """Monitor memory usage of a process."""
    max_memory = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        try:
            memory_info = process.memory_info()
            current_memory = memory_info.rss / (1024 * 1024)  # Convert to MB
            max_memory = max(max_memory, current_memory)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(interval)

    return max_memory


def benchmark_slaf_conversion(h5ad_path: str, output_dir: str) -> dict[str, float]:
    """
    Benchmark SLAF conversion using the SLAFConverter class.

    Args:
        h5ad_path: Path to input h5ad file
        output_dir: Directory to store output

    Returns:
        Dictionary with timing results
    """
    slaf_output = os.path.join(output_dir, "converted.slaf")

    # Clean up any existing output
    if os.path.exists(slaf_output):
        shutil.rmtree(slaf_output)

    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)
    max_memory = initial_memory

    # Run SLAF conversion
    start_time = time.time()

    try:
        # Monitor memory during conversion
        def memory_monitor():
            nonlocal max_memory
            while True:
                try:
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    max_memory = max(max_memory, current_memory)
                    time.sleep(0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()

        # Use SLAFConverter class directly
        converter = SLAFConverter(
            chunked=True,
            chunk_size=5000,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
            enable_v2_manifest=True,
            compact_after_write=False,
        )
        converter.convert(h5ad_path, slaf_output)

        end_time = time.time()
        conversion_time = end_time - start_time

        # Get final memory reading
        try:
            final_memory = process.memory_info().rss / (1024 * 1024)
            max_memory = max(max_memory, final_memory)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Get output size
        output_size = 0
        if os.path.exists(slaf_output):
            for root, _dirs, files in os.walk(slaf_output):
                for file in files:
                    output_size += os.path.getsize(os.path.join(root, file))

        return {
            "conversion_time": conversion_time,
            "output_size_mb": output_size / (1024 * 1024),
            "peak_memory_mb": max_memory,
            "success": True,
            "error": None,
        }

    except Exception as e:
        end_time = time.time()
        return {
            "conversion_time": end_time - start_time,
            "output_size_mb": 0,
            "peak_memory_mb": max_memory,
            "success": False,
            "error": f"SLAF error: {str(e)}",
        }


def benchmark_tiledb_conversion(h5ad_path: str, output_dir: str) -> dict[str, float]:
    """
    Benchmark TileDB SOMA conversion using from_h5ad function.

    Args:
        h5ad_path: Path to input h5ad file
        output_dir: Directory to store output

    Returns:
        Dictionary with timing results
    """
    tiledb_output = os.path.join(output_dir, "converted_tiledb")

    # Clean up any existing output
    if os.path.exists(tiledb_output):
        shutil.rmtree(tiledb_output)

    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)
    max_memory = initial_memory

    # Run TileDB conversion
    start_time = time.time()

    try:
        # Monitor memory during conversion
        def memory_monitor():
            nonlocal max_memory
            while True:
                try:
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    max_memory = max(max_memory, current_memory)
                    time.sleep(0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()

        # Use from_h5ad function as specified in the TileDB documentation
        soma_io.from_h5ad(
            tiledb_output,
            h5ad_path,
            measurement_name="RNA",
            obs_id_name="obs_id",
            var_id_name="var_id",
        )
        end_time = time.time()

        conversion_time = end_time - start_time

        # Get final memory reading
        try:
            final_memory = process.memory_info().rss / (1024 * 1024)
            max_memory = max(max_memory, final_memory)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Get output size
        output_size = 0
        if os.path.exists(tiledb_output):
            for root, _dirs, files in os.walk(tiledb_output):
                for file in files:
                    output_size += os.path.getsize(os.path.join(root, file))

        return {
            "conversion_time": conversion_time,
            "output_size_mb": output_size / (1024 * 1024),
            "peak_memory_mb": max_memory,
            "success": True,
            "error": None,
        }

    except Exception as e:
        end_time = time.time()
        return {
            "conversion_time": end_time - start_time,
            "output_size_mb": 0,
            "peak_memory_mb": max_memory,
            "success": False,
            "error": f"TileDB error: {str(e)}",
        }


def run_conversion_benchmark(
    h5ad_path: str, num_runs: int = 3
) -> dict[str, list[dict]]:
    """
    Run conversion benchmark comparing SLAF and TileDB.

    Args:
        h5ad_path: Path to input h5ad file
        num_runs: Number of runs for averaging

    Returns:
        Dictionary with results for both systems
    """
    print(f"🔄 Running conversion benchmark on {h5ad_path}")
    print(f"📊 Running {num_runs} iterations for each system")

    slaf_results = []
    tiledb_results = []

    for run in range(num_runs):
        print(f"\n🏃 Run {run + 1}/{num_runs}")

        # Create temporary directory for this run
        with tempfile.TemporaryDirectory() as temp_dir:
            # Benchmark SLAF
            print("  📦 Testing SLAF conversion...")
            slaf_result = benchmark_slaf_conversion(h5ad_path, temp_dir)
            slaf_result["run"] = run + 1
            slaf_results.append(slaf_result)

            if slaf_result["success"]:
                print(
                    f"    ✅ SLAF: {slaf_result['conversion_time']:.2f}s, {slaf_result['output_size_mb']:.1f}MB, {slaf_result['peak_memory_mb']:.1f}MB peak"
                )
            else:
                print(f"    ❌ SLAF failed: {slaf_result['error']}")

            # Benchmark TileDB
            print("  🧱 Testing TileDB conversion...")
            tiledb_result = benchmark_tiledb_conversion(h5ad_path, temp_dir)
            tiledb_result["run"] = run + 1
            tiledb_results.append(tiledb_result)

            if tiledb_result["success"]:
                print(
                    f"    ✅ TileDB: {tiledb_result['conversion_time']:.2f}s, {tiledb_result['output_size_mb']:.1f}MB, {tiledb_result['peak_memory_mb']:.1f}MB peak"
                )
            else:
                print(f"    ❌ TileDB failed: {tiledb_result['error']}")

    return {"slaf": slaf_results, "tiledb": tiledb_results}


def calculate_statistics(results: list[dict]) -> dict[str, float]:
    """Calculate statistics from benchmark results."""
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        return {
            "mean_time": 0,
            "std_time": 0,
            "min_time": 0,
            "max_time": 0,
            "mean_size": 0,
            "mean_memory": 0,
            "max_memory": 0,
            "success_rate": 0,
        }

    times = [r["conversion_time"] for r in successful_results]
    sizes = [r["output_size_mb"] for r in successful_results]
    memories = [r["peak_memory_mb"] for r in successful_results]

    return {
        "mean_time": sum(times) / len(times),
        "std_time": (
            sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)
        )
        ** 0.5,
        "min_time": min(times),
        "max_time": max(times),
        "mean_size": sum(sizes) / len(sizes),
        "mean_memory": sum(memories) / len(memories),
        "max_memory": max(memories),
        "success_rate": len(successful_results) / len(results),
    }


def print_benchmark_results(results: dict[str, list[dict]], dataset_name: str):
    """Print formatted benchmark results."""
    print(f"\n📊 Conversion Benchmark Results: {dataset_name}")
    print("=" * 60)

    slaf_stats = calculate_statistics(results["slaf"])
    tiledb_stats = calculate_statistics(results["tiledb"])

    print("\n🔄 SLAF Conversion:")
    if slaf_stats["success_rate"] > 0:
        print(
            f"  ⏱️  Time: {slaf_stats['mean_time']:.2f} ± {slaf_stats['std_time']:.2f}s (min: {slaf_stats['min_time']:.2f}s, max: {slaf_stats['max_time']:.2f}s)"
        )
        print(f"  📦 Size: {slaf_stats['mean_size']:.1f}MB")
        print(f"  🧠 Peak Memory: {slaf_stats['max_memory']:.1f}MB")
        print(f"  ✅ Success Rate: {slaf_stats['success_rate'] * 100:.0f}%")
    else:
        print("  ❌ All runs failed")

    print("\n🧱 TileDB Conversion:")
    if tiledb_stats["success_rate"] > 0:
        print(
            f"  ⏱️  Time: {tiledb_stats['mean_time']:.2f} ± {tiledb_stats['std_time']:.2f}s (min: {tiledb_stats['min_time']:.2f}s, max: {tiledb_stats['max_time']:.2f}s)"
        )
        print(f"  📦 Size: {tiledb_stats['mean_size']:.1f}MB")
        print(f"  🧠 Peak Memory: {tiledb_stats['max_memory']:.1f}MB")
        print(f"  ✅ Success Rate: {tiledb_stats['success_rate'] * 100:.0f}%")
    else:
        print("  ❌ All runs failed")

    # Calculate speedup and efficiency
    if slaf_stats["success_rate"] > 0 and tiledb_stats["success_rate"] > 0:
        speedup = tiledb_stats["mean_time"] / slaf_stats["mean_time"]
        memory_efficiency = (
            tiledb_stats["max_memory"] / slaf_stats["max_memory"]
            if slaf_stats["max_memory"] > 0
            else float("inf")
        )
        size_ratio = slaf_stats["mean_size"] / tiledb_stats["mean_size"]

        print("\n🚀 Performance Comparison:")
        print(f"  SLAF vs TileDB Speedup: {speedup:.1f}x faster")
        print(
            f"  SLAF vs TileDB Memory Efficiency: {memory_efficiency:.1f}x more memory efficient"
        )
        print(f"  SLAF vs TileDB Size Ratio: {size_ratio:.2f}x")

    return {
        "slaf_stats": slaf_stats,
        "tiledb_stats": tiledb_stats,
        "speedup": (
            tiledb_stats["mean_time"] / slaf_stats["mean_time"]
            if slaf_stats["success_rate"] > 0 and tiledb_stats["success_rate"] > 0
            else 0
        ),
        "memory_efficiency": (
            tiledb_stats["max_memory"] / slaf_stats["max_memory"]
            if slaf_stats["success_rate"] > 0
            and tiledb_stats["success_rate"] > 0
            and slaf_stats["max_memory"] > 0
            else 0
        ),
    }


def main():
    """Main benchmark function."""
    import argparse

    parser = argparse.ArgumentParser(description="Conversion Performance Benchmark")
    parser.add_argument("--h5ad-path", required=True, help="Path to h5ad file")
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs (default: 3)"
    )
    parser.add_argument("--dataset-name", help="Dataset name for reporting")

    args = parser.parse_args()

    if not os.path.exists(args.h5ad_path):
        print(f"❌ File not found: {args.h5ad_path}")
        return

    dataset_name = args.dataset_name or Path(args.h5ad_path).stem

    # Run benchmark
    results = run_conversion_benchmark(args.h5ad_path, args.runs)

    # Print results
    summary = print_benchmark_results(results, dataset_name)

    return results, summary


if __name__ == "__main__":
    main()
