import sys
from pathlib import Path

# Add the benchmarks directory to the path so we can import the modules
benchmarks_dir = Path(__file__).parent
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

# Import individual benchmark modules
from . import (  # noqa: E402
    benchmark_anndata_ops,
    benchmark_cell_filtering,
    benchmark_expression_queries,
    benchmark_gene_filtering,
    benchmark_scanpy_preprocessing,
)
from .benchmark import (  # noqa: E402
    generate_benchmark_summary,
    run_benchmark_suite,
    update_performance_docs,
)

"""
SLAF Benchmark Module

This module provides comprehensive benchmarking capabilities for SLAF,
comparing performance against traditional single-cell data formats.

Main functions:
- run_benchmark_suite: Run comprehensive benchmark suite
- generate_benchmark_summary: Generate summary from results
- update_performance_docs: Update documentation with benchmark data
"""

__all__ = [
    "run_benchmark_suite",
    "generate_benchmark_summary",
    "update_performance_docs",
    "benchmark_anndata_ops",
    "benchmark_cell_filtering",
    "benchmark_expression_queries",
    "benchmark_gene_filtering",
    "benchmark_scanpy_preprocessing",
]
