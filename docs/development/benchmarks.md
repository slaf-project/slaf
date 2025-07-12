# SLAF Benchmark System

This document describes the SLAF benchmark suite for performance testing and documentation generation.

## 🚀 Quick Start (Recommended)

Use the unified CLI interface for all benchmark operations:

```bash
# Run benchmarks
slaf benchmark run --datasets pbmc3k_processed --types cell_filtering,expression_queries --verbose

# Generate summary from results
slaf benchmark summary --results comprehensive_benchmark_results.json

# Update documentation
slaf benchmark docs --summary benchmark_summary.json

# Run complete workflow
slaf benchmark all --datasets pbmc3k_processed --auto-convert
```

## 📁 File Structure

### Core Files

- `benchmarks/benchmark.py` - Main benchmark runner with CLI integration
- `benchmarks/benchmark_utils.py` - Shared utilities for all benchmarks

### Individual Benchmark Modules

- `benchmarks/benchmark_cell_filtering.py` - Cell filtering performance tests
- `benchmarks/benchmark_gene_filtering.py` - Gene filtering performance tests
- `benchmarks/benchmark_expression_queries.py` - Expression query performance tests
- `benchmarks/benchmark_anndata_ops.py` - AnnData operation performance tests
- `benchmarks/benchmark_scanpy_preprocessing.py` - Scanpy preprocessing performance tests
- `benchmarks/benchmark_tokenizers.py` - Tokenizer throughput tests
- `benchmarks/benchmark_dataloaders.py` - Dataloader overhead tests

### Output Files

- `benchmarks/comprehensive_benchmark_results.json` - Complete benchmark results
- `benchmarks/benchmark_summary.json` - Documentation-ready summary
- `benchmarks/benchmark_output.txt` - Detailed benchmark output
- `benchmarks/benchmark_results.json` - Legacy results file

## 🔧 CLI Commands

### Run Benchmarks

```bash
# Run all benchmark types
slaf benchmark run --datasets pbmc3k_processed --auto-convert

# Run specific benchmark types
slaf benchmark run --datasets pbmc3k_processed --types cell_filtering,expression_queries

# Run with verbose output
slaf benchmark run --datasets pbmc3k_processed --verbose --auto-convert

# Run on multiple datasets
slaf benchmark run --datasets pbmc3k_processed pbmc_68k --auto-convert
```

### Generate Summary

```bash
# Generate summary from existing results
slaf benchmark summary --results comprehensive_benchmark_results.json

# Generate summary with custom output
slaf benchmark summary --results comprehensive_benchmark_results.json --output custom_summary.json
```

### Update Documentation

```bash
# Update performance.md with summary data
slaf benchmark docs --summary benchmark_summary.json

# Update with custom summary file
slaf benchmark docs --summary custom_summary.json
```

### Complete Workflow

```bash
# Run benchmarks, generate summary, and update docs
slaf benchmark all --datasets pbmc3k_processed --auto-convert --verbose
```

## 📊 Available Benchmark Types

- **cell_filtering** - Metadata-based cell filtering performance
- **gene_filtering** - Metadata-based gene filtering performance
- **expression_queries** - Expression matrix slicing performance
- **anndata_ops** - AnnData operation performance
- **scanpy_preprocessing** - Scanpy preprocessing pipeline performance
- **tokenizers** - Tokenizer throughput for transformer training
- **dataloaders** - Dataloader overhead analysis
- **multi_process_scaling** - Multi-process scaling analysis
- **data_vs_tokenization_timing** - Data loading vs tokenization timing

## 🎯 Usage Examples

### Development Workflow

```bash
# Quick test of cell filtering
slaf benchmark run --datasets pbmc3k_processed --types cell_filtering --verbose

# Comprehensive testing
slaf benchmark all --datasets pbmc3k_processed --auto-convert --verbose
```

### Performance Analysis

```bash
# Generate performance summary
slaf benchmark summary --results comprehensive_benchmark_results.json

# Update documentation with latest results
slaf benchmark docs --summary benchmark_summary.json
```

### Multi-Dataset Testing

```bash
# Test on multiple datasets
slaf benchmark run --datasets pbmc3k_processed pbmc_68k --types cell_filtering,expression_queries --auto-convert
```

## 📈 Output Files

### Results Files

- `comprehensive_benchmark_results.json` - Complete benchmark results with detailed timing and memory data
- `benchmark_summary.json` - Condensed summary for documentation updates
- `benchmark_output.txt` - Human-readable benchmark output with tables and analysis

### Documentation Integration

The benchmark system automatically updates `docs/benchmarks/performance.md` with the latest performance data, ensuring documentation stays current with benchmark results.

## 🔍 Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure datasets are in the correct directory and use `--auto-convert` to convert h5ad files
2. **Benchmark failures**: Check that SLAF files exist and are properly formatted
3. **Memory issues**: Some benchmarks require significant memory for large datasets

### Debug Mode

```bash
# Run with verbose output for debugging
slaf benchmark run --datasets pbmc3k_processed --types cell_filtering --verbose
```

## 📝 Contributing

When adding new benchmarks:

1. Create a new benchmark module following the existing pattern
2. Add the benchmark type to the CLI in `slaf/cli.py`
3. Update this documentation with the new benchmark type
4. Test with `slaf benchmark run --types your_new_benchmark`

## 🏗️ Architecture

The benchmark system uses a modular design:

- **CLI Interface**: Unified command-line interface in `slaf/cli.py`
- **Benchmark Runner**: Main orchestration in `benchmarks/benchmark.py`
- **Individual Modules**: Specialized benchmark tests in separate files
- **Utilities**: Shared functions in `benchmarks/benchmark_utils.py`
- **Documentation**: Automatic updates to `docs/benchmarks/performance.md`
