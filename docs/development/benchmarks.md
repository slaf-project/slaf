# SLAF Benchmark System

This document describes the SLAF benchmark suite for performance testing and documentation generation. The benchmark system has been refactored to separate bioinformatics and ML benchmarks.

## 🚀 Quick Start (Recommended)

### Bioinformatics Benchmarks (CLI Integration)

Use the unified CLI interface for bioinformatics benchmark operations:

```bash
# Run bioinformatics benchmarks
slaf benchmark run --datasets pbmc3k_processed --types cell_filtering,expression_queries --verbose

# Generate summary from results
slaf benchmark summary --results comprehensive_benchmark_results.json

# Update documentation
slaf benchmark docs --summary benchmark_summary.json

# Run complete workflow
slaf benchmark all --datasets pbmc3k_processed --auto-convert
```

### ML Benchmarks (Standalone Scripts)

ML benchmarks are run as standalone scripts:

```bash
# External dataloader comparisons
python benchmarks/benchmark_dataloaders_external.py

# Internal tokenization strategies
python benchmarks/benchmark_dataloaders_internal.py

# Prefetcher performance analysis
python benchmarks/benchmark_prefetcher.py
```

## 📁 File Structure

### Core Files

- `benchmarks/benchmark.py` - Main bioinformatics benchmark runner with CLI integration
- `benchmarks/benchmark_utils.py` - Shared utilities for bioinformatics benchmarks

### Bioinformatics Benchmark Modules (CLI Integrated)

- `benchmarks/benchmark_cell_filtering.py` - Cell filtering performance tests
- `benchmarks/benchmark_gene_filtering.py` - Gene filtering performance tests
- `benchmarks/benchmark_expression_queries.py` - Expression query performance tests
- `benchmarks/benchmark_anndata_ops.py` - AnnData operation performance tests
- `benchmarks/benchmark_scanpy_preprocessing.py` - Scanpy preprocessing performance tests

### ML Benchmark Modules (Standalone)

- `benchmarks/benchmark_dataloaders_external.py` - External dataloader comparisons (SLAF vs scDataset, BioNeMo, etc.)
- `benchmarks/benchmark_dataloaders_internal.py` - Internal tokenization strategy comparisons (scGPT, Geneformer, etc.)
- `benchmarks/benchmark_prefetcher.py` - Prefetcher pipeline performance analysis

### Output Files

- `benchmarks/comprehensive_benchmark_results.json` - Complete bioinformatics benchmark results
- `benchmarks/benchmark_summary.json` - Documentation-ready summary
- `benchmarks/benchmark_output.txt` - Detailed benchmark output
- `benchmarks/benchmark_results.json` - Legacy results file

## 🔧 CLI Commands (Bioinformatics Only)

### Run Benchmarks

```bash
# Run all bioinformatics benchmark types
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
# Update bioinformatics_benchmarks.md with summary data
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

### Bioinformatics Benchmarks (CLI Integrated)

- **cell_filtering** - Metadata-based cell filtering performance
- **gene_filtering** - Metadata-based gene filtering performance
- **expression_queries** - Expression matrix slicing performance
- **anndata_ops** - AnnData operation performance
- **scanpy_preprocessing** - Scanpy preprocessing pipeline performance

### ML Benchmarks (Standalone Scripts)

- **External Dataloader Comparisons** - SLAF vs scDataset, BioNeMo SCDL, AnnDataLoader
- **Internal Tokenization Strategies** - scGPT, Geneformer, raw data loading
- **Prefetcher Performance** - Pipeline timing analysis across configurations

## 🎯 Usage Examples

### Bioinformatics Development Workflow

```bash
# Quick test of cell filtering
slaf benchmark run --datasets pbmc3k_processed --types cell_filtering --verbose

# Comprehensive testing
slaf benchmark all --datasets pbmc3k_processed --auto-convert --verbose
```

### ML Development Workflow

```bash
# Compare against external dataloaders
python benchmarks/benchmark_dataloaders_external.py

# Test different tokenization strategies
python benchmarks/benchmark_dataloaders_internal.py

# Analyze prefetcher performance
python benchmarks/benchmark_prefetcher.py
```

### Performance Analysis

```bash
# Generate bioinformatics performance summary
slaf benchmark summary --results comprehensive_benchmark_results.json

# Update bioinformatics documentation with latest results
slaf benchmark docs --summary benchmark_summary.json
```

### Multi-Dataset Testing

```bash
# Test bioinformatics benchmarks on multiple datasets
slaf benchmark run --datasets pbmc3k_processed pbmc_68k --types cell_filtering,expression_queries --auto-convert
```

## 📈 Output Files

### Bioinformatics Results Files

- `comprehensive_benchmark_results.json` - Complete benchmark results with detailed timing and memory data
- `benchmark_summary.json` - Condensed summary for documentation updates
- `benchmark_output.txt` - Human-readable benchmark output with tables and analysis

### ML Results Files

- ML benchmarks output results directly to console with rich formatting
- Results are not automatically saved to files (manual documentation updates required)

### Documentation Integration

The bioinformatics benchmark system automatically updates `docs/benchmarks/bioinformatics_benchmarks.md` with the latest performance data, ensuring documentation stays current with benchmark results. ML benchmarks are documented separately in `docs/benchmarks/ml_benchmarks.md` and require manual updates.

## 🔍 Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure datasets are in the correct directory and use `--auto-convert` to convert h5ad files
2. **Benchmark failures**: Check that SLAF files exist and are properly formatted
3. **Memory issues**: Some benchmarks require significant memory for large datasets
4. **ML benchmark dependencies**: Ensure all ML dependencies are installed for standalone ML benchmarks

### Debug Mode

```bash
# Run bioinformatics benchmarks with verbose output for debugging
slaf benchmark run --datasets pbmc3k_processed --types cell_filtering --verbose

# Run ML benchmarks with debug output
python benchmarks/benchmark_dataloaders_external.py --debug
```

## 📝 Contributing

### Adding Bioinformatics Benchmarks

When adding new bioinformatics benchmarks:

1. Create a new benchmark module following the existing pattern
2. Add the benchmark type to the CLI in `slaf/cli.py`
3. Update this documentation with the new benchmark type
4. Test with `slaf benchmark run --types your_new_benchmark`

### Adding ML Benchmarks

When adding new ML benchmarks:

1. Create a new standalone benchmark script following the existing pattern
2. Add appropriate documentation in `docs/benchmarks/ml_benchmarks.md`
3. Test the standalone script directly
4. Consider integration with CLI system in the future

## 🏗️ Architecture

The benchmark system uses a modular design with two distinct approaches:

### Bioinformatics Benchmarks (CLI Integrated)

- **CLI Interface**: Unified command-line interface in `slaf/cli.py`
- **Benchmark Runner**: Main orchestration in `benchmarks/benchmark.py`
- **Individual Modules**: Specialized benchmark tests in separate files
- **Utilities**: Shared functions in `benchmarks/benchmark_utils.py`
- **Documentation**: Automatic updates to `docs/benchmarks/bioinformatics_benchmarks.md`

### ML Benchmarks (Standalone)

- **Standalone Scripts**: Independent benchmark scripts with rich console output
- **External Comparisons**: `benchmark_dataloaders_external.py` for competitor analysis
- **Internal Analysis**: `benchmark_dataloaders_internal.py` for tokenization strategies
- **Pipeline Analysis**: `benchmark_prefetcher.py` for prefetcher performance
- **Documentation**: Manual updates to `docs/benchmarks/ml_benchmarks.md`

## 🔄 Future Integration

The ML benchmarks are currently standalone but may be integrated with the CLI system in the future to provide:

- Unified benchmark execution
- Automatic result aggregation
- Integrated documentation updates
- Consistent output formatting

For now, ML benchmarks provide immediate value as standalone tools for development and performance analysis.
