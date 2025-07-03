# SLAF Comprehensive Benchmark Suite

This directory contains a comprehensive benchmark suite for comparing SLAF performance against h5ad across multiple domains of single-cell analysis.

## Overview

The benchmark suite is organized into modular components that test different aspects of single-cell data analysis:

- **Cell Filtering**: Metadata-based cell selection scenarios
- **Gene Filtering**: Gene selection and filtering operations
- **Expression Queries**: Expression data extraction and aggregation
- **AnnData Operations**: Standard AnnData interface operations
- **Scanpy Preprocessing**: Scanpy preprocessing pipeline operations

## File Structure

```
benchmarks/
├── README.md                           # This file
├── benchmark_utils.py                  # Shared utilities and table formatting
├── benchmark_cell_filtering.py         # Cell filtering benchmarks
├── benchmark_gene_filtering.py         # Gene filtering benchmarks
├── benchmark_expression_queries.py     # Expression query benchmarks
├── benchmark_anndata_ops.py            # AnnData operation benchmarks
├── benchmark_scanpy_preprocessing.py   # Scanpy preprocessing benchmarks
├── run_comprehensive_benchmarks.py     # Master benchmark runner
└── benchmarks.py                       # Legacy single-file benchmarks
```

## Quick Start

### Run All Benchmarks

```bash
# Run all benchmark types on pbmc3k dataset
python run_comprehensive_benchmarks.py --datasets pbmc3k --auto-convert

# Run with verbose output
python run_comprehensive_benchmarks.py --datasets pbmc3k --verbose --auto-convert
```

### Run Specific Benchmark Types

```bash
# Run only cell filtering and expression queries
python run_comprehensive_benchmarks.py --datasets pbmc3k --types cell_filtering expression_queries --auto-convert

# Run on multiple datasets
python run_comprehensive_benchmarks.py --datasets pbmc3k pbmc_68k --auto-convert
```

## Benchmark Types

### 1. Cell Filtering (`cell_filtering`)

Tests realistic cell filtering scenarios commonly used in single-cell analysis:

- **QC-based filtering**: Filter by number of genes, mitochondrial percentage
- **Cluster-based filtering**: Filter by cluster assignments
- **Batch filtering**: Filter by batch information
- **Combined filtering**: Multiple criteria simultaneously
- **Range queries**: Filter by value ranges

**Scenarios**: 10 realistic filtering scenarios

### 2. Gene Filtering (`gene_filtering`)

Tests gene selection and filtering operations:

- **Expression-based filtering**: Filter by expression levels, cell counts
- **Gene type filtering**: Filter by gene annotations
- **Highly variable genes**: Filter by variability metrics
- **Combined filtering**: Multiple gene criteria
- **Range queries**: Filter by expression ranges

**Scenarios**: 12 realistic gene filtering scenarios

### 3. Expression Queries (`expression_queries`)

Tests expression data extraction and aggregation:

- **Single cell queries**: Extract expression for individual cells
- **Multiple cell queries**: Extract expression for cell groups
- **Single gene queries**: Extract expression for individual genes
- **Multiple gene queries**: Extract expression for gene groups
- **Submatrix queries**: Extract expression submatrices
- **Aggregation queries**: Compute means, sums across cells/genes

**Scenarios**: 13 expression query scenarios

### 4. AnnData Operations (`anndata_ops`)

Tests standard AnnData interface operations:

- **Slicing operations**: Array-style indexing and slicing
- **Boolean indexing**: Filter-based subsetting
- **Metadata access**: Access cell and gene metadata
- **Expression matrix operations**: Shape, density, non-zero counts
- **Aggregation operations**: Mean, sum across dimensions
- **Statistical operations**: Variance, standard deviation, min/max

**Scenarios**: 16 AnnData operation scenarios

### 5. Scanpy Preprocessing (`scanpy_preprocessing`)

Tests Scanpy preprocessing pipeline operations:

- **QC metrics calculation**: Compute quality control metrics
- **Cell filtering**: Filter cells based on QC metrics
- **Gene filtering**: Filter genes based on expression
- **Normalization**: Total count normalization
- **Log transformation**: Log1p transformation
- **Highly variable genes**: Identify variable genes
- **Workflow combinations**: Complete preprocessing pipelines

**Scenarios**: 12 preprocessing scenarios

## Command Line Options

### Master Runner (`run_comprehensive_benchmarks.py`)

```bash
python run_comprehensive_benchmarks.py [OPTIONS]

Options:
  --datasets TEXT...           Dataset names to benchmark (default: pbmc3k)
  --data-dir PATH             Directory containing datasets
  --types [cell_filtering|gene_filtering|expression_queries|anndata_ops|scanpy_preprocessing]...
                               Specific benchmark types to run (default: all)
  --auto-convert              Auto-convert h5ad to SLAF before benchmarking
  --verbose, -v               Enable verbose output during benchmarking
  --output PATH               Output file for results
  --no-summary                Skip printing comprehensive summary
  --help                      Show help message
```

### Individual Benchmark Modules

Each benchmark module can be run independently:

```bash
# Cell filtering only
python benchmark_cell_filtering.py

# Gene filtering only
python benchmark_gene_filtering.py

# Expression queries only
python benchmark_expression_queries.py

# AnnData operations only
python benchmark_anndata_ops.py

# Scanpy preprocessing only
python benchmark_scanpy_preprocessing.py
```

## Output Format

### Console Output

The benchmark suite provides rich console output with:

- **Progress indicators**: Real-time progress for each benchmark type
- **Rich tables**: Formatted results tables with color coding
- **Summary statistics**: Average speedups and performance insights
- **Memory analysis**: Memory usage comparisons
- **Error handling**: Graceful handling of failed scenarios

### JSON Results

Results are saved to JSON files with the following structure:

```json
{
  "dataset_name": {
    "benchmark_type": [
      {
        "scenario_type": "cell_filtering",
        "h5ad_total_time": 1234.5,
        "slaf_total_time": 567.8,
        "total_speedup": 2.17,
        "query_speedup": 3.45,
        "load_speedup": 1.23,
        "h5ad_total_memory_mb": 45.6,
        "slaf_total_memory_mb": 12.3,
        "results_match": true,
        ...
      }
    ]
  }
}
```

## Performance Metrics

Each benchmark measures:

### Timing Metrics

- **Load time**: Time to load/initialize data structures
- **Query time**: Time to execute the specific operation
- **Total time**: Combined load + query time
- **Speedup ratios**: h5ad time / SLAF time

### Memory Metrics

- **Load memory**: Memory used during data loading
- **Query memory**: Memory used during operation execution
- **Total memory**: Combined memory usage
- **Memory efficiency**: h5ad memory / SLAF memory

### Validation Metrics

- **Results matching**: Whether h5ad and SLAF produce identical results
- **Result sizes**: Size of returned data structures

## Dataset Requirements

### Required Files

- `{dataset_name}_processed.h5ad`: Processed h5ad file with metadata
- `{dataset_name}.slaf`: SLAF format file (auto-converted if needed)

### Expected Metadata

- **Cell metadata**: `n_genes_by_counts`, `total_counts`, `pct_counts_mt`, `leiden`, `batch`
- **Gene metadata**: `n_cells_by_counts`, `total_counts`, `gene_type`, `highly_variable`

## Examples

### Basic Usage

```bash
# Run all benchmarks on pbmc3k
python run_comprehensive_benchmarks.py --datasets pbmc3k --auto-convert
```

### Advanced Usage

```bash
# Run specific benchmarks with verbose output
python run_comprehensive_benchmarks.py \
  --datasets pbmc3k pbmc_68k \
  --types cell_filtering expression_queries \
  --verbose \
  --auto-convert \
  --output my_results.json
```

### Custom Dataset Directory

```bash
# Use custom dataset directory
python run_comprehensive_benchmarks.py \
  --datasets my_dataset \
  --data-dir /path/to/my/datasets \
  --auto-convert
```

## Troubleshooting

### Common Issues

1. **Missing datasets**: Ensure h5ad files exist in the data directory
2. **Conversion failures**: Check that SLAF converter is properly installed
3. **Memory errors**: Reduce dataset size or run individual benchmark types
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

```bash
# Run with verbose output for debugging
python run_comprehensive_benchmarks.py --datasets pbmc3k --verbose --auto-convert
```

## Extending the Benchmarks

### Adding New Benchmark Types

1. Create a new benchmark module (e.g., `benchmark_new_type.py`)
2. Implement the required functions:
   - `demo_realistic_scenarios()`: Define test scenarios
   - `benchmark_new_type()`: Main benchmark function
   - Measurement functions for h5ad and SLAF
3. Add to the master runner in `run_comprehensive_benchmarks.py`

### Adding New Scenarios

1. Add scenario definitions to the appropriate `demo_realistic_*()` function
2. Implement measurement logic in the benchmark functions
3. Ensure proper error handling and result validation

## Performance Interpretation

### Speedup Guidelines

- **>2x**: Excellent performance improvement
- **1.5-2x**: Good performance improvement
- **1-1.5x**: Modest performance improvement
- **<1x**: Needs investigation

### Memory Efficiency

- **>1x**: SLAF uses less memory than h5ad
- **<1x**: SLAF uses more memory than h5ad
- **∞x**: SLAF uses negligible memory compared to h5ad

## Contributing

When adding new benchmarks:

1. Follow the existing code structure and patterns
2. Use shared utilities from `benchmark_utils.py`
3. Include proper error handling and validation
4. Add documentation for new scenarios
5. Test with multiple datasets

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scanpy`: Single-cell analysis
- `rich`: Console formatting
- `slaf`: SLAF library
- `scipy`: Sparse matrix operations
