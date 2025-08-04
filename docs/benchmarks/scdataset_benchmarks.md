# scDataset Benchmarks

This document reports comprehensive benchmark results for scDataset, a high-performance dataloader for single-cell data. These benchmarks were conducted using the Tahoe-100M dataset to evaluate scDataset's performance characteristics and scaling behavior.

## **Dataset and Hardware**

### **Dataset: Tahoe-100M**

- **Cells**: 5,481,420 cells
- **Genes**: 62,710 genes
- **Format**: h5ad (backed mode)
- **Batch Size**: 64 cells per batch

### **Hardware Configuration**

- **Machine**: Apple MacBook Pro with M1 Max
- **Memory**: 32 GB RAM
- **Storage**: 1 TB NVMe SSD
- **OS**: macOS 13.6.1

## **Parameter Scaling Benchmarks**

### **Methodology**

We tested scDataset performance across different combinations of `block_size` and `fetch_factor` parameters, which are key to scDataset's performance according to their paper. All tests used:

- **Single worker** (`num_workers=0`) to isolate parameter effects
- **10-second measurement duration**
- **Module-level callback** to avoid pickling issues
- **Backed AnnData** to match real-world usage

### **Results Summary**

| Parameter Range                          | Best Performance     | Best Configuration              | Scaling Factor |
| ---------------------------------------- | -------------------- | ------------------------------- | -------------- |
| `block_size`: [1, 2, 4, 8, 16, 32, 64]   | **10,976 cells/sec** | `block_size=8, fetch_factor=64` | **23.1x**      |
| `fetch_factor`: [1, 2, 4, 8, 16, 32, 64] |                      |                                 |                |

### **Key Findings**

#### **1. Strong Scaling with fetch_factor**

- **Minimal fetch_factor (1)**: ~500 cells/sec
- **Optimal fetch_factor (64)**: ~10,000+ cells/sec
- **Scaling factor**: 20x+ improvement with higher fetch_factor

#### **2. Moderate Scaling with block_size**

- **Small block_size (1-4)**: Good performance with high fetch_factor
- **Medium block_size (8-16)**: Optimal performance
- **Large block_size (32-64)**: Slightly reduced performance

### **Detailed Results Table**

| Block Size | Fetch Factor | Throughput (cells/sec) |
| ---------- | ------------ | ---------------------- |
| 1          | 1            | 474                    |
| 1          | 64           | 10,093                 |
| 4          | 16           | 5,103                  |
| 8          | 64           | **10,976**             |
| 16         | 64           | 10,484                 |
| 32         | 64           | 9,382                  |
| 64         | 64           | 10,159                 |

!!! success "Parameter Optimization"

    The optimal configuration for scDataset on our hardware is `block_size=8, fetch_factor=64`, achieving **10,976 cells/sec**.

## **Multiprocessing Benchmarks**

### **Methodology**

We tested scDataset's multiprocessing capabilities using the optimal parameters (`block_size=4, fetch_factor=16`) and varying `num_workers` values.

### **Results**

| Num Workers | Throughput (cells/sec) | Status            |
| ----------- | ---------------------- | ----------------- |
| 0           | 5,363                  | ✅ Success        |
| 1           | N/A                    | ❌ Pickling Error |
| 2           | N/A                    | ❌ Pickling Error |
| 4           | N/A                    | ❌ Pickling Error |
| 8           | N/A                    | ❌ Pickling Error |

### **Key Findings**

#### **1. Multiprocessing Limitations**

- **Single worker only**: scDataset works reliably with `num_workers=0`
- **Pickling errors**: All multiprocessing attempts failed with "h5py objects cannot be pickled"
- **Callback issues**: Module-level callbacks didn't resolve the pickling problem

#### **2. Performance Comparison**

- **Single worker**: 5,363 cells/sec with optimal parameters
- **No multiprocessing scaling**: Unable to test due to pickling limitations

!!! warning "Multiprocessing Limitation"

    scDataset's multiprocessing capabilities are limited by pickling issues with h5py-backed AnnData objects. This prevents the scaling benefits reported in their paper.

## **Comparison with Paper Results**

### **Reported vs Observed Performance**

| Metric                | Paper Claim      | Our Results      | Difference      |
| --------------------- | ---------------- | ---------------- | --------------- |
| **Best throughput**   | ~2,000 cells/sec | 10,976 cells/sec | **5.5x higher** |
| **Parameter scaling** | Significant      | 23.1x scaling    | **Matches**     |
| **Multiprocessing**   | 12 workers       | 0 workers only   | **Limited**     |

### **Possible Explanations**

1. **Hardware differences**: Our M1 Max may be faster than the paper's hardware
2. **Dataset differences**: Different datasets may have different characteristics
3. **Implementation differences**: Different scDataset versions or configurations
4. **Measurement methodology**: Different benchmark setups

!!! info "Performance Validation"

    Our results show that scDataset can achieve excellent performance with proper parameter tuning, even exceeding the paper's reported numbers on modern hardware.

## **Technical Challenges**

### **1. Pickling Issues**

- **Problem**: h5py objects cannot be pickled for multiprocessing
- **Impact**: Prevents multiprocessing scaling
- **Workaround**: Use single worker with optimized parameters

### **2. Parameter Sensitivity**

- **Problem**: Performance varies dramatically with parameters
- **Impact**: Requires careful tuning for optimal performance
- **Solution**: Systematic parameter sweeps

## **Recommendations**

### **For scDataset Users**

1. **Parameter Tuning**: Always test different `block_size` and `fetch_factor` combinations
2. **Single Worker**: Use `num_workers=0` to avoid pickling issues
3. **Hardware Testing**: Test on your specific hardware for optimal parameters

### **For Developers**

1. **Pickling Fix**: Address h5py pickling issues for multiprocessing support
2. **Parameter Documentation**: Provide clearer guidance on parameter selection
3. **Benchmark Suite**: Include comprehensive benchmark tools

## **Conclusion**

scDataset demonstrates excellent performance potential with proper parameter tuning, achieving **10,976 cells/sec** in our benchmarks. However, multiprocessing limitations prevent the scaling benefits reported in their paper. The strong parameter scaling validates their design approach, but the pickling issues need to be addressed for broader adoption.

The benchmark results show that scDataset can be a viable high-performance dataloader for single-cell data, but requires careful configuration and has limitations for multiprocessing scenarios.

---

_These benchmarks were conducted using scDataset with backed AnnData objects on the Tahoe-100M dataset. Results may vary with different datasets, hardware configurations, or scDataset versions._
