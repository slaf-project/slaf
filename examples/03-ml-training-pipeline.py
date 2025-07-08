import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import numpy as np

    from slaf import SLAFArray
    from slaf.ml.dataloaders import SLAFDataLoader
    from slaf.ml.tokenizers import SLAFTokenizer

    return SLAFArray, SLAFDataLoader, SLAFTokenizer, mo, np, time


@app.cell
def _(mo):
    mo.md(
        """
    # SLAF ML Training Pipeline

    This notebook demonstrates how to build complete ML training pipelines with SLAF, including:

    - Custom tokenizer configuration

    - Different tokenization strategies (scGPT, Geneformer)

    - DataLoader integration with PyTorch

    - Performance optimization techniques

    - Custom training loop examples

    **Key Benefits for ML Training:**

    🚀 **Fast Tokenization**: SQL-level performance for token generation

    💾 **Memory Efficient**: Stream data without loading everything into memory

    🔄 **Flexible**: Support for different tokenization strategies and model architectures

    🧬 **Production Ready**: Built for large-scale training with proper splits and error handling

    ⚡ **Optimized**: SQL binning, percentile filtering, and other performance optimizations
    """
    )
    return


@app.cell
def _(SLAFArray):
    # Load SLAF dataset for ML examples
    slaf = SLAFArray("../slaf-datasets/pbmc3k_processed.slaf")
    print(f"✅ Loaded SLAF dataset: {slaf.shape[0]:,} cells × {slaf.shape[1]:,} genes")

    # Show dataset info
    slaf.info()
    return (slaf,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 1. Understanding SLAF Tokenization

    SLAF provides efficient tokenization for single-cell data, supporting multiple strategies:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Tokenizer Configuration""")
    return


@app.cell
def _(SLAFTokenizer, slaf):
    # Create tokenizer with custom settings
    tokenizer = SLAFTokenizer(
        slaf_array=slaf,
        vocab_size=5000,  # Smaller vocab for demo
        n_expression_bins=20,  # More expression bins
        chunk_size=512,  # Smaller chunks for memory efficiency
    )

    # Get vocabulary information
    vocab_info = tokenizer.get_vocab_info()
    print("✅ Tokenizer initialized:")
    print(f"   Total vocabulary size: {vocab_info['total_vocab_size']:,}")
    print(f"   Special tokens: {vocab_info['special_tokens']}")
    print(f"   Expression bins: {vocab_info['n_expression_bins']}")
    print(f"   Gene vocabulary size: {vocab_info['vocab_size']:,}")

    # Show special tokens
    print("\nSpecial tokens:")
    for token_name, token_id in tokenizer.special_tokens.items():
        print(f"   {token_name}: {token_id}")

    return (tokenizer,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 2. Tokenization Strategies

    SLAF supports different tokenization strategies for different model architectures:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Geneformer and scGPT style tokenization""")
    return


@app.cell
def _(tokenizer):
    # Tokenize a small batch of cells for demonstration

    # 1. Geneformer tokenization
    print("\n1. Geneformer Tokenization:")
    print("   Format: [gene1, gene2, gene3, ...] (sorted by expression)")

    geneformer_tokens = tokenizer.tokenize_geneformer(
        cell_integer_id_range=(0, 10),  # Limit for demo
        max_genes=50,  # Limit for demo
        min_percentile=10,  # Filter low-expression genes
    )

    print(
        f"   Generated {len(geneformer_tokens)} sequences of length {len(geneformer_tokens[0])}"
    )

    # 2. scGPT tokenization
    print("\n2. scGPT Tokenization:")
    print("   Format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]")

    scgpt_tokens = tokenizer.tokenize_scgpt(
        cell_integer_id_range=(0, 10),  # Limit for demo (scGPT sequences are longer)
        max_genes=25,  # Limit for demo (scGPT sequences are longer)
        use_sql_binning=True,  # Use SQL for better performance
    )

    print(
        f"   Generated {len(scgpt_tokens)} sequences of length {len(scgpt_tokens[0])}"
    )

    return geneformer_tokens, scgpt_tokens


@app.cell
def _(mo):
    mo.md(r"""### Token decoding examples""")
    return


@app.cell
def _(geneformer_tokens, scgpt_tokens, tokenizer):
    # Demonstrate token decoding
    if geneformer_tokens:
        print("1. Geneformer sequence decoding:")
        decoded_geneformer = tokenizer.decode_tokens(geneformer_tokens[0])
        print(f"   Sequence length: {len(geneformer_tokens[0])}")
        print(f"   Genes: {len(decoded_geneformer['genes'])} genes")
        if decoded_geneformer["genes"]:
            print(f"   First few genes: {decoded_geneformer['genes'][:3]}")

    if scgpt_tokens:
        print("\n2. scGPT sequence decoding:")
        decoded_scgpt = tokenizer.decode_tokens(scgpt_tokens[0])
        print(f"   Sequence length: {len(scgpt_tokens[0])}")
        print(f"   Special tokens: {decoded_scgpt['special_tokens']}")
        print(f"   Genes: {len(decoded_scgpt['genes'])} genes")
        print(f"   Expression bins: {len(decoded_scgpt['expression_bins'])} bins")
        if decoded_scgpt["genes"]:
            print(f"   First few genes: {decoded_scgpt['genes'][:3]}")
            print(
                f"   First few expression bins: {decoded_scgpt['expression_bins'][:3]}"
            )

    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Performance Comparison - Different Tokenization Approaches

    Let's compare the performance of different tokenization strategies:
    """
    )
    return


@app.cell
def _(time):
    class Timer:
        def __init__(self, name=None):
            self.name = name
            self.elapsed = 0.0
            self._start_time = None

        def __enter__(self):
            self._start_time = (
                time.perf_counter()
            )  # Use perf_counter for more accurate timing
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._start_time is not None:
                end_time = time.perf_counter()
                self.elapsed = end_time - self._start_time

    # Usage
    with Timer("My Task") as t:
        # Code to be timed
        time.sleep(2)  # Simulate some work
        print("Task completed")

    print(f"Elapsed time captured from context manager object: {t.elapsed:.4f} seconds")

    return (Timer,)


@app.cell
def _(Timer, tokenizer):
    # Performance comparison
    print("⚡ Tokenization Performance Comparison")
    print("=" * 45)

    def demo_tokenizer_performance():
        # Test Geneformer with 100 cells
        print("\n1. Geneformer Performance:")

        # Standard Geneformer
        with Timer("geneformer_standard") as standard_time:
            geneformer_standard = tokenizer.tokenize_geneformer(
                (0, 100), max_genes=100, min_percentile=None
            )

        print(f"    Geneformer [standard]: {standard_time.elapsed:.4f}s")
        # Geneformer with percentile filtering
        with Timer("geneformer_percentile") as percentile_time:
            _ = tokenizer.tokenize_geneformer(
                (0, 100), max_genes=100, min_percentile=10
            )
        print(f"   Geneformer [percentile]: {percentile_time.elapsed:.4f}s")
        print(f"   Speedup: {standard_time.elapsed / percentile_time.elapsed:.2f}x")

        # Test scGPT with different settings
        print("\n2. scGPT Performance:")

        # scGPT with Python binning
        with Timer("scgpt_python") as python_time:
            _ = tokenizer.tokenize_scgpt((0, 100), max_genes=50, use_sql_binning=False)
        print(f"   scGPT [Python]: {python_time.elapsed:.4f}s")

        # scGPT with SQL binning
        with Timer("scgpt_sql") as sql_time:
            scgpt_sql = tokenizer.tokenize_scgpt(
                (0, 100), max_genes=50, use_sql_binning=True
            )
        print(f"   scGPT [SQL]: {sql_time.elapsed:.4f}s")
        print(f"   Speedup: {python_time.elapsed / sql_time.elapsed:.2f}x")

        # Calculate tokens per second
        total_tokens_geneformer = sum(len(seq) for seq in geneformer_standard)
        total_tokens_scgpt = sum(len(seq) for seq in scgpt_sql)

        print("\n3. Throughput:")
        print(
            f"   Geneformer [standard]: {total_tokens_geneformer / standard_time.elapsed:,.0f} tokens/sec"
        )
        print(
            f"   Geneformer [percentile]: {total_tokens_geneformer / percentile_time.elapsed:,.0f} tokens/sec"
        )
        print(
            f"   scGPT [Python]: {total_tokens_scgpt / python_time.elapsed:,.0f} tokens/sec"
        )
        print(
            f"   scGPT [SQL]: {total_tokens_scgpt / sql_time.elapsed:,.0f} tokens/sec"
        )

    demo_tokenizer_performance()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 4. SLAF DataLoader - Production-Ready Training

    SLAF provides a high-performance DataLoader for training:
    """
    )
    return


@app.cell
def _(SLAFDataLoader, slaf):
    # Initialize DataLoader
    print("📦 SLAF DataLoader Configuration")
    print("=" * 40)

    # Create DataLoader with custom settings
    dataloader = SLAFDataLoader(
        slaf_array=slaf,
        tokenizer_type="geneformer",  # or "scgpt"
        batch_size=16,  # Small batch for demo
        max_genes=100,
        num_workers=2,  # Number of worker processes
        vocab_size=5000,
        n_expression_bins=20,
        chunk_size=512,
    )

    print("✅ DataLoader initialized:")
    print(f"   Tokenizer type: {dataloader.tokenizer_type}")
    print(f"   Batch size: {dataloader.batch_size}")
    print(f"   Max genes: {dataloader.max_genes}")
    print(f"   Number of batches: {len(dataloader)}")
    print(f"   Special tokens: {dataloader.special_tokens}")

    return (dataloader,)


@app.cell
def _(dataloader):
    # Demonstrate DataLoader iteration
    print("🔄 DataLoader Iteration")
    print("=" * 25)

    def demo_dataloader_iteration(dataloader):
        # Get first batch
        print("1. First batch structure:")
        batch = next(iter(dataloader))

        for key, value in batch.items():
            if hasattr(value, "shape"):
                print(f"   {key}: {type(value)} with shape {value.shape}")
            else:
                print(f"   {key}: {type(value)} with length {len(value)}")

        # Show batch details
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        cell_ids = batch["cell_ids"]

        print("\n2. Batch details:")
        print(f"   Input IDs shape: {input_ids.shape}")
        print(f"   Attention mask shape: {attention_mask.shape}")
        print(f"   Cell IDs shape: {cell_ids.shape}")
        print(f"   Data type: {input_ids.dtype}")

        # Show sample tokens
        print("\n3. Sample tokens from first sequence:")
        first_seq = input_ids[0]
        print(f"   First 10 tokens: {first_seq[:10].tolist()}")
        print(f"   Sequence length: {len(first_seq)}")

    demo_dataloader_iteration(dataloader)
    return


@app.cell
def _(Timer, dataloader):
    # Performance testing of DataLoader
    print("⚡ DataLoader Performance")
    print("=" * 30)

    # Test iteration speed
    print("1. Iteration performance:")

    def demo_dataloader_iteration_performance():
        batch_count = 0
        total_tokens = 0

        for batch in dataloader:
            batch_count += 1
            total_tokens += batch["input_ids"].shape[0] * batch["input_ids"].shape[1]

            # Only process first few batches for demo
            if batch_count >= 5:
                break
        return batch_count, total_tokens

    with Timer("dataloader_iteration") as iteration_time:
        batch_count, total_tokens = demo_dataloader_iteration_performance()

    print(f"   Processed {batch_count} batches in {iteration_time.elapsed:.4f}s")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Tokens per second: {total_tokens / iteration_time.elapsed:,.0f}")
    print(f"   Batches per second: {batch_count / iteration_time.elapsed:.2f}")

    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 5. PyTorch Training Loop Integration

    Here's how to integrate SLAF DataLoader with PyTorch training:
    """
    )
    return


@app.cell
def _(dataloader):
    # Demonstrate PyTorch integration
    print("🔥 PyTorch Training Loop Integration")
    print("=" * 45)

    def demo_training_loop(dataloader):
        # Check if PyTorch is available
        try:
            import torch

            TORCH_AVAILABLE = True
            print("✅ PyTorch is available")
        except ImportError:
            TORCH_AVAILABLE = False
            print("⚠️ PyTorch not available - showing numpy-based approach")

        if TORCH_AVAILABLE:
            print("\n1. PyTorch tensor conversion:")
            batch = next(iter(dataloader))

            # Get device info
            from slaf.ml.dataloaders import get_device_info, get_optimal_device

            device_info = get_device_info()
            optimal_device = get_optimal_device()

            print(f"   Device info: {device_info}")
            print(f"   Using device: {optimal_device}")

            # Convert to PyTorch tensors on optimal device
            input_ids_tensor = torch.tensor(
                batch["input_ids"], dtype=torch.long, device=optimal_device
            )
            attention_mask_tensor = torch.tensor(
                batch["attention_mask"], dtype=torch.bool, device=optimal_device
            )
            cell_ids_tensor = torch.tensor(
                batch["cell_ids"], dtype=torch.long, device=optimal_device
            )

            print(
                f"   Input IDs tensor: {input_ids_tensor.shape}, {input_ids_tensor.dtype}"
            )
            print(
                f"   Attention mask tensor: {attention_mask_tensor.shape}, {attention_mask_tensor.dtype}"
            )
            print(
                f"   Cell IDs tensor: {cell_ids_tensor.shape}, {cell_ids_tensor.dtype}"
            )

            print("\n2. Simple training loop structure:")
            print(
                """
            # Training loop example with smart device detection
            from slaf.ml.dataloaders import get_optimal_device

            device = get_optimal_device()
            model = YourModel(vocab_size=tokenizer.get_vocab_info()['total_vocab_size'])
            model = model.to(device)  # Move model to optimal device
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            for epoch in range(num_epochs):
                model.train()
                for batch in dataloader:
                    # DataLoader already provides tensors on optimal device
                    input_ids = batch["input_ids"]  # Already on device
                    attention_mask = batch["attention_mask"]  # Already on device

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = outputs.loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            """
            )

        else:
            print("\n1. Numpy-based approach:")
            batch = next(iter(dataloader))
            print(
                f"   Input IDs: {batch['input_ids'].shape}, {batch['input_ids'].dtype}"
            )
            print(
                f"   Attention mask: {batch['attention_mask'].shape}, {batch['attention_mask'].dtype}"
            )
            print(f"   Cell IDs: {batch['cell_ids'].shape}, {batch['cell_ids'].dtype}")

    demo_training_loop(dataloader)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 6. Custom DataLoader Implementation

    Learn how to create custom DataLoaders for specific use cases:
    """
    )
    return


@app.cell
def _(SLAFTokenizer, np, slaf):
    # Custom DataLoader implementation
    print("🔧 Custom DataLoader Implementation")
    print("=" * 40)

    class CustomSLAFDataLoader:
        """Custom DataLoader with specific functionality"""

        def __init__(
            self,
            slaf_array,
            batch_size=32,
            max_genes=100,
            tokenizer_type="geneformer",
            **tokenizer_kwargs,
        ):
            self.slaf_array = slaf_array
            self.batch_size = batch_size
            self.max_genes = max_genes
            self.tokenizer_type = tokenizer_type

            # Initialize tokenizer
            self.tokenizer = SLAFTokenizer(slaf_array, **tokenizer_kwargs)

            # Get cell ranges for batching
            self.cell_ranges = self._get_cell_ranges()

        def _get_cell_ranges(self):
            """Get cell integer ID ranges for batching"""
            max_cell_id = int(self.slaf_array.obs["cell_integer_id"].astype(int).max())
            ranges = []
            for start in range(0, max_cell_id + 1, self.batch_size):
                end = min(start + self.batch_size, max_cell_id + 1)
                ranges.append((start, end))
            return ranges

        def __iter__(self):
            """Iterate through batches"""
            for cell_range in self.cell_ranges:
                # Tokenize based on type
                if self.tokenizer_type == "geneformer":
                    tokens = self.tokenizer.tokenize_geneformer(
                        cell_integer_id_range=cell_range, max_genes=self.max_genes
                    )
                elif self.tokenizer_type == "scgpt":
                    tokens = self.tokenizer.tokenize_scgpt(
                        cell_integer_id_range=cell_range, max_genes=self.max_genes
                    )
                else:
                    raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

                if not tokens:
                    continue

                # Convert to numpy arrays
                batch_tensors = np.array(tokens, dtype=np.int64)
                attention_mask = batch_tensors != self.tokenizer.special_tokens["PAD"]

                # Get cell IDs for this range
                start_cell, end_cell = cell_range
                cell_ids = list(range(start_cell, end_cell))

                yield {
                    "input_ids": batch_tensors,
                    "attention_mask": attention_mask,
                    "cell_ids": np.array(cell_ids[: len(tokens)], dtype=np.int64),
                }

        def __len__(self):
            return len(self.cell_ranges)

    # Test custom DataLoader
    print("✅ Custom DataLoader created")

    custom_dataloader = CustomSLAFDataLoader(
        slaf_array=slaf,
        batch_size=8,
        max_genes=50,
        tokenizer_type="geneformer",
        vocab_size=1000,
    )

    print(f"   Number of batches: {len(custom_dataloader)}")

    # Test iteration
    batch = next(iter(custom_dataloader))
    print(f"   First batch shape: {batch['input_ids'].shape}")

    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 7. Advanced Tokenization Features

    Explore advanced tokenization features and optimizations:
    """
    )
    return


@app.cell
def _(np, time, tokenizer):
    # Advanced tokenization features
    print("🚀 Advanced Tokenization Features")
    print("=" * 40)

    def demo_advanced_tokenization():
        print("1. Different max_genes settings:")
        for max_genes in [25, 50, 100]:
            tokens = tokenizer.tokenize_geneformer((0, 20), max_genes=max_genes)
            avg_length = np.mean([len(seq) for seq in tokens])
            print(f"   max_genes={max_genes}: avg_length={avg_length:.1f}")

        print("\n2. Different percentile filtering:")
        for percentile in [None, 5, 10, 20]:
            tokens = tokenizer.tokenize_geneformer(
                (0, 20), max_genes=50, min_percentile=percentile
            )
            avg_length = np.mean([len(seq) for seq in tokens])
            print(f"   min_percentile={percentile}: avg_length={avg_length:.1f}")

        print("\n3. SQL vs Python binning for scGPT:")
        # SQL binning
        start_time = time.time()
        _ = tokenizer.tokenize_scgpt((0, 20), max_genes=25, use_sql_binning=True)
        sql_time = time.time() - start_time

        # Python binning
        start_time = time.time()
        _ = tokenizer.tokenize_scgpt((0, 20), max_genes=25, use_sql_binning=False)
        python_time = time.time() - start_time

        print(f"   SQL binning: {sql_time:.4f}s")
        print(f"   Python binning: {python_time:.4f}s")
        print(f"   Speedup: {python_time / sql_time:.2f}x")

    demo_advanced_tokenization()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 8. Memory and Performance Optimization

    Learn how to optimize memory usage and performance for large-scale training:
    """
    )
    return


@app.cell
def _(SLAFDataLoader, slaf, time):
    # Memory and performance optimization
    print("💾 Memory and Performance Optimization")
    print("=" * 45)

    import gc

    import psutil

    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def demo_memory_and_performance_optimization():
        print("1. Memory usage comparison:")

        # Baseline memory
        gc.collect()
        baseline_memory = get_memory_usage()
        print(f"   Baseline memory: {baseline_memory:.1f} MB")

        # Memory with different batch sizes
        for batch_size in [8, 16, 32]:
            gc.collect()
            start_memory = get_memory_usage()

            dataloader = SLAFDataLoader(
                slaf_array=slaf, batch_size=batch_size, max_genes=100, vocab_size=1000
            )

            # Process one batch
            _ = next(iter(dataloader))
            end_memory = get_memory_usage()

            print(
                f"   Batch size {batch_size}: {end_memory:.1f} MB (+{end_memory - start_memory:.1f} MB)"
            )

        print("\n2. Performance with different settings:")

        # Test different configurations
        configs = [
            {"batch_size": 8, "max_genes": 50, "description": "Small batches"},
            {"batch_size": 16, "max_genes": 100, "description": "Medium batches"},
            {"batch_size": 32, "max_genes": 200, "description": "Large batches"},
        ]

        for config in configs:
            dataloader = SLAFDataLoader(
                slaf_array=slaf,
                **{k: v for k, v in config.items() if k != "description"},
            )

            start_time = time.time()
            batch_count = 0
            for _ in dataloader:
                batch_count += 1
                if batch_count >= 3:  # Test first 3 batches
                    break

            elapsed_time = time.time() - start_time
            print(
                f"   {config['description']}: {elapsed_time:.4f}s for {batch_count} batches"
            )

    demo_memory_and_performance_optimization()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 9. Production Training Workflow

    Complete example of a production-ready training workflow:
    """
    )
    return


@app.cell
def _(SLAFDataLoader, SLAFTokenizer, slaf):
    # Production training workflow
    print("🏭 Production Training Workflow")
    print("=" * 40)

    def create_production_dataloaders(
        slaf_array, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    ):
        """Create train/val/test dataloaders"""

        # Get total number of cells
        total_cells = len(slaf_array.obs)
        _ = int(total_cells * train_ratio)  # train_size
        _ = int(total_cells * val_ratio)  # val_size

        # Create tokenizer
        tokenizer = SLAFTokenizer(
            slaf_array=slaf_array,
            vocab_size=10000,
            n_expression_bins=50,
            chunk_size=1024,
        )

        print(
            f"✅ Created tokenizer with {tokenizer.get_vocab_info()['total_vocab_size']} total tokens"
        )

        # Create dataloaders (in production, you'd use proper splits)
        train_dataloader = SLAFDataLoader(
            slaf_array=slaf_array,
            tokenizer_type="geneformer",
            batch_size=32,
            max_genes=512,
            vocab_size=10000,
            n_expression_bins=50,
        )

        val_dataloader = SLAFDataLoader(
            slaf_array=slaf_array,
            tokenizer_type="geneformer",
            batch_size=32,
            max_genes=512,
            vocab_size=10000,
            n_expression_bins=50,
        )

        return train_dataloader, val_dataloader, tokenizer

    # Create production dataloaders
    train_dl, val_dl, prod_tokenizer = create_production_dataloaders(slaf)

    print("\n📊 Production Setup:")
    print(f"   Train batches: {len(train_dl)}")
    print(f"   Validation batches: {len(val_dl)}")
    print(f"   Batch size: {train_dl.batch_size}")
    print(f"   Max genes: {train_dl.max_genes}")

    # Test production workflow
    print("\n🧪 Testing production workflow:")

    # Test training batch
    train_batch = next(iter(train_dl))
    print(f"   Training batch shape: {train_batch['input_ids'].shape}")

    # Test validation batch
    val_batch = next(iter(val_dl))
    print(f"   Validation batch shape: {val_batch['input_ids'].shape}")

    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 10. Best Practices for ML Training

    Key best practices for using SLAF in ML training:
    """
    )
    return


@app.cell
def _():
    # Best practices for ML training
    print("💡 Best Practices for ML Training")
    print("=" * 40)

    print("1. Tokenizer Configuration:")
    print("   ✅ Choose appropriate vocab_size based on your dataset")
    print("   ✅ Use n_expression_bins=50 for fine-grained expression modeling")
    print("   ✅ Set chunk_size based on available memory")

    print("\n2. DataLoader Configuration:")
    print("   ✅ Start with small batch_size and increase gradually")
    print("   ✅ Use max_genes appropriate for your model architecture")
    print("   ✅ Set num_workers based on CPU cores")

    print("\n3. Performance Optimization:")
    print("   ✅ Use SQL binning for scGPT tokenization")
    print("   ✅ Leverage percentile filtering for Geneformer")
    print("   ✅ Monitor memory usage during training")

    print("\n4. Training Workflow:")
    print("   ✅ Create separate train/val/test splits")
    print("   ✅ Use consistent tokenizer across splits")
    print("   ✅ Implement proper error handling")

    print("\n5. Production Considerations:")
    print("   ✅ Use appropriate batch sizes for your hardware")
    print("   ✅ Implement checkpointing for long training runs")
    print("   ✅ Monitor tokenization throughput")
    print("   ✅ Consider distributed training for large datasets")

    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary

    **What you've learned about SLAF ML Training:**

    1. **Tokenizer Configuration**: Customize vocabulary size, expression bins, and chunking
    2. **Tokenization Strategies**: Geneformer and scGPT formats with different performance characteristics
    3. **DataLoader Integration**: High-performance data loading with PyTorch compatibility
    4. **Performance Optimization**: SQL-level performance with memory efficiency
    5. **Custom Implementation**: How to build custom DataLoaders for specific needs
    6. **Production Workflow**: Complete training pipeline setup
    7. **Best Practices**: Guidelines for optimal ML training performance
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
