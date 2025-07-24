import marimo

__generated_with = "0.14.0"
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

    - Streaming DataLoader with async prefetching
    - PyTorch-compatible datasets
    - Performance optimization techniques
    - Custom training loop examples

    **Key Benefits for ML Training:**

    💾 **Memory Efficient**: Stream data in chunks without loading everything into memory

    🔄 **Flexible**: Support for different tokenization strategies (scGPT, Geneformer)

    🧬 **High Throughput**: Load and tokenize cells at 10k cells / sec: fast enough to never let a 8 x H100 GPU node stay idle
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
    ## 2. Tokenization Strategies

    SLAF supports different tokenization strategies for different model architectures.
    Each strategy has its own format and vocabulary structure:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### GeneFormer vs scGPT Tokenization

    **GeneFormer**: Simple gene sequences sorted by expression
    - Format: `[CLS, gene1, gene2, gene3, ..., SEP]`
    - Vocabulary: Gene tokens only
    - Use case: Models that only need gene identity

    **scGPT**: Gene-expression pairs with special tokens
    - Format: `[CLS, gene1, expr1, gene2, expr2, ..., SEP]`
    - Vocabulary: Gene tokens + expression bin tokens
    - Use case: Models that need both gene identity and expression levels
    """
    )
    return


@app.cell
def _(SLAFTokenizer, slaf):
    def create_tokenizer():
        # Create tokenizer with custom settings
        tokenizer = SLAFTokenizer(
            slaf_array=slaf,
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=20,  # More expression bins
        )

        # Get vocabulary information
        vocab_info = tokenizer.get_vocab_info()
        print("✅ Tokenizer initialized:")
        print(f"   Total vocabulary size: {vocab_info['vocab_size']:,}")
        print(f"   Special tokens: {vocab_info['special_tokens']}")
        print(f"   Expression bins: {vocab_info['n_expression_bins']}")
        print(f"   Gene vocabulary size: {vocab_info['gene_vocab_size']:,}")

        # Show special tokens
        print("\nSpecial tokens:")
        for token_name, token_id in tokenizer.special_tokens.items():
            print(f"   {token_name}: {token_id}")

        return tokenizer

    tokenizer = create_tokenizer()
    return (tokenizer,)


@app.cell
def _(SLAFTokenizer, slaf):
    def demonstrate_geneformer():
        print("🧬 GeneFormer Tokenization & Decoding")
        print("=" * 40)

        # Create GeneFormer tokenizer
        tokenizer = SLAFTokenizer(
            slaf_array=slaf,
            tokenizer_type="geneformer",
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=20,
        )

        # Show vocabulary info
        vocab_info = tokenizer.get_vocab_info()
        print(
            f"✅ Vocabulary: {vocab_info['vocab_size']} total tokens, {vocab_info['gene_vocab_size']} genes"
        )
        print(f"   Special tokens: {vocab_info['special_tokens']}")

        # Create sample gene sequences
        gene_sequences = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [0, 2, 4, 6, 8]]

        # Tokenize
        input_ids, attention_mask = tokenizer.tokenize(
            gene_sequences=gene_sequences,
            max_genes=50,
        )

        print("\n📊 Tokenization Results:")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Attention mask shape: {attention_mask.shape}")
        print(f"   First sequence tokens: {input_ids[0].tolist()[:10]}...")
        print(f"   First sequence attention: {attention_mask[0].tolist()[:10]}...")

        # Decode first sequence
        print("\n🔍 Decoding Results:")
        decoded = tokenizer.decode_tokens(input_ids[0].tolist())
        print(f"   Sequence length: {len(input_ids[0])}")
        print(f"   Genes: {len(decoded['genes'])} genes")
        print(f"   First few genes: {decoded['genes'][:5]}")

        return tokenizer, input_ids, attention_mask

    geneformer_tokenizer, geneformer_input_ids, geneformer_attention_mask = (
        demonstrate_geneformer()
    )
    return geneformer_tokenizer, geneformer_input_ids, geneformer_attention_mask


@app.cell
def _(SLAFTokenizer, slaf):
    def demonstrate_scgpt():
        print("🧬 scGPT Tokenization & Decoding")
        print("=" * 40)

        # Create scGPT tokenizer
        tokenizer = SLAFTokenizer(
            slaf_array=slaf,
            tokenizer_type="scgpt",
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=20,
        )

        # Show vocabulary info
        vocab_info = tokenizer.get_vocab_info()
        print(
            f"✅ Vocabulary: {vocab_info['vocab_size']} total tokens, {vocab_info['gene_vocab_size']} genes"
        )
        print(f"   Special tokens: {vocab_info['special_tokens']}")
        print(
            f"   Expression bins: {vocab_info['n_expression_bins']} bins (start at token {tokenizer.expr_bin_start})"
        )

        # Create sample gene and expression sequences
        gene_sequences = [[0, 1, 2], [1, 2, 3]]
        expr_sequences = [[0.5, 0.8, 0.2], [0.9, 0.1, 0.7]]

        # Tokenize
        input_ids, attention_mask = tokenizer.tokenize(
            gene_sequences=gene_sequences,
            expr_sequences=expr_sequences,
            max_genes=25,
        )

        print("\n📊 Tokenization Results:")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Attention mask shape: {attention_mask.shape}")
        print("   Expected length: 1 + 2*25 + 1 = 52 (CLS + 25*(gene+expr) + SEP)")
        print(f"   First sequence tokens: {input_ids[0].tolist()[:10]}...")
        print(f"   First sequence attention: {attention_mask[0].tolist()[:10]}...")

        # Decode first sequence
        print("\n🔍 Decoding Results:")
        decoded = tokenizer.decode_tokens(input_ids[0].tolist())
        print(f"   Sequence length: {len(input_ids[0])}")
        print(f"   Genes: {len(decoded['genes'])} genes")
        print(f"   Expressions: {len(decoded['expressions'])} expressions")
        print(f"   First few genes: {decoded['genes'][:3]}")
        print(f"   First few expressions: {decoded['expressions'][:3]}")

        return tokenizer, input_ids, attention_mask

    scgpt_tokenizer, scgpt_input_ids, scgpt_attention_mask = demonstrate_scgpt()
    return scgpt_tokenizer, scgpt_input_ids, scgpt_attention_mask


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. SLAF DataLoader - Production-Ready Training

    SLAF provides a high-performance DataLoader with streaming and async prefetching:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### DataLoader Configuration""")
    return


@app.cell
def _(SLAFDataLoader, slaf):
    def create_dataloader():
        # Initialize DataLoader
        print("📦 SLAF DataLoader Configuration")
        print("=" * 40)

        # Create DataLoader with custom settings
        dataloader = SLAFDataLoader(
            slaf_array=slaf,
            tokenizer_type="geneformer",  # or "scgpt"
            batch_size=16,  # Small batch for demo
            max_genes=100,
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=20,
            n_epochs=1,  # Number of epochs
        )

        print("✅ DataLoader initialized:")
        print(f"   Tokenizer type: {dataloader.tokenizer_type}")
        print(f"   Batch size: {dataloader.batch_size}")
        print(f"   Max genes: {dataloader.max_genes}")
        print(f"   Special tokens: {dataloader.special_tokens}")
        print(f"   Number of epochs: {dataloader.n_epochs}")

        return dataloader

    create_dataloader()
    return


@app.cell
def _(mo):
    mo.md(r"""### DataLoader Iteration""")
    return


@app.cell
def _(SLAFDataLoader, slaf):
    def demonstrate_dataloader_iteration():
        # Demonstrate DataLoader iteration
        print("🔄 DataLoader Iteration")
        print("=" * 25)

        # Create dataloader
        dataloader = SLAFDataLoader(
            slaf_array=slaf,
            tokenizer_type="geneformer",  # or "scgpt"
            batch_size=16,  # Small batch for demo
            max_genes=100,
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=20,
            n_epochs=1,  # Number of epochs
        )

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

    demonstrate_dataloader_iteration()
    return


@app.cell
def _(mo):
    mo.md(r"""### Performance Testing""")
    return


@app.cell
def _(SLAFDataLoader, slaf, time):
    def test_dataloader_performance():
        # Performance testing of DataLoader
        print("⚡ DataLoader Performance")
        print("=" * 30)

        # Create dataloader
        dataloader = SLAFDataLoader(
            slaf_array=slaf,
            tokenizer_type="geneformer",  # or "scgpt"
            batch_size=16,  # Small batch for demo
            max_genes=100,
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=20,
            n_epochs=1,  # Number of epochs
        )

        # Test iteration speed
        print("1. Iteration performance:")

        batch_count = 0
        total_tokens = 0

        start_time = time.time()
        for batch in dataloader:
            batch_count += 1
            total_tokens += batch["input_ids"].shape[0] * batch["input_ids"].shape[1]

            # Only process first few batches for demo
            if batch_count >= 5:
                break

        elapsed_time = time.time() - start_time

        print(f"   Processed {batch_count} batches in {elapsed_time:.4f}s")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Tokens per second: {total_tokens / elapsed_time:,.0f}")
        print(f"   Batches per second: {batch_count / elapsed_time:.2f}")

    test_dataloader_performance()
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
def _(SLAFDataLoader, slaf):
    def demonstrate_pytorch_integration():
        # Demonstrate PyTorch integration
        print("🔥 PyTorch Training Loop Integration")
        print("=" * 45)

        # Reinitialize DataLoader since the previous one was exhausted
        dataloader = SLAFDataLoader(
            slaf_array=slaf,
            tokenizer_type="geneformer",
            batch_size=16,
            max_genes=100,
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=20,
            n_epochs=1,
        )

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
            model = YourModel(vocab_size=tokenizer.get_vocab_info()['vocab_size'])
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

    demonstrate_pytorch_integration()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 6. Streaming Dataset Features

    Learn about the new streaming dataset capabilities:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Async Prefetching""")
    return


@app.cell
def _(SLAFDataLoader, slaf):
    def demonstrate_streaming_features():
        print("🔄 Streaming Dataset Features")
        print("=" * 35)

        # Create dataloader with streaming features
        dataloader = SLAFDataLoader(
            slaf_array=slaf,
            tokenizer_type="geneformer",
            batch_size=8,
            max_genes=50,
            n_epochs=1,  # Single epoch for demo
        )

        print("✅ Streaming features:")
        print("   - Async fragment prefetching")
        print("   - Background fragment loading")
        print("   - Memory-efficient streaming")
        print("   - PyTorch IterableDataset compatibility")

        # Test streaming iteration
        print("\n🔄 Testing streaming iteration:")
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            print(f"   Batch {batch_count}: {batch['input_ids'].shape}")
            if batch_count >= 3:
                break

        print(f"   Successfully streamed {batch_count} batches")

    demonstrate_streaming_features()
    return


@app.cell
def _(mo):
    mo.md(r"""### Device Optimization""")
    return


@app.cell
def _(SLAFDataLoader, slaf):
    def demonstrate_device_optimization():
        print("⚡ Device Optimization")
        print("=" * 25)

        from slaf.ml.dataloaders import get_device_info, get_optimal_device

        # Get device information
        device_info = get_device_info()
        optimal_device = get_optimal_device()

        print("✅ Device detection:")
        print(f"   PyTorch available: {device_info['torch_available']}")
        print(f"   CUDA available: {device_info['cuda_available']}")
        print(f"   MPS available: {device_info['mps_available']}")
        print(f"   Optimal device: {optimal_device}")

        # Create dataloader with device optimization
        dataloader = SLAFDataLoader(
            slaf_array=slaf,
            tokenizer_type="geneformer",
            batch_size=8,
            max_genes=50,
            n_epochs=1,
        )

        print("\n✅ Device optimization:")
        print("   - Automatic device detection")
        print("   - Tensor transfer to optimal device")
        print("   - Memory-efficient device handling")

        # Test device transfer
        batch = next(iter(dataloader))
        if hasattr(batch["input_ids"], "device"):
            print(f"   Batch device: {batch['input_ids'].device}")

    demonstrate_device_optimization()
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
def _(SLAFTokenizer, np, time, tokenizer):
    def demonstrate_advanced_features():
        # Advanced tokenization features
        print("🚀 Advanced Tokenization Features")
        print("=" * 40)

        print("1. Different max_genes settings:")
        gene_sequences = [[i, i + 1, i + 2] for i in range(20)]

        for max_genes in [25, 50, 100]:
            input_ids, attention_mask = tokenizer.tokenize(
                gene_sequences=gene_sequences,
                max_genes=max_genes,
            )
            avg_length = np.mean([len(seq) for seq in input_ids])
            print(f"   max_genes={max_genes}: avg_length={avg_length:.1f}")

        print("\n2. Different vocabulary sizes:")
        for vocab_size in [1000, 5000, 10000]:
            # Create a new tokenizer with different vocab size
            test_tokenizer = SLAFTokenizer(
                slaf_array=tokenizer.slaf_array,
                vocab_size=vocab_size,
                n_expression_bins=10,
            )
            vocab_info = test_tokenizer.get_vocab_info()
            print(
                f"   vocab_size={vocab_size}: actual_vocab={vocab_info['vocab_size']}"
            )

        print("\n3. Expression binning for scGPT:")
        # Test scGPT with different expression bins
        gene_sequences = [[1, 2, 3], [2, 3, 4]]
        expr_sequences = [[0.5, 0.8, 0.2], [0.9, 0.1, 0.7]]

        for n_bins in [5, 10, 20]:
            scgpt_tokenizer = SLAFTokenizer(
                slaf_array=tokenizer.slaf_array,
                tokenizer_type="scgpt",
                vocab_size=2000,  # Dataset has <2000 genes
                n_expression_bins=n_bins,
            )

            start_time = time.time()
            input_ids, attention_mask = scgpt_tokenizer.tokenize(
                gene_sequences=gene_sequences,
                expr_sequences=expr_sequences,
                max_genes=25,
            )
            elapsed_time = time.time() - start_time

            print(f"   n_expression_bins={n_bins}: {elapsed_time:.4f}s")

    demonstrate_advanced_features()
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
    def demonstrate_memory_optimization():
        # Memory and performance optimization
        print("💾 Memory and Performance Optimization")
        print("=" * 45)

        import gc

        import psutil

        def get_memory_usage():
            """Get current memory usage in MB"""
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024

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
                slaf_array=slaf,
                batch_size=batch_size,
                max_genes=100,
                vocab_size=2000,  # Dataset has <2000 genes
                n_epochs=1,
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
                n_epochs=1,
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

    demonstrate_memory_optimization()
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
    def create_production_workflow():
        # Production training workflow
        print("🏭 Production Training Workflow")
        print("=" * 40)

        # Create tokenizer
        tokenizer = SLAFTokenizer(
            slaf_array=slaf,
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=50,
        )

        print(
            f"✅ Created tokenizer with {tokenizer.get_vocab_info()['vocab_size']} total tokens"
        )

        # Create dataloaders (in production, you'd use proper splits)
        train_dataloader = SLAFDataLoader(
            slaf_array=slaf,
            tokenizer_type="geneformer",
            batch_size=32,
            max_genes=512,
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=50,
            n_epochs=1,
        )

        val_dataloader = SLAFDataLoader(
            slaf_array=slaf,
            tokenizer_type="geneformer",
            batch_size=32,
            max_genes=512,
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=50,
            n_epochs=1,
        )

        print("\n📊 Production Setup:")
        print(f"   Batch size: {train_dataloader.batch_size}")
        print(f"   Max genes: {train_dataloader.max_genes}")

        # Test production workflow
        print("\n🧪 Testing production workflow:")

        # Test training batch
        train_batch = next(iter(train_dataloader))
        print(f"   Training batch shape: {train_batch['input_ids'].shape}")

        # Test validation batch
        val_batch = next(iter(val_dataloader))
        print(f"   Validation batch shape: {val_batch['input_ids'].shape}")

        return train_dataloader, val_dataloader, tokenizer

    create_production_workflow()
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
    def show_best_practices():
        # Best practices for ML training
        print("💡 Best Practices for ML Training")
        print("=" * 40)

        print("1. Tokenizer Configuration:")
        print("   ✅ Choose appropriate vocab_size based on your dataset")
        print("   ✅ Use n_expression_bins=50 for fine-grained expression modeling")
        print("   ✅ Use the correct tokenizer_type for your model architecture")

        print("\n2. DataLoader Configuration:")
        print("   ✅ Start with small batch_size and increase gradually")
        print("   ✅ Use max_genes appropriate for your model architecture")
        print("   ✅ Set n_epochs for multi-epoch training on small datasets")

        print("\n3. Performance Optimization:")
        print("   ✅ Fragment-based processing is much faster than SQL")
        print("   ✅ Use async prefetching to minimize GPU idle time")
        print("   ✅ Leverage device optimization for automatic tensor transfer")
        print("   ✅ Monitor memory usage during training")

        print("\n4. Training Workflow:")
        print("   ✅ Create separate train/val/test splits")
        print("   ✅ Use consistent tokenizer across splits")
        print("   ✅ Implement proper error handling")
        print("   ✅ Use streaming datasets for large datasets")

        print("\n5. Production Considerations:")
        print("   ✅ Use appropriate batch sizes for your hardware")
        print("   ✅ Implement checkpointing for long training runs")
        print("   ✅ Monitor tokenization throughput")
        print("   ✅ Consider distributed training for large datasets")

    show_best_practices()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary

    **What you've learned about SLAF ML Training:**

    1. **Streaming DataLoader**: Async prefetching with PyTorch compatibility
    2. **Tokenization Strategies**: Geneformer and scGPT formats with optimized processing
    3. **Memory Efficiency**: Streaming datasets for large-scale training
    4. **Production Workflow**: Complete training pipeline setup
    5. **Best Practices**: Guidelines for optimal ML training performance

    **Key Performance Improvements:**
    - Fragment processing for high throughput
    - Streaming datasets with async prefetching
    - Automatic device optimization
    - Memory-efficient processing for large datasets
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
