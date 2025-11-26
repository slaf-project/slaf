"""
SLAF-specific distributed dataloader.

Composes generic distributed components with SLAF-specific logic.
"""

from typing import Any

import modal

from slaf.core.slaf import SLAFArray
from slaf.distributed.coordinator import Coordinator

# Import generic distributed components
from slaf.distributed.data_source import LanceDataSource
from slaf.distributed.dataloader import DistributedDataLoader
from slaf.distributed.processor import DataSchema

# Import SLAF-specific components (for type hints and adapters)
from slaf.ml.aggregators import Window
from slaf.ml.samplers import Shuffle
from slaf.ml.tokenizers import SLAFTokenizer

# Configure Modal image for SLAF workers
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "python3-dev", "git")
    .pip_install(["torch", "polars", "pyarrow", "numpy", "psutil"])
    .run_commands(
        "pip install git+https://github.com/slaf-project/slaf.git@distributed_dataloader#egg=slafdb"
    )
)

# Create SLAF-specific Modal app
# IMPORTANT: This app must be deployed before use:
#   modal deploy slaf/ml/distributed.py
# This makes it a persistent deployed app that can be invoked from other Modal functions.
# See: https://modal.com/docs/guide/apps#deployed-apps
app = modal.App("slaf-distributed-dataloader")


@app.function(
    image=image,
    cpu=8,
    memory=32768,  # 32 GB per worker
    timeout=3600,
    secrets=[modal.Secret.from_name("s3-credentials")],
)
def distributed_prefetch_worker(
    worker_id: str,
    partition_indices: list[int],
    data_source_config: dict[str, Any],
    processor_config: dict[str, Any],
    queue_name: str,
    n_scanners: int = 8,
    batches_per_partition: int = 32,
    prefetch_batch_size: int = 262144,
    max_batches: int | None = None,
    partial_groups_kv_name: str | None = None,
) -> dict[str, Any]:
    """
    SLAF-specific Modal worker function with SLAF image.

    This function wraps the framework-agnostic worker implementation and
    handles all Modal-specific setup (Queue, KV store, etc.).
    """
    from slaf.distributed.worker import prefetch_worker

    # Create Modal Queue
    # Use create_if_missing=True to ensure queue exists (it should be created in __init__)
    queue = modal.Queue.from_name(queue_name, create_if_missing=True)

    # Create Modal Dict for cross-worker boundary merging if enabled
    partial_groups_kv = None
    if (
        processor_config.get("enable_cross_worker_boundary_merging", False)
        and partial_groups_kv_name
    ):
        partial_groups_kv = modal.Dict.from_name(
            partial_groups_kv_name, create_if_missing=True
        )

    # Call the framework-agnostic implementation
    return prefetch_worker(
        worker_id=worker_id,
        partition_indices=partition_indices,
        data_source_config=data_source_config,
        processor_config=processor_config,
        queue=queue,
        n_scanners=n_scanners,
        batches_per_partition=batches_per_partition,
        prefetch_batch_size=prefetch_batch_size,
        max_batches=max_batches,
        partial_groups_kv=partial_groups_kv,
    )


class DistributedSLAFDataLoader:
    """
    SLAF-specific distributed dataloader.

    Composes generic distributed components with SLAF-specific configuration.
    """

    def __init__(
        self,
        slaf_array: SLAFArray,
        tokenizer_type: str = "geneformer",
        window: Window | None = None,
        shuffle: Shuffle | None = None,
        tokenizer: SLAFTokenizer | None = None,
        n_workers: int = 64,
        n_scanners: int = 16,
        prefetch_batch_size: int = 4_194_304,
        batches_per_partition: int = 32,
        batch_size: int = 32,
        max_genes: int = 1024,
        vocab_size: int = 50000,
        n_expression_bins: int = 10,
        n_epochs: int = 1,
        raw_mode: bool = False,
        seed: int = 42,
        queue_name: str | None = None,
        **window_kwargs: Any,
    ):
        """
        Initialize distributed SLAF dataloader.

        Args:
            slaf_array: SLAFArray instance containing the data
            tokenizer_type: Tokenization strategy ("geneformer", "scgpt", or "raw")
                           If "raw", raw_mode is automatically enabled
            window: Window function instance (optional, created automatically if None)
            shuffle: Shuffle function instance (optional, created automatically if None)
            tokenizer: SLAFTokenizer instance (optional, created automatically from tokenizer_type if None)
            n_workers: Number of Modal workers
            n_scanners: Number of scanners per worker (for MoS)
            prefetch_batch_size: Batch size for reading from partitions
            batches_per_partition: Number of batches to read per partition before processing
            batch_size: Training batch size (not used in distributed mode, kept for compatibility)
            max_genes: Maximum genes per cell after window function
            vocab_size: Vocabulary size for tokenizer
            n_expression_bins: Number of expression bins for scGPT
            n_epochs: Number of epochs to process
            raw_mode: If True, return raw data without tokenization
            seed: Random seed for reproducibility
            queue_name: Name of Modal Queue (auto-generated if None)
            **window_kwargs: Additional window function parameters
        """
        # Handle raw_mode
        if tokenizer_type == "raw":
            raw_mode = True
            tokenizer = None
        elif raw_mode:
            tokenizer = None

        # Create tokenizer if not provided and not in raw mode
        if tokenizer is None and not raw_mode:
            tokenizer = SLAFTokenizer(
                slaf_array=slaf_array,
                tokenizer_type=tokenizer_type,
                vocab_size=vocab_size,
                n_expression_bins=n_expression_bins,
            )
        # Create data source
        lance_path = f"{slaf_array.slaf_path}/expression.lance"
        data_source = LanceDataSource(lance_path)

        # Create coordinator
        coordinator = Coordinator(data_source, n_workers)
        assignments = coordinator.assign_partitions(seed=seed)

        # Create queue and KV store
        if queue_name is None:
            queue_name = f"slaf-dataloader-{id(slaf_array)}"
        modal.Queue.from_name(queue_name, create_if_missing=True)

        # Create KV store for cross-worker boundary merging
        # We'll create it after processor_config is defined, but the name is deterministic
        partial_groups_kv_name = f"{queue_name}-partial-groups"

        # Prepare configs for workers
        data_source_config = {
            "type": "lance",
            "path": lance_path,
        }

        # Data schema for SLAF (maps generic schema to SLAF column names)
        schema = DataSchema(
            group_key="cell_integer_id",  # Group by cell
            item_key="gene_integer_id",  # Items are genes
            value_key="value",  # Values are expression
            group_key_out="cell_integer_id",  # Output keeps same group key
            item_list_key="gene_sequence",  # Aggregated gene list
            value_list_key="expr_sequence",  # Aggregated expression list (for scGPT)
        )

        processor_config = {
            "schema": {
                "group_key": schema.group_key,
                "item_key": schema.item_key,
                "value_key": schema.value_key,
                "group_key_out": schema.group_key_out,
                "item_list_key": schema.item_list_key,
                "value_list_key": schema.value_list_key,
            },
            # Window factory config (module path in config, not hardcoded)
            # For now, we'll use None and let the worker use the default Window
            # TODO: Add factory functions to slaf.ml.aggregators and slaf.ml.samplers
            "window_factory": None,  # Will use default Window from slaf.distributed
            "shuffle_factory": None,  # Will use default Shuffle from slaf.distributed
            "max_items": max_genes,  # Generic name
            "seed": seed,
            "n_epochs": n_epochs,
            "window_kwargs": window_kwargs,
            "continuity_check": "sequential",  # How to detect continuity between partitions
            "enable_cross_worker_boundary_merging": True,  # Enable cross-worker merging via KV store
        }

        # Create the KV dict to ensure it exists before workers try to access it
        if processor_config.get("enable_cross_worker_boundary_merging", True):
            modal.Dict.from_name(partial_groups_kv_name, create_if_missing=True)

        # Tokenizer factory config (for dynamic import in worker)
        # Only include tokenizer if not in raw mode
        if tokenizer and not raw_mode:
            processor_config["tokenizer_factory"] = {
                "module": "slaf.ml.tokenizers",
                "class": "SLAFTokenizer",
                "kwargs": {
                    # Serialize tokenizer config (vocab size, etc.)
                    # Tokenizer instance itself can't be serialized
                    "tokenizer_type": tokenizer.tokenizer_type.value,
                    "vocab_size": tokenizer.vocab_size,
                    "n_expression_bins": tokenizer.n_expression_bins,
                },
            }
        else:
            # Raw mode - no tokenizer
            processor_config["tokenizer_factory"] = None

        # Spawn workers
        # NOTE: The app must be deployed before spawning workers:
        #   modal deploy slaf/ml/distributed.py
        # Or ensure the app is running when called from another Modal function.
        # We cannot use app.run() here because we may be inside another Modal function.
        # Reference the deployed function by name to avoid hydration issues
        # when called from another Modal app
        # The app must be deployed first: modal deploy slaf/ml/distributed.py
        worker_function = modal.Function.from_name(
            "slaf-distributed-dataloader", "distributed_prefetch_worker"
        )
        # Explicitly hydrate the function to ensure it's synchronized with the Modal server
        # This is needed when referencing a function from a deployed app
        worker_function.hydrate()

        worker_handles = []
        print(f"Spawning {len(assignments)} workers...")
        for worker_id, assignment in assignments.items():
            print(
                f"  Spawning worker {worker_id} with {len(assignment.partition_indices)} partitions"
            )
            try:
                handle = worker_function.spawn(
                    worker_id=worker_id,
                    partition_indices=assignment.partition_indices,
                    data_source_config=data_source_config,
                    processor_config=processor_config,
                    queue_name=queue_name,
                    n_scanners=n_scanners,
                    batches_per_partition=batches_per_partition,
                    prefetch_batch_size=prefetch_batch_size,
                    partial_groups_kv_name=partial_groups_kv_name,
                )
                worker_handles.append(handle)
                print(
                    f"  ✅ Worker {worker_id} spawned successfully (handle: {handle})"
                )
            except Exception as e:
                print(f"  ❌ Error spawning worker {worker_id}: {e}")
                import traceback

                traceback.print_exc()
                raise

        print(f"✅ All {len(worker_handles)} workers spawned successfully")
        print(f"Queue name: {queue_name}")
        print(f"KV store name: {partial_groups_kv_name}")

        # Check worker status after a short delay
        import time

        print("Waiting 5 seconds for workers to initialize...")
        time.sleep(5)

        # Try to get worker status (non-blocking check)
        for worker_id, handle in zip(assignments.keys(), worker_handles, strict=True):
            try:
                # Try to get result with timeout=0.1 (non-blocking)
                # This will raise if worker hasn't started or has an error
                result = handle.get(timeout=0.1)
                print(f"  Worker {worker_id} completed: {result}")
            except Exception:
                # Expected - workers are still running
                print(f"  Worker {worker_id} status: running (expected)")

        # Create queue object for the dataloader
        queue = modal.Queue.from_name(queue_name, create_if_missing=True)

        # Create dataloader with queue object (framework-agnostic)
        self.dataloader = DistributedDataLoader(queue)
        self.worker_handles = worker_handles
        self.queue_name = queue_name  # Store queue name for external access
        self.partial_groups_kv_name = (
            partial_groups_kv_name  # Store KV store name for external access
        )

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
