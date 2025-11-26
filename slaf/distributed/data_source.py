"""
Generic data source interface for distributed dataloading.

Inspired by Ray Data's block-based approach - each partition is a unit
that can be processed independently. Works with Lance fragments, Parquet
files/row groups, MosaicML Streaming shards, Zarr arrays, Vortex datasets,
and many other formats.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

import lance
import polars as pl


class DataSource(ABC):
    """
    Abstract interface for loading raw data partitions.

    Similar to Ray Data's concept of blocks - each partition is a unit
    that can be processed independently. Works with Lance fragments,
    Parquet files/row groups, MosaicML Streaming shards, Zarr arrays,
    Vortex datasets, and many other formats.

    The term "partition" is database-inspired and format-agnostic:
    - Lance: fragments
    - Parquet: files or row groups
    - Zarr: chunks (but we call them partitions to avoid confusion)
    - MosaicML Streaming: shards
    - Vortex: fragments
    - WebDataset: tar files
    - etc.
    """

    @abstractmethod
    def get_partition_count(self) -> int:
        """
        Return total number of data partitions.

        Partitions are the unit of parallelization - could be:
        - Lance fragments
        - Parquet files or row groups
        - MosaicML Streaming shards
        - Zarr array chunks
        - Vortex fragments
        - WebDataset tar files
        - Any other partitionable data unit
        """
        pass

    @abstractmethod
    def create_reader(
        self, partition_index: int, batch_size: int
    ) -> Iterator[pl.DataFrame]:
        """
        Create a reader/generator for a specific partition.

        Args:
            partition_index: Index of the partition to read (0 to get_partition_count()-1)
            batch_size: Size of batches to yield from the reader

        Returns:
            Iterator of Polars DataFrames (raw data in tabular format)

            The DataFrame format depends on the data source:
            - For tabular data (Lance, Parquet): Returns DataFrames directly
            - For structured data (JSONL, TFRecord): Returns DataFrames with parsed records
            - For array data (Zarr, Vortex): Returns DataFrames with flattened/reshaped data
            - For multi-modal data (WebDataset): Returns DataFrames with file paths or decoded data

        Note:
            Lazy initialization - generator created on-demand, not upfront.
            First read from generator may include warmup cost.
        """
        pass


class LanceDataSource(DataSource):
    """
    Data source for Lance datasets.

    Works with any Lance dataset (not SLAF-specific).
    Uses fragments as partitions.
    """

    def __init__(self, lance_path: str):
        """
        Initialize Lance data source.

        Args:
            lance_path: Path to Lance dataset
        """
        self.lance_path = lance_path
        self._dataset: lance.Dataset | None = None
        self._fragment_count: int | None = None

    @property
    def dataset(self):
        """Lazy-load Lance dataset."""
        if self._dataset is None:
            self._dataset = lance.dataset(self.lance_path)
        return self._dataset

    def get_partition_count(self) -> int:
        """Return number of fragments in the Lance dataset."""
        if self._fragment_count is None:
            self._fragment_count = len(list(self.dataset.get_fragments()))
        # After the check above, _fragment_count is guaranteed to be int
        assert self._fragment_count is not None
        return self._fragment_count

    def create_reader(
        self, partition_index: int, batch_size: int
    ) -> Iterator[pl.DataFrame]:
        """
        Create a reader for a specific fragment.

        Args:
            partition_index: Fragment index (0 to get_partition_count()-1)
            batch_size: Size of batches to yield

        Returns:
            Iterator of Polars DataFrames from the fragment
        """
        fragments = list(self.dataset.get_fragments())
        if partition_index >= len(fragments):
            raise IndexError(
                f"Partition index {partition_index} out of range "
                f"(0 to {len(fragments) - 1})"
            )

        fragment = fragments[partition_index]
        # Create generator from fragment batches
        for batch in fragment.to_batches(batch_size=batch_size):
            yield pl.from_arrow(batch)
