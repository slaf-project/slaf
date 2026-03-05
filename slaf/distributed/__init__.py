"""
Distributed dataloader implementation.

This module provides generic distributed dataloader components that can be
extracted to a separate repository in the future. It separates generic
distributed coordination logic from SLAF-specific business logic.
"""

from slaf.distributed.boundary import GroupBoundaryHandler
from slaf.distributed.coordinator import Coordinator
from slaf.distributed.data_source import DataSource, LanceDataSource
from slaf.distributed.dataloader import DistributedDataLoader
from slaf.distributed.processor import BatchProcessor, DataSchema
from slaf.distributed.shuffle import Shuffle
from slaf.distributed.window import Window

__all__ = [
    "DataSource",
    "LanceDataSource",
    "DataSchema",
    "BatchProcessor",
    "GroupBoundaryHandler",
    "Window",
    "Shuffle",
    "Coordinator",
    "DistributedDataLoader",
]
