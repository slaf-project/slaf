try:
    from importlib.metadata import version

    __version__ = version("slaf")
except ImportError:
    __version__ = "unknown"

from slaf.core import SLAFArray
from slaf.data import SLAFConverter
from slaf.integrations import LazyAnnData, LazyExpressionMatrix, pp
from slaf.ml import (
    SLAFDataLoader,
    SLAFTokenizer,
    TokenizerType,
    create_shuffle,
    create_window,
)

__all__ = [
    # Core data structures
    "SLAFArray",
    "SLAFConverter",
    # Integrations
    "LazyAnnData",
    "LazyExpressionMatrix",
    "pp",
    # ML components
    "SLAFDataLoader",
    "SLAFTokenizer",
    "TokenizerType",
    "create_window",
    "create_shuffle",
]
