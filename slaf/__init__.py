try:
    from importlib.metadata import version

    __version__ = version("slafdb")
except ImportError:
    __version__ = "unknown"

# Core import - always available
from slaf.core import SLAFArray


# Lazy imports for heavy dependencies
def _import_converter():
    """Lazy import for SLAFConverter"""
    from slaf.data import SLAFConverter

    return SLAFConverter


def _import_integrations():
    """Lazy import for integration modules"""
    from slaf.integrations import LazyAnnData, LazyExpressionMatrix, pp

    return LazyAnnData, LazyExpressionMatrix, pp


def _import_ml_components():
    """Lazy import for ML components"""
    from slaf.ml import (
        SLAFDataLoader,
        SLAFTokenizer,
        TokenizerType,
        create_shuffle,
        create_window,
    )

    return SLAFDataLoader, SLAFTokenizer, TokenizerType, create_shuffle, create_window


# Expose lazy import functions
def get_converter():
    """Get SLAFConverter (lazy import)"""
    return _import_converter()


def get_integrations():
    """Get integration modules (lazy import)"""
    return _import_integrations()


def get_ml_components():
    """Get ML components (lazy import)"""
    return _import_ml_components()


__all__ = [
    # Core data structures (always available)
    "SLAFArray",
    # Lazy import functions
    "get_converter",
    "get_integrations",
    "get_ml_components",
]
