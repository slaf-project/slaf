try:
    from importlib.metadata import version

    __version__ = version("slaf")
except ImportError:
    __version__ = "unknown"

from slaf.core import SLAFArray
from slaf.data import SLAFConverter
from slaf.integrations import LazyAnnData, LazyExpressionMatrix, pp

__all__ = ["SLAFArray", "SLAFConverter", "LazyAnnData", "LazyExpressionMatrix", "pp"]
