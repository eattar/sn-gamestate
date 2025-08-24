from importlib import metadata
try:
    __version__ = metadata.version('sn-gamestate')
except metadata.PackageNotFoundError:
    __version__ = None

# Import our custom modules so TrackLab can discover them
from .calibration.unified_backbone import UnifiedBackboneModule

__all__ = ['UnifiedBackboneModule']