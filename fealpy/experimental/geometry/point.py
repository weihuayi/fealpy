from typing import Tuple
from ..backend import backend_manager as bm
from ..backend import TensorLike

from .geometry_base import GeometryBase

class Point(GeometryBase):
    def __init__(self, coords: TensorLike):
        """
        Initialize 2D or 3D points.
        
        Parameters:
            coords (TensorLike): Coordinates of the points, shape (NP, GD).
                Where NP is the number of points, GD is the geometric dimension.
        """
        self.data = bm.tensor(coords, dtype=bm.float64)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the point(s)."""
        return self.data.shape[-1]

    def top_dimension(self) -> int:
        """Return the toplogy dimension of the point(s)."""
        return 0

    GD = property(geo_dimension)
    TD = property(top_dimension)

    def measure(self) -> TensorLike:
        """Points have no measure; return zeros."""
        kwargs = bm.context(self.data)
        return bm.zeros(self.data.shape[0], **kwargs)

    def __repr__(self) -> str:
        return f"Point(coords={self.data})"
