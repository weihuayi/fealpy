
from typing import Any, Tuple
from ..backend import TensorLike


class GeometryBase:
    """Base class for geometric objects with common interfaces."""

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the object."""
        raise NotImplementedError("Subclasses should implement this method.")

    def top_dimension(self) -> int:
        """Return the toplogy dimension of the object."""
        raise NotImplementedError("Subclasses should implement this method.")

    def measure(self) -> TensorLike:
        """Compute the measure (e.g., area, volume) of the geometry object."""
        raise NotImplementedError("Subclasses should implement this method.")

    def contains(self, points: TensorLike) -> TensorLike:
        """Check if points are inside the geometry object."""
        raise NotImplementedError("Subclasses should implement this method.")
