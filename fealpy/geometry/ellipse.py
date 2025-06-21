from typing import Optional
from ..backend import backend_manager as bm
from ..backend import TensorLike
from .geometry_base import GeometryBase
from .functional import apply_rotation

class Ellipse(GeometryBase):
    def __init__(self, data: TensorLike, 
                 TD: int = 2, GD: Optional[int] = None, 
                 rotation: Optional[TensorLike] = None):
        """
        Initialize ellipses defined by centers, radii, and orientation.

        Parameters:
            data (TensorLike): Array with shape (NE, GD+2), where:
                - GD is the geometric dimension (2 or 3)
                - The last two entries represent the semi-major and semi-minor axes.
            TD (int): The topology dimension, typically 1 for circumference or 2 for area.
            GD (Optional[int]): The geometric dimension (default based on data).
            rotation (Optional[TensorLike]): Rotation angles (in radians) for the ellipses.
                - For 2D, this is a single angle.
                - For 3D, this can be a tensor of shape (NE, 3) representing rotation around each axis.
        """
        self.data = bm.tensor(data, dtype=bm.float64)
        self.TD = TD
        self.GD = GD if GD is not None else (self.data.shape[-1] - 2)
        self.rotation = rotation if rotation is not None else bm.zeros((self.data.shape[0], 3) if self.GD == 3 else 1)

    def geo_dimension(self) -> int:
        return self.GD

    def top_dimension(self) -> int:
        return self.TD

    def measure(self) -> TensorLike:
        """Calculate the measure (area or circumference) of the ellipse(s)."""
        semi_major = self.data[:, -2]  # Semi-major axis
        semi_minor = self.data[:, -1]  # Semi-minor axis
        
        if self.TD == 2:
            return bm.pi * semi_major * semi_minor
        elif self.TD == 1:
            a = semi_major
            b = semi_minor
            h = ((a - b) ** 2) / ((a + b) ** 2)
            return bm.pi * (a + b) * (1 + (3 * h) / (10 + bm.sqrt(4 - 3 * h)))

    def contains(self, points: TensorLike) -> TensorLike:
        """
        Check if given points are inside the ellipse(s).
        
        Parameters:
            points (TensorLike): Coordinates of the points to check, shape (NP, GD).
        
        Returns:
            TensorLike: Boolean tensor indicating containment status.
        """
        centers = self.data[:, :self.GD]
        semi_major = self.data[:, -2]
        semi_minor = self.data[:, -1]

        # Apply rotation to the points
        rotated_points = apply_rotation(points, centers, self.rotation, self.GD)

        distances_major = bm.abs(rotated_points[..., 0] - centers[:, None, 0]) / semi_major[:, None]
        distances_minor = bm.abs(rotated_points[..., 1] - centers[:, None, 1]) / semi_minor[:, None]

        return (distances_major ** 2 + distances_minor ** 2) <= 1

    def __repr__(self) -> str:
        return f"Ellipse(data={self.data}, rotation={self.rotation})"
