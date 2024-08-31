from typing import Optional
from ..backend import backend_manager as bm
from ..backend import TensorLike
from .geometry_base import GeometryBase
from .functional import apply_rotation

class Circle(GeometryBase):
    def __init__(self, data: TensorLike, 
                 TD: int = 2, GD: Optional[int] = None, 
                 rotation: Optional[TensorLike] = None):
        """
        Initialize circles defined by centers, radii, and optional rotation.

        Parameters:
            data (TensorLike): Array with shape (NC, GD+1), where:
                - GD is the geometric dimension (2 or 3)
                - The last entry represents the radius.
            TD (int): The topology dimension, 1 for circumference or 2 for area.
            GD (Optional[int]): The geometric dimension (default based on data).
            rotation (Optional[TensorLike]): Rotation angles (in radians) for the circles.
                - For 2D, this is a single angle.
                - For 3D, this can be a tensor of shape (NC, 3) representing rotation around each axis.
        """
        self.data = bm.tensor(data, dtype=bm.float64)
        self.TD = TD
        self.GD = GD if GD is not None else (self.data.shape[-1] - 1)
        self.rotation = rotation if rotation is not None else bm.zeros((self.data.shape[0], 3) if self.GD == 3 else 1)

    def geo_dimension(self) -> int:
        return self.GD

    def top_dimension(self) -> int:
        return self.TD

    def measure(self) -> TensorLike:
        """Calculate the measure (area or circumference) of the circle(s)."""
        radii = self.data[:, -1]

        if self.TD == 2:
            return bm.pi * radii ** 2
        elif self.TD == 1:
            return 2.0 * bm.pi * radii

    def contains(self, points: TensorLike) -> TensorLike:
        """
        Check if given points are inside the circle(s).
        
        Parameters:
            points (TensorLike): Coordinates of the points to check, shape (NP, GD).
        
        Returns:
            TensorLike: Boolean tensor indicating containment status.
        """
        centers = self.data[:, :self.GD]  # Extract circle centers
        radii = self.data[:, -1]          # Extract radii

        # Apply rotation to the points
        rotated_points = apply_rotation(points, centers, self.rotation, self.GD)

        # Distance from point to circle center
        distances = bm.linalg.norm(rotated_points - centers[:, None, :], axis=2)
        return distances <= radii[:, None]

    def __repr__(self) -> str:
        return f"Circle(data={self.data}, rotation={self.rotation})"
