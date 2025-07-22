from fealpy.mesh import EdgeMesh
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike


class TimoshenkoBeamData3D:
    """
    3D Timoshenko beam problem:
    """

    def __init__(self):
        """
        Initialize beam parameters.
        """
        self.mesh = self.init_mesh()

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 3