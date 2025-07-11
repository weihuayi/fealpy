from fealpy.mesh import EdgeMesh
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike

class BarData2D:
    """
    1D Axial Bar Problem:

        E A du/dx = p(x),    x ∈ (0, L)
        u(0) = 0,            (fixed boundary at node 4)
        p(x) = constant axial force

    Parameters:
        E: Young's modulus
        A: Cross-sectional area (used if axial terms are present)
        f: load 
        L: Bar length
        n: Number of mesh elements

    This class constructs a 1D axial bar mesh with specified boundary conditions and external loads.
    It also provides the source term for the axial force distribution.
    """

    def __init__(self):
        """
        Initialize bar parameters.

        Parameters:
            E (float): Young's modulus.
            A (float): Cross-sectional area.
            f (float): load.
            L (float): Beam length.
            n (int): Number of mesh elements.
        """
        self.mesh = self.init_mesh()

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 1

    def domain(self):
        """Return the computational domain [xmin, xmax]."""
        return [0.0, self.L]
    
    def init_mesh(self):
        """
        Construct a 2D EdgeMesh for the bar domain.

        Returns:
            EdgeMesh: 1D mesh from x=0 to x=L.
        """
        node = bm.array([[0, 0], [0 , 0.4],[0.4, 0.3], [0, 0.3]], dtype=bm.float64)
        cell = bm.array([[0, 1],[1, 2], [2, 0], [2, 3]] , dtype=bm.int32)
        return EdgeMesh(node, cell)