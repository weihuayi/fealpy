
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike

from fealpy.mesh import EdgeMesh

class Truss2D:
    """
    2 Axial Truss problem:

        E A du/dx = f(x),    x ∈ (0, L)
        f(x) = constant axial force

    Parameters
        E: Young's modulus
        A: Cross-sectional area (used if axial terms are present)
        f: load 
        L: Bar length
        n: Number of mesh elements

    This class constructs a 2D axial bar mesh with specified boundary conditions and external loads.
    It also provides the source term for the axial force distribution.
    """

    def __init__(self):
        super().__init__()

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 1

    def domain(self):
        """Return the computational domain [xmin, xmax]."""
        return [0.0, self.L]
    
    def init_mesh(self):
        """
        Construct a 2D EdgeMesh for the truss domain.

        Returns
            EdgeMesh: 2D mesh from x=0 to x=L.
        """
        node = bm.array([[0, 0], [0 , 0.4],[0.4, 0.3], [0, 0.3]], dtype=bm.float64)
        cell = bm.array([[0, 1],[2, 1], [0, 2], [3, 2]] , dtype=bm.int32)
        return EdgeMesh(node, cell)
    
    @cartesian
    def source(self, x):
        """
        Compute the distributed load f(x).

        Parameters
            x: Spatial coordinate(s).

        Returns
            Tensor: Distributed load at x.
        """
        return bm.ones_like(x) * self.f
    
    def dirichlet_dof_index(self) -> TensorLike:
        """
        Return the indices of degrees of freedom (DOFs) where Dirichlet boundary conditions are applied.

        Parameters
            total_dof : int
            Total number of global degrees of freedom.

        Returns
            Tensor[int]: Indices of boundary DOFs.
        """
        return bm.array([0, 1, 2, 3])
    
    @cartesian
    def dirichlet(self, x):
        """
        Compute the Dirichlet boundary condition.

        Parameters
            x: Spatial coordinate(s).

        Returns
            Tensor: Dirichlet boundary condition at x.
        """
        return bm.zeros()