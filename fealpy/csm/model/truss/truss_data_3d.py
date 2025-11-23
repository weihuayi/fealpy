from fealpy.decorator import cartesian
from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm

from fealpy.mesh import EdgeMesh


class TrussData3D:
    """
    A data structure class representing a 3D truss example.

    This class specifically handles 3D truss structures, initializing with predefined node coordinates and edge connectivity.
    It sets up the mesh, source term, distributed load, and boundary conditions tailored for truss geometries. Upon instantiation,
    it provides methods to construct the mesh and define problem-specific data for the truss model.

    Parameters:
        None

    Attributes:
        mesh(EdgeMesh): The edge mesh representing the truss structure.
        node(ndarray): Array of node coordinates.
        edge(ndarray): Array of edge connectivity.

    Methods:
        geo_dimension(): Return the geometric dimension of the domain.
        domain(): Return the computational domain for API compatibility.
        init_mesh(): Construct and return the EdgeMesh for the truss structure.
        source(x): Return the source term for the truss problem.
        load(x): Return the distributed load for the truss problem.
        is_displacement_boundary(): Return indices of nodes with displacement boundary conditions.
        displacement_bc(p): Return prescribed displacement values for boundary nodes.
    """

    def __init__(self):
        super().__init__()

        self.A = self.cross_section_area()
        self.GD = self.geo_dimension()
        
    def geo_dimension(self) -> int:
        """
        Returns the geometric dimension of the domain.

        Returns:
            int: The geometric dimension (3).
        """
        return 3

    def domain(self):
        """
        Returns the computational domain for API compatibility.

        Returns:
            list: The domain as [xmin, xmax].
        """
        return [0.0, self.L]
    
    def cross_section_area(self):
        """Return bar cross-sectional areas.
        """
        return 2000.0
        
    def init_mesh(self):
        """
        Constructs the 3D EdgeMesh for the truss.

        Returns:
            EdgeMesh: The edge mesh of the truss.
        """
        node = bm.array([
            [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540], 
            [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
            [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0], 
            [-2540, -2540, 0]], dtype=bm.float64)
        edge = bm.array([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]] , dtype=bm.int32)
        self.mesh = EdgeMesh(node, edge)
        return self.mesh
    
    @cartesian
    def source(self, x):
        """
        Returns the source term at x.

        Parameters:
            x (TensorLike): The spatial coordinates.

        Returns:
            TensorLike: Zero vector in 3D.
        """
        return bm.zeros_like(x)
    
    @cartesian
    def load(self, x):
        """
        Returns the distributed load at x.

        Parameters:
            x (TensorLike): The spatial coordinates.

        Returns:
            TensorLike: Constant vector with magnitude self.f.
        """
        return bm.ones_like(x) * self.f
    
    @cartesian
    def is_displacement_boundary(self) -> TensorLike:
        """
        Returns the indices of nodes with displacement boundary conditions.

        Returns:
            TensorLike: Indices of boundary nodes.
        """
        return bm.array([18, 19, 20, 21, 22, 23, 24, 
                         25, 26, 27, 28, 29], dtype=bm.int32)

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        """
        Returns the prescribed displacement values for boundary nodes.

        Parameters:
            p (TensorLike): The coordinates of boundary nodes.

        Returns:
            TensorLike: Zero displacement.
        """
        return bm.zeros_like(p)
