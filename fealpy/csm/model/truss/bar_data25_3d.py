from fealpy.decorator import cartesian
from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm

from fealpy.mesh import EdgeMesh


class BarData25:
    """
    A data structure class representing a 3D bar example.

    This class specifically handles 3D bar structures, initializing with predefined node coordinates and edge connectivity.
    It sets up the mesh, source term, distributed load, and boundary conditions tailored for bar geometries. Upon instantiation,
    it provides methods to construct the mesh and define problem-specific data for the bar model.

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
        load(): Return the nodal loads for the truss problem.
        is_dirichlet_boundary(): Return indices of nodes with displacement boundary conditions.
        dirichlet_bc(): Return prescribed displacement values for boundary nodes.
    """

    def __init__(self):
        super().__init__()

        self.A = self.cross_section_area()
        self.GD = self.geo_dimension()
        
        self.mesh = self.init_mesh()
        
    def geo_dimension(self) -> int:
        """Returns the geometric dimension of the domain."""
        return 3
    
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
        mesh = EdgeMesh(node, edge)
        return mesh
    
    @cartesian
    def load(self):
        """Return the nodal loads for the truss problem.
        
        Note:
            A concentrated force of [0, 900, 0] is applied at each top node, 
            representing a 900-unit force in the positive y-direction.
        """
        GD = self.GD
        NN = self.mesh.number_of_nodes()
        F = bm.zeros((NN, GD), dtype=bm.float64)
        
        node = self.mesh.entity('node')
        F[node[..., 2] == 5080] = bm.array([0, 900, 0])
        return F

    @cartesian
    def is_dirichlet_boundary(self) -> TensorLike:
        """Returns a boolean array indicating which DOFs have Dirichlet BCs.

        Returns:
            TensorLike: Boolean array of shape (NN*GD,).
        """
        node = self.mesh.entity('node')
        GD = self.GD
        
        # 底部节点 (z < 1e-12)
        is_bd_node = node[..., 2] < 1e-12
        is_bd_dof = bm.repeat(is_bd_node, GD)
    
        return is_bd_dof

    @cartesian
    def dirichlet_bc(self) -> TensorLike:
        """Returns the prescribed displacement values for ALL DOFs.


        Returns:
            TensorLike: Zero displacement vector of shape (NN*GD,).
        """
        NN = self.mesh.number_of_nodes()
        GD = self.GD
        
        return bm.zeros(NN * GD, dtype=bm.float64)
