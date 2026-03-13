from fealpy.decorator import cartesian
from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.mesh import EdgeMesh

from ...mesh.bar942 import Bar942


class BarData942:
    """A data structure class representing a 3D bar example.
    This class specifically handles 3D bar structures, initializing with predefined node coordinates and edge connectivity.
    It sets up the mesh, source term, distributed load, and boundary conditions tailored for bar geometries.Upon instantiation,
    it provides methods to construct the mesh and define problem-specific data for the bar model.
    
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
        return 4.0
        
    def init_mesh(self):
        """Constructs the 3D EdgeMesh for the bar942.

        Returns:
            EdgeMesh: The edge mesh of the bar942.
        """
        bar = Bar942()
        nodes, cells = bar.build_truss_3d()
        mesh = EdgeMesh(nodes, cells)
        return mesh
    
    @cartesian
    def load(self):
        """Return the nodal loads for the truss problem.
        
        Note:
            Concentrated forces are applied at nodes 1 and 2:
            - Node 1: [0, 400, -100] (y=400, z=-100)
            - Node 2: [0, 400, -100] (y=400, z=-100)
        """
        GD = self.GD
        NN = self.mesh.number_of_nodes()
        F = bm.zeros((NN, GD), dtype=bm.float64)
        
        # 在节点1和节点2施加载荷 [0, 400, -100]
        F[0] = bm.array([0, 400, -100])
        F[1] = bm.array([0, 400, -100])
        
        return F

    @cartesian
    def is_dirichlet_boundary(self) -> TensorLike:
        """Returns a boolean array indicating which DOFs have Dirichlet BCs.

        Returns:
            TensorLike: Boolean array of shape (NN*GD,).
        """
        NN = self.mesh.number_of_nodes()
        GD = self.GD
        
        is_bd_dof = bm.zeros(NN * GD, dtype=bm.bool)
        
        # 约束节点232-243的所有自由度
        for i in range(12):
                node_idx = i + 232  # 对应节点233-244
                is_bd_dof[node_idx * GD : (node_idx + 1) * GD] = True
    
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
