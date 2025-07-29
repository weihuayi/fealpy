from fealpy.typing import Optional, TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian

from fealpy.material import (
        LinearElasticMaterial,
        )
from fealpy.mesh import EdgeMesh

class TimoshenkoBeamData3D:
    """
    3D Timoshenko beam problem.
    """

    def __init__(self):
        """
        Initialize beam parameters.
        """
        # Axle structure parameters [phi diameter, length, number of segments]
        self.para = bm.array([
            [120, 141, 2],
            [150, 28, 2],
            [184, 177, 4], 
            [160, 268, 2],
            [184.2, 478, 2], 
            [160, 484, 2], 
            [184, 177, 4],
            [150, 28, 2], 
            [120, 141, 2]
        ])
        self.L = bm.sum(self.para[:, 1])  # 总长度
        self.mesh = self.init_mesh()

    @property
    def E(self, p: Optional[TensorLike] = None) -> TensorLike:
        # Young's modulus in Pascals
        return 2.07e11 
    
    @property
    def nu(self, p: Optional[TensorLike] = None) -> TensorLike:
        # Poisson's ratio
        return 0.276 
    
    @property
    def k_lunzhou(self, p: Optional[TensorLike] = None) -> TensorLike:
        #N/mm equivalent node stiffness for the axle, identical in all three translational directions.
        return 1.976e6 

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 3
    
    def domain(self):
        """Return the computational domain [xmin, xmax]."""
        return [0.0, self.L]

    def init_mesh(self):
        """
        Construct a 3D mesh for the Timoshenko beam domain.

        Returns:
            EdgeMesh: 3D mesh with nodes and cells for the beam domain.
        """
        node = bm.array([[0.0, 0.0, 0.0], [70.5, 0.0, 0.0], [141.0, 0.0, 0.0], [155.0, 0.0, 0.0],
            [169.0, 0.0, 0.0], [213.25, 0.0, 0.0], [257.5, 0.0, 0.0], [301.75, 0.0, 0.0],
            [346.0, 0.0, 0.0], [480.0, 0.0, 0.0], [614.0, 0.0, 0.0], [853.0, 0.0, 0.0],
            [1092.0, 0.0, 0.0], [1334.0, 0.0, 0.0], [1576.0, 0.0, 0.0], [1620.25, 0.0, 0.0],
            [1664.5, 0.0, 0.0], [1708.75, 0.0, 0.0], [1753.0, 0.0, 0.0], [1767.0, 0.0, 0.0],
            [1781.0, 0.0, 0.0], [1851.5, 0.0, 0.0],[1922.0, 0.0, 0.0], [169.0, 0.0, -100.0],
            [213.25, 0.0, -100.0], [257.5, 0.0, -100.0], [301.75, 0.0, -100.0], [346.0, 0.0, -100.0],
            [1576.0, 0.0, -100.0], [1620.25, 0.0, -100.0], [1664.5, 0.0, -100.0], 
            [1708.75, 0.0, -100.0], [1753.0, 0.0, -100.0]], dtype=bm.float64)

        cell = bm.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
             [5, 6], [6, 7], [7, 8],[8, 9], [9, 10], [10, 11],
             [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
             [16, 17],[17, 18], [18, 19], [19, 20], [20, 21], 
             [21, 22], [4, 23], [5, 24], [6, 25], [7, 26], [8, 27], 
             [14, 28], [15, 29],[16, 30], [17, 31], [18,32]], dtype=bm.int32)
        
        return EdgeMesh(node, cell)
    
    @cartesian
    def load(self):
        """
        The load applied to the node.
        load = [node_index, x, y, z, theta_x theta_y theta_z]
        """
        x = self.mesh.node[:, 0]
        dofs = 6 * len(x)
        load = bm.array([[1, 0, 0, -88200, 0, 0 , 0],
                         [21, 0, 0, -88200, 0, 0, 0],
                         [11, 3140, 0, 0, 14000e3, 0, 0]], dtype=bm.float64)

        return load
    
    
    def dirichlet_dof_index(self) -> TensorLike:
        """
        Return the indices of degrees of freedom (DOFs) where Dirichlet boundary conditions are applied.
        """
        return bm.array([23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
    
    @cartesian
    def dirichlet(self, x):
        """
        Compute the Dirichlet boundary condition.
        """
        return 0