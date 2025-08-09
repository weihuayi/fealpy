from typing import Optional

from fealpy.typing import Tuple, TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesh import EdgeMesh

from ...material import TimoshenkoBeamMaterial


class TimoshenkoBeamData3D:
    """
    3D Timoshenko beam problem.
    """

    def __init__(self):
        """
        A data structure class representing beam parameters and properties, including geometry characteristics.

        Parameters:
            para(Tensor): A list or tensor containing the axle structure parameters: [phi diameter, length, number of segments].
            D(float): Diameter of the beam.
            L(float): Length of the beam.
            FSY(float): Shear correction factor in the y-direction.
            FSZ(float): Shear correction factor in the z-direction.
            AX(float): Cross-sectional area in the x-direction.
            AY(float): Cross-sectional area in the y-direction.
            AZ(float): Cross-sectional area in the z-direction.
            Iy(float): Moment of inertia about the y-axis.
            Iz(float): Moment of inertia about the z-axis.
            Ix(float): Polar moment of inertia (for torsional effects).
            mesh(EdgeMesh): An `EdgeMesh` object representing the geometry of the beam.
            k_lunzhou(float): Equivalent node stiffness for the axle, which characterizes the rigidity of the beam.
        
        Notes:
            FSY and FSZ: The shear correction factor, 6/5 for rectangular and 10/9 for circular.
        """
        
        self.para = bm.array([
            [120, 141, 2],
            [150, 28, 2],
            [184, 177, 4], 
            [160, 268, 2],
            [184.2, 478, 2], 
            [160, 484, 2], 
            [184, 177, 4],
            [150, 28, 2], 
            [120, 141, 2],
            [1.976e6 , 100, 10]
        ])
        self.L = bm.sum(self.para[:, 1])
        self._D = bm.repeat(self.para[:, 0], self.para[:, 2].astype(int))
        self._FSY = 10/9
        self._FSZ = 10/9
        self.mesh = self.init_mesh()
        
        self._AX, self._AY, self._AZ = self._cross_sectional_areas()
        self._Iy, self._Iz, self._Ix = self._moments_of_inertia()
        
    def __str__(self) -> str:
        pass
    
    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 3
    
    def D(self) -> TensorLike:
        """the diameter of the wheel and axle."""
        return self._D
    
    def _cross_sectional_areas(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Cross-sectional area of the beam element."""
        AX = bm.pi * self._D**2 / 4
        AY = AX / self._FSY
        AZ = AX / self._FSZ

        return AX, AY, AZ
    
    def _moments_of_inertia(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Moment of inertia for the beam element."""
        Iy  = bm.pi * self._D**4 / 64
        Iz = Iy
        Ix = Iy + Iz

        return Ix, Iy, Iz
    
    def create_material(self, name='timo', E=None, nu=None):
        if E is None or nu is None:
            raise ValueError("Elastic modulus (E) and Poisson ratio (nu) must be provided externally.")
        
        material = TimoshenkoBeamMaterial(name=name,model=self, 
                                          elastic_modulus=E, poisson_ratio=nu)
        return material
    
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
    def external_load(self) -> TensorLike:
        """The load applied to the node.
        Notes:
            Each node has 6 DOFs: [u, v, w, θx, θy, θz].
            dof_map = {'u': 0,'v': 1,'w': 2,'θx': 3,'θy': 4,'θz': 5}
        """
        NN = self.mesh.number_of_nodes()
        dofs_per_node = 6
        n_dofs = NN * dofs_per_node
        F = bm.zeros(n_dofs)

        external_load = bm.array([-88200, 3140, 1.4e6, -88200])

        F[1 * dofs_per_node + 2] = external_load[0]
        F[11 * dofs_per_node] = external_load[1]
        F[11 * dofs_per_node + 3] = external_load[2]
        F[21 * dofs_per_node + 2] = external_load[3]

        return F 
    
    def dirichlet_dof_index(self) -> TensorLike:
        """Dirichlet boundary conditions are applied.

        Returns:
            A 1D tensor containing the global DOF indices corresponding to 
        the fixed boundary nodes.

        Notes:
            Each node has 6 DOFs: [u, v, w, θx, θy, θz].
            Global DOF index for a given node `n` and local DOF `i` is calculated as `n * 6 + i`.
            The fixed nodes here are nodes 23 through 32 inclusive.
        """
        fixed_nodes = bm.arange(23, 33) # 节点23到32
        dofs_per_node = 6

        bd_idx = []
        for node  in fixed_nodes:
            for i in range(dofs_per_node):
                bd_idx.append(node * dofs_per_node + i)
        return bm.array(bd_idx)
    
    @cartesian
    def dirichlet(self):
        """Compute the Dirichlet boundary condition."""
        return 0