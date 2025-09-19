from fealpy.typing import Tuple, TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesh import EdgeMesh


class TimobeamAxleData3D:
    """
        A data structure class representing beam and axle parameters and properties, including geometry characteristics.

        Parameters:
            para(TensorLike): A list or tensor containing the axle structure parameters: [phi diameter, length, number of segments].
            D(float): Diameter of the beam and lunzhou.
            FSY(float): Shear correction factor in the y-direction.
            FSZ(float): Shear correction factor in the z-direction.
            beam_E, beam_nu: beam elastic modulus & poisson_ratio.
            axle_E, axle_mu: axle elastic modulus & shear modulus.
            dofs_per_node(int): Each node has 6 DOFs,[u, v, w, θx, θy, θz].
        
        Notes:
            FSY and FSZ: The shear correction factor, 6/5 for rectangular and 10/9 for circular.
    """
    def __init__(self, para: TensorLike=None, 
                 FSY: float=10/9, FSZ: float=10/9):
        self.beam_para = bm.array([
            [120, 141, 2], [150, 28, 2], [184, 177, 4], [160, 268, 2],
            [184.2, 478, 2], [160, 484, 2], [184, 177, 4], [150, 28, 2],
            [120, 141, 2]], dtype=bm.float64)
        self.axle_para =  bm.array([[1.976e6, 100, 10]], dtype=bm.float64)
    
        # self.para = bm.concatenate((self.beam_para, self.axle_para), axis=0)
        
        # diameter
        self.beam_D = bm.repeat(self.beam_para[:, 0], self.beam_para[:, 2].astype(int))
        self.axle_D = bm.repeat(self.axle_para[:, 0], self.axle_para[:, 2].astype(int))
        
        self.FSY = FSY
        self.FSZ = FSZ
        
        # === 计算 beam 截面 & 惯性矩 ===
        self.beam_Ax, self.beam_Ay, self.beam_Az = self.calculate_beam_cross_section()
        self.beam_Ix, self.beam_Iy, self.beam_Iz = self.calculate_beam_inertia()
        
        self.dofs_per_node = 6
        self.mesh = self.init_mesh() 
        
    def __str__(self) -> str:
        """Returns a formatted multi-line string summarizing the configuration of the 3D Timoshenko beam data.
        
        Returns:
            str: A multi-line string showing key beam parameters, geometry, and mesh info.
    """
        s = f"{self.__class__.__name__}(\n"
        s += "  === Geometry & Mesh ===\n"
        s += f"  Total Length (L)      : {float(self.L):.3f} mm\n"
        s += f"  Diameters (D)         : {self._D.tolist()} mm\n"
        s += f"  Mesh Type             : {self.mesh.__class__.__name__}\n"
        s += f"  Number of Nodes       : {self.mesh.number_of_nodes()}\n"
        s += f"  Number of Elements    : {self.mesh.number_of_cells()}\n"
        s += f"  Geo Dimension         : {self.geo_dimension()}\n"
        s += f"  Shear Factors     : {self.FSY}, {self.FSZ}\n"
        s += f"  beam_Ax, beam_Ay, beam_Az : {self.beam_Ax[:5].tolist()} ...\n"
        s += f"  beam_Ix, beam_Iy, beam_Iz : {self.beam_Ix[:5].tolist()} ...\n"
        s += ")"
        return s
    
    def geo_dimension(self) -> int:
        """the geometric dimension."""
        return 3
    
    def shear_factors(self) -> Tuple[float, float]:
        """Shear correction factor in the y-direction and z-direction."""
        return self.FSY, self.FSZ
    
    def calculate_beam_cross_section(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Beam cross-sectional areas."""
        beam_Ax = bm.pi * self.beam_D**2 / 4
        beam_Ay = beam_Ax / self.FSY
        beam_Az = beam_Ax / self.FSZ
        return beam_Ax, beam_Ay, beam_Az
    
    def calculate_beam_inertia(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
       """Beam moments of inertia."""
       beam_Iy  = bm.pi * self.beam_D**4 / 64
       beam_Iz = beam_Iy
       beam_Ix = beam_Iy + beam_Iz
       return beam_Ix, beam_Iy, beam_Iz
    
    def init_mesh(self):
        """Construct a mesh for the beam and axle.

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
        
        mesh = EdgeMesh(node, cell)
        beam_props = {
            "Ax": self.beam_Ax,
            "Ay": self.beam_Ay,
            "Az": self.beam_Az,
            "Ix": self.beam_Ix,
            "Iy": self.beam_Iy,
            "Iz": self.beam_Iz,
        }
        
        for key, value in beam_props.items():
            arr = bm.zeros(22)
            arr[:22] = value
            mesh.celldata[key] = arr
        return mesh
    
    @cartesian
    def external_load(self) -> TensorLike:
        """Node-based concentrated external loads for the Timoshenko beam and Axle.
        
        Notes:
            Each node has 6 degrees of freedom (DOFs): [u, v, w, θx, θy, θz].
            dof_map = {'u': 0,'v': 1,'w': 2,'θx': 3,'θy': 4,'θz': 5}.
            
            Concentrated load definition:
                - Node 1: Fz = -88200
                - Node 11: Fx = 3140, Mx = 1.4e6
                - Node 21: Fz = -88200

        Returns:
            F(TensorLike):
                Global nodal force vector, with concentrated loads applied
                at the specified nodes.
        """
        NN = self.mesh.number_of_nodes()
        n_dofs = NN * self.dofs_per_node
        F = bm.zeros(n_dofs)

        external_load = bm.array([-88200, 3140, 1.4e7, -88200], dtype=bm.float64)

        F[1 * self.dofs_per_node + 2] = external_load[0]
        F[11 * self.dofs_per_node] = external_load[1]
        F[11 * self.dofs_per_node + 3] = external_load[2]
        F[21 * self.dofs_per_node + 2] = external_load[3]

        return F 
    
    @cartesian
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
        node_dofs = []
        for node in fixed_nodes:               # 遍历固定的节点
            for i in range(self.dofs_per_node): # 遍历该节点的所有自由度
                dof_index = node * self.dofs_per_node + i
                node_dofs.append(dof_index)     # 加入全局 DOF 列表
        
        return bm.array(node_dofs)
    
    @cartesian
    def dirichlet(self, x: TensorLike):
        """Dirichlet boundary condition function that accepts boundary integration point coordinates x,
        and returns the boundary values at the corresponding points with shape (N, dofs_per_node).
        
        Parameters:
            x (TensorLike): Coordinates of boundary integration points, shape (N, dim).
        
        Returns:
            TensorLike: Boundary values at the points, shape (N, dofs_per_node).
        """
        N = x.shape[0]  # 积分点数量
        return bm.zeros((N, self.dofs_per_node))
