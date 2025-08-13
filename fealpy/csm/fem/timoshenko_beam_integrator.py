from typing import Optional, Literal
from fealpy.typing import TensorLike, Index, _S
from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache)
from fealpy.functionspace.space import FunctionSpace as _FS


class TimoshenkoBeamIntegrator(LinearInt, OpInt, CellInt):
    """
    Integrator for Timoshenko beam problems.
    """

    def __init__(self, 
                 space: _FS, 
                 material, 
                 index: Index=_S,
                 method: Optional[str]=None )-> None:
        super().__init__()

        self.space = space
        self.material = material
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    def _coord_transfrom(self) -> TensorLike:
        """Construct the coordinate transformation matrix for 3D beam elements."""
        mesh = self.space.mesh
        node= mesh.entity('node')
        cell = mesh.entity('cell')
        bar_nodes = node[cell]
        
        NC = mesh.number_of_cells()
        x = bar_nodes[..., 0]
        y = bar_nodes[..., 1]
        z = bar_nodes[..., 2]

        bars_length = mesh.entity_measure('cell')
        
        # 构造第一行方向向量（杆轴方向单位向量）
        T11 = (x[..., 1] - x[..., 0]) / bars_length
        T12 = (y[..., 1] - y[..., 0]) / bars_length
        T13 = (z[..., 1] - z[..., 0]) / bars_length

        vy = bm.array([0, 1, 0], dtype=bm.float64)
        k1, k2, k3 = vy

        # 计算第二行方向向量（垂直于杆轴方向的 y 方向局部坐标单位向量）
        A = bm.sqrt((T12 * k3 - T13 * k2)**2 + 
                    (T13 * k1 - T11 * k3)**2 +
                    (T11 * k2 - T12 * k1)**2)

        T21 = -(T12 * k3 - T13 * k2) / A
        T22 = -(T13 * k1 - T11 * k3) / A
        T23 = -(T11 * k2 - T12 * k1) / A

        # 计算第三行方向向量（通过叉乘第一和第二行得到 z 方向单位向量）
        B = bm.sqrt((T12 * T23 - T13 * T22)**2 +
                    (T13 * T21 - T11 * T23)**2 +
                    (T11 * T22 - T12 * T21)**2)
        
        T31 = (T12 * T23 - T13 * T22) / B
        T32 = (T13 * T21 - T11 * T23) / B
        T33 = (T11 * T22 - T12 * T21) / B

        # 构造3x3基础旋转矩阵 T0
        T0 = bm.stack([
                    bm.stack([T11, T12, T13], axis=-1),  # shape: (32, 3)
                    bm.stack([T21, T22, T23], axis=-1),
                    bm.stack([T31, T32, T33], axis=-1)
                ], axis=1)  # shape: (32, 3, 3)

        # 构造12x12旋转变换矩阵 R
        O = bm.zeros((NC, 3, 3))
        row1 = bm.concatenate([T0   , O,  O,  O], axis=2)
        row2 = bm.concatenate([O,  T0, O,  O], axis=2)
        row3 = bm.concatenate([O,  O,  T0, O], axis=2)
        row4 = bm.concatenate([O,  O,  O,  T0], axis=2)

        R = bm.concatenate([row1, row2, row3, row4], axis=1)  #shape: (32, 12, 12)
        return R
    
    @enable_cache
    def fetch(self, space: _FS):
        """Retrieve material and geometric parameters for the 3D Timoshenko beam.
        Parameters:
            E(float) : Young's modulus.
            mu(float): shear modulus.
            l(float): Length of the beam element.
            AX(float): Cross-sectional area in the x-direction.
            AY(float): Cross-sectional area in the y-direction.
            AZ(float): Cross-sectional area in the z-direction.
            Iy(float): Moment of inertia about the y-axis.
            Iz(float): Moment of inertia about the z-axis.
            Ix(float): Polar moment of inertia (for torsional effects).
        """
        assert space is self.space  

        beam_E, beam_mu = self.material.E, self.material.mu
        axle_E, axle_mu = self.material.axle_material_paras()

        mesh = space.mesh
        NC = mesh.number_of_cells()
        
        axle_cells = bm.arange(NC - 10, NC)
        
        # 构造每个单元的E和mu数组
        E = bm.full(NC, beam_E, dtype=float)
        mu = bm.full(NC, beam_mu, dtype=float)
        E[axle_cells] = axle_E
        mu[axle_cells] = axle_mu
        
        # 几何参数
        l = mesh.entity_measure('cell')
        NC = mesh.number_of_cells()

        Ax, Ay, Az = self.material.cross_sectional_areas()
        Ix, Iy, Iz = self.material.moments_of_inertia()
        
        # 坐标变换矩阵
        R = self._coord_transfrom()
        
        return E, mu, l, Ax, Ay, Az, Ix, Iy, Iz, R
    
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        """Construct the stiffness matrix for 3D beam elements.This function computes the (12, 12) stiffness matrix for each element.
            
        Returns:
            Ke(ndarray),The 3D beam element stiffness matrix, shape (NC, 12, 12).
        """
        E, mu, l, Ax, Ay, Az, Ix, Iy, Iz, R = self.fetch(space)
        mesh = space.mesh
        NC = mesh.number_of_cells()
        
        phi_y = 12 * E * Iz / mu / Ay / (l**2) 
        phi_z = 12 * E * Iy / mu / Az / (l**2) 

        KE = bm.zeros((NC, 12, 12))

        Ke = bm.zeros((NC, 12, 12))

        Ke[:, 0, 0] = E * Ax / l
        Ke[:, 0, 6] = -Ke[:, 0, 0]

        Ke[:, 1, 1] = 12 * E * Iz / (1+phi_y) /(l**3)
        Ke[:, 1, 5] = 6 * E *Iz / (1+phi_y) / (l**2)
        Ke[:, 1, 7] = -Ke[:, 1, 1]
        Ke[:, 1, 11] = -Ke[:, 1, 5]
        

        Ke[:, 2, 2] = 12 * E * Iy / (1+phi_z) /(l**3)
        Ke[:, 2, 4] = -6 * E *Iy / (1+phi_z) / (l**2)
        Ke[:, 2, 8] = -Ke[:, 2, 2]
        Ke[:, 2, 10] = Ke[:, 2, 4]

        Ke[:, 3, 3] = mu * Ix / l
        Ke[:, 3, 9] = -Ke[:, 3, 3]

        Ke[:, 4, 4] = (4+phi_z) * E * Iy / (1+phi_z) / l
        Ke[:, 4, 8] = 6 * E * Iy / (1+phi_z) / (l**2)
        Ke[:, 4, 10] = (2-phi_z) * E * Iy / (1+phi_z) / l

        Ke[:, 5, 5] = (4+phi_y) * E * Iz / (1+phi_y) / l
        Ke[:, 5, 7] = -6 * E * Iz / (1+phi_y) / (l**2)
        Ke[:, 5, 11] = (2-phi_y) * E * Iz / (1+phi_y) / l

        Ke[:, 6, 6] = Ke[:, 0, 0]
        Ke[:, 7, 7] = -Ke[:, 1, 7]
        Ke[:, 7, 11] = -Ke[:, 1, 11]

        Ke[:, 8, 8] = -Ke[:, 2, 8]
        Ke[:, 8, 10] = -Ke[:, 2, 10]
        Ke[:, 9, 9] = Ke[:, 3, 3]
        Ke[:, 10, 10] = Ke[:, 4, 4]
        Ke[:, 11, 11] = Ke[:, 6, 6]
        
        # Symmetrize
        for j in range(11):
            for k in range(j + 1, 12):
                Ke[:, k, j] = Ke[:, j, k]
                   
        KE = bm.einsum('cij, cjk, clk -> cil', R, Ke, R)

        return KE