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
        x, y, z = bar_nodes[..., 0], bar_nodes[..., 1], bar_nodes[..., 2]
        bars_length = mesh.entity_measure('cell')
        
        # 第一行（轴向单位向量）
        T11 = (x[..., 1] - x[..., 0]) / bars_length
        T12 = (y[..., 1] - y[..., 0]) / bars_length
        T13 = (z[..., 1] - z[..., 0]) / bars_length
        
        # 固定的参考向量 (全局y方向)
        vy = bm.array([0, 1, 0], dtype=bm.float64)
        k1, k2, k3 = vy

        # 第二行（局部y方向）
        A = bm.sqrt((T12 * k3 - T13 * k2)**2 + 
                    (T13 * k1 - T11 * k3)**2 +
                    (T11 * k2 - T12 * k1)**2)

        T21 = -(T12 * k3 - T13 * k2) / A
        T22 = -(T13 * k1 - T11 * k3) / A
        T23 = -(T11 * k2 - T12 * k1) / A

         # 第三行（局部z方向 = 第一行 × 第二行）
        B = bm.sqrt((T12 * T23 - T13 * T22)**2 +
                    (T13 * T21 - T11 * T23)**2 +
                    (T11 * T22 - T12 * T21)**2)
        
        T31 = (T12 * T23 - T13 * T22) / B
        T32 = (T13 * T21 - T11 * T23) / B
        T33 = (T11 * T22 - T12 * T21) / B

        # 构造3x3基础旋转矩阵 T0
        T0 = bm.stack([
                    bm.stack([T11, T12, T13], axis=-1),  # shape: (NC, 3)
                    bm.stack([T21, T22, T23], axis=-1),
                    bm.stack([T31, T32, T33], axis=-1)
                ], axis=1)  # shape: (NC, 3, 3)

        # 构造12x12旋转变换矩阵 R
        O = bm.zeros((NC, 3, 3))
        row1 = bm.concatenate([T0   , O,  O,  O], axis=2)
        row2 = bm.concatenate([O,  T0, O,  O], axis=2)
        row3 = bm.concatenate([O,  O,  T0, O], axis=2)
        row4 = bm.concatenate([O,  O,  O,  T0], axis=2)

        R = bm.concatenate([row1, row2, row3, row4], axis=1)  #shape: (NC, 12, 12)
        return R
    
    @enable_cache
    def fetch(self, space: _FS):
        """Retrieve material and geometric parameters for the 3D Timoshenko beam.
        
        Parameters:
            beam_E(float) : Young's modulus.
            beam_mu(float): shear modulus.
            l(float): Length of the beam element.
            AX(float): Cross-sectional area in the x-direction.
            AY(float): Cross-sectional area in the y-direction.
            AZ(float): Cross-sectional area in the z-direction.
            Iy(float): Moment of inertia about the y-axis.
            Iz(float): Moment of inertia about the z-axis.
            Ix(float): Polar moment of inertia (for torsional effects).
        """
        assert space is self.space  

        beam_E, beam_mu = self.material.beam_E, self.material.beam_mu
        mesh = space.mesh
        
        # 几何参数
        l = mesh.entity_measure('cell')
        beam_Ax, beam_Ay, beam_Az = self.material.beam_cross_section()
        beam_Ix, beam_Iy, beam_Iz = self.material.beam_inertia()

        # 坐标变换矩阵
        R = self._coord_transfrom()

        return beam_E, beam_mu, l, beam_Ax, beam_Ay, beam_Az, beam_Ix, beam_Iy, beam_Iz, R

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        """Construct the stiffness matrix for 3D beam elements.This function computes the (12, 12) stiffness matrix for each element.
            
        Returns:
            Ke(ndarray),The 3D beam element stiffness matrix, shape (NC, 12, 12).
        """
        E, mu, l, Ax, Ay, Az, Ix, Iy, Iz, R = self.fetch(space)
        mesh = space.mesh
        NC = mesh.number_of_cells()
        beam_cells = bm.arange(0, NC - 10)
        
        l = l[beam_cells]
        R = R[beam_cells]
        
        phi_y = 12 * E * Iz / mu / Ay / (l**2) 
        phi_z = 12 * E * Iy / mu / Az / (l**2) 
        
        KE = bm.zeros((NC, 12, 12))
        beam_Ke = bm.zeros((len(beam_cells), 12, 12))

        beam_Ke[:, 0, 0] = E * Ax / l
        beam_Ke[:, 0, 6] = -beam_Ke[:, 0, 0]

        beam_Ke[:, 1, 1] = 12 * E * Iz / (1+phi_y) /(l**3)
        beam_Ke[:, 1, 5] = 6 * E *Iz / (1+phi_y) / (l**2)
        beam_Ke[:, 1, 7] = -beam_Ke[:, 1, 1]
        beam_Ke[:, 1, 11] = -beam_Ke[:, 1, 5]

        beam_Ke[:, 2, 2] = 12 * E * Iy / (1+phi_z) /(l**3)
        beam_Ke[:, 2, 4] = -6 * E *Iy / (1+phi_z) / (l**2)
        beam_Ke[:, 2, 8] = -beam_Ke[:, 2, 2]
        beam_Ke[:, 2, 10] = beam_Ke[:, 2, 4]

        beam_Ke[:, 3, 3] = mu * Ix / l
        beam_Ke[:, 3, 9] = -beam_Ke[:, 3, 3]

        beam_Ke[:, 4, 4] = (4+phi_z) * E * Iy / (1+phi_z) / l
        beam_Ke[:, 4, 8] = 6 * E * Iy / (1+phi_z) / (l**2)
        beam_Ke[:, 4, 10] = (2-phi_z) * E * Iy / (1+phi_z) / l

        beam_Ke[:, 5, 5] = (4+phi_y) * E * Iz / (1+phi_y) / l
        beam_Ke[:, 5, 7] = -6 * E * Iz / (1+phi_y) / (l**2)
        beam_Ke[:, 5, 11] = (2-phi_y) * E * Iz / (1+phi_y) / l

        beam_Ke[:, 6, 6] = beam_Ke[:, 0, 0]
        beam_Ke[:, 7, 7] = -beam_Ke[:, 1, 7]
        beam_Ke[:, 7, 11] = -beam_Ke[:, 1, 11]

        beam_Ke[:, 8, 8] = -beam_Ke[:, 2, 8]
        beam_Ke[:, 8, 10] = -beam_Ke[:, 2, 10]
        beam_Ke[:, 9, 9] = beam_Ke[:, 3, 3]
        beam_Ke[:, 10, 10] = beam_Ke[:, 4, 4]
        beam_Ke[:, 11, 11] = beam_Ke[:, 6, 6]

        # Symmetrize
        for j in range(11):
            for k in range(j + 1, 12):
                beam_Ke[:, k, j] = beam_Ke[:, j, k]

        beam_KE = bm.einsum('cij, cjk, clk -> cil', R, beam_Ke, R)
        
        k_axle = 1.976e6  # Axle stiffness
        
        axle_cells = bm.arange(NC - 10, NC)
        # 坐标变换矩阵
        R = self._coord_transfrom()[axle_cells]

        kx, ky, kz = k_axle, k_axle, k_axle

        K0 = bm.array([[kx, 0, 0],
                      [0, ky, 0],
                      [0, 0, kz]], dtype=bm.float64)
        
        # 转动刚度矩阵（假设远大于平动）
        K_zeros = bm.ones((3, 3), dtype=bm.float64)*kx*1e3
        
        # 水平拼接每一行
        row1 = bm.concatenate(( K0,      K_zeros,  -K0,     K_zeros), axis=1)
        row2 = bm.concatenate(( K_zeros, K_zeros,   K_zeros, K_zeros), axis=1)
        row3 = bm.concatenate((-K0,      K_zeros,    K0,     K_zeros), axis=1)
        row4 = bm.concatenate(( K_zeros, K_zeros,   K_zeros, K_zeros), axis=1)
        Ke = bm.concatenate((row1, row2, row3, row4), axis=0)
        
        # 刚度矩阵
        Ke_batch = bm.repeat(Ke[None, :, :], len(axle_cells), axis=0)  # (NC_axle, 12, 12)
        axle_KE = bm.einsum('cij, cjk, clk -> cil', R, Ke_batch, R)
        
        # 将轴承刚度矩阵与梁刚度矩阵相加
        KE[beam_cells] += beam_KE
        KE[axle_cells] += axle_KE

        return KE