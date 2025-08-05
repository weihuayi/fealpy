from typing import Optional, Literal
from fealpy.typing import TensorLike, Index, _S
from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache)
from fealpy.functionspace.space import FunctionSpace as _FS

# from ..material.timoshenko_beam_material import TimoshenkoBeamMaterial


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
        A = bm.sqrt((T12 * k3 - T13 * k2)**2 + (T13 * k1 - T11 * k3)**2 + (T11 * k2 - T12 * k1)**2)

        T21 = -(T12 * k3 - T13 * k2) / A
        T22 = -(T13 * k1 - T11 * k3) / A
        T23 = -(T11 * k2 - T12 * k1) / A

        # 计算第三行方向向量（通过叉乘第一和第二行得到 z 方向单位向量）
        B = bm.sqrt((T12 * T23 - T13 * T22)**2 + (T13 * T21 - T11 * T23)**2 + (T11 * T22 - T12 * T21)**2)
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
 
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        """Construct the stiffness matrix for 3D beam elements.This function computes the (12, 12) stiffness matrix for each element.
        
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
            
        Returns:
            Ke(ndarray),The 3D beam element stiffness matrix, shape (NC, 12, 12).
        """
        assert space is self.space  
        E = self.material.E
        mu = self.material.mu

        mesh = space.mesh
        bar_length = mesh.entity_measure('cell')
        NC = mesh.number_of_cells()

        AX, AY, AZ = self.material.calculate_cross_sectional_areas()
        Iy, Iz, Ix = self.material.calculate_moments_of_inertia()

        R = self._coord_transfrom()

        FY = 12 * E * Iz / mu / AY / (bar_length**2)  # Phi_y
        FZ = 12 * E * Iy / mu / AZ / (bar_length**2)  # Phi_x

        KE = bm.zeros((NC, 12, 12))

        for i in range(NC):
            ke = bm.zeros((12, 12))
            li = bar_length[i]
            axi, ayi, azi =  AX[i], AY[i], AZ[i]
            iyi, izi, ixi = Iy[i], Iz[i], Ix[i]
            fy = FY[i]
            fz = FZ[i]

            ke[0, 0] = E * axi / li
            ke[0, 6] = -ke[0, 0]
            ke[1, 1] = 12 * E * izi / (1 + fy) / (li**3)
            ke[1, 5] = 6 * E * izi / (1 + fy) / (li**2)
            ke[1, 7] = -ke[1, 1]
            ke[1, 11] = ke[1, 5]
            ke[2, 2] = 12 * E * iyi / (1 + fz) / (li**3)
            ke[2, 4] = -6 * E * iyi / (1 + fz) / (li**2)
            ke[2, 8] = -ke[2, 2]
            ke[2, 10] = ke[2, 4]
            ke[3, 3] = mu * ixi / li
            ke[3, 9] = -ke[3, 3]
            ke[4, 4] = (4 + fz) * E * iyi / (1 + fz) / li
            ke[4, 8] = 6 * E * iyi / (1 + fz) / (li**2)
            ke[4, 10] = (2 - fz) * E * iyi / (1 + fz) / li
            ke[5, 5] = (4 + fy) * E * izi / (1 + fy) / li
            ke[5, 7] = -6 * E * izi / (1 + fy) / (li**2)
            ke[5, 11] = (2 - fy) * E * izi / (1 + fy) / li
            ke[6, 6] = ke[0, 0]
            ke[7, 7] = -ke[1, 7]
            ke[7, 11] = -ke[1, 11]
            ke[8, 8] = -ke[2, 8]
            ke[8, 10] = -ke[2, 10]
            ke[9, 9] = ke[3, 3]
            ke[10, 10] = ke[4, 4]
            ke[11, 11] = ke[5, 5]

            # Symmetrize
            for j in range(11):
                for k in range(j + 1, 12):
                    ke[k, j] = ke[j, k]

            # Apply transformation
            KE[i] = R[i].T @ ke @ R[i]

        return KE