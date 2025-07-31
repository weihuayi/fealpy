from typing import Optional, Literal
from fealpy.typing import TensorLike, Index, _S
from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache)
from fealpy.functionspace.space import FunctionSpace as _FS

from ..material.timoshenko_beam_material import TimoshenkoBeamMaterial


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
        self.type = beam_type.lower()
        self.meterial = material
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]
    
    def _bars_length(self) -> TensorLike:
        """Calculate the length of each beam element."""
        mesh = self.space.mesh
        NC = self.space.number_of_cells()

        node = mesh.entity('node')
        idx = self.space.cell_to_dof()
        barslength = bm.zeros(NC)

        node = self.mesh.entity('node')
        for i in range(NC):
            dof = idx[i]
            x1, y1, z1 = node[dof[0]]
            x2, y2, z2 = node[dof[1]]
            barslength[i] = bm.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return barslength

    def _coord_transfrom(self, x, y,z) -> TensorLike:
        """Construct the coordinate transformation matrix for 3D beam elements."""
        mesh = self.space.mesh
        node = mesh.entity('node')
        NC = self.space.number_of_cells()

         # 构造第一行方向向量（杆轴方向单位向量）
        T11 = (x[1] - x[0]) / self.BarLength
        T12 = (y[1] - y[0]) / self.BarLength
        T13 = (z[1] - z[0]) / self.BarLength


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
        T0 = bm.array([
            [T11, T12, T13],
            [T21, T22, T23],
            [T31, T32, T33]])

        # 构造12x12旋转变换矩阵 R
        O = bm.zeros((3, 3))
        R = bm.block([
            [T0, O,  O,  O],
            [O,  T0, O,  O],
            [O,  O,  T0, O],
            [O,  O,  O,  T0]])
        return R
 
    @variantmethod
    def assembly(self) -> TensorLike:
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
        E = self.material.E
        mu = self.material.mu
        l = self.space.cell_length()

        AX, AY, AZ = self.material.calculate_cross_sectional_areas()
        Iy, Iz, Ix = self.material.calculate_moments_of_inertia()

        R = self._coord_transfrom()

        FY = 12 * E * Iz / mu / AY / (l**2)  # Phi_y
        FZ = 12 * E * Iy / mu / AZ / (l**2)  # Phi_x

        KE = bm.zeros((12, 12))

        KE[0, 0] = E * AX / l
        KE[0, 6] = -KE[0, 0]
        KE[1, 1] = 12 * E * Iz / (1 + FY) / (l**3)
        KE[1, 5] = 6 * E * Iz / (1 + FY) / (l**2)
        KE[1, 7] = -KE[1, 1]
        KE[1, 11] = KE[1, 5]
        KE[2, 2] = 12 * E * Iy / (1 + FZ) / (l**3)
        KE[2, 4] = -6 * E * Iy / (1 + FZ) / (l**2)
        KE[2, 8] = -KE[2, 2]
        KE[2, 10] = KE[2, 4]
        KE[3, 3] = mu * Ix / l
        KE[3, 9] = -KE[3, 3]
        KE[4, 4] = (4 + FZ) * E * Iy / (1 + FZ) / l
        KE[4, 8] = 6 * E * Iy / (1 + FZ) / (l**2)
        KE[4, 10] = (2 - FZ) * E * Iy / (1 + FZ) / l
        KE[5, 5] = (4 + FY) * E * Iz / (1 + FY) / l
        KE[5, 7] = -6 * E * Iz / (1 + FY) / (l**2)
        KE[5, 11] = (2 - FY) * E * Iz / (1 + FY) / l
        KE[6, 6] = KE[0, 0]
        KE[7, 7] = -KE[1, 7]
        KE[7, 11] = -KE[1, 11]
        KE[8, 8] = -KE[2, 8]
        KE[8, 10] = -KE[2, 10]
        KE[9, 9] = KE[3, 3]
        KE[10, 10] = KE[4, 4]
        KE[11, 11] = KE[5, 5]

        # Symmetry of KE matrix
        for j in range(11):
            for i in range(j + 1, 12):
                KE[i, j] = KE[j, i]

        Ke = R.T @ KE @ R  # Matrix multiplication with R

        return Ke