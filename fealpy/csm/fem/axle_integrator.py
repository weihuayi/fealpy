from typing import Optional, Literal
from fealpy.typing import TensorLike, Index, _S
from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache)
from fealpy.functionspace.space import FunctionSpace as _FS


class AxleIntegrator(LinearInt, OpInt, CellInt):
    """
    Integrator for axle problems.
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
    
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        """Construct the stiffness matrix for 3D beam elements.This function computes the (12, 12) stiffness matrix for each element.
            
        Returns:
            Ke(ndarray),The 3D beam element stiffness matrix, shape (NC, 12, 12).
        """
        assert space is self.space  

        mesh = space.mesh
        NC = mesh.number_of_cells()
        cells = bm.arange(NC - 10, NC)
        
        k_axle = 1.976e6  # Axle stiffness

        R = self._coord_transfrom()[cells] # 坐标变换矩阵
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
        Ke_batch = bm.repeat(Ke[None, :, :], len(cells), axis=0)  # (NC_axle, 12, 12)
        KE = bm.einsum('cij, cjk, clk -> cil', R, Ke_batch, R)
        return KE