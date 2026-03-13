from typing import Optional, Literal
from fealpy.typing import TensorLike, Index, _S
from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache)
from fealpy.functionspace.space import FunctionSpace as _FS

from ..utils import CoordTransform


class AxleIntegrator(LinearInt, OpInt, CellInt):
    """Integrator for axle problems."""

    def __init__(self, 
                 space: _FS,
                 model, 
                 material, 
                 index: Index=_S,
                 method: Optional[str]=None )-> None:
        super().__init__()

        self.space = space
        self.model = model
        self.material = material
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]
    
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        """Construct the stiffness matrix for 3D beam elements.This function computes the (12, 12) stiffness matrix for each element.
            
        Returns:
            Ke(ndarray),The 3D beam element stiffness matrix, shape (NC, 12, 12).
        """
        assert space is self.space  

        mesh = space.mesh
        cells = bm.arange(mesh.number_of_cells()) if self.index is _S else self.index
        NC = len(cells)
        
        k_axle = getattr(self.material, "k_axle", 1.976e6) # Axle stiffness
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
        Ke = bm.concatenate((row1, row2, row3, row4), axis=0) # (12,12)
        
        # 刚度矩阵
        coord_trans = CoordTransform(method='beam3d')
        R = coord_trans.coord_transform_beam3d(mesh, vref=[0, 1, 0], index=cells)
        KE = bm.einsum('cji, ...jk, ckl -> cil', R, Ke, R)
       
        return KE