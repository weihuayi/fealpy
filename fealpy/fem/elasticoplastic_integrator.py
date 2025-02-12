from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from ..mesh import HomogeneousMesh, SimplexMesh, TensorMesh
from ..functionspace.space import FunctionSpace as _FS
from ..functionspace.tensor_space import TensorFunctionSpace as _TS
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod
)
from fealpy.fem.utils import SymbolicIntegration
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator

class TransitionElasticIntegrator(LinearElasticIntegrator):
    def __init__(self, D_ep, material, q, method=None):
        # 传递 method 参数并调用父类构造函数
        super().__init__(material, q, method=method)
        self.D_ep = D_ep  # 弹塑性材料矩阵

    def assembly(self, space, mesh, cellidx):
        # 获取单元信息
        cell = mesh.entity('cell', cellidx)
        NC = len(cellidx)
        gphi = space.grad_basis(bcs, cellidx=cellidx)  # (NC, NQ, ldof, GD)
        
        # 获取当前单元对应的弹塑性矩阵
        cell_D_ep = self.D_ep[cellidx]  # (NC, NQ, N, N)
        
        # 使用Voigt记法的组装
        mesh = getattr(space, 'mesh', None)
        cm, ws, detJ, D, B = self.fetch_voigt_assembly(space)
        
        if isinstance(mesh, TensorMesh):
            KK = bm.einsum('q, cq, cqki, cqkl, cqlj -> cij',
                            ws, detJ, B, D_ep, B)
        else:
                KK = bm.einsum('q, c, cqki, cqkl, cqlj -> cij',
                                ws, cm, B, D_ep, B)
        
        return K_cell.sum(axis=1)  # (NC, tdof, tdof)
        
    @assemblymethod('standard')
    def assembly_standard(self, space, mesh, cellidx):
        # 获取单元信息
        cell = mesh.entity('cell', cellidx)
        NC = len(cellidx)
        gphi = space.grad_basis(bcs, cellidx=cellidx)  # (NC, NQ, ldof, GD)

        # 获取当前单元对应的弹塑性矩阵
        cell_D_ep = self.D_ep[cellidx]  # (NC, NQ, N, N)

        #使用标准的线性弹性组装
        B = self.material.strain_matrix(True, gphi)  # (NC, NQ, 3/6, tdof)
        K_cell = bm.einsum('nqij,nqjk,nqkl,nq->nil', 
                            B.transpose(0,1,3,2), 
                            cell_D_ep, 
                            B, 
                            ws * mesh.cell_volume()[cellidx, None])
        
        return K_cell.sum(axis=1)  # (NC, tdof, tdof)