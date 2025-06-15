from typing import Optional
from functools import partial

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.utils import process_coef_func, is_scalar, is_tensor, fill_axis
from fealpy.functional import bilinear_integral, linear_integral, get_semilinear_coef
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod,
    CoefLike
)


class OPCSIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, source: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.source = source
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]
    
    def find_min_component_indices(self,coords):
        indices = []
        for coord in coords:
            # 找到最小值及其索引
            min_val = bm.min(coord)
            min_indices = bm.where(coord == min_val)[0]
            
            if len(min_indices) == 1:
                # 如果只有一个最小值，直接使用其索引
                index = min_indices[0]
            else:
                # 找到最大值及其索引
                max_val_index = bm.argmax(coord)
                
                # 计算下一个索引（循环）
                index = (max_val_index + 1) % len(coord)
            
            indices.append(index)
        return indices
    
    def transform_subtriangle_points(self, quad_points):
        """
        将每个小三角形的积分点转换到原大三角形的全局坐标中。
        
        参数:
            quad_points (ndarray): 小三角形的积分点数组，形状为(NQ, 3)，使用重心坐标。
            
        返回:
            global_quad_points (ndarray): 大三角形中的全局积分点数组，形状为(3*NQ, 3)。
            subtri1 (ndarray): 第一个小三角形的全局积分点，形状为(NQ, 3)。
            subtri2 (ndarray): 第二个小三角形的全局积分点，形状为(NQ, 3)。
            subtri3 (ndarray): 第三个小三角形的全局积分点，形状为(NQ, 3)。
        """
        # 定义变换矩阵
        # 第0条边
        M0 = bm.array([
            [0, 0, 1/3],
            [1, 0, 1/3],
            [0, 1, 1/3]
        ])
    
        
        # 第1条边
        M1 = bm.array([
            [0, 1, 1/3],
            [0, 0, 1/3],
            [1, 0, 1/3]
        ])
        # 第2条边
        M2 = bm.array([
            [1, 0, 1/3],
            [0, 1, 1/3],
            [0, 0, 1/3]
        ])


        # 对每个小三角形应用变换
        subtri1 = bm.dot(quad_points, M0.T)  # 第一个小三角形
        subtri2 = bm.dot(quad_points, M1.T)  # 第二个小三角形
        subtri3 = bm.dot(quad_points, M2.T)  # 第三个小三角形

        # 合并所有积分点
        global_quad_points = bm.stack([subtri1, subtri2, subtri3])
        return global_quad_points
    
    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        global_points =self.transform_subtriangle_points(bcs)
        bcs_dual = bm.array([[0, 0.5, 0.5],[0.5,0, 0.5],[0.5,0.5,0]])
        phi_dual = space.basis(bcs_dual, index=index)
        #phi_new_1: 第0个基函数  
        phi_new_1 = phi_dual[:, 0, 0, :]  
        # phi_new_2: 第1个基函数
        phi_new_2 = phi_dual[:, 1, 1, :]  
        # phi_new_3: 第2个基函数
        phi_new_3 = phi_dual[:, 2, 2, :]  
        # 将 phi_new_1, phi_new_2, phi_new_3 堆叠起来
        phi_dual = bm.stack([phi_new_1, phi_new_2, phi_new_3], axis=1) 
        phi_dual = phi_dual[:,bm.newaxis,:,:] 
        return bcs, global_points, ws, 2*phi_dual, cm, index
    
    
    def assembly(self,  space: _FS) -> TensorLike:
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, global_points, ws, phi_dual, cm, index = self.fetch(space)
        val0 = process_coef_func(f, bcs=global_points[0], mesh=mesh, etype='cell', index=index)
        val1 = process_coef_func(f, bcs=global_points[1], mesh=mesh, etype='cell', index=index)
        val2 = process_coef_func(f, bcs=global_points[2], mesh=mesh, etype='cell', index=index)
        val = bm.stack([val0, val1, val2])
        if isinstance(f, int):
            result = bm.einsum('q,c,i,cqik->ci', ws, 1/3*cm, val, phi_dual)
        else:
            result = bm.einsum('q, c, icqk, cqik -> ci', ws, 1/3*cm, val, phi_dual)
        return result