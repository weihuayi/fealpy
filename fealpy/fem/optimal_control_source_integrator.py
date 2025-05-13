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
    def __init__(self, ph, pde, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched
        self.ph = ph
        self.pde = pde

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
    
    def transform_subtriangle_points(self,quad_points):
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
        # 第2条边 [0.5,0.5,0]
        M1 = bm.array([
            [1, 0, 1/3],
            [0, 1, 1/3],
            [0, 0, 1/3]
        ])
        
        # 第1条边 [0.5,0,0.5]
        M2 = bm.array([
            [0, 1, 1/3],
            [0, 0, 1/3],
            [1, 0, 1/3]
        ])
        
        # 第0条边 [0,0.5,0.5]
        M3 = bm.array([
            [0, 0, 1/3],
            [1, 0, 1/3],
            [0, 1, 1/3]
        ])
        
        # 对每个小三角形应用变换
        subtri1 = bm.dot(quad_points, M1.T)  # 第一个小三角形
        subtri2 = bm.dot(quad_points, M2.T)  # 第二个小三角形
        subtri3 = bm.dot(quad_points, M3.T)  # 第三个小三角形
        
        # 合并所有积分点
        global_quad_points = bm.stack([subtri1, subtri2, subtri3])      #(3,NQ,3)
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
        bcs_dual = bm.array([[0.0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])
        phi_dual = space.basis(bcs_dual, index=index)
        # 计算每个积分点重心坐标的最小分量的索引
        k_indices_1 = self.find_min_component_indices(global_points[0])
        k_indices_2 = self.find_min_component_indices(global_points[1])
        k_indices_3 = self.find_min_component_indices(global_points[2])
        # 由于bcs2中的点按边0、边1、边2排列
        phi_new_1 = phi_dual[:, k_indices_1, :, :] #[0.5,0.5,0]
        phi_new_2 = phi_dual[:, k_indices_2, :, :] #[0.5,0,0.5]
        phi_new_3 = phi_dual[:, k_indices_3, :, :] #[0,0.5,0.5]
        # phi_new_1: 保留第 3 轴的第 3 分量
        phi_new_1 = phi_new_1[:, :, 2, :]  # 第 3 轴索引从 0 开始，2 表示第 3 分量
        # phi_new_2: 保留第 3 轴的第 2 分量
        phi_new_2 = phi_new_2[:, :, 1, :]  # 1 表示第 2 分量
        # phi_new_3: 保留第 3 轴的第 1 分量
        phi_new_3 = phi_new_3[:, :, 0, :]  # 0 表示第 1 分量
        # 将 phi_new_1, phi_new_2, phi_new_3 堆叠起来
        phi_dual = bm.stack([phi_new_1, phi_new_2, phi_new_3], axis=0)  # 在第 0 轴堆叠
        return bcs, global_points, ws, phi_dual, cm, index
    
    
    def assembly(self,  space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, global_points, ws, phi_dual, cm, index = self.fetch(space)
        pd_function = space.interpolation(self.pde.pd_fun)
        '''
        def source_function1(bcs):
            ps = mesh.bc_to_point(bcs)
            return self.pde.pd_fun(ps)-self.ph(bcs)        
        '''
        def source_function(bcs,index=None):
            return pd_function(bcs)-self.ph(bcs)
        val0 = source_function(global_points[0])
        val1 = source_function(global_points[1])
        val2 = source_function(global_points[2])
        val = bm.stack([val0, val1, val2])
        result = bm.einsum('q, c, icqk, icqk -> ci', ws, 1/3*cm, val, phi_dual)
        return result