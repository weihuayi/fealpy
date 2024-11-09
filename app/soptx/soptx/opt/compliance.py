from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.opt.base import ObjectiveBase

class ComplianceObjective(ObjectiveBase):
    """结构柔度最小化目标函数"""
    
    def __init__(self, 
                tensor_space,
                material_properties,
                ke0,
                filter_properties=None):
        """初始化柔度目标函数"""
        self.space = tensor_space
        self.material_properties = material_properties
        self.ke0 = ke0
        self.filter_properties = filter_properties
        
    def compute_element_compliance(self, rho: TensorLike, uh: TensorLike) -> TensorLike:
        """计算单元柔度"""
        self.material_properties.rho = rho
        
        cell2ldof = self.space.cell_to_dof()
        uhe = uh[cell2ldof]
        
        ce = bm.einsum('ci, cik, ck -> c', uhe, self.ke0, uhe)
        
        return ce
        
    def fun(self, rho: TensorLike, uh: TensorLike) -> float:
        """计算总柔度"""
        ce = self.compute_element_compliance(rho, uh)
        self.ce = ce  # 缓存单元柔度用于梯度计算
        
        E = self.material_properties.material_model()
        
        c = bm.einsum('c, c -> ', E, ce)
        
        return c
        
    def jac(self, rho: TensorLike, beta: float = None, rho_tilde: TensorLike = None) -> TensorLike:
        """计算柔度关于密度的梯度"""
        # 获取缓存的单元柔度
        ce = self.ce
        if ce is None:
            raise ValueError("必须先调用fun()计算柔度值")
            
        dE = self.material_properties.material_model_derivative()
        dce = -bm.einsum('c, c -> c', dE, ce)
        
        # 如果没有过滤器，直接返回梯度
        if self.filter_properties is None:
            return dce
            
        # 应用过滤器
        ft = self.filter_properties.ft
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs
        cell_measure = self.space.mesh.entity_measure('cell')
        
        if ft == 0:  # 灵敏度过滤
            rho_dce = bm.einsum('c, c -> c', rho[:], dce)
            filtered_dce = H.matmul(rho_dce)
            dce[:] = filtered_dce / Hs / bm.maximum(bm.array(0.001), rho[:])
            
        elif ft == 1:  # 密度过滤
            dce[:] = H.matmul(dce * cell_measure / H.matmul(cell_measure))
            
        elif ft == 2:  # Heaviside投影
            if beta is None or rho_tilde is None:
                raise ValueError("Heaviside projection filter requires both 'beta' and 'rho_tilde'.")
            dxe = beta * bm.exp(-beta * rho_tilde) + bm.exp(-beta)
            dce[:] = H.matmul(dce * dxe * cell_measure / H.matmul(cell_measure))
            
        return dce