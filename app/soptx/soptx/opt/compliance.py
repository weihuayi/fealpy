from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from typing import Dict, Optional, Any

from soptx.material import ElasticMaterialProperties
from soptx.solver import ElasticFEMSolver
from soptx.opt import ObjectiveBase
from soptx.filter import Filter

class ComplianceObjective(ObjectiveBase):
    """结构柔度最小化目标函数
    
    该类负责：
    1. 计算目标函数值（柔顺度）
    2. 计算目标函数对密度的梯度
    3. 管理状态变量（位移场）的更新和缓存
    4. 管理密度与柔顺度的对应关系
    
    变量说明：
    - rho: density, 密度场
    - u: displacement, 位移场
    - ce: element compliance, 单元柔顺度
    """
    
    def __init__(self,
                 material_properties: ElasticMaterialProperties,
                 solver: ElasticFEMSolver,
                 filter: Optional[Filter] = None):
        """
        Parameters
        ----------
        material_properties : 材料属性计算器
        solver : 有限元求解器
        filter : 可选的滤波器
        """
        self.material_properties = material_properties
        self.solver = solver
        self.filter = filter

        # 缓存状态
        self._current_rho = None  # 当前密度场
        self._current_u = None    # 当前位移场
        self._element_compliance = None   # 单元柔顺度
        
    #---------------------------------------------------------------------------
    # 内部方法
    #---------------------------------------------------------------------------
    def _update_u(self, rho: TensorLike) -> TensorLike:
        """更新位移场
        
        如果密度发生变化，重新求解状态方程；否则使用缓存的状态
        
        Parameters
        ----------
        rho : 密度场
        
        Returns
        -------
        u : 位移场
        """
        # 检查是否需要更新
        if (self._current_rho is None or 
            self._current_u is None or
            not bm.all(rho == self._current_rho)):
            
            # 更新求解器中的密度并求解
            self.solver.update_density(rho)
            self._current_u = self.solver.solve().displacement 
            # self._current_u = self.solver.solve_cg().displacement
            # self._current_rho = bm.copy(rho)  # 这里的 copy 导致内部状态与外部不同步
            self._current_rho = rho  # 直接引用，内部状态会随外部更新
            
        return self._current_u

    def _compute_element_compliance(self, u: TensorLike) -> TensorLike:
        """计算单元柔顺度
        
        Parameters
        ----------
        u : 位移场
        
        Returns
        -------
        ce : 单元柔顺度向量
        """
        ke0 = self.solver.get_base_local_stiffness_matrix()
        cell2dof = self.solver.tensor_space.cell_to_dof()
        ue = u[cell2dof]
        
        # 更新缓存
        self._element_compliance = bm.einsum('ci, cik, ck -> c', ue, ke0, ue)
        return self._element_compliance
    
    def get_element_compliance(self) -> TensorLike:
        """获取单元柔顺度"""
        if self._element_compliance is None:
            raise ValueError("必须先调用 fun() 计算柔顺度")
        return self._element_compliance

    #---------------------------------------------------------------------------
    # 优化相关方法
    #---------------------------------------------------------------------------
    def fun(self, 
            rho: TensorLike, 
            u: Optional[TensorLike] = None) -> float:
        """计算总柔度值
        
        Parameters
        ----------
        rho : 密度场
        u : 可选的位移场，如果为 None 则自动计算或使用缓存的位移场
        
        Returns
        -------
        c : 总柔顺度值
        """
        # 获取位移场
        if u is None:
            u = self._update_u(rho)
            
        # 计算单元柔度
        ce = self._compute_element_compliance(u)
        
        # 计算总柔度
        E = self.material_properties.calculate_elastic_modulus(rho)
        c = bm.einsum('c, c -> ', E, ce)
        
        return c
        
    def jac(self,
            rho: TensorLike,
            u: Optional[TensorLike] = None,
            filter_params: Optional[Dict[str, Any]] = None) -> TensorLike:
        """计算目标函数梯度
        
        Parameters
        ----------
        rho : 密度场
        u : 可选的位移场，如果为 None 则自动计算或使用缓存的位移场
        filter_params : 滤波器参数
        
        Returns
        -------
        dc : 目标函数对密度的梯度
        """
        # 获取位移场
        if u is None:
            u = self._update_u(rho)
            
        # 获取单元柔度
        ce = (self.get_element_compliance() 
              if self._element_compliance is not None 
              else self._compute_element_compliance(u))
        
        # 计算梯度
        dE = self.material_properties.calculate_elastic_modulus_derivative(rho)
        dc = -bm.einsum('c, c -> c', dE, ce)
        
        # 应用滤波
        if self.filter is None:
            return dc
        
        # 明确指定这是目标函数的梯度    
        return self.filter.filter_sensitivity(dc, rho, 'objective', filter_params)
    
    def hess(self, rho: TensorLike, lambda_: dict) -> TensorLike:
        """计算目标函数 Hessian 矩阵（未实现）"""
        pass