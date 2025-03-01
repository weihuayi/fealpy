from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Literal

from typing import Dict, Optional, Any
from dataclasses import dataclass

from soptx.solver import ElasticFEMSolver
from soptx.opt import ObjectiveBase
from soptx.utils import timer

@dataclass
class ComplianceConfig:
    """Configuration for compliance objective computation"""
    diff_mode: Literal["auto", "manual"] = "manual"  # 微分模式选择

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
    
    def __init__(self, solver: ElasticFEMSolver):
        """
        Parameters
        - solver : 有限元求解器
        """

        self.solver = solver
        self.materials = solver.materials

        # 缓存状态
        self._current_rho = None          # 当前密度场
        self._current_u = None            # 当前位移场
        self._element_compliance = None   # 单元柔顺度
        
    #---------------------------------------------------------------------------
    # 内部方法
    #---------------------------------------------------------------------------
    def _update_u(self, rho: TensorLike) -> TensorLike:
        """更新位移场
        
        如果密度发生变化，重新求解状态方程; 否则使用缓存的状态
        
        Parameters
        - rho : 密度场
        
        Returns
        - u : 位移场
        """
        # 检查是否需要更新
        if (self._current_rho is None or 
            self._current_u is None or
            not bm.all(rho == self._current_rho)):
            
            # 更新求解器中的密度并求解
            self.solver.update_status(rho)
            self._current_u = self.solver.solve().displacement 
            self._current_rho = rho  # 直接引用，内部状态会随外部更新
            
        return self._current_u

    def _compute_element_compliance(self, u: TensorLike) -> TensorLike:
        """计算单元柔顺度向量
        
        Parameters
        - u : 位移场
        
        Returns
        - ce : 单元柔顺度向量
        """
        ke0 = self.solver.get_base_local_stiffness_matrix()
        cell2dof = self.solver.tensor_space.cell_to_dof()
        ue = u[cell2dof]
        
        # 更新缓存
        self._element_compliance = bm.einsum('ci, cik, ck -> c', ue, ke0, ue)

        return self._element_compliance
    
    def _compute_gradient_manual(self, 
                               rho: TensorLike,
                               u: Optional[TensorLike] = None) -> TensorLike:
        """使用解析方法计算梯度"""
        if u is None:
            u = self._update_u(rho)
            
        ce = (self.get_element_compliance() 
              if self._element_compliance is not None 
              else self._compute_element_compliance(u))
        
        dE = self.materials.calculate_elastic_modulus_derivative(rho)
        dc = -bm.einsum('c, c -> c', dE, ce)

        return dc

    def _compute_gradient_auto(self, 
                             rho: TensorLike,
                             u: Optional[TensorLike] = None) -> TensorLike:
        """使用自动微分计算梯度"""
        # 首先获取位移场（只需要计算一次, 在自动微分之外完成）
        if u is None:
            u = self._update_u(rho)

        # 获取基础刚度矩阵和单元自由度映射
        ke0 = self.solver.get_base_local_stiffness_matrix()
        cell2dof = self.solver.tensor_space.cell_to_dof()
        ue = u[cell2dof]  # 获取单元位移

        def compliance_contribution(rho_i: float, ue_i: TensorLike, ke0_i: TensorLike) -> float:
            """计算单个单元的柔顺度贡献
            
            Parameters
            ----------
            rho_i : 单个单元的密度值
            ue_i : (tldof, ), 单个单元的位移向量
            ke0_i : (tldof, tldof), 单个单元的基础刚度矩阵
            """
            # 计算该单元的材料属性
            E = self.materials.calculate_elastic_modulus(rho_i)
            
            # 计算单元柔顺度并取负值 : -(E * u^T * K * u)
            dE = -E * bm.einsum('i, ij, j', ue_i, ke0_i, ue_i)
            
            return dE
        
        # 创建向量化的梯度计算函数
        # 最内层：lambda x: compliance_contribution(x, u, k)
        # 这创建了一个关于单个密度值的函数
        # x 是将要求导的变量（单元密度）
        # u 和 k 是固定的参数（位移和刚度矩阵）

        # 中间层：bm.grad(...)(r)
        # grad 计算上述函数关于 x 的导数
        # (r) 表示在点 r 处求值

        # 外层：bm.vmap(lambda r, u, k: ...)
        # vmap 将这个操作向量化，使其可以并行处理所有单元
        vmap_grad = bm.vmap(lambda r, u, k: bm.jacrev(
            lambda x: compliance_contribution(x, u, k)
        )(r))
        
        # 直接对所有单元进行并行计算
        # 这一步同时处理所有单元，对应关系是：

        # rho 对应每个单元的密度值
        # ue 对应每个单元的位移向量
        # ke0 对应每个单元的基础刚度矩阵
        return vmap_grad(rho, ue, ke0)
    
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
        - rho : 密度场
        - u : 可选的位移场，如果为 None 则自动计算或使用缓存的位移场
        """
        # 获取位移场
        if u is None:
            u = self._update_u(rho)
            
        # 计算单元柔度
        ce = self._compute_element_compliance(u)
        
        # 计算总柔度
        E = self.materials.elastic_modulus
        c = bm.einsum('c, c -> ', E, ce)
        
        return c
    
    def jac(self,
            rho: TensorLike,
            u: Optional[TensorLike] = None,
            diff_mode: Literal["auto", "manual"] = "manual") -> TensorLike:
        """计算目标函数梯度
        
        Parameters
        - rho : 密度场
        - u : 可选的位移场，如果为 None 则自动计算或使用缓存的位移场
        - diff_mode : 梯度计算方式
            - "manual": 使用解析推导的梯度公式（默认）
            - "auto": 使用自动微分技术
        """
        # # 创建计时器
        # t = timer(f"Gradient Computation ({diff_mode} mode)")
        # next(t)

        # 选择计算方法
        if diff_mode == "manual":
            dc = self._compute_gradient_manual(rho, u)
            # t.send('Manual gradient computed')
        elif diff_mode == "auto":  
            dc = self._compute_gradient_auto(rho)
            # t.send('Automatic gradient computed')

        # 结束计时
        # t.send(None)
        
        return dc
    
    def hess(self, rho: TensorLike, lambda_: dict) -> TensorLike:
        """计算目标函数 Hessian 矩阵（未实现）"""
        pass