from typing import Dict, Any, Optional, Tuple
from time import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.opt import ObjectiveBase, ConstraintBase, OptimizerBase
from soptx.filter import Filter

@dataclass
class MMAOptions:
    """MMA 算法的配置选项"""
    max_iterations: int = 100      # 最大迭代次数
    tolerance: float = 0.01       # 收敛容差
    move_limit: float = 0.2       # 移动限制
    asymp_init: float = 0.5      # 渐近初始系数
    asymp_incr: float = 1.2      # 渐近递增系数
    asymp_decr: float = 0.7      # 渐近递减系数
    elastic_weight: float = 1e3   # 弹性权重
    min_asymp: float = 1e-12     # 最小渐近系数

@dataclass
class OptimizationHistory:
    """优化过程的历史记录"""
    densities: list       # 密度场历史
    
    def __init__(self):
        """初始化各个记录列表"""
        self.densities = []
        
    def log_iteration(self, iter_idx: int, obj_val: float, volume: float, 
                     change: float, time: float, density: TensorLike):
        """记录一次迭代的信息"""
        self.densities.append(density.copy())
        
        print(f"Iteration: {iter_idx + 1}, "
              f"Objective: {obj_val:.3f}, "
              f"Volume: {volume:.12f}, "
              f"Change: {change:.3f}, "
              f"Time: {time:.3f} sec")

class MMAOptimizer(OptimizerBase):
    """Method of Moving Asymptotes (MMA) 优化器
    
    用于求解拓扑优化问题的 MMA 方法实现。该方法通过动态调整渐近线位置
    来控制优化过程，具有良好的收敛性能。
    """
    
    def __init__(self,
                 objective: ObjectiveBase,
                 constraint: ConstraintBase,
                 filter: Optional[Filter] = None,
                 options: Optional[Dict[str, Any]] = None):
        """初始化 MMA 优化器
        
        Parameters
        ----------
        objective : 目标函数对象
        constraint : 约束条件对象
        filter : 滤波器对象
        options : 算法参数配置
        """
        self.objective = objective
        self.constraint = constraint
        self.filter = filter
        
        # 设置默认参数
        self.options = MMAOptions()
        if options is not None:
            for key, value in options.items():
                if hasattr(self.options, key):
                    setattr(self.options, key, value)
                    
        # MMA 内部状态
        self._epoch = 0
        self._xold1 = None
        self._xold2 = None
        self._low = None
        self._upp = None
        
    def _update_asymptotes(self, 
                          rho: TensorLike, 
                          xmin: TensorLike,
                          xmax: TensorLike) -> Tuple[TensorLike, TensorLike]:
        """更新渐近线位置
        
        Parameters
        ----------
        rho : 当前密度场
        xmin : 设计变量下界
        xmax : 设计变量上界
        
        Returns
        -------
        low : 下渐近线
        upp : 上渐近线
        """
        n = len(rho)
        xmami = xmax - xmin
        xmamieps = 0.00001 * np.ones((n, 1))
        xmami = np.maximum(xmami, xmamieps)
        
        if self._epoch <= 2:
            # 初始化渐近线
            self._low = rho - self.options.asymp_init * xmami
            self._upp = rho + self.options.asymp_init * xmami
        else:
            # 基于历史信息调整渐近线
            factor = np.ones((n, 1))
            xxx = (rho - self._xold1) * (self._xold1 - self._xold2)
            
            # 根据变化趋势调整系数
            factor[xxx > 0] = self.options.asymp_incr
            factor[xxx < 0] = self.options.asymp_decr
            
            # 更新渐近线位置
            self._low = rho - factor * (self._xold1 - self._low)
            self._upp = rho + factor * (self._upp - self._xold1)
            
            # 限制渐近线范围
            lowmin = rho - 10 * xmami
            lowmax = rho - 0.01 * xmami
            uppmin = rho + 0.01 * xmami
            uppmax = rho + 10 * xmami
            
            self._low = np.maximum(self._low, lowmin)
            self._low = np.minimum(self._low, lowmax)
            self._upp = np.minimum(self._upp, uppmax)
            self._upp = np.maximum(self._upp, uppmin)
            
        return self._low, self._upp
        
    def _solve_subproblem(self, 
                         rho: TensorLike,
                         dc: TensorLike,
                         dg: TensorLike,
                         xmin: TensorLike,
                         xmax: TensorLike) -> TensorLike:
        """求解 MMA 子问题
        
        Parameters
        ----------
        rho : 当前密度场
        dc : 目标函数梯度
        dg : 约束函数梯度
        xmin : 设计变量下界
        xmax : 设计变量上界
        
        Returns
        -------
        rho_new : 更新后的密度场
        """
        n = len(rho)      # 设计变量数量
        m = 1             # 约束数量
        
        # 更新渐近线
        low, upp = self._update_asymptotes(rho, xmin, xmax)
        
        # 计算移动限制
        move = self.options.move_limit
        alpha = np.maximum(low + 0.1 * (rho - low), 
                         rho - move * (xmax - xmin))
        beta = np.minimum(upp - 0.1 * (upp - rho),
                        rho + move * (xmax - xmin))
        
        # 构建子问题的参数
        ux1 = upp - rho
        xl1 = rho - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        
        # 目标函数的二次近似项
        p0 = np.maximum(dc, 0)
        q0 = np.maximum(-dc, 0)
        p0 = p0 * ux2
        q0 = q0 * xl2
        
        # 约束函数的二次近似项
        P = np.maximum(dg, 0)
        Q = np.maximum(-dg, 0)
        P = (diags(ux2.flatten(), 0).dot(P.T)).T
        Q = (diags(xl2.flatten(), 0).dot(Q.T)).T
        
        # 求解子问题
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = self._mma_sub_solver(
            m, n, self.options.min_asymp, 
            low, upp, alpha, beta,
            p0, q0, P, Q,
            self.options.elastic_weight
        )
        
        return xmma
        
    def optimize(self, rho: TensorLike, **kwargs) -> Tuple[TensorLike, OptimizationHistory]:
        """运行 MMA 优化算法
        
        Parameters
        ----------
        rho : 初始密度场
        **kwargs : 其他参数，例如：
            - beta: Heaviside 投影参数
        
        Returns
        -------
        rho : 最优密度场
        history : 优化历史记录
        """
        # 获取参数
        max_iters = self.options.max_iterations
        tol = self.options.tolerance
        
        # 设置变量界限
        xmin = np.zeros_like(rho)
        xmax = np.ones_like(rho)
        
        # 准备 Heaviside 投影的参数
        filter_params = {'beta': kwargs.get('beta')} if 'beta' in kwargs else None
        
        # 获取物理密度
        rho_phys = (self.filter.get_physical_density(rho, filter_params) 
                   if self.filter is not None else rho)
        
        # 初始化历史记录
        history = OptimizationHistory()
        
        # 优化主循环
        for iter_idx in range(max_iters):
            start_time = time()
            
            # 更新迭代计数
            self._epoch = iter_idx + 1
            
            # 保存历史信息
            if self._xold1 is None:
                self._xold1 = rho.copy()
                self._xold2 = rho.copy()
            else:
                self._xold2 = self._xold1.copy()
                self._xold1 = rho.copy()
            
            # 计算目标函数值和梯度
            obj_val = self.objective.fun(rho_phys)
            obj_grad = self.objective.jac(rho_phys)
            
            # 计算约束值和梯度
            con_val = self.constraint.fun(rho_phys)
            con_grad = self.constraint.jac(rho_phys)
            
            # 求解 MMA 子问题
            rho_new = self._solve_subproblem(rho, obj_grad, con_grad, xmin, xmax)
            
            # 计算收敛性
            change = np.max(np.abs(rho_new - rho))
            
            # 更新密度场
            rho = rho_new
            
            # 更新物理密度
            if self.filter is not None:
                rho_phys = self.filter.filter_density(rho, filter_params)
            else:
                rho_phys = rho
                
            # 记录当前迭代信息
            iteration_time = time() - start_time
            history.log_iteration(iter_idx, obj_val, np.mean(rho_phys), 
                                change, iteration_time, rho_phys)
            
            # 收敛检查
            if change <= tol:
                print(f"Converged after {iter_idx + 1} iterations")
                break
                
        return rho, history