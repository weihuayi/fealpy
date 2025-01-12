from typing import Dict, Any, Optional
from time import time
from dataclasses import dataclass

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.opt import ObjectiveBase, ConstraintBase, OptimizerBase
from soptx.filter import Filter

@dataclass
class OCOptions:
    """OC 算法的配置选项"""
    max_iterations: int = 100     # 最大迭代次数
    move_limit: float = 0.2       # 正向移动限制 m
    damping_coef: float = 0.5     # 阻尼系数 η
    tolerance: float = 0.01       # 收敛容差
    initial_lambda: float = 1e9   # 初始 lambda 值
    bisection_tol: float = 1e-3   # 二分法收敛容差

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
              f"Objective: {obj_val:.4f}, "
              f"Volume: {volume:.4f}, "
              f"Change: {change:.4f}, "
              f"Time: {time:.3f} sec")

class OCOptimizer(OptimizerBase):
    """Optimality Criteria (OC) 优化器"""
    
    def __init__(self,
                 objective: ObjectiveBase,
                 constraint: ConstraintBase,
                 filter: Optional[Filter] = None,
                 options: Optional[Dict[str, Any]] = None):
        """
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
        self.options = OCOptions()
        if options is not None:
            for key, value in options.items():
                if hasattr(self.options, key):
                    setattr(self.options, key, value)
                    
    def _update_density(self,
                       rho: TensorLike,
                       dc: TensorLike,
                       dg: TensorLike,
                       lmid: float) -> TensorLike:
        """使用 OC 准则更新密度"""
        m = self.options.move_limit
        eta = self.options.damping_coef
        
        B_e = -dc / (dg * lmid)
        B_e_damped = bm.power(B_e, eta)

        # OC update scheme
        rho_new = bm.maximum(
            bm.tensor(0.0, dtype=rho.dtype), 
            bm.maximum(
                rho - m, 
                bm.minimum(
                    bm.tensor(1.0, dtype=rho.dtype), 
                    bm.minimum(
                        rho + m, 
                        rho * B_e_damped
                    )
                )
            )
        )
        
        return rho_new
        
    def optimize(self, rho: TensorLike, **kwargs) -> TensorLike:
        """运行 OC 优化算法

        Parameters
        ----------
        - rho : 初始密度场
        - **kwargs : 其他参数，例如：
            -- beta: Heaviside 投影参数
        """
        # 获取优化参数
        max_iters = self.options.max_iterations
        tol = self.options.tolerance
        bisection_tol = self.options.bisection_tol

        # 准备 Heaviside 投影的参数 (如果需要)
        filter_params = {'beta': kwargs.get('beta')} if 'beta' in kwargs else None
        
        # 获取物理密度(对于非 Heaviside 投影，就是设计密度本身)
        rho_phys = (self.filter.get_physical_density(rho, filter_params) 
                   if self.filter is not None else rho)
        
        # 初始化历史记录
        history = OptimizationHistory()
        
        # 优化主循环
        for iter_idx in range(max_iters):
            start_time = time()
            
            # 使用物理密度计算目标函数值和梯度
            obj_val = self.objective.fun(rho_phys)
            obj_grad = self.objective.jac(rho_phys)  # (NC, )
            if self.filter is not None:
                obj_grad = self.filter.filter_sensitivity(
                                        obj_grad, rho_phys, 'objective', filter_params)
            
            # 使用物理密度计算约束值和梯度
            con_val = self.constraint.fun(rho_phys)
            con_grad = self.constraint.jac(rho_phys)  # (NC, )
            if self.filter is not None:
                con_grad = self.filter.filter_sensitivity(
                                        con_grad, rho_phys, 'constraint', filter_params)
            
            # 二分法求解拉格朗日乘子
            l1, l2 = 0.0, self.options.initial_lambda
            
            while (l2 - l1) / (l2 + l1) > bisection_tol:
                lmid = 0.5 * (l2 + l1)
                rho_new = self._update_density(rho, obj_grad, con_grad, lmid)
                
                # 计算新的物理密度
                if self.filter is not None:
                    rho_phys = self.filter.filter_density(rho_new, filter_params)
                else:
                    rho_phys = rho_new
                    
                # 检查约束
                if self.constraint.fun(rho_phys) > 0:
                    l1 = lmid
                else:
                    l2 = lmid

            # 计算收敛性
            change = bm.max(bm.abs(rho_new - rho))
            # 更新设计变量，确保目标函数内部状态同步
            rho = rho_new
            
            iteration_time = time() - start_time

            history.log_iteration(iter_idx, obj_val, bm.mean(rho_phys[:]), 
                                change, iteration_time, rho_phys[:])
            
            # 收敛检查
            if change <= tol:
                print(f"Converged after {iter_idx + 1} iterations")
                break
                
        return rho, history
    
def save_optimization_history(mesh, history: OptimizationHistory, save_path: str):
    """保存优化过程的所有迭代结果
    
    Parameters
    ----------
    mesh : 有限元网格对象
    history : 优化历史记录
    save_path : 保存路径
    """
    for i, density in enumerate(history.densities):
        mesh.celldata['density'] = density
        mesh.to_vtk(f"{save_path}/density_iter_{i:03d}.vts")