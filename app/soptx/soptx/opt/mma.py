from typing import Dict, Any, Optional, Tuple
from time import time
from dataclasses import dataclass

from scipy.sparse import csr_matrix, spdiags

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.opt import ObjectiveBase, ConstraintBase, OptimizerBase
from .utils import solve_mma_subproblem
from soptx.filter import Filter

@dataclass
class MMAOptions:
    """MMA 算法的配置选项"""
    # 问题规模参数
    m: int                          # 约束函数的数量
    n: int                          # 设计变量的数量
    # 算法控制参数
    max_iterations: int = 100       # 最大迭代次数
    tolerance: float = 0.001         # 收敛容差
    # MMA 子问题参数
    a0: float = 1.0                 # a_0*z 项的常数系数 a_0
    a: Optional[TensorLike] = None  # a_i*z 项的线性系数 a_i
    c: Optional[TensorLike] = None  # c_i*y_i 项的线性系数 c_i
    d: Optional[TensorLike] = None  # 0.5*d_i*(y_i)**2 项的二次项系数 d_i

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
              f"Objective: {obj_val:.6f}, "
              f"Volume: {volume:.6f}, "
              f"Change: {change:.6f}, "
              f"Time: {time:.3f} sec")

class MMAOptimizer(OptimizerBase):
    """Method of Moving Asymptotes (MMA) 优化器
    
    用于求解拓扑优化问题的 MMA 方法实现. 该方法通过动态调整渐近线位置
    来控制优化过程, 具有良好的收敛性能
    """

    # MMA 算法的固定参数
    _ASYMP_INIT = 0.5      # 渐近线初始距离的因子
    _ASYMP_INCR = 1.2      # 渐近线矩阵减小的因子
    _ASYMP_DECR = 0.7      # 渐近线矩阵增加的因子
    _MOVE_LIMIT = 0.2      # 移动限制
    _ALBEFA = 0.1          # 计算边界 alpha 和 beta 的因子
    _RAA0 = 1e-5           # 函数近似精度的参数
    _EPSILON_MIN = 1e-7    # 最小容差
    
    def __init__(self,
                objective: ObjectiveBase,
                constraint: ConstraintBase,
                m: int,
                n: int,
                filter: Optional[Filter] = None,
                options: Optional[Dict[str, Any]] = None):
        """初始化 MMA 优化器 """
        self.objective = objective
        self.constraint = constraint
        self.filter = filter

        # 设置默认参数
        self.options = MMAOptions(m=m, n=n)
        
        # 更新用户提供的参数
        if options is not None:
            for key, value in options.items():
                if hasattr(self.options, key):
                    setattr(self.options, key, value)
        
        # 初始化未设置的 MMA 参数
        if self.options.a is None:
            self.options.a = bm.zeros((m, 1))
        if self.options.c is None:
            self.options.c = 1e4 * bm.ones((m, 1))
        if self.options.d is None:
            self.options.d = bm.zeros((m, 1))

        # 初始化 xmin 和 xmax
        self.xmin = bm.zeros((n, 1))  # 下界
        self.xmax = bm.ones((n, 1))   # 上界
                    
        # MMA 内部状态
        self._epoch = 0
        self._low = None
        self._upp = None
        
    def _update_asymptotes(self, 
                          xval: TensorLike, 
                          xmin: TensorLike,
                          xmax: TensorLike,
                          xold1: TensorLike,
                          xold2: TensorLike) -> Tuple[TensorLike, TensorLike]:
        """更新渐近线位置"""
        asyinit = self._ASYMP_INIT
        asyincr = self._ASYMP_INCR
        asydecr = self._ASYMP_DECR

        xmami = xmax - xmin

        if self._epoch <= 2:
            self._low = xval - asyinit * xmami
            self._upp = xval + asyinit * xmami
        else:
            factor = bm.ones((xval.shape[0], 1))
            xxx = (xval - xold1) * (xold1 - xold2)
            factor[xxx > 0] = asyincr
            factor[xxx < 0] = asydecr
            factor[xxx == 0] = 1.0
            
            self._low = xval - factor * (xold1 - self._low)
            self._upp = xval + factor * (self._upp - xold1)
            
            lowmin = xval - 10 * xmami
            lowmax = xval - 0.01 * xmami
            uppmin = xval + 0.01 * xmami
            uppmax = xval + 10 * xmami
            
            self._low = bm.maximum(self._low, lowmin)
            self._low = bm.minimum(self._low, lowmax)
            self._upp = bm.minimum(self._upp, uppmax)
            self._upp = bm.maximum(self._upp, uppmin)
            
        return self._low, self._upp
        
    def _solve_subproblem(self, 
                        xval: TensorLike,
                        fval: TensorLike,
                        df0dx: TensorLike,
                        dfdx: TensorLike,
                        low: TensorLike,
                        upp: TensorLike,
                        xold1: TensorLike,
                        xold2: TensorLike) -> TensorLike:
        """求解 MMA 子问题"""
        m = self.options.m    # 使用配置的约束数量
        n = self.options.n    # 使用配置的设计变量数量
        a0 = self.options.a0
        a = self.options.a
        c = self.options.c
        d = self.options.d

        move = self._MOVE_LIMIT
        albefa = self._ALBEFA
        raa0 = self._RAA0
        epsimin = self._EPSILON_MIN

        xmin = self.xmin
        xmax = self.xmax

        eeen = bm.ones((n, 1), dtype=bm.float64)
        eeem = bm.ones((m, 1), dtype=bm.float64)
        
        # 更新渐近线
        low, upp = self._update_asymptotes(xval, xmin, xmax, xold1, xold2)
        
        # 计算边界 alpha, beta
        xxx1 = low + albefa * (xval - low)
        xxx2 = xval - move * (xmax - xmin)
        xxx = bm.maximum(xxx1, xxx2)
        alpha = bm.maximum(xmin, xxx)
        xxx1 = upp - albefa * (upp - xval)
        xxx2 = xval + move * (xmax - xmin)
        xxx = bm.minimum(xxx1, xxx2)
        beta = bm.minimum(xmax, xxx)

        # 计算 p0, q0, P, Q 和 b
        xmami = xmax - xmin
        xmami_eps = raa0 * eeen
        xmami = bm.maximum(xmami, xmami_eps)
        xmami_inv = eeen / xmami
        # 定义当前设计点
        ux1 = upp - xval
        xl1 = xval - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv = eeen / ux1
        xlinv = eeen / xl1
        # 构建 p0, q0
        p0 = bm.maximum(df0dx, 0)   
        q0 = bm.maximum(-df0dx, 0) 
        pq0 = 0.001 * (p0 + q0) + raa0 * xmami_inv
        p0 = p0 + pq0
        q0 = q0 + pq0
        p0 = p0 * ux2
        q0 = q0 * xl2
        # 构建 P, Q
        P = bm.zeros((m, n), dtype=bm.float64)
        Q = bm.zeros((m, n), dtype=bm.float64)
        P = bm.maximum(dfdx, 0)
        Q = bm.maximum(-dfdx, 0)
        PQ = 0.001 * (P + Q) + raa0 * bm.dot(eeem, xmami_inv.T)
        P = P + PQ
        Q = Q + PQ
        from numpy import diag as diags
        P = (diags(ux2.flatten(), 0) @ P.T).T
        Q = (diags(xl2.flatten(), 0) @ Q.T).T
        b = bm.dot(P, uxinv) + bm.dot(Q, xlinv) - fval
        
        # 求解子问题
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = solve_mma_subproblem(
                                                            m, n, epsimin, low, upp, alpha, beta,
                                                            p0, q0, P, Q,
                                                            a0, a, b, c, d
                                                        )
        
        return xmma.reshape(-1), low, upp
        
    def optimize(self, rho: TensorLike, **kwargs) -> Tuple[TensorLike, OptimizationHistory]:
        """运行 MMA 优化算法"""
        # 获取参数
        max_iters = self.options.max_iterations
        tol = self.options.tolerance

        low = bm.ones_like(rho)
        upp = bm.ones_like(rho)
        
        # 准备 Heaviside 投影的参数
        filter_params = {'beta': kwargs.get('beta')} if 'beta' in kwargs else None
        
        # 获取物理密度
        rho_phys = (self.filter.get_physical_density(rho, filter_params) 
                   if self.filter is not None else rho)
        
        # 初始化历史记录
        history = OptimizationHistory()

        xold1 = bm.copy(rho)  # 当前的设计变量
        xold2 = bm.copy(rho)  # 初始化为当前的设计变量
        
        # 优化主循环
        for iter_idx in range(max_iters):
            start_time = time()
            
            # 更新迭代计数
            self._epoch = iter_idx + 1
            
            # 计算目标函数值和梯度
            obj_val = self.objective.fun(rho_phys)
            obj_grad = self.objective.jac(rho_phys)
            
            # 计算约束值和约束值梯度
            con_val = self.constraint.fun(rho_phys)
            con_grad = self.constraint.jac(rho_phys)
            
            # 求解 MMA 子问题
            volfrac = self.constraint.volume_fraction
            dfdx = con_grad[:, None].T / (volfrac * con_grad.shape[0])
            rho_new, low, upp = self._solve_subproblem(rho[:, None], 
                                                    con_val, obj_grad[:, None], 
                                                    dfdx, 
                                                    low, upp,
                                                    xold1[:, None], xold2[..., None])
            
            # 更新物理密度
            if self.filter is not None:
                rho_phys = self.filter.filter_density(rho_new, filter_params)
            else:
                rho_phys = rho_new

            xold2 = xold1
            xold1 = rho
            
            # 计算收敛性
            change = bm.max(bm.abs(rho_new - rho))
            
            # 更新密度场
            rho = rho_new
                
            # 记录当前迭代信息
            iteration_time = time() - start_time
            history.log_iteration(iter_idx, obj_val, bm.mean(rho_phys), 
                                change, iteration_time, rho_phys)
            
            # 收敛检查
            if change <= tol:
                print(f"Converged after {iter_idx + 1} iterations")
                break
                
        return rho, history