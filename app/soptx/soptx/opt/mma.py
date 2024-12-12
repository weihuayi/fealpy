from typing import Dict, Any, Optional, Tuple
from time import time
from dataclasses import dataclass

from scipy.sparse import csr_matrix, spdiags

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.opt import ObjectiveBase, ConstraintBase, OptimizerBase
from soptx.filter import Filter

@dataclass
class MMAOptions:
    """MMA 算法的配置选项"""
    # 问题规模参数
    m: int                          # 约束函数的数量
    n: int                          # 设计变量的数量
    # 算法控制参数
    max_iterations: int = 100       # 最大迭代次数
    tolerance: float = 0.01         # 收敛容差
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
              f"Objective: {obj_val:.3f}, "
              f"Volume: {volume:.12f}, "
              f"Change: {change:.3f}, "
              f"Time: {time:.3f} sec")

class MMAOptimizer(OptimizerBase):
    """Method of Moving Asymptotes (MMA) 优化器
    
    用于求解拓扑优化问题的 MMA 方法实现. 该方法通过动态调整渐近线位置
    来控制优化过程, 具有良好的收敛性能
    """

    # MMA 算法的固定参数
    _MOVE_LIMIT = 0.01     # 移动限制
    _ASYMP_INIT = 0.01     # 渐近初始系数
    _ASYMP_INCR = 1.2      # 渐近递增系数
    _ASYMP_DECR = 0.4      # 渐近递减系数
    _ALBEFA = 0.1          # 渐近线移动系数
    _RAA0 = 1e-5           # 正则化参数
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
                    
        # MMA 内部状态
        self._epoch = 0
        self._xold1 = None
        self._xold2 = None
        self._low = None
        self._upp = None
        
    def _update_asymptotes(self, 
                          xval: TensorLike, 
                          xmin: TensorLike,
                          xmax: TensorLike) -> Tuple[TensorLike, TensorLike]:
        """更新渐近线位置"""
        asyinit = self._ASYMP_INIT
        asyincr = self._ASYMP_INCR
        asydecr = self._ASYMP_DECR

        xmami = xmax - xmin

        if self._epoch <= 2:
            # 初始化渐近线
            self._low = xval - asyinit * xmami
            self._upp = xval + asyinit * xmami
        else:
            # 基于历史信息调整渐近线
            factor = bm.ones((xval.shape[0], 1))
            xxx = (xval - self._xold1) * (self._xold1 - self._xold2)
            # 根据变化趋势调整系数
            factor[xxx > 0] = asyincr
            factor[xxx < 0] = asydecr
            
            # 更新渐近线位置
            self._low = xval - factor * (self._xold1 - self._low)
            self._upp = xval + factor * (self._upp - self._xold1)
            
            # 限制渐近线范围
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
                        xmin: TensorLike,
                        xmax: TensorLike) -> TensorLike:
        """求解 MMA 子问题"""
        m = self.options.m    # 使用配置的约束数量
        n = self.options.n    # 使用配置的设计变量数量
        a0 = self.options.a0
        a = self.options.a
        c = self.options.c
        d = self.options.d

        raa0 = self._RAA0
        epsimin = self._EPSILON_MIN
        move = self._MOVE_LIMIT
        
        # 更新渐近线
        low, upp = self._update_asymptotes(xval, xmin, xmax)
        
        # 计算移动限制
        alpha = bm.maximum(low + 0.1 * (xval - low), xval - move * (xmax - xmin))
        beta = bm.minimum(upp - 0.1 * (upp - xval), xval + move * (xmax - xmin))

        # 一些辅助量
        eeen = bm.ones(n)
        eeem = bm.ones((m, 1))

        # 计算 xmami, xmamiinv 等参数
        xmami = xmax - xmin
        xmamieps = raa0 * eeen
        xmami = bm.maximum(xmami, xmamieps)
        xmamiinv = eeen / xmami

        # 定义当前设计点
        ux1 = upp - xval
        xl1 = xval - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv = eeen / ux1
        xlinv = eeen / xl1
        
        # 构建 p0, q0
        p0 = bm.maximum(df0dx, 0)   # (NC, )
        q0 = bm.maximum(-df0dx, 0)  # (NC, )
        pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
        p0 = p0 + pq0
        q0 = q0 + pq0
        p0 = p0 * ux2
        q0 = q0 * xl2
        
        # 构建 P, Q
        P = csr_matrix((m, n)) 
        Q = csr_matrix((m, n))
        P_data = bm.maximum(dfdx.reshape(m, n), 0)
        Q_data = bm.maximum(-dfdx.reshape(m, n), 0)
        P = csr_matrix(P_data)
        Q = csr_matrix(Q_data)

        PQ = 0.001 * (P + Q) + raa0*(eeem @ xmamiinv[None,:])
        P = P + PQ
        Q = Q + PQ

        P = P @ spdiags(ux2, 0, n, n)
        Q = Q @ spdiags(xl2, 0, n, n)
        
        # 计算 b
        b = (P @ uxinv + Q @ xlinv - fval)
        
        # 求解子问题
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = self._mma_sub_solver(
            m, n, epsimin, low, upp, alpha, beta,
            p0, q0, P, Q,
            a0, a, b, c, d)
        
        return xmma
        
    def optimize(self, rho: TensorLike, **kwargs) -> Tuple[TensorLike, OptimizationHistory]:
        """运行 MMA 优化算法
        
        Parameters
        ----------
        rho-(NC, ): 初始密度场 
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
        xmin = bm.zeros_like(rho)
        xmax = bm.ones_like(rho)
        
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
            rho_new = self._solve_subproblem(rho, con_val, obj_grad, con_grad, xmin, xmax)
            
            # 计算收敛性
            change = bm.max(bm.abs(rho_new - rho))
            
            # 更新密度场
            rho = rho_new
            
            # 更新物理密度
            if self.filter is not None:
                rho_phys = self.filter.filter_density(rho, filter_params)
            else:
                rho_phys = rho
                
            # 记录当前迭代信息
            iteration_time = time() - start_time
            history.log_iteration(iter_idx, obj_val, bm.mean(rho_phys), 
                                change, iteration_time, rho_phys)
            
            # 收敛检查
            if change <= tol:
                print(f"Converged after {iter_idx + 1} iterations")
                break
                
        return rho, history