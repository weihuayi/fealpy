from typing import Dict, Any, Optional, Tuple
from time import time
from dataclasses import dataclass

from scipy.sparse import csr_matrix, spdiags

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from soptx.opt import ObjectiveBase, ConstraintBase, OptimizerBase
from soptx.filter import Filter
from soptx.opt.utils import solve_mma_subproblem

@dataclass
class MMAOptions:
    """MMA 算法的配置选项"""
    # 算法控制参数
    max_iterations: int = 200       # 最大迭代次数
    tolerance: float = 0.001        # 收敛容差

    # 问题规模参数
    m: int = 1                         # 约束函数的数量, 默认 1 个约束
    n: Optional[int] = None            # 设计变量的数量

    # MMA 子问题参数
    xmin: Optional[TensorLike] = None  # 设计变量的下界
    xmax: Optional[TensorLike] = None  # 设计变量的上界
    a0: float = 1.0                    # a_0*z 项的常数系数 a_0
    a: Optional[TensorLike] = None     # a_i*z 项的线性系数 a_i
    c: Optional[TensorLike] = None     # c_i*y_i 项的线性系数 c_i
    d: Optional[TensorLike] = None     # 0.5*d_i*(y_i)**2 项的二次项系数 d_i

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
    _ALBEFA = 0.1          # 计算边界 alfa 和 beta 的因子
    _RAA0 = 1e-5           # 函数近似精度的参数
    _EPSILON_MIN = 1e-7    # 最小容差
    
    def __init__(self,
                objective: ObjectiveBase,
                constraint: ConstraintBase,
                filter: Optional[Filter] = None,
                options: Dict[str, Any] = None):
        """初始化 MMA 优化器 """
        self.objective = objective
        self.constraint = constraint
        self.filter = filter

        # 使用用户提供的参数初始化选项
        self.options = MMAOptions(**(options or {}))

        # 设置问题规模参数的默认值
        if self.options.n is None:
            self.options.n = constraint.mesh.number_of_cells()  # 默认使用网格单元数
        
        # 初始化未设置的 MMA 参数
        m = self.options.m
        n = self.options.n
        if self.options.xmin is None:
            self.options.xmin = bm.zeros((n, 1))
        if self.options.xmax is None:
            self.options.xmax = bm.ones((n, 1))
        if self.options.a is None:
            self.options.a = bm.zeros((m, 1))
        if self.options.c is None:
            self.options.c = 1e4 * bm.ones((m, 1))
        if self.options.d is None:
            self.options.d = bm.zeros((m, 1))
                    
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
        asyinit = self._ASYMP_INIT # 0.5
        asyincr = self._ASYMP_INCR # 1.2
        asydecr = self._ASYMP_DECR # 0.7

        xmami = xmax - xmin
        if self._epoch <= 2:
            self._low = xval - asyinit * xmami
            self._upp = xval + asyinit * xmami
        else:
            factor = bm.ones((xval.shape[0], 1))
            xxx = (xval - xold1) * (xold1 - xold2)
            epsilon = 1e-12
            factor[xxx > epsilon] = asyincr
            factor[xxx < -epsilon] = asydecr
            
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
        """求解 MMA 子问题
        
        Paramters
        - xval (n, 1): 当前设计变量
        - df0dx (n, 1): 目标函数的梯度
        - dfdx (m, n): 约束函数的梯度

        Returns
        - xmma (n, 1): 当前 MMA 子问题中变量 x_j 的最优值.
        - low (n, 1): 当前 MMA 子问题中计算和使用的下渐近线.
        - upp (n, 1): 当前 MMA 子问题中计算和使用的上渐近线.
        """
        xmin = self.options.xmin # (n, )
        xmax = self.options.xmax # (n, )
        m = self.options.m    # 使用配置的约束数量
        n = self.options.n    # 使用配置的设计变量数量

        a0 = self.options.a0
        a = self.options.a
        c = self.options.c
        d = self.options.d

        move = self._MOVE_LIMIT # 0.2
        albefa = self._ALBEFA   # 0.1
        raa0 = self._RAA0       # 1e-5   
        epsimin = self._EPSILON_MIN

        eeen = bm.ones((n, 1), dtype=bm.float64) # (n, 1)
        eeem = bm.ones((m, 1), dtype=bm.float64) # (m, 1)
        
        # 更新渐近线
        low, upp = self._update_asymptotes(xval, xmin, xmax, xold1, xold2) # (n, 1), (n, 1)
        
        # 计算变量边界 alfa, beta
        xxx1 = low + albefa * (xval - low)
        xxx2 = xval - move * (xmax - xmin)
        xxx = bm.maximum(xxx1, xxx2)
        alfa = bm.maximum(xmin, xxx)       # (n, 1) 
        xxx1 = upp - albefa * (upp - xval)
        xxx2 = xval + move * (xmax - xmin)
        xxx = bm.minimum(xxx1, xxx2)
        beta = bm.minimum(xmax, xxx)       # (n, 1)

        # 计算 p0, q0 构建目标函数的近似
        xmami = xmax - xmin # (n, 1)
        xmami_eps = raa0 * eeen
        xmami = bm.maximum(xmami, xmami_eps)
        xmami_inv = eeen / xmami # (n, 1)
        ux1 = upp - xval
        xl1 = xval - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv = eeen / ux1
        xlinv = eeen / xl1
        p0 = bm.maximum(df0dx, 0)   
        q0 = bm.maximum(-df0dx, 0) 
        pq0 = 0.001 * (p0 + q0) + raa0 * xmami_inv
        p0 = p0 + pq0
        q0 = q0 + pq0
        p0 = p0 * ux2 # (n, 1)
        q0 = q0 * xl2 # (n, 1)
        # 构建 P, Q 和 b 构建约束函数的近似
        P = bm.zeros((m, n), dtype=bm.float64)
        Q = bm.zeros((m, n), dtype=bm.float64)
        P = bm.maximum(dfdx, 0)
        Q = bm.maximum(-dfdx, 0)
        PQ = 0.001 * (P + Q) + raa0 * bm.dot(eeem, xmami_inv.T)
        P = P + PQ
        Q = Q + PQ
        # TODO 使用 einsum 替代对角矩阵乘法
        P = bm.einsum('j, ij -> ij', ux2.flatten(), P)
        Q = bm.einsum('j, ij -> ij', xl2.flatten(), Q)
        # from numpy import diag as diags
        # P = (diags(ux2.flatten(), 0) @ P.T).T
        # Q = (diags(xl2.flatten(), 0) @ Q.T).T
        b = bm.dot(P, uxinv) + bm.dot(Q, xlinv) - fval  # (1, 1)
        
        # 求解子问题
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = solve_mma_subproblem(
                                                        m=m, n=n, 
                                                        epsimin=epsimin, 
                                                        low=low, upp=upp, 
                                                        alfa=alfa, beta=beta,
                                                        p0=p0, q0=q0, P=P, Q=Q,
                                                        a0=a0, a=a, b=b, c=c, d=d
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
            obj_grad = self.objective.jac(rho_phys) # (NC, )
            if self.filter is not None:
                obj_grad = self.filter.filter_sensitivity(
                                        obj_grad, rho_phys, 'objective', filter_params)
            
            # 计算约束值和约束值梯度
            con_val = self.constraint.fun(rho_phys)
            con_grad = self.constraint.jac(rho_phys) # (NC, )
            if self.filter is not None:
                con_grad = self.filter.filter_sensitivity(
                                        con_grad, rho_phys, 'constraint', filter_params)
            
            # MMA 方法
            volfrac = self.constraint.volume_fraction
            # 标准化的约束值梯度
            dfdx = con_grad[:, None].T / (volfrac * con_grad.shape[0])     # (m, n)
            rho_new, low, upp = self._solve_subproblem(
                                        xval=rho[:, None], fval=con_val, 
                                        df0dx=obj_grad[:, None], dfdx=dfdx, 
                                        low=low, upp=upp,
                                        xold1=xold1[:, None], xold2=xold2[..., None]
                                    )

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