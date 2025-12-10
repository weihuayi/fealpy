from . import Monitor
from . import Interpolater
from .config import *
from scipy.integrate import solve_ivp
from .tool import _compute_coef_2d, quad_equ_solver , linear_surfploter
from scipy.sparse import coo_matrix
from fealpy.utils import timer
time = timer()
next(time)

class LMEAGAdaptive(Monitor, Interpolater):
    def __init__(self, mesh, beta, space, config:Config):
        super().__init__(mesh, beta, space, config)
        self.config = config
        self.alpha = config.alpha
        self.tau = config.tau
        self.t_max = config.t_max
        self.tol = config.tol
        self.maxit = config.maxit
        self.pre_steps = config.pre_steps
        self.gamma = config.gamma
        self.dt = self.tau * self.t_max
        self.R = self.R_matrix()

        self.cell2cell = self.mesh.cell_to_cell()
        self.total_steps = 10
        self.t_span = 0.1
        self.step = 10
        self.BD_projector()
        self._build_jac_pattern()
    
    def _prepare_ivp_cache(self, X, M, M_inv , theta):
        """
        预计算在一个 solve_ivp 步进内不变的量，减少 jac/ode 回调重复开销。
        """
        E_K     = self.edge_matrix(X)           # (NC,d,d)
        E_K_inv = bm.linalg.inv(E_K)            # (NC,d,d)
        det_E_K = bm.linalg.det(E_K)            # (NC,)
        B       = E_K_inv @ M_inv               # (NC,d,d)
        rho     = self.rho(M)                   # (NC,)
        P_diag  = self.balance(self.M_node)     # (NN,)
        cache = {
            'E_K': E_K, 'E_K_inv': E_K_inv, 'det_E_K': det_E_K,
            'B': B, 'rho': rho, 'cm': self.cm, 'P_diag': P_diag,'theta': theta,
            'rxx': self.rxx, 'ryy': self.ryy, 'rxy': self.rxy, 'ryx': self.ryx
        }
        self._ivp_cache = cache
    
    def edge_matrix(self,X):
        """
        边矩阵 E
        E = [x_1 - x_0, x_2 - x_0, ..., x_d - x_0]
        """
        cell = self.cell
        X0 = X[cell[:, 0], :]
        E = X[cell[:, 1:], :] - X0[:, None, :] # (NC, GD, GD)
        E = bm.permute_dims(E, axes=(0,2,1))
        return E
    
    def A(self,E , E_hat , M_inv):
        """
        基本的雅可比算子组件 A
        A = E_hat E_K^{-1}M^{-1} E_K^{-T} E_hat^T
        """
        E_K_inv = bm.linalg.inv(E)
        E_K_inv_T = bm.swapaxes(E_K_inv , -1,-2)
        E_hat_T = bm.swapaxes(E_hat , -1,-2)
        A = E_hat @ E_K_inv @ M_inv @ E_K_inv_T @ E_hat_T
        return A
    
    def rho(self,M):
        """
        权重函数 rho , 一般形式
        rho = (det(M))^{1/2}
        Parameters
           M: (NC, GD, GD)
        """
        rho = bm.sqrt(bm.linalg.det(M))
        return rho
    
    def theta(self, M):
        """
        积分乘子 theta
        theta = 
        """
        d = self.GD
        cm = self.cm
        NC = self.NC
        det_M = bm.linalg.det(M)
        rho_K = bm.sqrt(det_M)
        sigma = bm.sum(cm * rho_K , axis=0)
        area = bm.sum(cm , axis=0)
        theta = (sigma/area)**(-2/d)
        return theta
        
    def T(self, theta ,A):
        """
        (...)^gamma 型
        T = (1/2 * |A|^2_F + theta*(tr(A) - det(A)^{1/d}))^gamma
          = H^gamma
        Parameters:
            theta(float): 积分全局乘子
            A(Tensor): 雅可比算子
        Return:
            T(float): T 函数值
        """
        gamma = self.gamma
        T = self.H(theta , A)**gamma
        return T

    def H(self, theta , A):
        """
        H = 1/2 * |A|^2_F + theta*(tr(A) -d*det(A)^{1/d})

        Parameters:
            theta(float): 积分全局乘子
            A(Tensor): 雅可比算子
        Return:
            H(Tensor): H 函数值
        """
        d = self.GD
        trace_A = bm.trace(A,axis1=-2,axis2=-1)
        det_A = bm.linalg.det(A)
        A_Fnorm = bm.sum(A**2 , axis=(-2,-1))
        H = 0.5 * A_Fnorm + theta * (trace_A - d * det_A**(1/d))
        return H
    
    def TdA(self,theta,A):
        """
        (...)^gamma 型
        基于几何离散的 T 泛函关于算子 A 的导数
        T = (H)^r
        A = E_hat E_K^{-1}M^{-1} E_K^{-T} E_hat^T
        
        pT/pA = gamma * H^{gamma-1} *(A + theta I)
        """
        gamma = self.gamma
        H = self.H(theta , A)
        d = A.shape[-1]
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(A , **self.kwargs0)
        I += I_base
        pT_pA = (gamma * H**(gamma-1))[...,None,None] * (A + theta * I)
        return pT_pA
    
    def Tdg(self, theta,A ,det_A):
        """
        (...)^gamma 型
        基于几何离散的 T 泛函关于标量 g 的导数
        g = det(A)

        pT/pg = - gamma* H^{gamma-1} * theta * g^{1/d - 1}
        """
        gamma = self.gamma
        H = self.H(theta , A)
        d = self.GD
        pT_pg = - gamma * H**(gamma-1) * theta * det_A**(1/d - 1)
        return pT_pg
    
    def TdA_3(self,theta,A):
        """
        测试去掉 tr(A)
        """
        gamma = self.gamma
        d = self.GD
        det_A = bm.linalg.det(A)
        A_Fnorm = bm.sum(A**2 , axis=(-2,-1))
        H = 0.5 * A_Fnorm + theta * ( - d * det_A**(1/d)) + 1/2 * d * theta**2
        d = A.shape[-1]
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(A , **self.kwargs0)
        I += I_base
        pT_pA = (gamma * H**(gamma-1))[...,None,None] * A
        return pT_pA
    
    def Tdg_3(self,theta,A , det_A):
        """
        测试去掉 tr(A)
        """
        gamma = self.gamma
        d = self.GD
        A_Fnorm = bm.sum(A**2 , axis=(-2,-1))
        H = 0.5 * A_Fnorm + theta * ( - d * det_A**(1/d)) + 1/2 * d * theta**2
        pT_pg = - gamma * H**(gamma-1) * theta * det_A**(1/d - 1)
        return pT_pg
    
    def TdA_2(self,theta ,A):
        """
        缩放指数
        基于几何离散的 T 泛函关于标量 A 的导数
        T = 1/2 * |A|_F^{d*gamma/2} + (theta*tr(A))^{d*gamma/4} - (d*theta*det(A)^{1/d})^{d*gamma/4}
        pT/pA = d*gamma/4 *(|A|_F^{d*gamma/2 - 2} * A + theta^{d*gamma/4} * tr(A)^{d*gamma/4 - 1} * I)
        """
        gamma = self.gamma
        d = self.GD
        A_Fnorm = bm.sum(A**2 , axis=(-2,-1))
        trace_A = bm.trace(A,axis1=-2,axis2=-1)
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(A , **self.kwargs0)
        I += I_base
        term1 = (d*gamma/4) * (A_Fnorm**(d*gamma/4 - 1))[...,None,None] * A
        term2 = (d*gamma/4) * theta**(d*gamma/4) * (trace_A**(d*gamma/4 - 1))[...,None,None] * I
        pT_pA = term1 + term2
        return pT_pA

    def Tdg_2(self, theta , det_A):
        """
        缩放指数
        基于几何离散的 T 泛函关于标量 g 的导数
        g = det(A)
        pT/pg = - gamma/4 * (d*theta)^{d*gamma/4} * det(A)^{gamma/4 - 1}
        """
        gamma = self.gamma
        d = self.GD
        pT_pg = - gamma/4 * (d*theta)**(d*gamma/4) * det_A**(gamma/4 - 1)
        return pT_pg
    
    def TdA_4(self,theta ,A):
        """
        缩放指数
        基于几何离散的 T 泛函关于标量 A 的导数
        T = (H)^gamma
        H = 1/2 * |A|_F^{d/2} + (theta*tr(A))^{d/4} - (d*theta*det(A)^{1/d})^{d/4}
        pT/pA = gamma * H^{gamma-1} * [ d/4 * |A|_F^{d/2 - 2} * A + d/4 * theta^{d/4} * tr(A)^{d/4 - 1} * I ]
        """
        gamma = self.gamma
        d = self.GD
        A_Fnorm = bm.sqrt(bm.sum(A**2 , axis=(-2,-1)))
        trace_A = bm.trace(A,axis1=-2,axis2=-1)
        det_A = bm.linalg.det(A)
        H = 0.5 * A_Fnorm**(d/2) + (theta*trace_A)**(d/4) - (d*theta*det_A**(1/d))**(d/4)
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(A , **self.kwargs0)
        I += I_base
        term1 = (d/4) * (A_Fnorm**(d/2 - 2))[...,None,None] * A
        term2 = (d/4) * theta**(d/4) * (trace_A**(d/4 - 1))[...,None,None] * I
        pT_pA = (gamma * H**(gamma-1))[...,None,None] * (term1 + term2)
        return pT_pA
    
    def Tdg_4(self, theta ,A, det_A):
        """
        缩放指数
        基于几何离散的 T 泛函关于标量 g 的导数
        g = det(A)
        pT/pg = - gamma * H^{gamma-1} * 1/4 * (d*theta)^{d/4} * det(A)^{1/4 - 1}
        """
        gamma = self.gamma
        d = self.GD
        det_A = bm.linalg.det(A)
        A_Fnorm = bm.sqrt(bm.sum(A**2 , axis=(-2,-1)))
        trace_A = bm.trace(A,axis1=-2,axis2=-1)
        H = 0.5 * A_Fnorm**(d/2) + (theta*trace_A)**(d/4) - (d*theta*det_A**(1/d))**(d/4)
        pT_pg = - gamma * H**(gamma-1) * 1/4 * (d*theta)**(d/4) * det_A**(1/4 - 1)
        return pT_pg
    
    def TdA_5(self,theta ,A):
        """
        T = (H)^gamma
        H = d^{1/2} * |A|_F + theta*(tr(A) -2 * d * det(A)^{1/d})
        pT/pA = gamma * H^{gamma-1} * [ d^{1/2} * A / |A|_F + theta * I ]
        """
        gamma = self.gamma
        d = self.GD
        A_Fnorm = bm.sqrt(bm.sum(A**2 , axis=(-2,-1)))
        trace_A = bm.trace(A,axis1=-2,axis2=-1)
        det_A = bm.linalg.det(A)
        H = d**(1/2) * A_Fnorm + theta * (trace_A - 2 * d * det_A**(1/d))
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(A , **self.kwargs0)
        I += I_base
        term1 = d**(1/2) * (A_Fnorm**(-1))[...,None,None] * A
        term2 = theta * I
        pT_pA = (gamma * H**(gamma-1))[...,None,None] * (term1 + term2)
        return pT_pA
    
    def Tdg_5(self, theta ,A, det_A):
        """
        T = (H)^gamma
        H = d^{1/2} * |A|_F + theta*(tr(A) -2 * d * det(A)^{1/d})
        pT/pg = - gamma * H^{gamma-1} * theta * 2 * det(A)^{1/d - 1}
        """
        gamma = self.gamma
        d = self.GD
        A_Fnorm = bm.sqrt(bm.sum(A**2 , axis=(-2,-1)))
        trace_A = bm.trace(A,axis1=-2,axis2=-1)
        det_A = bm.linalg.det(A)
        H = d**(1/2) * A_Fnorm + theta * (trace_A - 2 * d * det_A**(1/d))
        pT_pg = - gamma * H**(gamma-1) * theta * 2 * det_A**(1/d - 1)
        return pT_pg
    
    def TdA_6(self,theta,A):
        """
        T = lambda *(tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        TdA = lambda * d*gamma/2 * tr(A)^{d*gamma/2 - 1} * I 
        """
        lam = self.lam(theta)
        gamma = self.gamma
        d = self.GD
        trace_A = bm.trace(A,axis1=-2,axis2=-1)
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(A , **self.kwargs0)
        I += I_base
        pT_pA = 1/lam * (d*gamma/2) * (trace_A**(d*gamma/2 - 1))[...,None,None] * I
        return pT_pA
    
    def Tdg_6(self, theta ,det_A):
        """
        T = lambda *(tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        Tdg = -lambda * d^{d*gamma/2} * gamma/2* theta^{d*gamma/2} * 1/det(A)
        """
        lam = self.lam(theta)
        gamma = self.gamma
        d = self.GD
        pT_pg = - 1/lam * d**(d*gamma/2) * gamma/2 * theta**(d*gamma/2) * det_A**(-1)
        return pT_pg

    def lam(self , theta):
        """
        拉伸因子 lambda
        """
        d = self.GD
        gamma = self.gamma
        power = d * gamma / 2
        lam = d**(power) * theta**(power)*(1- power * bm.log(theta))
        return lam

    def balance(self,M_node):
        """
        (...)^gamma 型
        平衡函数 P 为 (NN,NN) 的对角矩阵,实际组装时只需对角元
        P = diag( det(M)^{n})
        n = 1/d * (2 * gamma - d/2)
        Parameters:
            M_node(Tensor): 目标单元度量张量 (NN, GD, GD)
            theta(float): 积分全局乘子
        """
        d = self.GD
        gamma = self.gamma
        det_M_node = bm.linalg.det(M_node)
        n = 1/d * (gamma - d/2)
        P_diag = det_M_node**n # (NN,)
        return P_diag
    
    def balance_2(self,M_node):
        """
        缩放指数
        平衡函数 P 为 (NN,NN) 的对角矩阵,实际组装时只需对角元
        P = diag( det(M)^{m})
        m = 1/2 * (gamma - 1)
        Parameters:
            M_node(Tensor): 目标单元度量张量 (NN, GD, GD)
            theta(float): 积分全局乘子
        """
        d = self.GD
        gamma = self.gamma
        det_M_node = bm.linalg.det(M_node)
        m = 1/2 * (gamma - 1)
        P_diag = det_M_node**m # (NN,)
        return P_diag
    
    def balance_3(self,M_node):
        """
        T = (H)^gamma
        H = d^{1/2} * |A|_F + theta*(tr(A) -2 * det(A)^{1/d})
        平衡函数 P 为 (NN,NN) 的对角矩阵,实际组装时只需对角元
        P = diag( det(M)^{k})
        k = 1/d * ( gamma -d/2)
        Parameters:
            M_node(Tensor): 目标单元度量张量 (NN, GD, GD)
            theta(float): 积分全局乘子
        """
        d = self.GD
        gamma = self.gamma
        det_M_node = bm.linalg.det(M_node)
        k = 1/d * ( gamma - d/2)
        P_diag = det_M_node**k # (NN,)
        return P_diag
    
    def balance_4(self,M_node):
        d = self.GD
        det_M_node = bm.linalg.det(M_node)
        k = -1/2
        P_diag = det_M_node**k # (NN,)
        return P_diag
    
    def R_matrix(self):
        """
        局部赋值矩阵 R
        R = [-1, -1 ,..., -1
              1,  0 ,...,  0
              0,  1 ,...,  0
              ...
              0,  0 ,...,  1]
        """
        dim = self.mesh.localFace.shape[0]
        R = bm.zeros((dim, dim-1), **self.kwargs0)
        R[0, :] = -1
        R[1:, :] = bm.eye(dim-1, **self.kwargs0)
        return R

    def Idxi_from_Ehat(self, E_hat, E_K, rho, M_inv, theta):
        """
        泛函局部导数 Idxi 的计算
        Idxi = 2 * rho * R @ [ E_hat^{-1} A TdA + (Tdg * g) E_hat^{-1} ]
        Parameters:
            E_hat(Tensor): 参考单元边矩阵 (NC, GD, GD)
            E_K(Tensor): 物理单元边矩阵 (NC, GD, GD)
            rho(Tensor): 权重函数 (NC,)
            M_inv(Tensor): 目标度量张量的逆 (NC, GD, GD)
        """
        E_hat_inv = bm.linalg.inv(E_hat)
        A = self.A(E_K , E_hat , M_inv)
        g = bm.linalg.det(A)
        TdA = self.TdA(theta, A)
        Tdg = self.Tdg(theta,A,g)
        
        term0 = E_hat_inv @ A @ TdA
        term1 = (Tdg * g)[..., None, None] * E_hat_inv
        Idxi_grad_part = rho[..., None, None] * (term0 + term1) # (NC, GD, GD)
        Idxi = self.R[None,...] @ Idxi_grad_part # (NC, GD+1, GD)
        return Idxi
    
    def BD_projector(self):
        """
        构造按“变量列”右乘的边界投影矩阵系数（2D）
        全局变量排列为 [X(:,0); X(:,1)] （Fortran/列优先展开）
        列投影: [Jx Jy] @ [[Pxx Pxy],[Pyx Pyy]]
        其中 P** 为 NN×NN 的对角稀疏矩阵（非边界为单位/零，边界为对应系数）
        """
        assert self.GD == 2, "当前 BD_projector 仅示例 2D，3D 请仿照扩展"

        NN = self.NN
        idx = self.Bdinnernode_idx              # 边界节点索引 (nb,)
        n = self.Bi_Lnode_normal                # (nb, 2), 每个边界节点的单位法向

        # 默认对角：非边界 Px=Py=1，交叉为 0
        one = bm.ones(NN, **self.kwargs0)
        zero = bm.zeros(NN, **self.kwargs0)
        d_xx = one.copy()
        d_yy = one.copy()
        d_xy = zero.copy()
        d_yx = zero.copy()

        # 边界节点的投影系数
        nx, ny = n[:, 0], n[:, 1]
        d_xx = bm.set_at(d_xx, idx, 1.0 - nx*nx)
        d_yy = bm.set_at(d_yy, idx, 1.0 - ny*ny)
        d_xy = bm.set_at(d_xy, idx, -nx*ny)
        d_yx = bm.set_at(d_yx, idx, -ny*nx)

        # 角点（完全固定）列也置零
        if hasattr(self, 'Vertices_idx'):
            vi = self.Vertices_idx
            d_xx = bm.set_at(d_xx, vi, 0.0)
            d_yy = bm.set_at(d_yy, vi, 0.0)
            d_xy = bm.set_at(d_xy, vi, 0.0)
            d_yx = bm.set_at(d_yx, vi, 0.0)

        # 存成稀疏对角矩阵
        self.Rxx = spdiags(d_xx, 0, NN, NN, format='coo').to_scipy()
        self.Ryy = spdiags(d_yy, 0, NN, NN, format='coo').to_scipy()
        self.Rxy = spdiags(d_xy, 0, NN, NN, format='coo').to_scipy()
        self.Ryx = spdiags(d_yx, 0, NN, NN, format='coo').to_scipy()
        
        self.rxx = self.Rxx.diagonal()
        self.ryy = self.Ryy.diagonal()
        self.rxy = self.Rxy.diagonal()
        self.ryx = self.Ryx.diagonal()
    
    def _build_jac_pattern(self):
        """
        只建一次：雅可比稀疏结构图样（行列下标），便于每次仅填 data。
        """
        if hasattr(self, "_jac_pat_ready") and self._jac_pat_ready:
            return

        cell = self.cell              # (NC, d+1)
        NN   = self.NN
        d    = self.GD
        NC   = self.NC

        # 每个 c=1..d 的“列节点索引向量”（与行 rows 对齐）
        rows_base = cell.reshape(-1)  # (NC*(d+1),)
        cols_c = [bm.repeat(cell[:, c], d+1) for c in range(1, d+1)]  # d 个 (NC*(d+1),)
        cols_0 = bm.repeat(cell[:, 0], d+1)                           # (NC*(d+1),)

        # 行索引：对 c=1..d 拼接（同样的 rows 重复 d 次），以及 c=0 一份
        self._rows_tile_d = bm.tile(rows_base, d)     # (d*NC*(d+1),)
        self._rows_base   = rows_base                 # (NC*(d+1),)

        # 列索引：x/y 两组，c=1..d 和 c=0
        self._cols_x_tile_d = bm.concat(cols_c, axis=0)        # (d*NC*(d+1),)
        self._cols_y_tile_d = self._cols_x_tile_d + NN
        self._cols_x_0      = cols_0                           # (NC*(d+1),)
        self._cols_y_0      = cols_0 + NN

        # 行偏移（vx 行: 0..NN-1；vy 行: NN..2NN-1）
        self._row_off_x_tile_d = self._rows_tile_d
        self._row_off_y_tile_d = self._rows_tile_d + NN
        self._row_off_x_0      = self._rows_base
        self._row_off_y_0      = self._rows_base + NN

        # 完成
        self._jac_pat_ready = True
    
    def vector_construction(self,Xi , M_inv):
        """
        构造全局移动向量场
        Parameters:
            X: (NN, GD) 当前网格节点坐标
            Xi: (NN, GD) 参考网格节点坐标
            M: (NC, GD, GD) 当前网格度量张量
            M_inv: (NC, GD, GD) 当前网格度量张量的逆
        Returns:
            v: (NN, GD) 参考网格上的全局移动向量场
            Idxi: (NC, GD+1, GD) 单元局部参考坐标的泛函导数
        """
        E_hat = self.edge_matrix(Xi)  # (NC, GD, GD)

        cache = self._ivp_cache
        if cache is not None:
            E_K     = cache['E_K'];     E_K_inv = cache['E_K_inv']
            det_E_K = cache['det_E_K']; rho     = cache['rho']     
            cm      = cache['cm'];      P_diag  = cache['P_diag']
            theta   = cache['theta']

        Idxi = self.Idxi_from_Ehat(E_hat, E_K, rho, M_inv, theta)  # (NC, GD+1, GD)
        cell = self.cell
        cm = self.cm
        global_vector = bm.zeros((self.NN, self.GD), dtype=bm.float64)
        global_vector = bm.index_add(global_vector , cell , cm[:,None,None] * Idxi)

        tau = self.tau
        v = -1/tau * global_vector * P_diag[:, None]  # (NN, GD)
        
        # 边界投影和角点固定
        Bi_Lnode_normal = self.Bi_Lnode_normal
        Bdinnernode_idx = self.Bdinnernode_idx
        dot = bm.sum(Bi_Lnode_normal * v[Bdinnernode_idx],axis=1)
        v = bm.set_at(v , Bdinnernode_idx ,
                        v[Bdinnernode_idx] - dot[:,None] * Bi_Lnode_normal)
        vertice_idx = self.Vertices_idx
        v = bm.set_at(v , vertice_idx , 0)
        
        return v

    def JAC_functional(self,Xi,M_inv):
        """
        (...)^gamma 型
        构造算法：
        d E_hat = e_k x e_c^T
        d E_hat_inv = - E_hat_inv dE_hat E_hat_inv
        d A =  2*M_inv (dE_hat E_K^{-1})^T
        d g =  g * tr(A^{-1} dA)
        d H = (A + theta I) : dA - theta * g^{1/d-1} * dg
        d (TdA) = gamma *(gamma-1)* H^{gamma-2} dH * (A + theta I) + gamma * H^{gamma-1} * dA
        d (Tdg) = - gamma *(gamma-1)* H^{gamma-2} dH * theta * g^{1/d-1} - 
                    gamma * H^{gamma-1} * theta * (1/d - 1) * g^{1/d-2} * dg
        P1 = 2 * rho * d E_hat_inv (A TdA + Tdg * g * I)
        P2 = 2 * rho * E_hat_inv (dA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        J = R @ (P1 + P2)
        """
        d   = self.GD
        NN  = self.NN
        NC  = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同张量结构扩展"

        # 缓存（在 solve_ivp 步内不变）
        cache = getattr(self, '_ivp_cache', None)
        if cache is None:
            raise RuntimeError("ivp 缓存未准备，请先调用 _prepare_ivp_cache")
        E_K     = cache['E_K']
        E_K_inv = cache['E_K_inv']
        rho     = cache['rho']
        theta   = cache['theta']

        # 基本量
        E_hat    = self.edge_matrix(Xi)                     # (NC,d,d)
        Einv     = bm.linalg.inv(E_hat)                     # (NC,d,d)
        Ehat_T   = bm.swapaxes(E_hat, -1, -2)               # (NC,d,d)
        C        = E_K_inv @ M_inv @ bm.swapaxes(E_K_inv, -1, -2)  # (NC,d,d) = E_K^{-1} M^{-1} E_K^{-T}
        A        = E_hat @ C @ Ehat_T                       # (NC,d,d)
        I_d      = bm.eye(d, **self.kwargs0)
        I_mat    = bm.zeros_like(A, **self.kwargs0); I_mat += I_d
        trA      = bm.trace(A, axis1=-2, axis2=-1)          # (NC,)
        A_F2     = bm.sum(A*A, axis=(-2,-1))                # (NC,)
        g        = bm.linalg.det(A)                          # (NC,)
        # 数值稳健性
        eps = 1e-14
        g_pos    = bm.maximum(g, eps)
        H        = 0.5*A_F2 + theta*(trA - d * g_pos**(1.0/d))   # (NC,)
        H_pos    = bm.maximum(H, eps)

        gamma = self.gamma
        S     = A + theta * I_mat                             # (NC,d,d)
        H_gm1 = H_pos**(gamma-1.0)
        TdA   = (gamma * H_gm1)[..., None, None] * S          # (NC,d,d)
        Tdg   = -gamma * H_gm1 * theta * (g_pos**(1.0/d - 1.0))   # (NC,)

        # 预备构件（用于 δA ）
        U = E_hat @ C                        # (NC,d,d)
        W = C @ Ehat_T                       # (NC,d,d)
        # δE_hat^{-1} = −(b_k ⊗ row_c(Einv))，一次性生成 (NC,c,k,d,d)
        b_all        = bm.swapaxes(Einv, -1, -2)              # (NC,d,d) 列向量转行表示：b_all[:,k,:] = Einv[:, :, k]^T
        row_c_Einv   = Einv                                   # (NC,c,d)
        dEinv_all = -( b_all[:, None, :, :, None] * row_c_Einv[:, :, None, None, :] )  # (NC,c,k,d,d)

        # δA_all = (e_k ⊗ row_c(W)) + (col_c(U) ⊗ e_k^T)，全向量化
        eye = bm.eye(d, **self.kwargs0)                       # (d,d)
        row_sel = eye[None, None, :, :, None]                 # (1,1,k,d,1) 选定“行 k”
        col_sel = eye[None, None, :, None, :]                 # (1,1,k,1,d) 选定“列 k”
        # row_c(W): (NC,c,d), col_c(U): (NC,c,d)
        drow_all = W                                          # (NC,c,d) 使用 W 的第 c 行
        ucol_all = bm.permute_dims(U, (0, 2, 1))              # (NC,c,d) 使用 U 的第 c 列
        dA_row   = row_sel * drow_all[:, :, None, None, :]    # (NC,c,k,d,d)，仅第 k 行非零
        dA_col   = ucol_all[:, :, None, :, None] * col_sel    # (NC,c,k,d,d)，仅第 k 列非零
        dA_all   = dA_row + dA_col                            # (NC,c,k,d,d)
    
        # δg_all = g * tr(A^{-1} δA)
        Ainv    = bm.linalg.inv(A)                                           # (NC,d,d)
        tr_term = bm.sum(Ainv[:, None, None, :, :] * dA_all, axis=(3,4))   # (NC,c,k)
        dg_all  = (g_pos[:, None, None]) * tr_term                           # (NC,c,k)

        # δH_all = S:δA − theta*g^{1/d-1}*δg
        S_dot_dA = bm.sum(S[:, None, None, :, :] * dA_all, axis=(3,4))        # (NC,c,k)
        g_pow    = g_pos**(1.0/d - 1.0)                                       # (NC,)
        dH_all   = S_dot_dA - theta * g_pow[:, None, None] * dg_all            # (NC,c,k)

        # d(TdA)_all
        f1 = gamma*(gamma-1.0) * (H_pos**(gamma-2.0))                          # (NC,)
        term1 = (f1[:, None, None] * dH_all)[..., None, None] * S[:, None, None, :, :]   # (NC,c,k,d,d)
        term2 = (gamma * H_gm1)[..., None, None, None, None] * dA_all                      # (NC,c,k,d,d)
        dTdA_all = term1 + term2

        # d(Tdg)_all（标量）
        coef1 = -f1 * theta * g_pow      # (NC,)
        coef2 = -gamma * H_gm1 * theta * (1.0/d - 1.0) * (g_pos**(1.0/d - 2.0))# (NC,)
        dTdg_all = coef1[:, None, None] * dH_all + coef2[:, None, None] * dg_all   # (NC,c,k)

        # P1 = 2ρ·δÊ^{-1}·(A TdA + Tdg g I)
        M0 = A @ TdA + (Tdg * g_pos)[..., None, None] * I_mat                   # (NC,d,d)
        P1_all = 2.0 * rho[:, None, None, None, None] * bm.einsum('nckij,njl->nckil', dEinv_all, M0)

        # P2 = 2ρ·Ê^{-1}·(δA TdA + A d(TdA) + d(Tdg) g I + Tdg δg I)
        part_a = bm.einsum('nckij,njl->nckil', dA_all, TdA)                     # δA TdA
        part_b = bm.einsum('nij,nckjl->nckil', A, dTdA_all)                     # A d(TdA)
        part_c = (dTdg_all * g_pos[:, None, None])[..., None, None] * I_mat[:, None, None, :, :]   # d(Tdg) g I
        part_d = (Tdg[:, None, None] * dg_all)[..., None, None] * I_mat[:, None, None, :, :]       # Tdg δg I
        bracket = part_a + part_b + part_c + part_d                              # (NC,c,k,d,d)
        P2_all  = 2.0 * rho[:, None, None, None, None] * bm.einsum('nij,nckjl->nckil', Einv, bracket)

        # 局部二阶块（对 j'=1..d 的“梯度部分”），形状 (NC,c,k,d,matrix_cols=j')
        D2_all = P1_all + P2_all                                                # (NC,c,k,d,d)

        JAC = self.JAC_assembly(D2_all)
        return JAC
    
    def JAC_functional_3(self,Xi,M_inv):
        """
        (...)^gamma 型
        构造算法：
        d E_hat = e_k x e_c^T
        d E_hat_inv = - E_hat_inv dE_hat E_hat_inv
        d A =  2*M_inv (dE_hat E_K^{-1})^T
        d g =  g * tr(A^{-1} dA)
        d H = (A) : dA - theta * g^{1/d-1} * dg
        d (TdA) = gamma *(gamma-1)* H^{gamma-2} dH * (A) + gamma * H^{gamma-1} * dA
        d (Tdg) = - gamma *(gamma-1)* H^{gamma-2} dH * theta * g^{1/d-1} - 
                    gamma * H^{gamma-1} * theta * (1/d - 1) * g^{1/d-2} * dg
        P1 = 2 * rho * d E_hat_inv (A TdA + Tdg * g * I)
        P2 = 2 * rho * E_hat_inv (dA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        J = R @ (P1 + P2)
        """
        d   = self.GD
        NN  = self.NN
        NC  = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同张量结构扩展"

        # 缓存（在 solve_ivp 步内不变）
        cache = getattr(self, '_ivp_cache', None)
        if cache is None:
            raise RuntimeError("ivp 缓存未准备，请先调用 _prepare_ivp_cache")
        E_K     = cache['E_K']
        E_K_inv = cache['E_K_inv']
        rho     = cache['rho']
        theta   = cache['theta']

        # 基本量
        E_hat    = self.edge_matrix(Xi)                     # (NC,d,d)
        Einv     = bm.linalg.inv(E_hat)                     # (NC,d,d)
        Ehat_T   = bm.swapaxes(E_hat, -1, -2)               # (NC,d,d)
        C        = E_K_inv @ M_inv @ bm.swapaxes(E_K_inv, -1, -2)  # (NC,d,d) = E_K^{-1} M^{-1} E_K^{-T}
        A        = E_hat @ C @ Ehat_T                       # (NC,d,d)
        I_d      = bm.eye(d, **self.kwargs0)
        I_mat    = bm.zeros_like(A, **self.kwargs0); I_mat += I_d
  
        A_F2     = bm.sum(A*A, axis=(-2,-1))                # (NC,)
        g        = bm.linalg.det(A)                          # (NC,)
        # 数值稳健性
        eps = 1e-14
        g_pos    = bm.maximum(g, eps)
        H        = 0.5*A_F2 + theta*( - d * g_pos**(1.0/d)) + 1/2 * d * theta**2   # (NC,)
        H_pos    = bm.maximum(H, eps)

        gamma = self.gamma
        S     = A 
        H_gm1 = H_pos**(gamma-1.0)
        TdA   = (gamma * H_gm1)[..., None, None] * S          # (NC,d,d)
        Tdg   = -gamma * H_gm1 * theta * (g_pos**(1.0/d - 1.0))   # (NC,)

        # 预备构件（用于 δA ）
        U = E_hat @ C                        # (NC,d,d)
        W = C @ Ehat_T                       # (NC,d,d)
        # δE_hat^{-1} = −(b_k ⊗ row_c(Einv))，一次性生成 (NC,c,k,d,d)
        b_all        = bm.swapaxes(Einv, -1, -2)              # (NC,d,d) 列向量转行表示：b_all[:,k,:] = Einv[:, :, k]^T
        row_c_Einv   = Einv                                   # (NC,c,d)
        dEinv_all = -( b_all[:, None, :, :, None] * row_c_Einv[:, :, None, None, :] )  # (NC,c,k,d,d)

        # δA_all = (e_k ⊗ row_c(W)) + (col_c(U) ⊗ e_k^T)，全向量化
        eye = bm.eye(d, **self.kwargs0)                       # (d,d)
        row_sel = eye[None, None, :, :, None]                 # (1,1,k,d,1) 选定“行 k”
        col_sel = eye[None, None, :, None, :]                 # (1,1,k,1,d) 选定“列 k”
        # row_c(W): (NC,c,d), col_c(U): (NC,c,d)
        drow_all = W                                          # (NC,c,d) 使用 W 的第 c 行
        ucol_all = bm.permute_dims(U, (0, 2, 1))              # (NC,c,d) 使用 U 的第 c 列
        dA_row   = row_sel * drow_all[:, :, None, None, :]    # (NC,c,k,d,d)，仅第 k 行非零
        dA_col   = ucol_all[:, :, None, :, None] * col_sel    # (NC,c,k,d,d)，仅第 k 列非零
        dA_all   = dA_row + dA_col                            # (NC,c,k,d,d)
    
        # δg_all = g * tr(A^{-1} δA)
        Ainv    = bm.linalg.inv(A)                                           # (NC,d,d)
        tr_term = bm.sum(Ainv[:, None, None, :, :] * dA_all, axis=(3,4))   # (NC,c,k)
        dg_all  = (g_pos[:, None, None]) * tr_term                           # (NC,c,k)

        # δH_all = S:δA − theta*g^{1/d-1}*δg
        S_dot_dA = bm.sum(S[:, None, None, :, :] * dA_all, axis=(3,4))        # (NC,c,k)
        g_pow    = g_pos**(1.0/d - 1.0)                                       # (NC,)
        dH_all   = S_dot_dA - theta * g_pow[:, None, None] * dg_all            # (NC,c,k)

        # d(TdA)_all
        f1 = gamma*(gamma-1.0) * (H_pos**(gamma-2.0))                          # (NC,)
        term1 = (f1[:, None, None] * dH_all)[..., None, None] * S[:, None, None, :, :]   # (NC,c,k,d,d)
        term2 = (gamma * H_gm1)[..., None, None, None, None] * dA_all                      # (NC,c,k,d,d)
        dTdA_all = term1 + term2

        # d(Tdg)_all（标量）
        coef1 = -f1 * theta * g_pow      # (NC,)
        coef2 = -gamma * H_gm1 * theta * (1.0/d - 1.0) * (g_pos**(1.0/d - 2.0))# (NC,)
        dTdg_all = coef1[:, None, None] * dH_all + coef2[:, None, None] * dg_all   # (NC,c,k)

        # P1 = 2ρ·δÊ^{-1}·(A TdA + Tdg g I)
        M0 = A @ TdA + (Tdg * g_pos)[..., None, None] * I_mat                   # (NC,d,d)
        P1_all = 2.0 * rho[:, None, None, None, None] * bm.einsum('nckij,njl->nckil', dEinv_all, M0)

        # P2 = 2ρ·Ê^{-1}·(δA TdA + A d(TdA) + d(Tdg) g I + Tdg δg I)
        part_a = bm.einsum('nckij,njl->nckil', dA_all, TdA)                     # δA TdA
        part_b = bm.einsum('nij,nckjl->nckil', A, dTdA_all)                     # A d(TdA)
        part_c = (dTdg_all * g_pos[:, None, None])[..., None, None] * I_mat[:, None, None, :, :]   # d(Tdg) g I
        part_d = (Tdg[:, None, None] * dg_all)[..., None, None] * I_mat[:, None, None, :, :]       # Tdg δg I
        bracket = part_a + part_b + part_c + part_d                              # (NC,c,k,d,d)
        P2_all  = 2.0 * rho[:, None, None, None, None] * bm.einsum('nij,nckjl->nckil', Einv, bracket)

        # 局部二阶块（对 j'=1..d 的“梯度部分”），形状 (NC,c,k,d,matrix_cols=j')
        D2_all = P1_all + P2_all                                                # (NC,c,k,d,d)
        JAC = self.JAC_assembly(D2_all)
        return JAC
    
    def JAC_functional_2(self,Xi,M_inv):
        """
        T = 1/2 * |A|_F^{d*gamma/2} + (theta*tr(A))^{d*gamma/4} - (d*theta*det(A)^{1/d})^{d*gamma/4}
        TdA = d*gamma/4 *(|A|_F^{d*gamma/2 - 2} * A + theta^{d*gamma/4} * tr(A)^{d*gamma/4 - 1} * I)
        Tdg = - gamma/4 * (d*theta)^{d*gamma/4} * det(A)^{gamma/4 - 1}
        d E_hat = e_k x e_c^T
        d E_hat_inv = - E_hat_inv dE_hat E_hat_inv
        d A =  2*M_inv (dE_hat E_K^{-1})^T
        d g =  g * tr(A^{-1} dA)
        d (TdA) = d*gamma/4 * ( (d*gamma/2 - 2) * |A|_F^{d*gamma/2 - 4} * (A : dA) * A + 
                    |A|_F^{d*gamma/2 - 2} * dA + 
                    theta^{d*gamma/4} * (d*gamma/4 - 1) * tr(A)^{d*gamma/4 - 2} * tr(dA) * I )
        d (Tdg) = - gamma/4 * (gamma/4 - 1) * (d*theta)^{d*gamma/4} * det(A)^{gamma/4 - 2} * dg
        P1 = 2 * rho * d E_hat_inv (A TdA + Tdg * g * I)
        P2 = 2 * rho * E_hat_inv (dA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        J = R @ (P1 + P2)
        """
        d   = self.GD
        NN  = self.NN
        NC  = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同结构扩展"
        print("使用 JAC_functional_2")
        # 缓存
        cache = getattr(self, '_ivp_cache', None)
        if cache is None:
            raise RuntimeError("ivp 缓存未准备，请先调用 _prepare_ivp_cache")
        E_K     = cache['E_K']
        E_K_inv = cache['E_K_inv']
        rho     = cache['rho']
        theta   = cache['theta']

        # 基本量
        E_hat    = self.edge_matrix(Xi)                     # (NC,d,d)
        Einv     = bm.linalg.inv(E_hat)                     # (NC,d,d)
        Ehat_T   = bm.swapaxes(E_hat, -1, -2)               # (NC,d,d)
        C        = E_K_inv @ M_inv @ bm.swapaxes(E_K_inv, -1, -2)  # (NC,d,d)
        A        = E_hat @ C @ Ehat_T                       # (NC,d,d)
        I_d      = bm.eye(d, **self.kwargs0)
        I_mat    = bm.zeros_like(A, **self.kwargs0); I_mat += I_d
        trA      = bm.trace(A, axis1=-2, axis2=-1)          # (NC,)
        A_F2     = bm.sqrt(bm.sum(A*A, axis=(-2,-1)))                # (NC,)
        g        = bm.linalg.det(A)                         # (NC,)
        eps = 1e-14
        g_pos    = bm.maximum(g, eps)

        # 目标泛函及导数
        gamma = self.gamma
        # T = 1/2 * |A|_F^{d*gamma/2} + (theta*tr(A))^{d*gamma/4} - (d*theta*det(A)^{1/d})^{d*gamma/4}

        # TdA = d*gamma/4 *(|A|_F^{d*gamma/2 - 2} * A + theta^{d*gamma/4} * tr(A)^{d*gamma/4 - 1} * I)
        A_Fnorm_pow2 = (A_F2) ** (d*gamma/2 - 2)
        trA_pow2 = (theta * trA) ** (d*gamma/4 - 1)
        TdA = (d*gamma/4)* (A_Fnorm_pow2[...,None,None] * A + (theta**(d*gamma/4)) * trA_pow2[...,None,None] * I_mat)

        # Tdg = - gamma/4 * (d*theta)^{d*gamma/4} * det(A)^{gamma/4 - 1}
        Tdg = - gamma/4 * (d*theta)**(d*gamma/4) * g_pos**(gamma/4 - 1)

        # 预备构件（用于 δA ）
        U = E_hat @ C                        # (NC,d,d)
        W = C @ Ehat_T                       # (NC,d,d)
        # δE_hat^{-1} = −(b_k ⊗ row_c(Einv))，一次性生成 (NC,c,k,d,d)
        b_all        = bm.swapaxes(Einv, -1, -2)              # (NC,d,d)
        row_c_Einv   = Einv                                   # (NC,c,d)
        dEinv_all = -( b_all[:, None, :, :, None] * row_c_Einv[:, :, None, None, :] )  # (NC,c,k,d,d)

        # δA_all = (e_k ⊗ row_c(W)) + (col_c(U) ⊗ e_k^T)，全向量化
        eye = bm.eye(d, **self.kwargs0)                       # (d,d)
        row_sel = eye[None, None, :, :, None]                 # (1,1,k,d,1)
        col_sel = eye[None, None, :, None, :]                 # (1,1,k,1,d)
        drow_all = W                                          # (NC,c,d)
        ucol_all = bm.permute_dims(U, (0, 2, 1))              # (NC,c,d)
        dA_row   = row_sel * drow_all[:, :, None, None, :]    # (NC,c,k,d,d)
        dA_col   = ucol_all[:, :, None, :, None] * col_sel    # (NC,c,k,d,d)
        dA_all   = dA_row + dA_col                            # (NC,c,k,d,d)

        # δg_all = g * tr(A^{-1} δA)
        Ainv    = bm.linalg.inv(A)                                           # (NC,d,d)
        tr_term = bm.sum(Ainv[:, None, None, :, :] * dA_all, axis=(3,4))   # (NC,c,k)
        dg_all  = (g_pos[:, None, None]) * tr_term                           # (NC,c,k)

        # d (TdA)_all
        # 先准备常用项
        A_Fnorm_pow3 = (A_F2) ** (d*gamma/2 - 4)
        trA_pow3 = (theta * trA) ** (d*gamma/4 - 2)
        A_dot_dA = bm.sum(A[:, None, None, :, :] * dA_all, axis=(3,4))  # (NC,c,k)
        tr_dA = bm.trace(dA_all, axis1=3, axis2=4)                      # (NC,c,k)
        # d (TdA) = d*gamma/4 * [ (d*gamma/2 - 2) * |A|_F^{d*gamma/2 - 4} * (A : dA) * A
        #                        + |A|_F^{d*gamma/2 - 2} * dA
        #                        + theta^{d*gamma/4} * (d*gamma/4 - 1) * tr(A)^{d*gamma/4 - 2} * tr(dA) * I ]
        term1 = (d*gamma/4) * (d*gamma/2 - 2) * \
                A_Fnorm_pow3[..., None, None, None, None] * \
                A_dot_dA[..., None, None] * \
                A[:, None, None, ...]
        term2 = (d*gamma/4) * A_Fnorm_pow2[...,None, None,None, None] * dA_all
        term3 = (d*gamma/4) * (d*gamma/4 - 1) *\
                trA_pow3[...,None,None,None, None] * \
                tr_dA[...,None,None] * I_mat[:, None, None, :, :]
        dTdA_all = term1 + term2 + term3  # (NC,c,k,d,d)

        # d (Tdg)_all
        # d (Tdg) = - gamma/4 * (gamma/4 - 1) * (d*theta)^{d*gamma/4} * det(A)^{gamma/4 - 2} * dg
        dTdg_all = - gamma/4 * (gamma/4 - 1) * (d*theta)**(d*gamma/4) *\
                    (g_pos**(gamma/4 - 2))[...,None,None] * dg_all  # (NC,c,k)

        # P1 = 2 * rho * d E_hat_inv (A TdA + Tdg * g * I)
        M0 = A @ TdA + (Tdg * g_pos)[..., None, None] * I_mat                   # (NC,d,d)
        P1_all = 2.0 * rho[:, None, None, None, None] * bm.einsum('nckij,njl->nckil', dEinv_all, M0)

        # P2 = 2 * rho * E_hat_inv (dA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        part_a = bm.einsum('nckij,njl->nckil', dA_all, TdA)                     # dA TdA
        part_b = bm.einsum('nij,nckjl->nckil', A, dTdA_all)                     # A d(TdA)
        part_c = (dTdg_all * g_pos[:, None, None])[..., None, None] * I_mat[:, None, None, :, :]   # d(Tdg) g I
        part_d = (Tdg[:, None, None] * dg_all)[..., None, None] * I_mat[:, None, None, :, :]       # Tdg dg I
        bracket = part_a + part_b + part_c + part_d                              # (NC,c,k,d,d)
        P2_all  = 2.0 * rho[:, None, None, None, None] * bm.einsum('nij,nckjl->nckil', Einv, bracket)

        # 局部二阶块（对 j'=1..d 的“梯度部分”），形状 (NC,c,k,d,matrix_cols=j')
        D2_all = P1_all + P2_all                                                # (NC,c,k,d,d)

        JAC = self.JAC_assembly(D2_all) 
        return JAC
     
    def JAC_functional_4(self,Xi,M_inv):
        """
        T = (H)^gamma
        H = 1/2 * |A|_F^{d/2} + (theta*tr(A))^{d/4} - (d*theta*det(A)^{1/d})^{d/4}
        TdA = gamma * H^{gamma-1} * [ d/4 * |A|_F^{d/2 - 2} * A + d/4 * theta^{d/4} * tr(A)^{d/4 - 1} * I ]
        Tdg = - gamma * H^{gamma-1} * 1/4 * (d*theta)^{d/4} * det(A)^{1/4 - 1}
        d E_hat = e_k x e_c^T
        d E_hat_inv = - E_hat_inv dE_hat E_hat_inv
        d A =  2*M_inv (dE_hat E_K^{-1})^T
        d g =  g * tr(A^{-1} dA)
        d H = S : dA -  (d * theta)^{d/4} * 1/4 * g^{1/4 - 1} * dg
        S = 1/2 * d/2 * |A|_F^{d/2 - 2} * A + 1/4 * d * theta^{d/4} * tr(A)^{d/4 - 1} * I
        d (TdA) = gamma*(gamma-1) * (H)^{gamma-2} * (dH) * 
                [ d/4 * |A|_F^{d/2 - 2} * A + d/4 * theta^{d/4} * tr(A)^{d/4 - 1} * I ] +
                 gamma * H^{gamma-1} * [ d/4 * (d/2 - 2) * |A|_F^{d/2 - 4} * (A : dA) * A + d/4 * |A|_F^{d/2 - 2} * dA +
                 d/4 * theta^{d/4} * (d/4 - 1) * tr(A)^{d/4 - 2} * tr(dA) * I ]
        d (Tdg) = - gamma*(gamma-1) * (H)^{gamma-2} * (dH) * 1/4 * (d*theta)^{d/4} * det(A)^{1/4 - 1} 
                 - gamma * H^{gamma-1} * 1/4 * (d*theta)^{d/4} * (1/4 - 1) * det(A)^{1/4 - 2} * dg
        P1 = 2 * rho * d E_hat_inv (A TdA + Tdg * g * I)
        P2 = 2 * rho * E_hat_inv (dA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        J = R @ (P1 + P2)
        """
        d   = self.GD
        NN  = self.NN
        NC  = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同结构扩展"
        print("使用 JAC_functional_4")
        # 缓存
        cache = getattr(self, '_ivp_cache', None)
        if cache is None:
            raise RuntimeError("ivp 缓存未准备，请先调用 _prepare_ivp_cache")
        E_K     = cache['E_K']
        E_K_inv = cache['E_K_inv']
        rho     = cache['rho']
        theta   = cache['theta']

        # 基本量
        E_hat    = self.edge_matrix(Xi)                     # (NC,d,d)
        Einv     = bm.linalg.inv(E_hat)                     # (NC,d,d)
        Ehat_T   = bm.swapaxes(E_hat, -1, -2)               # (NC,d,d)
        C        = E_K_inv @ M_inv @ bm.swapaxes(E_K_inv, -1, -2)  # (NC,d,d)
        A        = E_hat @ C @ Ehat_T                       # (NC,d,d)
        I_d      = bm.eye(d, **self.kwargs0)
        I_mat    = bm.zeros_like(A, **self.kwargs0); I_mat += I_d
        trA      = bm.trace(A, axis1=-2, axis2=-1)          # (NC,)
        A_Fnorm  = bm.sqrt(bm.sum(A*A, axis=(-2,-1)))       # (NC,)
        g        = bm.linalg.det(A)                         # (NC,)
        eps = 1e-14
        g_pos    = bm.maximum(g, eps)

        # 目标泛函及导数
        gamma = self.gamma
        # H = 1/2 * |A|_F^{d/2} + (theta*tr(A))^{d/4} - (d*theta*det(A)^{1/d})^{d/4}
        H = 0.5 * (A_Fnorm**(d/2)) + (theta*trA)**(d/4) - (d*theta*g_pos**(1/d))**(d/4)
        H_pos = bm.maximum(H, eps)

        # TdA = gamma * H^{gamma-1} * [ d/4 * |A|_F^{d/2 - 2} * A + d/4 * theta^{d/4} * tr(A)^{d/4 - 1} * I ]
        A_Fnorm_pow = (A_Fnorm**(d/2 - 2))
        trA_pow = (theta*trA)**(d/4 - 1)
        TdA = (gamma * H_pos**(gamma-1))[...,None,None] * (
            (d/4) * A_Fnorm_pow[...,None,None] * A +
            (d/4) * theta**(d/4) * trA_pow[...,None,None] * I_mat
        )  # (NC,d,d)

        # Tdg = - gamma * H^{gamma-1} * 1/4 * (d*theta)^{d/4} * det(A)^{1/4 - 1}
        Tdg = - (gamma * H_pos**(gamma-1)) * 0.25 * (d*theta)**(d/4) * g_pos**(0.25 - 1)

        # S = 1/2 * d/2 * |A|_F^{d/2 - 2} * A + 1/4 * d * theta^{d/4} * tr(A)^{d/4 - 1} * I
        S = 0.5 * d/2 * A_Fnorm_pow[...,None,None] * A + 0.25 * d * theta**(d/4) * trA_pow[...,None,None] * I_mat

        # 预备构件（用于 δA ）
        U = E_hat @ C                        # (NC,d,d)
        W = C @ Ehat_T                       # (NC,d,d)
        # δE_hat^{-1} = −(b_k ⊗ row_c(Einv))，一次性生成 (NC,c,k,d,d)
        b_all        = bm.swapaxes(Einv, -1, -2)              # (NC,d,d)
        row_c_Einv   = Einv                                   # (NC,c,d)
        dEinv_all = -( b_all[:, None, :, :, None] * row_c_Einv[:, :, None, None, :] )  # (NC,c,k,d,d)

        # δA_all = (e_k ⊗ row_c(W)) + (col_c(U) ⊗ e_k^T)，全向量化
        eye = bm.eye(d, **self.kwargs0)                       # (d,d)
        row_sel = eye[None, None, :, :, None]                 # (1,1,k,d,1)
        col_sel = eye[None, None, :, None, :]                 # (1,1,k,1,d)
        drow_all = W                                          # (NC,c,d)
        ucol_all = bm.permute_dims(U, (0, 2, 1))              # (NC,c,d)
        dA_row   = row_sel * drow_all[:, :, None, None, :]    # (NC,c,k,d,d)
        dA_col   = ucol_all[:, :, None, :, None] * col_sel    # (NC,c,k,d,d)
        dA_all   = dA_row + dA_col                            # (NC,c,k,d,d)

        # δg_all = g * tr(A^{-1} δA)
        Ainv    = bm.linalg.inv(A)                                           # (NC,d,d)
        tr_term = bm.sum(Ainv[:, None, None, :, :] * dA_all, axis=(3,4))   # (NC,c,k)
        dg_all  = (g_pos[:, None, None]) * tr_term                           # (NC,c,k)

        # dH_all = S : dA - (d*theta)^{d/4} * 1/4 * g^{1/4 - 1} * dg
        g_pow = g_pos**(0.25 - 1)
        dH_all = bm.sum(S[:, None, None, :, :] * dA_all, axis=(3,4)) - (d*theta)**(d/4) * 0.25 * g_pow[:, None, None] * dg_all  # (NC,c,k)

        # d (TdA)_all
        # 第一项
        f1 = gamma*(gamma-1) * H_pos**(gamma-2)
        TdA_bracket = (d/4) * A_Fnorm_pow[...,None,None] * A + (d/4) * theta**(d/4) * trA_pow[...,None,None] * I_mat
        term1 = (f1[:, None, None] * dH_all)[..., None, None] * TdA_bracket[:, None, None, :, :]
        # 第二项
        A_Fnorm_pow3 = (A_Fnorm**(d/2 - 4))
        A_dot_dA = bm.sum(A[:, None, None, :, :] * dA_all, axis=(3,4))  # (NC,c,k)
        trA_pow3 = (theta * trA) ** (d/4 - 2)
        tr_dA = bm.trace(dA_all, axis1=3, axis2=4)                      # (NC,c,k)
        term2 = (gamma * H_pos**(gamma-1))[...,None,None,None,None] * (
            (d/4)*(d/2-2) * A_Fnorm_pow3[...,None,None,None,None] * A_dot_dA[...,None,None] * A[:, None, None, ...] +
            (d/4) * A_Fnorm_pow[...,None,None,None,None] * dA_all +
            (d/4) * theta**(d/4) * (d/4-1) * trA_pow3[...,None,None,None,None] * tr_dA[...,None,None] * I_mat[:, None, None, :, :]
        )
        dTdA_all = term1 + term2  # (NC,c,k,d,d)

        # d (Tdg)_all
        # d (Tdg) = - gamma*(gamma-1) * (H)^{gamma-2} * (dH) * 1/4 * (d*theta)^{d/4} * det(A)^{1/4 - 1} 
        #         - gamma * H^{gamma-1} * 1/4 * (d*theta)^{d/4} * (1/4 - 1) * det(A)^{1/4 - 2} * dg
        coef1 = -gamma*(gamma-1) * H_pos**(gamma-2) * 0.25 * (d*theta)**(d/4) * g_pos**(0.25 - 1)
        coef2 = -gamma * H_pos**(gamma-1) * 0.25 * (d*theta)**(d/4) * (0.25 - 1) * g_pos**(0.25 - 2)
        dTdg_all = coef1[:, None, None] * dH_all + coef2[:, None, None] * dg_all   # (NC,c,k)

        # P1 = 2 * rho * d E_hat_inv (A TdA + Tdg * g * I)
        M0 = A @ TdA + (Tdg * g_pos)[..., None, None] * I_mat                   # (NC,d,d)
        P1_all = 2.0 * rho[:, None, None, None, None] * bm.einsum('nckij,njl->nckil', dEinv_all, M0)

        # P2 = 2 * rho * E_hat_inv (dA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        part_a = bm.einsum('nckij,njl->nckil', dA_all, TdA)                     # dA TdA
        part_b = bm.einsum('nij,nckjl->nckil', A, dTdA_all)                     # A d(TdA)
        part_c = (dTdg_all * g_pos[:, None, None])[..., None, None] * I_mat[:, None, None, :, :]   # d(Tdg) g I
        part_d = (Tdg[:, None, None] * dg_all)[..., None, None] * I_mat[:, None, None, :, :]       # Tdg dg I
        bracket = part_a + part_b + part_c + part_d                              # (NC,c,k,d,d)
        P2_all  = 2.0 * rho[:, None, None, None, None] * bm.einsum('nij,nckjl->nckil', Einv, bracket)

        # 局部二阶块（对 j'=1..d 的“梯度部分”），形状 (NC,c,k,d,matrix_cols=j')
        D2_all = P1_all + P2_all                                                # (NC,c,k,d,d)
        JAC = self.JAC_assembly(D2_all)
        return JAC
    
    def JAC_functional_5(self,Xi,M_inv):
        """
        T = (H)^gamma
        H = d^{1/2} * |A|_F + theta*(tr(A) -2 *d * det(A)^{1/d})
        TdA = gamma * H^{gamma-1} * [ d^{1/2} * |A|_F^{-1} * A + theta *I ]
        Tdg = -  2 * gamma * H^{gamma-1} * theta * det(A)^{1/d - 1}
        d E_hat = e_k x e_c^T
        d E_hat_inv = - E_hat_inv dE_hat E_hat_inv
        d A =  2*M_inv (dE_hat E_K^{-1})^T
        d g =  g * tr(A^{-1} dA)
        d H = S : dA -  theta  * g^{1/d - 1} * dg
        S = d^{1/2} * |A|_F^{-1} * A + theta * I
        d (TdA) = gamma*(gamma-1) * (H)^{gamma-2} * (dH) * 
                [ d^{1/2} * |A|_F^{-1} * A + theta * I ] +
                 gamma * H^{gamma-1} * [- d^{1/2} * |A|_F^{-3} * (A : dA) * A + d^{1/2} * |A|_F^{-1} * dA ]
        d (Tdg) = - gamma*(gamma-1) * (H)^{gamma-2} * (dH) * 2 * theta * det(A)^{1/d - 1}
                 - gamma * H^{gamma-1} * 2 * theta * (1/d - 1) * det(A)^{1/d - 2} * dg
        P1 = 2 * rho * d E_hat_inv (A TdA + Tdg * g * I)
        P2 = 2 * rho * E_hat_inv (dA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        J = R @ (P1 + P2)
        """
        d   = self.GD
        NN  = self.NN
        NC  = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同张量结构扩展"

        cache = getattr(self, '_ivp_cache', None)
        if cache is None:
            raise RuntimeError("ivp 缓存未准备，请先调用 _prepare_ivp_cache")
        E_K     = cache['E_K']
        E_K_inv = cache['E_K_inv']
        rho     = cache['rho']
        theta   = cache['theta']

        # 基本量
        E_hat    = self.edge_matrix(Xi)                     # (NC,d,d)
        Einv     = bm.linalg.inv(E_hat)                     # (NC,d,d)
        Ehat_T   = bm.swapaxes(E_hat, -1, -2)               # (NC,d,d)
        C        = E_K_inv @ M_inv @ bm.swapaxes(E_K_inv, -1, -2)  # (NC,d,d)
        A        = E_hat @ C @ Ehat_T                       # (NC,d,d)

        I_d      = bm.eye(d, **self.kwargs0)
        I_mat    = bm.zeros_like(A, **self.kwargs0); I_mat += I_d

        trA      = bm.trace(A, axis1=-2, axis2=-1)          # (NC,)
        A_Fnorm  = bm.sqrt(bm.sum(A * A, axis=(-2, -1)))    # (NC,)
        g        = bm.linalg.det(A)                         # (NC,)

        # 数值稳健性
        eps = 1e-14
        g_pos    = bm.maximum(g, eps)
        A_Fnorm_safe = bm.maximum(A_Fnorm, eps)
        # H 与其幂
        gamma = self.gamma
        H = (d**0.5) * A_Fnorm + theta * (trA - 2.0 *d * g_pos**(1.0 / d))   # (NC,)
        H_pos = bm.maximum(H, eps)
        H_gm1 = H_pos**(gamma - 1.0)

        # TdA 与 Tdg（直接复用已实现的 TdA_5 / Tdg_5）
        TdA = self.TdA_5(theta, A)    # (NC,d,d)
        Tdg = self.Tdg_5(theta, A, g_pos)  # (NC,)

        # 预备构件（用于 δA）
        U = E_hat @ C                        # (NC,d,d)
        W = C @ Ehat_T                       # (NC,d,d)

        # δE_hat^{-1} 结构：-(b_k ⊗ row_c(Einv))
        b_all        = bm.swapaxes(Einv, -1, -2)              # (NC,d,d)
        row_c_Einv   = Einv                                   # (NC,c,d)
        dEinv_all = -( b_all[:, None, :, :, None] * row_c_Einv[:, :, None, None, :] )  # (NC,c,k,d,d)

        # δA_all = (e_k ⊗ row_c(W)) + (col_c(U) ⊗ e_k^T)，全向量化
        eye = bm.eye(d, **self.kwargs0)                       # (d,d)
        row_sel = eye[None, None, :, :, None]                 # (1,1,k,d,1)
        col_sel = eye[None, None, :, None, :]                 # (1,1,k,1,d)
        drow_all = W                                          # (NC,c,d)
        ucol_all = bm.permute_dims(U, (0, 2, 1))              # (NC,c,d)
        dA_row   = row_sel * drow_all[:, :, None, None, :]    # (NC,c,k,d,d)
        dA_col   = ucol_all[:, :, None, :, None] * col_sel    # (NC,c,k,d,d)
        dA_all   = dA_row + dA_col                            # (NC,c,k,d,d)

        # δg_all = g * tr(A^{-1} δA)
        Ainv    = bm.linalg.inv(A)                                           # (NC,d,d)
        tr_term = bm.sum(Ainv[:, None, None, :, :] * dA_all, axis=(3,4))   # (NC,c,k)
        dg_all  = (g_pos[:, None, None]) * tr_term                           # (NC,c,k)

        # S 矩阵与 δH（注释中 S = d^{1/2} * |A|_F^{-1} * A + theta * I）
        A_Fnorm_inv = (1.0 / A_Fnorm_safe)
        S_mat = (d**0.5) * A_Fnorm_inv[..., None, None] * A + theta * I_mat  # (NC,d,d)

        # dH_all = S : dA - theta  * g^{1/d - 1} * dg
        S_dot_dA = bm.sum(S_mat[:, None, None, :, :] * dA_all, axis=(3,4))    # (NC,c,k)
        g_pow = g_pos**(1.0 / d - 1.0)
        dH_all = S_dot_dA - theta * g_pow[:, None, None] * dg_all  # (NC,c,k)

        # d(TdA)_all：
        # 第一项: gamma*(gamma-1) * H^{gamma-2} * dH * S_mat
        f1 = gamma * (gamma - 1.0) * (H_pos**(gamma - 2.0))    # (NC,)
        term1 = (f1[:, None, None] * dH_all)[..., None, None] * S_mat[:, None, None, :, :]  # (NC,c,k,d,d)

        # 第二项: gamma * H^{gamma-1} * [ - d^{1/2} * |A|_F^{-3} * (A : dA) * A + d^{1/2} * |A|_F^{-1} * dA ]
        A_dot_dA = bm.sum(A[:, None, None, :, :] * dA_all, axis=(3,4))  # (NC,c,k)
        A_Fnorm_inv3 = A_Fnorm_inv**3
        part_neg = - (d**0.5) * A_Fnorm_inv3[..., None, None, None, None] * A_dot_dA[..., None, None] * A[:, None, None, :, :]  # (NC,c,k,d,d)
        part_pos = (d**0.5) * A_Fnorm_inv[..., None, None, None, None] * dA_all  # (NC,c,k,d,d)
        term2 = (gamma * H_gm1)[..., None, None, None, None] * (part_neg + part_pos)

        dTdA_all = term1 + term2  # (NC,c,k,d,d)

        # d(Tdg)_all：
        # - gamma*(gamma-1) * H^{gamma-2} * dH * 2 * theta * det(A)^{1/d - 1}
        # - gamma * H^{gamma-1} * 2 * theta * (1/d - 1) * det(A)^{1/d - 2} * dg
        det_pow1 = g_pos**(1.0 / d - 1.0)
        det_pow2 = g_pos**(1.0 / d - 2.0)
        coef_a = - gamma * (gamma - 1.0) * (H_pos**(gamma - 2.0)) * (2.0 * theta) * det_pow1  # (NC,)
        coef_b = - gamma * H_gm1 * (2.0 * theta) * (1.0 / d - 1.0) * det_pow2  # (NC,)
        dTdg_all = coef_a[:, None, None] * dH_all + coef_b[:, None, None] * dg_all  # (NC,c,k)

        # P1 = 2ρ·δÊ^{-1}·(A TdA + Tdg * g * I)
        M0 = A @ TdA + (Tdg * g_pos)[..., None, None] * I_mat                   # (NC,d,d)
        P1_all = 2.0 * rho[:, None, None, None, None] * bm.einsum('nckij,njl->nckil', dEinv_all, M0)

        # P2 = 2ρ·Ê^{-1}·(δA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        part_a = bm.einsum('nckij,njl->nckil', dA_all, TdA)                     # δA TdA
        part_b = bm.einsum('nij,nckjl->nckil', A, dTdA_all)                     # A d(TdA)
        part_c = (dTdg_all * g_pos[:, None, None])[..., None, None] * I_mat[:, None, None, :, :]   # d(Tdg) g I
        part_d = (Tdg[:, None, None] * dg_all)[..., None, None] * I_mat[:, None, None, :, :]       # Tdg δg I
        bracket = part_a + part_b + part_c + part_d                              # (NC,c,k,d,d)
        P2_all  = 2.0 * rho[:, None, None, None, None] * bm.einsum('nij,nckjl->nckil', Einv, bracket)

        # 局部二阶块（对 j'=1..d 的“梯度部分”），形状 (NC,c,k,d,matrix_cols=j')
        D2_all = P1_all + P2_all                                                # (NC,c,k,d,d)

        # 使用统一装配函数把 D2_all 装配为全局 JAC
        JAC = self.JAC_assembly(D2_all)
        return JAC
    
    def JAC_functional_6(self,Xi,M_inv):
        """
        T = tr(A)^{d*gamma/2} - d*gamma/2 * theta^(d*gamma/2) * ln (det(A))
        TdA = (d*gamma/2) * tr(A)^{d*gamma/2 - 1} * I
        Tdg = - d*gamma/2 * theta^(d*gamma/2) * det(A)^{-1}
        d E_hat = e_k x e_c^T
        d E_hat_inv = - E_hat_inv dE_hat E_hat_inv
        d A =  2*M_inv (dE_hat E_K^{-1})^T
        d g =  g * tr(A^{-1} dA)
        d(TdA) = (d*gamma/2) * (d*gamma/2 - 1) * tr(A)^{d*gamma/2 - 2} * tr(dA) * I
        d(Tdg) = d*gamma/2 * theta^(d*gamma/2) * det(A)^{-2} * dg
        P1 = 2 * rho * d E_hat_inv (A TdA + Tdg * g * I)
        P2 = 2 * rho * E_hat_inv (dA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        J = R @ (P1 + P2)
        """
        d   = self.GD
        NN  = self.NN
        NC  = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同张量结构扩展"

        cache = getattr(self, '_ivp_cache', None)
        if cache is None:
            raise RuntimeError("ivp 缓存未准备，请先调用 _prepare_ivp_cache")
        E_K     = cache['E_K']
        E_K_inv = cache['E_K_inv']
        rho     = cache['rho']
        theta   = cache['theta']

        # 基本量
        E_hat    = self.edge_matrix(Xi)                     # (NC,d,d)
        Einv     = bm.linalg.inv(E_hat)                     # (NC,d,d)
        Ehat_T   = bm.swapaxes(E_hat, -1, -2)               # (NC,d,d)
        C        = E_K_inv @ M_inv @ bm.swapaxes(E_K_inv, -1, -2)  # (NC,d,d)
        A        = E_hat @ C @ Ehat_T                       # (NC,d,d)

        I_d      = bm.eye(d, **self.kwargs0)
        I_mat    = bm.zeros_like(A, **self.kwargs0); I_mat += I_d

        trA      = bm.trace(A, axis1=-2, axis2=-1)          # (NC,)
        g        = bm.linalg.det(A)                         # (NC,)
        lam      = self.lam(theta)
        # 数值稳健性
        eps = 1e-14
        g_pos    = bm.maximum(g, eps)
        trA_safe = bm.maximum(trA, eps)

        # 目标泛函及导数（无 H）
        gamma = self.gamma
        power = (d * gamma / 2.0)
        # TdA = (d*gamma/2) * tr(A)^{d*gamma/2 - 1} * I
        TdA = self.TdA_6(theta,A)  # (NC,d,d)
        Tdg = self.Tdg_6(theta, g_pos)  # (NC,)

        # 预备构件（用于 δA）
        U = E_hat @ C                        # (NC,d,d)
        W = C @ Ehat_T                       # (NC,d,d)

        # δE_hat^{-1} 结构：-(b_k ⊗ row_c(Einv))
        b_all        = bm.swapaxes(Einv, -1, -2)              # (NC,d,d)
        row_c_Einv   = Einv                                   # (NC,c,d)
        dEinv_all = -( b_all[:, None, :, :, None] * row_c_Einv[:, :, None, None, :] )  # (NC,c,k,d,d)

        # δA_all = (e_k ⊗ row_c(W)) + (col_c(U) ⊗ e_k^T)，全向量化
        eye = bm.eye(d, **self.kwargs0)                       # (d,d)
        row_sel = eye[None, None, :, :, None]                 # (1,1,k,d,1)
        col_sel = eye[None, None, :, None, :]                 # (1,1,k,1,d)
        drow_all = W                                          # (NC,c,d)
        ucol_all = bm.permute_dims(U, (0, 2, 1))              # (NC,c,d)
        dA_row   = row_sel * drow_all[:, :, None, None, :]    # (NC,c,k,d,d)
        dA_col   = ucol_all[:, :, None, :, None] * col_sel    # (NC,c,k,d,d)
        dA_all   = dA_row + dA_col                            # (NC,c,k,d,d)

        # δg_all = g * tr(A^{-1} δA)
        Ainv    = bm.linalg.inv(A)                                           # (NC,d,d)
        tr_term = bm.sum(Ainv[:, None, None, :, :] * dA_all, axis=(3,4))     # (NC,c,k)
        dg_all  = (g_pos[:, None, None]) * tr_term                           # (NC,c,k)

        # d(TdA)_all = (d*gamma/2) * (d*gamma/2 - 1) * tr(A)^{d*gamma/2 - 2} * tr(dA) * I
        tr_dA = bm.trace(dA_all, axis1=3, axis2=4)                            # (NC,c,k)
        coef_td = d**(-d*gamma/2) * (power) * (power - 1.0) * (trA_safe ** (power - 2.0))        # (NC,)
        dTdA_all = 1/lam * coef_td[:, None, None, None, None] * tr_dA[..., None, None] * I_mat[:, None, None, :, :]  # (NC,c,k,d,d)

        # d(Tdg)_all = d*gamma/2 * theta^(d*gamma/2) * det(A)^{-2} * dg
        coef_tdg = gamma/2  * (theta ** power) * (g_pos ** (-2.0))             # (NC,)
        dTdg_all =  1/lam * coef_tdg[:, None, None] * dg_all                           # (NC,c,k)

        # P1 = 2 * rho * d E_hat_inv (A TdA + Tdg * g * I)
        M0 = A @ TdA + (Tdg * g_pos)[..., None, None] * I_mat                 # (NC,d,d)
        P1_all = 2.0 * rho[:,None,None,None,None] * bm.einsum('nckij,njl->nckil', dEinv_all, M0)

        # P2 = 2 * rho * E_hat_inv (dA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        part_a = bm.einsum('nckij,njl->nckil', dA_all, TdA)                     # dA TdA
        part_b = bm.einsum('nij,nckjl->nckil', A, dTdA_all)                     # A d(TdA)
        part_c = (dTdg_all * g_pos[:, None, None])[..., None, None] * I_mat[:, None, None, :, :]   # d(Tdg) g I
        part_d = (Tdg[:, None, None] * dg_all)[..., None, None] * I_mat[:, None, None, :, :]       # Tdg dg I
        bracket = part_a + part_b + part_c + part_d                              # (NC,c,k,d,d)
        P2_all  = 2.0 * rho[:,None,None,None,None] * bm.einsum('nij,nckjl->nckil', Einv, bracket)

        # 局部二阶块（对 j'=1..d 的“梯度部分”），形状 (NC,c,k,d,matrix_cols=j')
        D2_all = P1_all + P2_all                                                # (NC,c,k,d,d)

        # 使用统一装配函数把 D2_all 装配为全局 JAC
        JAC = self.JAC_assembly(D2_all)
        return JAC
        
    
    def JAC_assembly(self,D2_all):
        """
        依据局部二阶块 D2_all 装配全局雅可比
        """
        d   = self.GD
        NN  = self.NN
        NC  = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同结构扩展"
        cm = self.cm
        rxx, ryy, rxy, ryx = self.rxx, self.ryy, self.rxy, self.ryx
        P_diag = self._ivp_cache['P_diag']
        # 打包：把 (NC,c,d) 压成按 c 分段的 (c*NC*(d+1),)，并在最前加 j=0 列（负和）
        def pack_all(D2_comp_all):
            D1 = -bm.sum(D2_comp_all, axis=2, keepdims=True)                    # (NC,c,1)
            V  = bm.concat([D1, D2_comp_all], axis=2)                           # (NC,c,d+1)
            V  = bm.permute_dims(V, (1, 0, 2)).reshape(-1)                      # (c*NC*(d+1),)
            return V

        # k=0 对 δx，k=1 对 δy；m=0/1 取 vx/vy 分量
        vx_seg_x_all = pack_all(D2_all[:, :, 0, :, 0])  # (d*NC*(d+1),)
        vy_seg_x_all = pack_all(D2_all[:, :, 0, :, 1])
        vx_seg_y_all = pack_all(D2_all[:, :, 1, :, 0])
        vy_seg_y_all = pack_all(D2_all[:, :, 1, :, 1])

        # 行索引与系数（与稀疏图样对齐）
        rr_x_all  = self._row_off_x_tile_d % NN
        rr_y_all  = self._row_off_y_tile_d % NN
        rc_x_tile = (-1.0 / self.tau) * P_diag[rr_x_all]                        # (d*NC*(d+1),)
        rc_y_tile = (-1.0 / self.tau) * P_diag[rr_y_all]
        cm_rep    = bm.repeat(cm, d+1)                                          # (NC*(d+1),)
        cm_rep_all = bm.tile(cm_rep, d)                                         # (d*NC*(d+1),)

        # 四个 tile 段（投影右乘 + 行缩放 + 单元权）
        data00_tile = rc_x_tile * cm_rep_all * ( rxx[rr_x_all] * vx_seg_x_all + rxy[rr_x_all] * vy_seg_x_all )
        data10_tile = rc_y_tile * cm_rep_all * ( ryx[rr_y_all] * vx_seg_x_all + ryy[rr_y_all] * vy_seg_x_all )
        data01_tile = rc_x_tile * cm_rep_all * ( rxx[rr_x_all] * vx_seg_y_all + rxy[rr_x_all] * vy_seg_y_all )
        data11_tile = rc_y_tile * cm_rep_all * ( ryx[rr_y_all] * vx_seg_y_all + ryy[rr_y_all] * vy_seg_y_all )

        # j=0 列（为 −sum_c），与 pack_all 逻辑等价（此处直接从 D2_all 聚合）
        def pack0(D2_0_comp):
            D1 = -bm.sum(D2_0_comp, axis=1, keepdims=True)                      # (NC,1)
            V  = bm.concat([D1, D2_0_comp], axis=1)                             # (NC,d+1)
            return V.reshape(-1)                                                # (NC*(d+1),)

        D2_0_x = -bm.sum(D2_all[:, :, 0, :, :], axis=1)                         # (NC,d,d)
        D2_0_y = -bm.sum(D2_all[:, :, 1, :, :], axis=1)
        vx_0_x = pack0(D2_0_x[:, :, 0]); vy_0_x = pack0(D2_0_x[:, :, 1])
        vx_0_y = pack0(D2_0_y[:, :, 0]); vy_0_y = pack0(D2_0_y[:, :, 1])

        rr_x0  = self._row_off_x_0 % NN
        rr_y0  = self._row_off_y_0 % NN
        coef_x0 = (-1.0 / self.tau) * P_diag[rr_x0]
        coef_y0 = (-1.0 / self.tau) * P_diag[rr_y0]

        data00_0 = coef_x0 * cm_rep * ( rxx[rr_x0] * vx_0_x + rxy[rr_x0] * vy_0_x )
        data10_0 = coef_y0 * cm_rep * ( ryx[rr_y0] * vx_0_x + ryy[rr_y0] * vy_0_x )
        data01_0 = coef_x0 * cm_rep * ( rxx[rr_x0] * vx_0_y + rxy[rr_x0] * vy_0_y )
        data11_0 = coef_y0 * cm_rep * ( ryx[rr_y0] * vx_0_y + ryy[rr_y0] * vy_0_y )

        # 一次性装配（COO -> CSR）
        I = bm.concat([
            self._row_off_x_tile_d, self._row_off_y_tile_d,
            self._row_off_x_0,      self._row_off_y_0,
            self._row_off_x_tile_d, self._row_off_y_tile_d,
            self._row_off_x_0,      self._row_off_y_0,
        ], axis=0)
        J = bm.concat([
            self._cols_x_tile_d, self._cols_x_tile_d,
            self._cols_x_0,      self._cols_x_0,
            self._cols_y_tile_d, self._cols_y_tile_d,
            self._cols_y_0,      self._cols_y_0,
        ], axis=0)
        V = bm.concat([
            data00_tile, data10_tile, data00_0, data10_0,
            data01_tile, data11_tile, data01_0, data11_0
        ], axis=0)

        from scipy.sparse import coo_matrix
        JAC = coo_matrix((V.astype(float), (I.astype(int), J.astype(int))),
                        shape=(2*NN, 2*NN)).tocsr()
        return JAC
    
    
    def linear_interpolate(self, Xi, Xi_new , X):
        """
        linear interpolation method
        
        Parameters
            moved_node: TensorLike, new node positions
        """
        node2cell = self.node2cell
        i, j = node2cell.row, node2cell.col
        p = self.pspace.p # physical space polynomial degree
        Xnew = bm.zeros_like(X, **self.kwargs0) # 初始化新的解向量,对节点优先进行赋值
        interpolated = bm.zeros(self.NN, dtype=bool, device=self.device)

        current_i, current_j = self.tri_interpolate_batch(i, j, Xnew,X, 
                                                      interpolated,Xi,Xi_new) 
        # 迭代扩展 - 添加循环上限
        max_iterations = min(30, int(bm.log(self.NC)) + 20)
        iteration_count = 0
        # 迭代扩展
        while len(current_i) > 0 and iteration_count < max_iterations:
            iteration_count += 1
            # 扩展邻居
            neighbors = self.cell2cell[current_j].flatten()
            expanded_i = bm.repeat(current_i, self.cell2cell.shape[1])
            valid_mask = neighbors >= 0

            if not bm.any(valid_mask):
                break

            combined = expanded_i[valid_mask] * self.NC + neighbors[valid_mask]
            unique_combined = bm.unique(combined)
            
            unique_i = unique_combined // self.NC
            unique_j = unique_combined % self.NC
            current_i, current_j = self.tri_interpolate_batch(unique_i, unique_j,
                                                    Xnew,X, interpolated,Xi,Xi_new)
        if iteration_count >= max_iterations:
            print(f"Warning: Maximum iterations reached ({max_iterations}) without full interpolation.")

        return Xnew

    def tri_interpolate_batch(self,nodes, cells,Xnew,X, interpolated,Xi,Xi_new):
        """
        triangle mesh interpolation batch processing
        
        Parameters
            nodes: TensorLike, nodes to be interpolated
            cells: TensorLike, cells corresponding to the nodes
            new_uh: TensorLike, new solution vector
            interpolated: TensorLike, boolean mask indicating if nodes are already interpolated
            moved_node: TensorLike, moved node positions
        Returns
            nodes: TensorLike, nodes that still need interpolation
            cells: TensorLike, cells corresponding to the nodes that still need interpolation
        """
        if len(nodes) == 0:
            return bm.array([], **self.kwargs1), bm.array([], **self.kwargs1)
            
        # 计算重心坐标
        v_matrix = bm.permute_dims(
            Xi_new[self.cell[cells, 1:]] - Xi_new[self.cell[cells, 0:1]], 
            axes=(0, 2, 1)
        )
        v_b = Xi[nodes] - Xi_new[self.cell[cells, 0]]
        
        inv_matrix = bm.linalg.inv(v_matrix)
        lam = bm.einsum('cij,cj->ci', inv_matrix, v_b)
        lam = bm.concat([(1 - bm.sum(lam, axis=-1, keepdims=True)), lam], axis=-1)
        valid = bm.all(lam > -1e-10, axis=-1) & ~interpolated[nodes]
        
        if bm.any(valid):
            valid_nodes = nodes[valid]
            phi = self.mesh.shape_function(lam[valid], self.pspace.p)
            valid_value = bm.sum(phi[...,None] * X[self.pcell2dof[cells[valid]]], axis=1)

            Xnew = bm.set_at(Xnew, valid_nodes, valid_value)
            interpolated = bm.set_at(interpolated, valid_nodes, True)
        
        return nodes[~interpolated[nodes]], cells[~interpolated[nodes]]
    
    def AC_generator(self,delta_x):
        """
        @brief generate the tensor A and C to construct the equation
        @param scell: the cell has been splited
        @param node: the physical node
        """
        node = self.node
        scell = self.cell
        A = bm.permute_dims((node[scell[:,1:]] - node[scell[:,0,None]]),axes=(0,2,1))
        C = bm.permute_dims((delta_x[scell[:,1:]] - delta_x[scell[:,0,None]]),axes=(0,2,1))
        return A,C
    
    def _construct(self,moved_node:TensorLike):
        """
        @brief construct information for the harmap method before the next iteration
        """
        self.mesh.node = moved_node
        self.node = moved_node
        self.cm = self.mesh.entity_measure('cell')
        self.sm = bm.zeros(self.NN, **self.kwargs0)
        self.sm = bm.index_add(self.sm , self.mesh.cell , self.cm[:, None])
    
    def mesh_redistributor(self,maxit = 1000):
        """
        协变拉格朗日乘子法自适应网格算法
        1. 初始化 E, E_hat, M_inv (边矩阵, 目标度量张量的逆)
        2. 计算初始的梯度信息
        3. 构造梯度流
        4. 构造逻辑网格到物理网格的插值算子,并防止翻转
        5. 完成物理网格的解插值
        6. 更新物理网格信息
        7. 时间步进迭代 
        """
        init_gamma = self.gamma
        for it in range(self.total_steps):
            self.monitor()
            self.mol_method()
            atol = 1e-6
            rtol = atol * 100
            M = self.M
            M_inv = bm.linalg.inv(M)
            X = self.mesh.node
            Xi = self.logic_mesh.node
            self.gamma = 1 + (init_gamma - 1) * (it / self.total_steps)
            # if it == 0:
            theta = self.theta(M)
            # s = it / self.total_steps
            # theta = s * theta
            print(theta)
            self._prepare_ivp_cache(X, M, M_inv,theta)
            P_diag = self._ivp_cache['P_diag']
            eps = 1e-12
            s_node = bm.maximum(P_diag, eps)                     # (NN,)
            s_vec  = bm.concat([s_node, s_node], axis=0)         # (2NN,)
            s_inv  = 1.0 / s_vec
            
            time.send("begin...")
            def ode_system(t, y):
                y = (s_vec * y)
                Xi_current = y.reshape(self.GD, self.NN).T
                v = self.vector_construction(Xi_current , M_inv)
                time.send("ODE vector field")
                return s_inv *v.ravel(order = 'F')
            
            def jac(t, y):
                y = (s_vec * y)
                Xi_current = y.reshape(self.GD, self.NN).T
                J_y = self.JAC_functional(Xi_current, M_inv)
                time.send("ODE Jacobian matrix computation complete")
                J_z = J_y.multiply(s_inv[:, None])             # 行缩放
                J_z = J_z.multiply(s_vec[None, :])             # 列缩放
                return J_z
    
            # ODE 求解器需要的初值
            t_span = [0,self.t_span]
            y0 = Xi.ravel(order = 'F')
            z0 = (s_inv * y0)
            sol = solve_ivp(ode_system, t_span, z0, jac=jac, method='BDF',
                                        first_step=self.t_span/self.step,
                                        atol=atol, rtol=rtol)
            y_last = (s_vec * sol.y[:, -1])
            Xinew = y_last.reshape(self.GD, self.NN).T
            
            Xnew = self.linear_interpolate(Xi, Xinew , X)
            self.uh = self.interpolate(Xnew)
            self._construct(Xnew)
            print(f"LEAGAdaptive: step {it+1}/{self.total_steps} completed.")
        time.send("Mesh redistribution complete")
        next(time)
        return Xnew
        
        


       