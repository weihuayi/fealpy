from . import Monitor
from . import Interpolater
from .config import *
from scipy.integrate import solve_ivp
from .tool import _compute_coef_2d, quad_equ_solver , linear_surfploter
from scipy.sparse import coo_matrix
from fealpy.utils import timer
time = timer()
next(time)

class TargetAdaptive(Monitor, Interpolater):
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
    
    def sqrt_M(self,M):
        """
        目标度量张量 M 的平方根
        Parameters:
            M: (NC, GD, GD)
        Return:
            sqrt_M: (NC, GD, GD)
        """
        eigvals, eigvecs = bm.linalg.eigh(M)  # (NC, GD), (NC, GD, GD)
        sqrt_eigvals = bm.sqrt(bm.maximum(eigvals, 1e-16))  # 防止负数开根号
        D_sqrt = bm.zeros_like(M , **self.kwargs0)          # (NC, GD, GD)
        for i in range(self.GD):
            D_sqrt = bm.set_at(D_sqrt , (..., i, i), sqrt_eigvals[..., i])
        eigvecs_T = bm.swapaxes(eigvecs , -1,-2)           # (NC, GD, GD)
        sqrt_M = eigvecs @ D_sqrt @ eigvecs_T               # (NC, GD, GD)
        return sqrt_M
    
    def T(self,E , E_hat , M , theta):
        """
        基本的雅可比算子组件 T
        T = theta^{-1/2}
        A = E_hat E_K^{-1}M^{-1} E_K^{-T} E_hat^T
        """
        E_K_inv = bm.linalg.inv(E)
        M_inv = bm.linalg.inv(M)
        sqrt_M = self.sqrt_M(M_inv)
        sqrt_theta = bm.sqrt(theta)
        T = 1/sqrt_theta * E_hat @ E_K_inv @ sqrt_M
        return T
    
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
        area = bm.sum(cm)
        theta = (sigma/area)**(-2/d)
        return theta
        
    def G(self,T , rho):
        """
        (...)^gamma 型
        G = H^gamma
        H = 1/(2*tau) * |T - I|_F^2
        Parameters:
            theta(float): 积分全局乘子
            A(Tensor): 雅可比算子
        Return:
            T(float): T 函数值
        """
        gamma = self.gamma
        G = self.H(T)**gamma
        return G

    def H(self,T):
        """
        H = 1/(2*tau) * |T - I|_F^2
        tau = det(T)
        Parameters:
            theta(float): 积分全局乘子
            T(Tensor): 雅可比算子
        Return:
            H(Tensor): H 函数值
        """
        d = self.GD
        tau = bm.linalg.det(T)
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(T , **self.kwargs0)
        I += I_base
        frob_norm = bm.sum((T - I)**2 , axis=(-2,-1))
        H = 1/(2*tau) * frob_norm
        return H
    
    def H_2(self,T):
        """
        H = 1/2 * |T|_F^2 - ln(tau) - 1
        tau = det(T)
        Parameters:
            theta(float): 积分全局乘子
            T(Tensor): 雅可比算子
        Return:
            H(Tensor): H 函数值
        """
        d = self.GD
        tau = bm.linalg.det(T)
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(T , **self.kwargs0)
        I += I_base
        frob_norm = bm.sum((T)**2 , axis=(-2,-1))
        H = 1/2 * frob_norm - bm.log(tau) - 1
        return H
    
    def GdT(self,T):
        """
        (...)^gamma 型
        基于几何离散的 G 泛函关于算子 T 的导数
        T = theta^{-1/2} * E_hat E_K^{-1}M^{1/2}
        
        GdT =  gamma * H^{gamma-1} * 1/tau * (T - I)
        """
        gamma = self.gamma
        H = self.H(T)
        d = self.GD
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(T , **self.kwargs0)
        I += I_base
        tau = bm.linalg.det(T)
        GdT = (gamma * H**(gamma-1) * 1/tau)[...,None,None] * (T - I)
        return GdT

    def Gdt(self,T ):
        """
        (...)^gamma 型
        基于几何离散的 G 泛函关于标量 tau 的导数
        tau = det(T)

        Gdt = - gamma * H^{gamma-1} * 1/(2*tau^2) * |T - I|_F^2
        """
        gamma = self.gamma
        H = self.H(T)
        d = self.GD
        frob_norm = bm.sum((T - bm.eye(d, **self.kwargs0))**2 , axis=(-2,-1))
        tau = bm.linalg.det(T)
        Gdt = - gamma * H**(gamma-1) * 1/(2*tau**2) * frob_norm
        return Gdt
    
    def GdT_2(self,T):
        """
        (...)^gamma 型
        基于几何离散的 G 泛函关于算子 T 的导数
        T = theta^{-1/2} * E_hat E_K^{-1}M^{1/2}
        
        GdT = gamma * H^{gamma-1} * T
        """
        gamma = self.gamma
        H = self.H_2(T)
        d = self.GD
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(T , **self.kwargs0)
        I += I_base
        GdT = (gamma * H**(gamma-1) )[...,None,None] * T
        return GdT
    
    def Gdt_2(self,T ):
        """
        (...)^gamma 型
        基于几何离散的 G 泛函关于标量 tau 的导数
        tau = det(T)

        Gdt = - gamma * H^{gamma-1}/tau
        """
        gamma = self.gamma
        H = self.H_2(T)
        tau = bm.linalg.det(T)
        Gdt = - gamma * H**(gamma-1)/tau
        return Gdt
    
    def balance(self,M_node):
        """
        (...)^gamma 型
        平衡函数 P 为 (NN,NN) 的对角矩阵,实际组装时只需对角元
        P = diag( det(M)^{n})
        n = 1/d * (-2*gamma + d * gamma - d/2)
        Parameters:
            M_node(Tensor): 目标单元度量张量 (NN, GD, GD)
            theta(float): 积分全局乘子
        """
        d = self.GD
        gamma = self.gamma
        det_M_node = bm.linalg.det(M_node)
        n = 1/d * (- d/2)
        P_diag = det_M_node**n # (NN,)
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

    def Idxi_from_Ehat(self, E_hat, E_K, M, theta):
        """
        泛函局部导数 Idxi 的计算
        Idxi = E_K^{-1} @ dG/dJ^{-T} + dG/dr * det(J^{-1}) * E_hat_K^{-1}
        J^{-1} = E_hat @ E_K^{-1}
        r = det(J^{-1})
        dG/dJ^{-T} = theta^{-1/2} * E_K^{-1}M^{1/2} @ GdT
        dG/dr = tau * Gdt
        Parameters:
            E_hat(Tensor): 参考单元边矩阵 (NC, GD, GD)
            E_K(Tensor): 物理单元边矩阵 (NC, GD, GD)
            rho(Tensor): 权重函数 (NC,)
            M_inv(Tensor): 目标度量张量的逆 (NC, GD, GD)
        """
        E_hat_inv = bm.linalg.inv(E_hat)
        T = self.T(E_K , E_hat , M , theta)
        tau = bm.linalg.det(T)
        GdT = self.GdT_2(T )  # (NC, GD, GD)
        Gdt = self.Gdt_2(T )  # (NC,)

        term0 =  E_hat_inv @ T  @ GdT
        term1 = (tau * Gdt)[..., None, None] * E_hat_inv
        Idxi_grad_part = (term0 + term1)
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
    
    def vector_construction(self,Xi , M):
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
        import matplotlib.pyplot as plt
        E_hat = self.edge_matrix(Xi)  # (NC, GD, GD)

        cache = self._ivp_cache
        if cache is not None:
            E_K     = cache['E_K'];     E_K_inv = cache['E_K_inv']
            det_E_K = cache['det_E_K']; rho     = cache['rho']     
            cm      = cache['cm'];      P_diag  = cache['P_diag']
            theta   = cache['theta']

        Idxi = self.Idxi_from_Ehat(E_hat, E_K, M, theta)  # (NC, GD+1, GD)
        cell = self.cell
        cm = self.cm
        global_vector = bm.zeros((self.NN, self.GD), dtype=bm.float64)
        global_vector = bm.index_add(global_vector , cell , cm[:,None,None] * Idxi)

        tau = self.tau
        v = -1/tau * global_vector 
        
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
        与当前 T 型能量一致的雅可比:
        G = rho * H(T)^gamma,  H = (1/(2*tau))*||T - I||_F^2,  tau = det(T)
        Idxi = R @ [ rho * ( E_hat^{-1} T GdT + (tau*Gdt) E_hat^{-1} ) ]
        其中 GdT = gamma*H^{gamma-1}*(1/tau)*(T - I),  tau*Gdt = - rho*gamma*H^(gamma-1)/tau * ||T - I||_F^2
        """
        d   = self.GD
        NN  = self.NN
        NC  = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同结构扩展"

        # 缓存
        cache = getattr(self, '_ivp_cache', None)
        if cache is None:
            raise RuntimeError("ivp 缓存未准备，请先调用 _prepare_ivp_cache")
        E_K     = cache['E_K']
        E_K_inv = cache['E_K_inv']
        rho     = cache['rho']
        cm      = cache['cm']
        P_diag  = cache['P_diag']
        theta   = cache['theta']
        rxx, ryy, rxy, ryx = cache['rxx'], cache['ryy'], cache['rxy'], cache['ryx']

        # 基本量
        E_hat   = self.edge_matrix(Xi)                 # (NC,d,d)
        Einv    = bm.linalg.inv(E_hat)                 # (NC,d,d)
        sqrt_M  = self.sqrt_M(M_inv)    # (NC,d,d) 由 M_inv 还原 M 再开根
        sqrt_th = bm.sqrt(theta)
        B       = E_K_inv @ sqrt_M                     # (NC,d,d)
        T       = (1/sqrt_th * (E_hat @ B))              # (NC,d,d)
        I_d     = bm.eye(d, **self.kwargs0)
        I_mat   = bm.zeros_like(T, **self.kwargs0); I_mat += I_d

        # 标量与导数准备
        tau     = bm.linalg.det(T)                     # (NC,)
        eps     = 1e-14
        tau_pos = bm.maximum(tau, eps)
        Tinv    = bm.linalg.inv(T)                     # (NC,d,d)
        F2      = bm.sum((T - I_mat)*(T - I_mat), axis=(-2,-1))  # ||T-I||_F^2
        H       = 0.5 * F2 / tau_pos                               # (NC,)
        H_pos   = bm.maximum(H, eps)

        gamma   = self.gamma
        # 与 Idxi_from_Ehat 保持一致（其内部 GdT/Gdt 已含 rho，Idxi 外面还会再乘 rho）
        GdT     = (gamma * (H_pos**(gamma-1)) / tau_pos)[..., None, None] * (T - I_mat)  # (NC,d,d)
        tauGdt  = - gamma * (H_pos**(gamma-1))/tau_pos* F2                                    # (NC,)

        # δE_hat^{-1} （对 (c,k) 的全张量）
        b_all        = bm.swapaxes(Einv, -1, -2)  # (NC,d,d)
        row_c_Einv   = Einv                       # (NC,d,d)
        dEinv_all = -( b_all[:, None, :, :, None] * row_c_Einv[:, :, None, None, :] )  # (NC,c,k,d,d)

        # δT_all = √θ · δE_hat · B = √θ · (e_k ⊗ row_c(B))
        eye     = bm.eye(d, **self.kwargs0)
        row_sel = eye[None, None, :, :, None]     # (1,1,k,d,1)
        Brow    = B                                # (NC,c,d) 取第 c 行
        dT_all  = (1/sqrt_th * 1.0) * (row_sel * Brow[:, :, None, None, :])  # (NC,c,k,d,d)

        # δH_all = [1/τ (T-I) - H T^{-T}] : δT
        S_H      = ( (T - I_mat) / tau_pos[..., None, None] ) - (H_pos[..., None, None] * bm.swapaxes(Tinv, -1, -2) )
        dH_all   = bm.sum(S_H[:, None, None, :, :] * dT_all, axis=(3,4))  # (NC,c,k)

        # δ(1/τ) = -(1/τ) tr(T^{-1} δT) = -(1/τ) (T^{-T} : δT)
        trTD     = bm.sum(bm.swapaxes(Tinv, -1, -2)[:, None, None, :, :] * dT_all, axis=(3,4))  # (NC,c,k)
        dinv_tau = -(1.0 / tau_pos)[:, None, None] * trTD                                        # (NC,c,k)

        # δGdT_all = rho*gamma*[ (γ-1)H^{γ-2} δH * (1/τ)(T-I) + H^{γ-1} δ(1/τ)(T-I) + H^{γ-1} (1/τ) δT ]
        gm    = gamma                           # (NC,)
        H_gm1 = H_pos**(gamma-1.0)
        H_gm2 = H_pos**(gamma-2.0)
        termA = (gm * (gamma-1.0) * H_gm2)[:, None, None, None, None] * dH_all[..., None, None] \
                * (1.0 / tau_pos)[:, None, None, None, None] * (T - I_mat)[:, None, None, :, :]
        termB = (gm * H_gm1)[:, None, None, None, None] * dinv_tau[..., None, None] * (T - I_mat)[:, None, None, :, :]
        termC = ((gm * H_gm1) / tau_pos)[:, None, None, None, None] * dT_all
        dGdT_all = termA + termB + termC  # (NC,c,k,d,d)

        # tau*Gdt = - gamma*H^gamma/tau * ||T - I||_F^2
        # δ(tau*Gdt) = - gamma * [
        #   (γ-1) H^{γ-2} δH / tau * ||T-I||_F^2
        # - H^γ / tau^2 δtau * ||T-I||_F^2
        # + H^γ / tau * 2⟨T-I, δT⟩
        # ]
        H_gm1 = H_pos**(gamma-2.0)
        H_gm  = H_pos**(gamma-1.0)
        norm2 = bm.sum((T - I_mat)**2, axis=(-2,-1))  # (NC,)
        # 1) (γ-1) H^{γ-2} δH / tau * ||T-I||_F^2
        term1 = ((gamma-1) * H_gm1 / tau_pos)[:, None, None] * dH_all * norm2[:, None, None]
        # 2) - H^γ / tau^2 δtau * ||T-I||_F^2
        # δtau = tau * tr(T^{-1} δT) = tau * (T^{-T} : δT)
        trTD = bm.sum(bm.swapaxes(Tinv, -1, -2)[:, None, None, :, :] * dT_all, axis=(3,4))  # (NC,c,k)
        dtau = tau_pos[:, None, None] * trTD
        term2 = - (H_gm / tau_pos**2)[:, None, None] * dtau * norm2[:, None, None]
        # 3) + H^γ / tau * 2⟨T-I, δT⟩
        inner = 2 * bm.sum((T - I_mat)[:, None, None, :, :] * dT_all, axis=(3,4))  # (NC,c,k)
        term3 = (H_gm / tau_pos)[:, None, None] * inner
        # 合并
        dtauGdt_all = - gamma * (term1 + term2 + term3)  # (NC,c,k)

        # 组合各项：δ[E^{-1} T GdT] + δ[(tau Gdt) E^{-1}]
        TGdT = T @ GdT  # (NC,d,d)
        # 1) δ(E^{-1}) T GdT
        part1 = bm.einsum('nckij,njl->nckil', dEinv_all, TGdT)
        # 2) E^{-1} δT GdT
        dT_GdT_all = bm.einsum('nckij,njl->nckil', dT_all, GdT)
        part2 = bm.einsum('nij,nckjl->nckil', Einv, dT_GdT_all)
        # 3) E^{-1} T δGdT
        T_dGdT_all = bm.einsum('nij,nckjl->nckil', T, dGdT_all)
        part3 = bm.einsum('nij,nckjl->nckil', Einv, T_dGdT_all)
        # 4) δ(tau Gdt) E^{-1}
        part4 = dtauGdt_all[..., None, None] * Einv[:, None, None, :, :]
        # 5) (tau Gdt) δ(E^{-1})
        part5 = tauGdt[:, None, None, None, None] * dEinv_all

        # 局部导数（对 Idxi_grad_part），再乘外层的 rho（与 Idxi_from_Ehat 保持一致）
        D2_all = (part1 + part2 + part3 + part4 + part5)  # (NC,c,k,d,d)

        # 打包（应用 R：第 0 行为 −sum_c）
        def pack_all(D2_comp_all):
            D1 = -bm.sum(D2_comp_all, axis=2, keepdims=True)  # (NC,c,1)
            V  = bm.concat([D1, D2_comp_all], axis=2)         # (NC,c,d+1)
            V  = bm.permute_dims(V, (1, 0, 2)).reshape(-1)    # (c*NC*(d+1),)
            return V

        vx_seg_x_all = pack_all(D2_all[:, :, 0, :, 0])  # δv_x / δx
        vy_seg_x_all = pack_all(D2_all[:, :, 0, :, 1])  # δv_y / δx
        vx_seg_y_all = pack_all(D2_all[:, :, 1, :, 0])  # δv_x / δy
        vy_seg_y_all = pack_all(D2_all[:, :, 1, :, 1])  # δv_y / δy

        # 行列索引与缩放
        rr_x_all  = self._row_off_x_tile_d % NN
        rr_y_all  = self._row_off_y_tile_d % NN
        rc_x_tile = (-1.0 / self.tau) 
        rc_y_tile = (-1.0 / self.tau) 
        cm_rep    = bm.repeat(cm, d+1)
        cm_rep_all = bm.tile(cm_rep, d)

        # 边界投影（列右乘）
        data00_tile = rc_x_tile * cm_rep_all * ( rxx[rr_x_all] * vx_seg_x_all + rxy[rr_x_all] * vy_seg_x_all )
        data10_tile = rc_y_tile * cm_rep_all * ( ryx[rr_y_all] * vx_seg_x_all + ryy[rr_y_all] * vy_seg_x_all )
        data01_tile = rc_x_tile * cm_rep_all * ( rxx[rr_x_all] * vx_seg_y_all + rxy[rr_x_all] * vy_seg_y_all )
        data11_tile = rc_y_tile * cm_rep_all * ( ryx[rr_y_all] * vx_seg_y_all + ryy[rr_y_all] * vy_seg_y_all )

        # j=0 列（−sum_c）
        def pack0(D2_0_comp):
            D1 = -bm.sum(D2_0_comp, axis=1, keepdims=True)
            V  = bm.concat([D1, D2_0_comp], axis=1)
            return V.reshape(-1)

        D2_0_x = -bm.sum(D2_all[:, :, 0, :, :], axis=1)
        D2_0_y = -bm.sum(D2_all[:, :, 1, :, :], axis=1)
        vx_0_x = pack0(D2_0_x[:, :, 0]); vy_0_x = pack0(D2_0_x[:, :, 1])
        vx_0_y = pack0(D2_0_y[:, :, 0]); vy_0_y = pack0(D2_0_y[:, :, 1])

        rr_x0  = self._row_off_x_0 % NN
        rr_y0  = self._row_off_y_0 % NN
        coef_x0 = (-1.0 / self.tau) 
        coef_y0 = (-1.0 / self.tau) 

        data00_0 = coef_x0 * cm_rep * ( rxx[rr_x0] * vx_0_x + rxy[rr_x0] * vy_0_x )
        data10_0 = coef_y0 * cm_rep * ( ryx[rr_y0] * vx_0_x + ryy[rr_y0] * vy_0_x )
        data01_0 = coef_x0 * cm_rep * ( rxx[rr_x0] * vx_0_y + rxy[rr_x0] * vy_0_y )
        data11_0 = coef_y0 * cm_rep * ( ryx[rr_y0] * vx_0_y + ryy[rr_y0] * vy_0_y )

        # 一次性装配
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

        JAC = coo_matrix((V.astype(float), (I.astype(int), J.astype(int))),
                         shape=(2*NN, 2*NN)).tocsr()
        return JAC

    def JAC_functional_2(self,Xi,M_inv):
        """
        与当前 T 型能量一致的雅可比:
        G = H(T)^gamma,  H = 0.5 *|T|_F^2 - ln(tau) - 1,  tau = det(T)
        Idxi = R @ [ rho * ( E_hat^{-1} T GdT + (tau*Gdt) E_hat^{-1} ) ]
        其中 GdT = gamma*H^{gamma-1}*T,  Gdt = - gamma*H^gamma/tau
        """
        d   = self.GD
        NN  = self.NN
        NC  = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同结构扩展"

        # 缓存
        cache = getattr(self, '_ivp_cache', None)
        if cache is None:
            raise RuntimeError("ivp 缓存未准备，请先调用 _prepare_ivp_cache")
        E_K     = cache['E_K']
        E_K_inv = cache['E_K_inv']
        rho     = cache['rho']
        cm      = cache['cm']
        P_diag  = cache['P_diag']
        theta   = cache['theta']
        rxx, ryy, rxy, ryx = cache['rxx'], cache['ryy'], cache['rxy'], cache['ryx']

        # 基本量
        E_hat   = self.edge_matrix(Xi)                 # (NC,d,d)
        Einv    = bm.linalg.inv(E_hat)                 # (NC,d,d)
        sqrt_M  = self.sqrt_M(M_inv)                   # (NC,d,d)
        sqrt_th = bm.sqrt(theta)
        B       = E_K_inv @ sqrt_M                     # (NC,d,d)
        T       = (1/sqrt_th * (E_hat @ B))            # (NC,d,d)
        I_d     = bm.eye(d, **self.kwargs0)
        I_mat   = bm.zeros_like(T, **self.kwargs0); I_mat += I_d

        # 标量与导数准备
        tau     = bm.linalg.det(T)                     # (NC,)
        eps     = 1e-14
        tau_pos = bm.maximum(tau, eps)
        Tinv    = bm.linalg.inv(T)                     # (NC,d,d)
        F2      = bm.sum(T**2, axis=(-2,-1))           # ||T||_F^2
        H       = 0.5 * F2  - bm.log(tau_pos) - 1      # (NC,)
        H_pos   = bm.maximum(H, eps)

        gamma   = self.gamma
        # GdT = gamma * H^{gamma-1} * T
        GdT     = (gamma * (H_pos**(gamma-1)))[..., None, None] * T  # (NC,d,d)
        # Gdt = - gamma * H^gamma / tau
        Gdt     = - gamma * (H_pos**gamma) / tau_pos                 # (NC,)

        # δE_hat^{-1} （对 (c,k) 的全张量）
        b_all        = bm.swapaxes(Einv, -1, -2)  # (NC,d,d)
        row_c_Einv   = Einv                       # (NC,d,d)
        dEinv_all = -( b_all[:, None, :, :, None] * row_c_Einv[:, :, None, None, :] )  # (NC,c,k,d,d)

        # δT_all = √θ · δE_hat · B = √θ · (e_k ⊗ row_c(B))
        eye     = bm.eye(d, **self.kwargs0)
        row_sel = eye[None, None, :, :, None]     # (1,1,k,d,1)
        Brow    = B                                # (NC,c,d) 取第 c 行
        dT_all  = (1/sqrt_th * 1.0) * (row_sel * Brow[:, :, None, None, :])  # (NC,c,k,d,d)

        # δH_all = [T - H T^{-T}] : δT
        S_H      = T - (H_pos[..., None, None] * bm.swapaxes(Tinv, -1, -2))
        dH_all   = bm.sum(S_H[:, None, None, :, :] * dT_all, axis=(3,4))  # (NC,c,k)

        # δGdT_all = gamma*[ (γ-1)H^{γ-2} δH * T + H^{γ-1} δT ]
        gm    = gamma
        H_gm1 = H_pos**(gamma-1.0)
        H_gm2 = H_pos**(gamma-2.0)
        termA = (gm * (gamma-1.0) * H_gm2)[:, None, None, None, None] * dH_all[..., None, None] * T[:, None, None, :, :]
        termB = (gm * H_gm1)[:, None, None, None, None] * dT_all
        dGdT_all = termA + termB  # (NC,c,k,d,d)

        # δGdt_all = -gamma*[ γ H^{γ-1} δH / tau + H^γ * ( -δtau / tau^2 ) ]
        # δtau = tau * tr(T^{-1} δT) = tau * (T^{-T} : δT)
        trTD = bm.sum(bm.swapaxes(Tinv, -1, -2)[:, None, None, :, :] * dT_all, axis=(3,4))  # (NC,c,k)
        dtau = tau_pos[:, None, None] * trTD
        H_gm  = H_pos**gamma
        term1 = (gamma * H_gm1 / tau_pos)[:, None, None] * dH_all
        term2 = - (H_gm / tau_pos**2)[:, None, None] * dtau
        dGdt_all = -gamma * (term1 + term2)  # (NC,c,k)

        # 组合各项：δ[E^{-1} T GdT] + δ[(tau*Gdt) E^{-1}]
        TGdT = T @ GdT  # (NC,d,d)
        # 1) δ(E^{-1}) T GdT
        part1 = bm.einsum('nckij,njl->nckil', dEinv_all, TGdT)
        # 2) E^{-1} δT GdT
        dT_GdT_all = bm.einsum('nckij,njl->nckil', dT_all, GdT)
        part2 = bm.einsum('nij,nckjl->nckil', Einv, dT_GdT_all)
        # 3) E^{-1} T δGdT
        T_dGdT_all = bm.einsum('nij,nckjl->nckil', T, dGdT_all)
        part3 = bm.einsum('nij,nckjl->nckil', Einv, T_dGdT_all)
        # 4) δ(tau*Gdt) E^{-1}
        part4 = dGdt_all[..., None, None] * Einv[:, None, None, :, :]
        # 5) (tau*Gdt) δ(E^{-1})
        part5 = (tau * Gdt)[..., None, None, None, None] * dEinv_all

        # 局部导数（对 Idxi_grad_part），再乘外层的 rho（与 Idxi_from_Ehat 保持一致）
        D2_all = (part1 + part2 + part3 + part4 + part5)  # (NC,c,k,d,d)

        # 打包（应用 R：第 0 行为 −sum_c）
        def pack_all(D2_comp_all):
            D1 = -bm.sum(D2_comp_all, axis=2, keepdims=True)  # (NC,c,1)
            V  = bm.concat([D1, D2_comp_all], axis=2)         # (NC,c,d+1)
            V  = bm.permute_dims(V, (1, 0, 2)).reshape(-1)    # (c*NC*(d+1),)
            return V

        vx_seg_x_all = pack_all(D2_all[:, :, 0, :, 0])  # δv_x / δx
        vy_seg_x_all = pack_all(D2_all[:, :, 0, :, 1])  # δv_y / δx
        vx_seg_y_all = pack_all(D2_all[:, :, 1, :, 0])  # δv_x / δy
        vy_seg_y_all = pack_all(D2_all[:, :, 1, :, 1])  # δv_y / δy

        # 行列索引与缩放
        rr_x_all  = self._row_off_x_tile_d % NN
        rr_y_all  = self._row_off_y_tile_d % NN
        rc_x_tile = (-1.0 / self.tau) 
        rc_y_tile = (-1.0 / self.tau) 
        cm_rep    = bm.repeat(cm, d+1)
        cm_rep_all = bm.tile(cm_rep, d)

        # 边界投影（列右乘）
        data00_tile = rc_x_tile * cm_rep_all * ( rxx[rr_x_all] * vx_seg_x_all + rxy[rr_x_all] * vy_seg_x_all )
        data10_tile = rc_y_tile * cm_rep_all * ( ryx[rr_y_all] * vx_seg_x_all + ryy[rr_y_all] * vy_seg_x_all )
        data01_tile = rc_x_tile * cm_rep_all * ( rxx[rr_x_all] * vx_seg_y_all + rxy[rr_x_all] * vy_seg_y_all )
        data11_tile = rc_y_tile * cm_rep_all * ( ryx[rr_y_all] * vx_seg_y_all + ryy[rr_y_all] * vy_seg_y_all )

        # j=0 列（−sum_c）
        def pack0(D2_0_comp):
            D1 = -bm.sum(D2_0_comp, axis=1, keepdims=True)
            V  = bm.concat([D1, D2_0_comp], axis=1)
            return V.reshape(-1)

        D2_0_x = -bm.sum(D2_all[:, :, 0, :, :], axis=1)
        D2_0_y = -bm.sum(D2_all[:, :, 1, :, :], axis=1)
        vx_0_x = pack0(D2_0_x[:, :, 0]); vy_0_x = pack0(D2_0_x[:, :, 1])
        vx_0_y = pack0(D2_0_y[:, :, 0]); vy_0_y = pack0(D2_0_y[:, :, 1])

        rr_x0  = self._row_off_x_0 % NN
        rr_y0  = self._row_off_y_0 % NN
        coef_x0 = (-1.0 / self.tau) 
        coef_y0 = (-1.0 / self.tau) 

        data00_0 = coef_x0 * cm_rep * ( rxx[rr_x0] * vx_0_x + rxy[rr_x0] * vy_0_x )
        data10_0 = coef_y0 * cm_rep * ( ryx[rr_y0] * vx_0_x + ryy[rr_y0] * vy_0_x )
        data01_0 = coef_x0 * cm_rep * ( rxx[rr_x0] * vx_0_y + rxy[rr_x0] * vy_0_y )
        data11_0 = coef_y0 * cm_rep * ( ryx[rr_y0] * vx_0_y + ryy[rr_y0] * vy_0_y )

        # 一次性装配
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

        JAC = coo_matrix((V.astype(float), (I.astype(int), J.astype(int))),
                        shape=(2*NN, 2*NN)).tocsr()
        return JAC
    
    
    # 这个目前存在没解决的问题
    def JAC_functional_simple(self, Xi, M_inv):
        """
        利用 edge_matrix2 + Idxi_from_Ehat 实现有限差分扰动的全局雅可比组装
        Xi: (NN, d) 当前逻辑网格节点
        M_inv: (NC, d, d) 目标度量张量的逆
        返回：scipy.sparse.csr_matrix
        """
        cell = self.cell      # (NC, d+1)
        NN = self.NN
        NC = self.NC
        d = self.GD

        # 1. 未扰动下的单元梯度
        cache = getattr(self, '_ivp_cache', None)
        assert cache is not None, "ivp_cache未初始化"
        E_K = cache['E_K']

        theta = cache['theta']
        M = bm.linalg.inv(M_inv)
        E_hat = self.edge_matrix(Xi)
        Idxi0 = self.Idxi_from_Ehat(E_hat, E_K, M, theta)  # (NC, d+1, d)

        # 预备常量
        Rlen = (d + 1) * d            # 每单元受影响的行数 (局部节点*(分量数))
        P = d * (d + 1)               # 扰动总数

        # 预计算每单元的行索引块（按 单元顺序扁平化）
        # base = cell * d -> 每个局部节点对应的起始自由度
        base = (cell * d).astype(int)                              # (NC, d+1)
        tile_k = bm.tile(bm.arange(d, dtype=int), (d+1,))          # (Rlen,)
        rows_block = bm.repeat(base, repeats=d, axis=1) + tile_k[None, :]  # (NC, Rlen)
        rows_flat_all = rows_block.reshape(-1)                    # (NC*Rlen,)

        rows_parts = []
        cols_parts = []
        data_parts = []
        # 2. 一次性批量扰动所有 (j1, k1)
        j1 = d+1
        k1 = d
        for j in range(j1):
            for k in range(k1):
                # 扰动 Xi
                
                eps = 1e-16
                EE = Xi[cell]
                PEE = EE[:,j,k]  # (NC,)
                ones = bm.ones_like(PEE, **self.kwargs0)
                h_mag = bm.maximum(bm.abs(PEE), ones) * eps**0.5
                h = bm.where(PEE >= 0,  h_mag, -h_mag)  # (NC,)
                ss = PEE >= 0
                h = ss - (~ss) * bm.abs(h) # (NC,)
                
                EE[:,j,k] += h
                Ec = EE[:,1:,:] - EE[:,0,:][:,None,:] # (NC, d, d)  扰动边矩阵
                
                Idxi1 = self.Idxi_from_Ehat(Ec, E_K, M, theta)  # (NC, d+1, d)
                dIdxi = (Idxi1 - Idxi0) / h[:,None,None]  # (NC, d+1, d)
                
                data_flat = dIdxi.reshape(-1).astype(float)   # (NC*Rlen,)

                # 列索引：每个单元的列自由度为 (cell[:, j] * d + k_pert)，重复 Rlen 次
                cols_vec = (cell[:, j] * d + k).astype(int)   # (NC,)
                cols_flat = bm.repeat(cols_vec, Rlen)               # (NC*Rlen,)

                # 行索引复用预计算的 rows_flat_all（与 data_flat 对齐）
                rows_flat = rows_flat_all  # (NC*Rlen,)
                
                rows_parts.append(rows_flat)
                cols_parts.append(cols_flat)
                data_parts.append(data_flat)
                self._debug_fd_column(Xi, M_inv, j, k)
                
                
            # 拼接所有扰动块并组装稀疏矩阵
        rows = bm.concat(rows_parts, axis=0).astype(int)
        cols = bm.concat(cols_parts, axis=0).astype(int)
        data = bm.concat(data_parts, axis=0).astype(float)

        # 转为 scipy.sparse 并返回
        JAC = coo_matrix((data, (rows, cols)), shape=(NN * d, NN * d)).tocsr()
        JAC = JAC.multiply(1.0 / self.tau)
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
    
    def get_physical_node(self,Xinew,X,vector_field):
        """
        计算物理网格的新节点位置
        x_{n+1} = x_n + eta * J * vector_field
        J = E_K E_hat_K^{-1} 为局部雅可比矩阵将逻辑网格的位移场拉回物理网格
        注意上述得到的位移需要在边界处进行修正,以保持边界形状
        eta 步长控制,其为了了防止网格翻转,一般取 eta in (0,1]
         
        Parameters:
            vector_field(Tensor): 逻辑网格节点的速度场 (NN, GD)
        Return:
            x_new(Tensor): 物理网格的新节点位置 (NN, GD)
        """
        cell = self.cell
        alpha = self.alpha
        E = self.edge_matrix(X) # (NC, GD, GD)
        Xinew_0 = Xinew[cell[:,0],:] # (NC, GD)
        E_map = Xinew[cell[:,1:],:] - Xinew_0[:, None , :] # (NC, GD, GD)
        E_map = bm.permute_dims(E_map, (0,2,1))  # (NC, GD, GD)
        J = E @ bm.linalg.inv(E_map) # (NC, GD, GD)
        vf_cell = bm.mean(vector_field[self.mesh.cell], axis=1) # (NC, GD)
        vf_physical_cell = bm.einsum('ijk,ik->ij', J , vf_cell) # (NC, GD)
        
        sm = self.sm
        cm = self.cm
        vf_physical = bm.zeros_like(vector_field , **self.kwargs0) # (NN, GD)
        vf_physical = bm.index_add(vf_physical , self.mesh.cell , (vf_physical_cell*cm[:, None])[:, None , :])
        vf_physical /= sm[:, None]
        
        Bdinnernode_idx = self.Bdinnernode_idx
        dot = bm.sum(self.Bi_Pnode_normal * vf_physical[Bdinnernode_idx],axis=1)
        vf_physical = bm.set_at(vf_physical , Bdinnernode_idx ,
                                vf_physical[Bdinnernode_idx] - dot[:,None] * self.Bi_Pnode_normal)
        vf_physical = bm.set_at(vf_physical , self.Vertices_idx , 0)
        
        coef = _compute_coef_2d(vf_physical,self.AC_generator)
        k = quad_equ_solver(coef)
        positive_k = bm.where(k>0, k, 1)
        eta = bm.min(positive_k)
        Xnew = self.mesh.node + alpha * eta * vf_physical

        return Xnew
    
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
        import matplotlib.pyplot as plt
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
            # self.gamma = 1 + (init_gamma - 1) * (it / self.total_steps)
            # if it == 0:
            theta = self.theta(M)
            # s = it / self.total_steps
            # theta = s * theta

            self._prepare_ivp_cache(X, M, M_inv,theta)
            P_diag = self._ivp_cache['P_diag']
            # eps = 1e-12
            # s_node = bm.maximum(P_diag, eps)                     # (NN,)
            # s_vec  = bm.concat([s_node, s_node], axis=0)         # (2NN,)
            # s_inv  = 1.0 / s_vec
            
            time.send("begin...")
            def ode_system(t, y):
                Xi_current = y.reshape(self.GD, self.NN).T
                v = self.vector_construction(Xi_current , M)
                time.send("ODE vector field")
                return v.ravel(order = 'F')
            
            def jac(t, y):
                # y = (s_vec * y)
                Xi_current = y.reshape(self.GD, self.NN).T
                J_y = self.JAC_functional_2(Xi_current, M_inv)
  
                time.send("ODE Jacobian matrix computation complete")
                # J_z = J_y.multiply(s_inv[:, None])             # 行缩放
                # J_z = J_z.multiply(s_vec[None, :])             # 列缩放
                return J_y
    
            # ODE 求解器需要的初值
            t_span = [0,self.t_span]
            y0 = Xi.ravel(order = 'F')
            # z0 = (s_inv * y0)
            sol = solve_ivp(ode_system, t_span, y0, jac=jac, method='BDF',
                                        first_step=self.t_span/self.step,
                                        atol=atol, rtol=rtol)
            # y_last = (s_vec * sol.y[:, -1])
            y_last = sol.y[:, -1]
            Xinew = y_last.reshape(self.GD, self.NN).T

            
            Xnew = self.linear_interpolate(Xi, Xinew , X)
            self.uh = self.interpolate(Xnew)
            self._construct(Xnew)
            print(f"TargetAdaptive: step {it+1}/{self.total_steps} completed.")
        time.send("Mesh redistribution complete")
        next(time)
        return Xnew