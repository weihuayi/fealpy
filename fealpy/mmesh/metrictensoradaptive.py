from . import Monitor
from . import Interpolater
from .tool import linear_surfploter
from .config import *
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from fealpy.utils import timer
from scipy.sparse.linalg import gmres, LinearOperator,cg ,lgmres
from scipy.sparse.linalg import spsolve as sv
from scipy.sparse import csc_matrix,block_diag,bmat
import scipy.sparse as sp
import matplotlib.pyplot as plt
time = timer()
next(time)

class MetricTensorAdaptive(Monitor, Interpolater):
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
        det_M = bm.linalg.det(M)
        rho_K = bm.sqrt(det_M)
        # sigma = bm.sum(rho_K , axis=0)
        sigma2 = bm.sum(cm * rho_K , axis=0)
        area = bm.sum(cm , axis=0)
        NC = self.NC
        theta = (sigma2/area)**(-2/d)
        return theta
        
    def I_func(self, theta , lam , trA , detA):
        """
        I = lambda *(tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        Parameters:
            theta(float): 积分全局乘子
            A(Tensor): 雅可比算子
        Return:
            T(float): T 函数值
        """
        d = self.GD
        gamma = self.gamma
        p = d * gamma / 2
        I  = 1/lam * ( trA**p - d**p * gamma/2 * theta**p * bm.log(detA) )
        return I
    
    def IdA(self,theta , trA):
        """
        I = 1/ambda *(tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        IdA = 1/lambda * d*gamma/2 * tr(A)^{d*gamma/2 - 1} * I 
        """
        lam = self.lam(theta)
        gamma = self.gamma
        d = self.GD
        p = d * gamma / 2
        pT_pA = 1/lam * (p) * (trA**(p - 1))[...,None,None] * self.I_p
        return pT_pA
    
    def Idg(self,theta ,det_A):
        """
        I = 1/lambda *(tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        Idg = -1/lambda * d^{d*gamma/2} * gamma/2* theta^{d*gamma/2} * 1/det(A)
        """
        lam = self.lam(theta)
        gamma = self.gamma
        d = self.GD
        p = d * gamma / 2
        Idg = - 1/lam * d**(p) * gamma/2 * theta**(p) * det_A**(-1)
        return Idg
    
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
        gamma = self.gamma
        det_M_node = bm.linalg.det(M_node)
        theta = self.theta(self.M)
        d = self.GD
        M_dim = (1/theta * det_M_node**(1/d))**0.5

        k = -d/2
        P_diag = M_dim**k # (NN,)
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

    def Idxi_from_Ehat(self,A ,g, trA, E_hat, rho, theta):
        """
        泛函局部导数 Idxi 的计算
        Idxi = 2 * rho * R @ [ E_hat^{-1} A TdA + (Tdg * g) E_hat^{-1} ]
        Parameters:
            A (Tensor): 雅可比算子 J^{-1}M^{-1}J^-T (NC, GD, GD)
            g (Tensor): 算子 A 行列式 (NC,)
            trA (Tensor): A 的迹 (NC,)
            E_hat (Tensor): 参考单元边矩阵 (NC, GD, GD)
            rho (Tensor): 权重函数 rho (NC,)
            theta (float): 积分全局乘子
        """
        E_hat_inv = bm.linalg.inv(E_hat)

        IdA = self.IdA(theta, trA)
        Idg = self.Idg(theta ,g)
        
        term0 = E_hat_inv @ A @ IdA
        term1 = (Idg * g)[..., None, None] * E_hat_inv
        
        Idxi_grad_part = 2 * rho[..., None, None] * (term0 + term1) # (NC, GD, GD)
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
    
    def vector_construction(self, A , g ,trA , E_hat):
        """
        构造全局移动向量场
        Parameters:
            A (Tensor): 雅可比算子 J^{-1}M^{-1}J^-T (NC, GD, GD)
            g (Tensor): 算子 A 行列式 (NC,)
            trA (Tensor): A 的迹 (NC,)
            E_hat (Tensor): 参考单元边矩阵 (NC, GD, GD)
        Returns:
            v: (NN, GD) 参考网格上的全局移动向量场
        """
        cache = self._ivp_cache
        if cache is not None:
            rho     = cache['rho']     
            cm      = cache['cm']      
            P_diag  = cache['P_diag']
            theta   = cache['theta']

        Idxi = self.Idxi_from_Ehat(A , g , trA , E_hat, rho, theta)  # (NC, GD+1, GD)
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
        v = bm.set_at(v , vertice_idx , 0.0)
        return v
    
    def JAC_functional(self,A,trA , g ,E_hat,M_inv):
        """
        I = 1/lambda *(tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        IdA = 1/lambda * d*gamma/2 * tr(A)^{d*gamma/2 - 1} * I 
        Idg = -1/lambda * d^{d*gamma/2} * gamma/2* theta^{d*gamma/2} * 1/det(A)
        d E_hat = e_k x e_c^T
        d E_hat_inv = - E_hat_inv dE_hat E_hat_inv
        d A =  2*M_inv (dE_hat E_K^{-1})^T
        d g =  g * tr(A^{-1} dA)
        
        d(IdA) = 1/lambda *(d*gamma/2) * (d*gamma/2 - 1) * tr(A)^{d*gamma/2 - 2} * tr(dA) * I
        d(Idg) = 1/lambda * d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * det(A)^{-2} * dg
        
        P1 = 2 * rho * d E_hat_inv (A IdA + Idg * g * I)
        P2 = 2 * rho * E_hat_inv (dA IdA + A d(IdA) + d(Idg) * g * I + Idg * dg * I)
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
        Einv     = bm.linalg.inv(E_hat)                     # (NC,d,d)
        Ehat_T   = bm.swapaxes(E_hat, -1, -2)               # (NC,d,d)
        C        = E_K_inv @ M_inv @ bm.swapaxes(E_K_inv, -1, -2)  # (NC,d,d)

        I_mat = self.I_p

        lam      = self.lam(theta)
        # 数值稳健性
        eps = 1e-14
        g_pos    = bm.maximum(g, eps)
        trA_safe = bm.maximum(trA, eps)

        # 目标泛函及导数（无 H）
        gamma = self.gamma
        power = (d * gamma / 2.0)
        # IdA = (d*gamma/2) * tr(A)^{d*gamma/2 - 1} * I
        IdA = self.IdA(theta,trA_safe)  # (NC,d,d)
        Idg = self.Idg(theta, g_pos)  # (NC,)

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

        # d(TdA)_all = 1/lambda *(d*gamma/2) * (d*gamma/2 - 1) * tr(A)^{d*gamma/2 - 2} * tr(dA) * I
        tr_dA = bm.trace(dA_all, axis1=3, axis2=4)                            # (NC,c,k)
        coef_td = 1/lam * (power) * (power - 1.0) * (trA_safe ** (power - 2.0))        # (NC,)
        dTdA_all = coef_td[:, None, None, None, None] * tr_dA[..., None, None] * I_mat[:, None, None, :, :]  # (NC,c,k,d,d)
 
        # d(Tdg)_all = 1/lambda * d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * det(A)^{-2} * dg
        coef_tdg = 1/lam * d**(power) * gamma/2  * (theta ** power) * (g_pos ** (-2.0))             # (NC,)
        dTdg_all = coef_tdg[:, None, None] * dg_all                           # (NC,c,k)

        # P1 = 2 * rho * d E_hat_inv (A IdA + Idg * g * I)
        M0 = A @ IdA + (Idg * g_pos)[..., None, None] * I_mat                 # (NC,d,d)
        P1_all = 2.0 * rho[:,None,None,None,None] * bm.einsum('nckij,njl->nckil', dEinv_all, M0)

        # P2 = 2 * rho * E_hat_inv (dA TdA + A d(TdA) + d(Tdg) * g * I + Tdg * dg * I)
        part_a = bm.einsum('nckij,njl->nckil', dA_all, IdA)                     # dA TdA
        part_b = bm.einsum('nij,nckjl->nckil', A, dTdA_all)                     # A d(TdA)
        part_c = (dTdg_all * g_pos[:, None, None])[..., None, None] * I_mat[:, None, None, :, :]   # d(Tdg) g I
        part_d = (Idg[:, None, None] * dg_all)[..., None, None] * I_mat[:, None, None, :, :]       # Tdg dg I
        bracket = part_a + part_b + part_c + part_d                                                 # (NC,c,k,d,d)

        P2_all  = 2.0 * rho[:,None,None,None,None] * bm.einsum('nij,nckjl->nckil', Einv, bracket)
        # 局部二阶块（对 j'=1..d 的“梯度部分”），形状 (NC,c,k,d,matrix_cols=j')
        D2_all = (P1_all + P2_all)                                                # (NC,c,k,d,d)

        # D2_all 装配为全局 JAC
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
        rc_x_tile = (-1.0 / self.tau) * P_diag[rr_x_all]                    # (d*NC*(d+1),)
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
    
    def _construct(self,moved_node:TensorLike):
        """
        @brief construct information for the harmap method before the next iteration
        """
        self.mesh.node = moved_node
        self.node = moved_node
        self.cm = self.mesh.entity_measure('cell')
        self.sm = bm.zeros(self.NN, **self.kwargs0)
        self.sm = bm.index_add(self.sm , self.mesh.cell , self.cm[:, None])
    
    def mesh_redistributor(self , total_steps=None, h = None,method='scipy'):
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
        if total_steps is None:
            total_steps = self.total_steps
        if h is None:
            h = self.t_span/self.step
        atol = 1e-6
        rtol = atol * 100
        
        for it in range(total_steps):
            self.monitor()
            self.mol_method()
            M = self.M
            M_inv = bm.linalg.inv(M)
            X = self.mesh.node
            Xi = self.logic_mesh.node
            
            theta = self.theta(M)
            print(theta)
            self._prepare_ivp_cache(X, M, M_inv,theta)
            E_K = self._ivp_cache['E_K']

            I_base = bm.eye(self.GD, **self.kwargs0)
            self.I_p = bm.zeros_like(E_K, **self.kwargs0)
            self.I_p += I_base
            if method == 'scipy':
                def ode_system(t, y):
                    Xi_current = y.reshape(self.GD, self.NN).T
                    E_hat = self.edge_matrix(Xi_current)
                    A = self.A(E_K , E_hat , M_inv)
                    g = bm.linalg.det(A)
                    trA = bm.trace(A, axis1=1, axis2=2)
                    v = self.vector_construction(A , g ,trA , E_hat)
                    time.send("ODE vector field")
                    return v.ravel(order = 'F')
                
                def jac(t, y):
                    Xi_current = y.reshape(self.GD, self.NN).T
                    E_hat = self.edge_matrix(Xi_current)
                    A = self.A(E_K , E_hat , M_inv)
                    g = bm.linalg.det(A)
                    trA = bm.trace(A, axis1=1, axis2=2)
                    J_y = self.JAC_functional(A , trA , g , E_hat, M_inv)
                    return J_y
                
                t_span = [0,self.t_span]
                y0 = Xi.ravel(order = 'F')
                sol = solve_ivp(ode_system, t_span, y0, jac=jac, method='BDF',
                                            first_step=self.t_span/self.step,
                                            atol=atol, rtol=rtol)
                y_last = sol.y[:, -1]
                Xinew = y_last.reshape(self.GD, self.NN).T
            else:
                Xinew = self.integrater(Xi, M_inv, h, atol=atol, rtol=rtol,
                                        newton_tol=1e-6, newton_maxit=20,)
            
            Xnew = self.linear_interpolate(Xi, Xinew , X)
            self.uh = self.interpolate(Xnew)
            self._construct(Xnew)
            print(f"MTAdaptive: step {it+1}/{self.total_steps} completed.")
        time.send("Mesh redistribution complete")
        next(time)
        return Xnew

    def two_level_mesh_redistributor(self,coarsen_mesh ,
                                          prolong,
                                          refine_level = 2):
        """
        两网格度量张量自适应网格算法
        从一个粗网格开始，逐步细化到目标加密层级的网格结构.
        """
        # 粗网格层的自适应
        Xi_refine = self.logic_mesh.node.copy()
        self.mesh = coarsen_mesh

        ones_f = bm.ones(prolong.shape[0] , **self.kwargs0)
        weights = prolong.T @ ones_f            # 长度 = NN_coarse
        R = sp.diags(1.0 / weights) @ prolong.T
        uh = R @ self.uh[:]
        
        self.pspace = LagrangeFESpace(self.mesh, p=1)
        self.uh = self.pspace.function()
        self.uh[:] = uh 

        self.__init__(self.mesh, self.beta , self.pspace , self.config)
        X_coarse = self.mesh_redistributor()
    
        h = self.t_span/self.step
        self.mesh.uniform_refine(refine_level)
        
        X = self.mesh.node
        cell = self.mesh.cell
        self.isBdNode = self.mesh.boundary_node_flag()
        def laplace_smooth(X , iterations):
            for _ in range(iterations):
                mean_node = bm.mean(X[cell], axis=1)
                neighbor_sum = bm.zeros_like(X, **self.kwargs0)
                counts = bm.zeros((X.shape[0], 1), dtype=bm.float64, device=self.device)

                neighbor_sum = bm.index_add(neighbor_sum , cell , mean_node[:, None , :])
                counts = bm.index_add(counts , cell , 1)
                avg_position = neighbor_sum / counts
                avg_position[self.isBdNode] = X[self.isBdNode]
                X = 0.5 * X + 0.5 * avg_position
            return X
        X = laplace_smooth(X, iterations=1)
        self.mesh.node = X
        
        self.pspace = LagrangeFESpace(self.mesh, p=1)
        Xnew = self.mesh.node
        self.uh = self.interpolate(Xnew)

        self.__init__(self.mesh, self.beta , self.pspace , self.config)
        self.logic_mesh.node = Xi_refine
        Xi = self.logic_mesh.node

        self.tau = 0.005
        X_refine = self.mesh_redistributor(total_steps=5, h=h )
        
    def mulit_preconditioner(self ,A_c,A_r, v , omega=0.8):
        """
        多重网格预条件器构造
        1.初始猜测 z = 0
        2.平滑处理 z = S_pre(z)
        3.计算残差 r = v - A z
        4.计算粗网格残差 r_c = R r
        5.粗网格上求解 A_c e_c = r_c
        6.细网格上插值 e_r = P e_c
        7.修正 z = z + e_r
        8.平滑处理 z = S_post(z)
        
        Parameters:
            dt: 时间步长
            coarsen_jac: 粗网格雅可比矩阵
            refine_jac: 细网格雅可比矩阵
            v: 待预条件向量
        """
        def smoother(Aop, x, b, iterations):
            if iterations <= 0:
                return x
            D = bm.array(Aop.diagonal())
            for _ in range(iterations):
                r = b - Aop.dot(x)
                x = x + omega * (r / D)
            return x
        z = bm.zeros_like(v , **self.kwargs0)
        # 预平滑
        z = smoother(A_r ,z , v ,  iterations = 7)
        # 计算残差
        r = v - A_r @ z
        # 计算粗网格残差
        r_c = self.R_block @ r
        
        # 粗网格上求解
        e_c = sv(A_c , r_c)
        # 细网格上插值
        e_r = self.P_block @ e_c
        # 修正
        z = z + e_r
        # 后平滑
        z = smoother(A_r ,z , v , iterations = 7)
        return z
    
    def integrater(self, Xi, M_inv, h, atol=1e-6, rtol=1e-4,
                                      newton_tol=1e-6, newton_maxit=20, 
                                      cg_tol=1e-8,
                                      h_min=None, h_max=None):
        """
        用隐式Euler(BDF1) + 拟牛顿 + cg 做时间积分替代 solve_ivp
        """
        NN = self.NN
        GD = self.GD
        Nvar = NN * GD
        y = Xi.ravel(order='F')
        last_delta = None  # GMRES 的初始猜测
        h_max = h_max or h * 10
        h_min = h_min or h * 0.1
        
        def info_update(y_new):
            Yi = y_new.reshape(GD, NN).T
            E_hat = self.edge_matrix(Yi)
            A = self.A(self._ivp_cache['E_K'], E_hat, M_inv)
            g = bm.linalg.det(A)
            trA = bm.trace(A, axis1=-2, axis2=-1)
            return A, g, trA, E_hat
        
        def single_step(y_new,y , h , last_delta):
            r_pre = None
            for nit in range(newton_maxit):
                # 计算 f 和残差 r = F(y_new)
                time.send("Starting single time step")
                A, g, trA, E_hat = info_update(y_new)
                f = self.vector_construction(A, g, trA, E_hat).ravel(order='F')
 
                r = y_new - y - h * f
                res_norm = (r**2).sum()**0.5
                if res_norm < newton_tol:
                    break
                lin_tol = min(0.5, bm.sqrt(res_norm)) * min(1.0, cg_tol)
                # J = d f / d y 
                if nit == 0:
                    J = self.JAC_functional(A, trA, g, E_hat, M_inv)
                    def Aop_mv(v):
                        Av = v - h * (J @ v)
                        return Av
                    A_op = LinearOperator((Nvar, Nvar), matvec=Aop_mv)
                    A_base_mv = A_op.dot
                else:
                    s = delta
                    t = r - r_pre
                    denom = float(bm.dot(s, s)) + 1e-14
                    yTs = bm.dot(t, s)
                    if yTs < 1e-14:  # 回退到Broyden
                        alpha = t - A_base_mv(s)
                        def Aop_mv(v):
                            Av = A_base_mv(v)
                            return Av + alpha * ((s @ v) / denom)
                    else:
                        Bks = A_base_mv(s)
                        sBks = bm.dot(s, Bks)
                        t1 = t / yTs
                        def Aop_mv(v):
                            Av = A_base_mv(v)
                            term1 = (bm.dot(Bks, v)/ sBks) * Bks
                            term2 = bm.dot(t1,v) * t
                            return Av + term2 - term1
                    A_op = LinearOperator((Nvar, Nvar), matvec=Aop_mv)

                # 线性系统: (I - h * J) delta = -r
                rhs = -r
                x0 = last_delta if last_delta is not None else None

                delta, info = cg(A_op, rhs, x0=x0, atol=lin_tol,maxiter=200,)
    
                last_delta =  delta
                y_new = y_new + delta
                r_pre = r
    
            return y_new , last_delta , r
            
        total_time = 0.0

        while total_time < self.t_span:
            if total_time + h > self.t_span:
                h = self.t_span - total_time
            # Newton 迭代求解 F(y_new) = y_new - y - h * f(t_np1, y_new) = 0
            y_new = y.copy()
            
            y_new , last_delta, res = single_step( y_new , y , h , last_delta)
            
            tol_vector = atol + rtol * bm.abs(y)
            scaled_error = bm.max(bm.abs(res) / tol_vector)

            # 5. 步长自适应 (理论正确)
            if scaled_error <= 1.0:
                # 接受步长
                total_time += h
                y = y_new
            h = bm.clip((1.0 / scaled_error)**0.5 * h, h_min, h_max)

        Xi_new = y.reshape(GD, NN).T
        return Xi_new