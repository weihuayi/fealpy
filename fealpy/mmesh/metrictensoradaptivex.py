from . import Monitor
from . import Interpolater
from .config import *
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from fealpy.utils import timer
from scipy.sparse.linalg import LinearOperator,cg
from scipy.sparse.linalg import spsolve as sv
import scipy.sparse as sp
time = timer()
next(time)

class MetricTensorAdaptiveX(Monitor, Interpolater):
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
    
    def _prepare_ivp_cache(self, xi):
        """
        预计算在一个 solve_ivp 步进内不变的量，减少 jac/ode 回调重复开销。
        """
        E_hat = self.edge_matrix(xi)  # (NC, GD, GD)
        E_hat_inv = bm.linalg.inv(E_hat)  # (NC, GD, GD)
        cache = {
            'E_hat': E_hat, 'E_hat_inv': E_hat_inv,
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
        
    def I_func(self, theta , lam , trA , detA , rho):
        """
        I = lambda *(tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        Parameters:
            theta(float): 积分全局乘子
            lam(float): 拉伸因子 lambda
            trA(Tensor): A 的迹
            detA(Tensor): A 的行列式
            rho(Tensor): 权重函数 rho
        Return:
            I(float): I 函数值
        """
        T = self.T( theta , trA , detA)
        I  = rho/lam * T
        I = bm.sum(self.cm * I)
        return I
    
    def T(self, theta , trA , det_A):
        """
        T = (tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        """
        gamma = self.gamma
        d = self.GD
        p = d * gamma / 2
        T = trA**p - d**p * gamma/2 * theta**p * bm.log(det_A)
        return T
    
    def TdA(self,trA):
        """
        T = (tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        TdA = d*gamma/2 * tr(A)^{d*gamma/2 - 1} * I 
        """
        gamma = self.gamma
        d = self.GD
        p = d * gamma / 2
        TdA = (p) * (trA**(p - 1))[...,None,None] * self.I_p
        return TdA
    
    def Tdg(self,theta ,det_A):
        """
        T = (tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        Tdg = -d^{d*gamma/2} * gamma/2* theta^{d*gamma/2} * 1/det(A)
        """
        gamma = self.gamma
        d = self.GD
        p = d * gamma / 2
        Tdg = - d**(p) * gamma/2 * theta**(p) * det_A**(-1)
        return Tdg
    
    def U_matrix(self,T , TdA , Tdg , A , g , E_K):
        """
        U = J (- 1/2 *T I + A TdA + (Tdg * g) I ) J^{-1}
        """
        E_hat_inv = self._ivp_cache['E_hat_inv']
        J_K = E_K @ E_hat_inv
        J_K_inv = bm.linalg.inv(J_K)
        H = A @ TdA + (Tdg * g - 0.5 * T)[..., None, None] * self.I_p
        U = J_K @ H @ J_K_inv
        return U
    
    def GdM(self,U , M , rho , lam):
        """
        GdM = -1/lambda * rho * U M^{-1}
        """
        M_inv = bm.linalg.inv(M)
        GdM = -1/lam * rho[..., None, None] * (U @ M_inv )
        return GdM
    
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

        k = -d
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

    def Idx_from_E_K(self,A ,g, trA, E_K, rho, theta):
        """
        泛函局部导数 Idxi 的计算
        Idx = rho/lambda * J (-0.5 * T I +  A TdA + (Tdg * g) I ) J^{-1} - 
             1/(d+1) * e ( sum_i trace( GdM M_node_i ) * V_i )
        Parameters:
            A (Tensor): 雅可比算子 J^{-1}M^{-1}J^-T (NC, GD, GD)
            g (Tensor): 算子 A 行列式 (NC,)
            trA (Tensor): A 的迹 (NC,)
            E_K (Tensor): 单元边矩阵 (NC, GD, GD)
            rho (Tensor): 权重函数 rho (NC,)
            theta (float): 积分全局乘子
        """
        GD = self.GD
        E_K_inv = bm.linalg.inv(E_K)
        V = self.R @ E_K_inv  # (NC, GD+1, GD)
        T = self.T(theta , trA , g)
        TdA = self.TdA(trA)
        Tdg = self.Tdg(theta , g)
        U = self.U_matrix(T , TdA , Tdg , A , g , E_K)
        
        rho = self.rho(self.M)
        lam = self.lam(theta)
        part0 = 2*rho[...,None,None]/lam * V @ U  # (NC, GD+1, GD)
        
        cell = self.cell
        GdM = self.GdM(U , self.M , rho , lam)  # (NC, GD, GD)
        M_cell = self.M_node[cell[:,1:]] - self.M_node[cell[:,0]][:,None,...]  # (NC, GD , GD , GD)
        u = bm.einsum('cij,clij->cl', GdM, M_cell)  # (NC, GD)
        u_Einv = bm.einsum('clj,cl->cj', E_K_inv, u)/(GD+1)  # (NC, GD)
        part1 = u_Einv[:, None, :]  # (NC, 1, GD)
        
        loacl_vector = -part0 + part1  # (NC, GD+1, GD)
        return loacl_vector
    
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

        self.I = bm.concat([
            self._row_off_x_tile_d, self._row_off_y_tile_d,
            self._row_off_x_0,      self._row_off_y_0,
            self._row_off_x_tile_d, self._row_off_y_tile_d,
            self._row_off_x_0,      self._row_off_y_0,
        ], axis=0)
        self.J = bm.concat([
            self._cols_x_tile_d, self._cols_x_tile_d,
            self._cols_x_0,      self._cols_x_0,
            self._cols_y_tile_d, self._cols_y_tile_d,
            self._cols_y_0,      self._cols_y_0,
        ], axis=0)
        # 完成
        self._jac_pat_ready = True
        ones = bm.ones(self.I.shape[0], dtype=bm.bool)
        S = coo_matrix((ones, (self.I.astype(int), self.J.astype(int))), shape=(2*NN, 2*NN)).tocsr()
        self._jac_sparsity = S
    
    def vector_construction(self, A , g ,trA , E_K, theta):
        """
        构造全局移动向量场
        v = rho/lambda * J (-1/2 *T I + A TdA + (Tdg * g) I )J^{-1}\
            - 1/(d+1) * e ( sum_i trace( GdM M_node_i ) * V_i )
        Parameters:
            A (Tensor): 雅可比算子 J^{-1}M^{-1}J^-T (NC, GD, GD)
            g (Tensor): 算子 A 行列式 (NC,)
            trA (Tensor): A 的迹 (NC,)
            E_K (Tensor): 单元边矩阵 (NC, GD, GD)
        Returns:
            v: (NN, GD) 参考网格上的全局移动向量场
        """
        local_vector = self.Idx_from_E_K(A , g , trA , E_K , 
                                         self.rho(self.M) , theta)
        cell = self.cell
        cm = self.cm
        global_vector = bm.zeros((self.NN, self.GD), dtype=bm.float64)
        global_vector = bm.index_add(global_vector , cell , cm[:,None,None] * local_vector)
        
        P_diag = self.balance(self.M_node)
        tau = self.tau
        v = -1/tau * global_vector * P_diag[:, None]  # (NN, GD)
        
        # 边界投影和角点固定
        Bi_Pnode_normal = self.Bi_Pnode_normal
        Bdinnernode_idx = self.Bdinnernode_idx
        dot = bm.sum(Bi_Pnode_normal * v[Bdinnernode_idx],axis=1)
        v = bm.set_at(v , Bdinnernode_idx ,
                        v[Bdinnernode_idx] - dot[:,None] * Bi_Pnode_normal)
        vertice_idx = self.Vertices_idx
        v = bm.set_at(v , vertice_idx , 0.0)
        return v
    
    def JAC_functional(self,A, g, trA , E_K,M_inv, theta):
        """
        I = 1/lambda *(tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        IdA = 1/lambda * d*gamma/2 * tr(A)^{d*gamma/2 - 1} * I 
        Idg = -1/lambda * d^{d*gamma/2} * gamma/2* theta^{d*gamma/2} * 1/det(A)
        
        v = rho/lambda * J (-0.5 * T I +  A TdA + (Tdg * g) I ) J^{-1} - 
            1/(d+1) * e ( sum_i trace( GdM M_node_i ) * V_i )
        """
        d = self.GD
        NC = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同张量结构扩展"
        E_hat = self._ivp_cache['E_hat']
        rho = self.rho(self.M)
        # 差分步长（相对尺度 + 绝对下限，数值稳健）
        cm = self.cm
        local = self.Idx_from_E_K(A , g , trA , E_K , rho , theta)   # (NC, d+1, d)
        local = local * cm[:, None, None]  # 去权重后的局部向量
        
        B = d * d
        k_idx, c_idx = bm.meshgrid(bm.arange(d), bm.arange(d), indexing='ij')
        k_idx = k_idx.reshape(-1)   # (B,)
        c_idx = c_idx.reshape(-1)   # (B,)
        K =bm.permute_dims(E_K, axes=(2,1,0))  # (NC, d, d)
        K_all = K.reshape((B,NC))  # (B, NC)
        
        eps = bm.finfo(E_K.dtype).eps
        h_mag = (K_all + bm.maximum(bm.abs(K_all), 1.0) * bm.sqrt(eps)) - K_all   # (B, NC)
        sgn   = bm.where(K_all >= 0, 1.0, -1.0)
        h_entry = sgn * bm.abs(h_mag)                 # (B, NC)
        
        basis = bm.zeros((B, d, d), **self.kwargs0)   # (B, d, d) one-hot
        basis = bm.set_at(basis, (bm.arange(B), k_idx, c_idx), 1.0)
        dE_all = h_entry[:, :, None, None] * basis[:, None, :, :]
        
        E_pos_all     = E_K[None, ...] + dE_all                         # (B, NC, d, d)
        cm_pos_all    = 0.5 * bm.abs(bm.linalg.det(E_pos_all))          # (B, NC)

        local_pos_list = []
        for b in range(B):
            A_pos_b = self.A(E_pos_all[b], E_hat, M_inv)
            g_pos_b = bm.linalg.det(A_pos_b)
            trA_pos_b = bm.trace(A_pos_b, axis1=1, axis2=2)
            local_pos_b = self.Idx_from_E_K(A_pos_b, g_pos_b, trA_pos_b, E_pos_all[b], rho, theta)  
            local_pos_list.append(local_pos_b)
        local_pos_all = bm.stack(local_pos_list, axis=0)
        local_pos_all = cm_pos_all[:, :, None, None] * local_pos_all 
        d_local_all = (local_pos_all - local) / h_entry[:, :, None, None]
        
        D2_all = bm.zeros((NC, d, d, d, d), **self.kwargs0)
        diff_jm = d_local_all[:, :, 1:, :]   # (B, NC, d, d) -> (B, NC, j, m)

        for b in range(B):
            c = c_idx[b]
            k = k_idx[b]
            D2_all = bm.set_at(D2_all, (slice(None), c, k, slice(None), slice(None)), diff_jm[b])
        # 交给装配器；它会根据 D2_all 自动补全 j=0 列（负和）并乘以权与投影
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
        rxx, ryy, rxy, ryx = self.rxx, self.ryy, self.rxy, self.ryx
        P_diag = self.balance(self.M_node)
        
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

        # 四个 tile 段（投影右乘 + 行缩放 + 单元权）
        data00_tile = rc_x_tile * ( rxx[rr_x_all] * vx_seg_x_all + rxy[rr_x_all] * vy_seg_x_all )
        data10_tile = rc_y_tile * ( ryx[rr_y_all] * vx_seg_x_all + ryy[rr_y_all] * vy_seg_x_all )
        data01_tile = rc_x_tile * ( rxx[rr_x_all] * vx_seg_y_all + rxy[rr_x_all] * vy_seg_y_all )
        data11_tile = rc_y_tile * ( ryx[rr_y_all] * vx_seg_y_all + ryy[rr_y_all] * vy_seg_y_all )

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

        data00_0 = coef_x0 * ( rxx[rr_x0] * vx_0_x + rxy[rr_x0] * vy_0_x )
        data10_0 = coef_y0 * ( ryx[rr_y0] * vx_0_x + ryy[rr_y0] * vy_0_x )
        data01_0 = coef_x0 * ( rxx[rr_x0] * vx_0_y + rxy[rr_x0] * vy_0_y )
        data11_0 = coef_y0 * ( ryx[rr_y0] * vx_0_y + ryy[rr_y0] * vy_0_y )

        # 一次性装配（COO -> CSR）
        V = bm.concat([
            data00_tile, data10_tile, data00_0, data10_0,
            data01_tile, data11_tile, data01_0, data11_0
        ], axis=0)

        from scipy.sparse import coo_matrix
        JAC = coo_matrix((V.astype(float), (self.I, self.J)),
                        shape=(2*NN, 2*NN)).tocsr()
        return JAC
    
    def _construct(self,moved_node:TensorLike):
        """
        @brief construct information for the harmap method before the next iteration
        """
        self.mesh.node = moved_node
        self.node = moved_node
        self.cm = self.mesh.entity_measure('cell')
        self.cm = bm.maximum(self.cm , 1e-14)
        self.sm = bm.zeros(self.NN, **self.kwargs0)
        self.sm = bm.index_add(self.sm , self.mesh.cell , self.cm[:, None])
    
    def mesh_redistributor(self , total_steps=None, h = None,
                           method='scipy',return_info = False):
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
        atol = 1e-5
        rtol = atol * 100
        
        I_h = []
        cm_min = []
        
        for it in range(total_steps):
            X = self.mesh.node
            Xi = self.logic_mesh.node
            
            self._prepare_ivp_cache(Xi)
            E_hat = self._ivp_cache['E_hat']

            I_base = bm.eye(self.GD, **self.kwargs0)
            self.I_p = bm.zeros_like(E_hat, **self.kwargs0)
            self.I_p += I_base
            
            def info_generator(x):
                E_K = self.edge_matrix(x)
                self.uh = self.interpolate(x)
                self._construct(x)
                self.monitor()
                self.mol_method()
                M = self.M
                theta = self.theta(M)
                M_inv = bm.linalg.inv(M)
                A = self.A(E_K , E_hat , M_inv)
                g = bm.linalg.det(A)
                g = bm.maximum(g , 1e-14)
                trA = bm.trace(A, axis1=1, axis2=2)
                trA = bm.maximum(trA , 1e-14)
                return E_K , A , g , trA, theta,M_inv
            
            if method == 'scipy':
                def ode_system(t, y):
                    X_current = y.reshape(self.GD, self.NN).T
                    E_K, A , g , trA, theta,_ = info_generator(X_current)
                    v = self.vector_construction(A , g ,trA , E_K , theta)
                    time.send("ODE vector field")
                    return v.ravel(order = 'F')
                
                def jac(t, y):
                    X_current = y.reshape(self.GD, self.NN).T
                    E_K, A , g , trA, theta, M_inv = info_generator(X_current)
                    J_y = self.JAC_functional(A , g , trA , E_K, M_inv, theta)
                    return J_y
                
                t_span = [0,self.t_span]
                y0 = X.ravel(order = 'F')
                sol = solve_ivp(ode_system, t_span, y0,jac = jac, method='BDF',
                                            jac_sparsity=self._jac_sparsity,
                                            first_step=self.t_span/self.step,
                                            atol=atol, rtol=rtol)
                y_last = sol.y[:, -1]
                Xnew = y_last.reshape(self.GD, self.NN).T
            else:
                Xnew = self.integrater(X, h, atol=atol, rtol=rtol,
                                        newton_tol=1e-6, newton_maxit=40,)
            
            if return_info:
                E_hat , A , g , trA, theta, _ = info_generator(Xnew)
                lam = self.lam(theta)
                rho = self.rho(self.M)
                I = self.I_func(theta, lam , trA , g, rho)
                I_h.append(I)
                cm_min.append(bm.min(self.cm).item())
        
            print(f"MTAdaptive: step {it+1}/{self.total_steps} completed.")
        time.send("Mesh redistribution complete")
        next(time)
        return Xnew , (I_h , cm_min) if return_info else Xnew

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
    
    def integrater(self, X, h, atol=1e-5, rtol=1e-3,
                                      newton_tol=1e-6, newton_maxit=20, 
                                      cg_tol=1e-8,
                                      h_min=None, h_max=None):
        """
        用隐式Euler(BDF1) + 拟牛顿 + cg 做时间积分替代 solve_ivp
        """
        NN = self.NN
        GD = self.GD
        Nvar = NN * GD
        y = X.ravel(order='F')
        last_delta = None  
        h_max = h_max or h * 10
        h_min = h_min or h * 0.1
        E_hat = self._ivp_cache['E_hat']
        def info_update(y_new):
            x = y_new.reshape(GD, NN).T
            E_K = self.edge_matrix(x)
            self.uh = self.interpolate(x)
            self._construct(x)
            self.monitor()
            self.mol_method()
            M = self.M
            theta = self.theta(M)
            M_inv = bm.linalg.inv(M)
            A = self.A(E_K , E_hat , M_inv)
            g = bm.linalg.det(A)
            g = bm.maximum(g , 1e-14)
            trA = bm.trace(A, axis1=1, axis2=2)
            trA = bm.maximum(trA , 1e-14)
            return E_K , A , g , trA, theta,M_inv
        
        def single_step(y_new,y , h , last_delta):
            r_pre = None
            for nit in range(newton_maxit):
                # 计算 f 和残差 r = F(y_new)
                time.send("Starting single time step")
                print(f"  Newton iteration {nit+1}/{newton_maxit}")
                E_K , A , g , trA, theta, M_inv = info_update(y_new)
                f = self.vector_construction(A, g, trA, E_K , theta).ravel(order='F')

                r = y_new - y - h * f
                res_norm = (r**2).sum()**0.5
                print(f"    Residual norm: {res_norm:.6e}")
                if res_norm < newton_tol:
                    break
                lin_tol = min(0.5, bm.sqrt(res_norm)) * min(1.0, cg_tol)
                # J = d f / d y 
                if nit == 0:
                    J = self.JAC_functional(A , g , trA , E_K, M_inv, theta)
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

            # 5. 步长自适应
            if scaled_error <= 1.0:
                # 接受步长
                total_time += h
                y = y_new
            h = bm.clip((1.0 / scaled_error)**0.5 * h, h_min, h_max)

        X_new = y.reshape(GD, NN).T
        return X_new