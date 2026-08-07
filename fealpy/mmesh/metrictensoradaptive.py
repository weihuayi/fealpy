from . import Monitor
from . import Interpolater
from .config import *
from fealpy.utils import timer
from .metric_shared import GeometricDiscreteCore
from scipy.sparse.linalg import LinearOperator,cg
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve as sv
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


if njit is not None:
    @njit(cache=True)
    def _mta_c_kernel(E_K_inv, M_inv):
        nc = E_K_inv.shape[0]
        C = np.empty((nc, 2, 2), dtype=np.float64)
        for n in range(nc):
            e00 = E_K_inv[n, 0, 0]
            e01 = E_K_inv[n, 0, 1]
            e10 = E_K_inv[n, 1, 0]
            e11 = E_K_inv[n, 1, 1]
            m00 = M_inv[n, 0, 0]
            m01 = M_inv[n, 0, 1]
            m10 = M_inv[n, 1, 0]
            m11 = M_inv[n, 1, 1]
            b00 = e00 * m00 + e01 * m10
            b01 = e00 * m01 + e01 * m11
            b10 = e10 * m00 + e11 * m10
            b11 = e10 * m01 + e11 * m11
            C[n, 0, 0] = b00 * e00 + b01 * e01
            C[n, 0, 1] = b00 * e10 + b01 * e11
            C[n, 1, 0] = b10 * e00 + b11 * e01
            C[n, 1, 1] = b10 * e10 + b11 * e11
        return C


    @njit(cache=True)
    def _mta_A_from_C_kernel(E_hat, C):
        nc = E_hat.shape[0]
        A = np.empty((nc, 2, 2), dtype=np.float64)
        for n in range(nc):
            e00 = E_hat[n, 0, 0]
            e01 = E_hat[n, 0, 1]
            e10 = E_hat[n, 1, 0]
            e11 = E_hat[n, 1, 1]
            c00 = C[n, 0, 0]
            c01 = C[n, 0, 1]
            c10 = C[n, 1, 0]
            c11 = C[n, 1, 1]
            b00 = e00 * c00 + e01 * c10
            b01 = e00 * c01 + e01 * c11
            b10 = e10 * c00 + e11 * c10
            b11 = e10 * c01 + e11 * c11
            A[n, 0, 0] = b00 * e00 + b01 * e01
            A[n, 0, 1] = b00 * e10 + b01 * e11
            A[n, 1, 0] = b10 * e00 + b11 * e01
            A[n, 1, 1] = b10 * e10 + b11 * e11
        return A


    @njit(cache=True)
    def _mta_idxi_kernel(A, g, trA, E_hat, rho, theta, gamma):
        nc = A.shape[0]
        local = np.empty((nc, 3, 2), dtype=np.float64)
        p = gamma
        theta_p = theta ** p
        d_p = 2.0 ** p
        lam = d_p * theta_p * (1.0 - p * np.log(theta))
        t_const = -d_p * gamma * 0.5 * theta_p
        for n in range(nc):
            e00 = E_hat[n, 0, 0]
            e01 = E_hat[n, 0, 1]
            e10 = E_hat[n, 1, 0]
            e11 = E_hat[n, 1, 1]
            det = e00 * e11 - e01 * e10
            inv00 = e11 / det
            inv01 = -e01 / det
            inv10 = -e10 / det
            inv11 = e00 / det

            td = p * (trA[n] ** (p - 1.0))
            b00 = td * (inv00 * A[n, 0, 0] + inv01 * A[n, 1, 0]) + t_const * inv00
            b01 = td * (inv00 * A[n, 0, 1] + inv01 * A[n, 1, 1]) + t_const * inv01
            b10 = td * (inv10 * A[n, 0, 0] + inv11 * A[n, 1, 0]) + t_const * inv10
            b11 = td * (inv10 * A[n, 0, 1] + inv11 * A[n, 1, 1]) + t_const * inv11
            scale = 2.0 * rho[n] / lam
            b00 *= scale
            b01 *= scale
            b10 *= scale
            b11 *= scale

            local[n, 0, 0] = -(b00 + b10)
            local[n, 0, 1] = -(b01 + b11)
            local[n, 1, 0] = b00
            local[n, 1, 1] = b01
            local[n, 2, 0] = b10
            local[n, 2, 1] = b11
        return local


    @njit(cache=True)
    def _mta_vector_assembly_kernel(cell, cm, local, p_diag, bd_idx, bd_normal,
                                    vertices_idx, tau, nn):
        global_vector = np.zeros((nn, 2), dtype=np.float64)
        nc = cell.shape[0]
        for c in range(nc):
            weight = cm[c]
            for j in range(3):
                node = cell[c, j]
                global_vector[node, 0] += weight * local[c, j, 0]
                global_vector[node, 1] += weight * local[c, j, 1]

        v = np.empty((nn, 2), dtype=np.float64)
        inv_tau = -1.0 / tau
        for i in range(nn):
            scale = inv_tau * p_diag[i]
            v[i, 0] = scale * global_vector[i, 0]
            v[i, 1] = scale * global_vector[i, 1]

        for q in range(bd_idx.shape[0]):
            node = bd_idx[q]
            nx = bd_normal[q, 0]
            ny = bd_normal[q, 1]
            dot = nx * v[node, 0] + ny * v[node, 1]
            v[node, 0] -= dot * nx
            v[node, 1] -= dot * ny

        for q in range(vertices_idx.shape[0]):
            node = vertices_idx[q]
            v[node, 0] = 0.0
            v[node, 1] = 0.0
        return v


    @njit(cache=True)
    def _mta_jac_values_kernel(D2_all, cm, p_diag, rxx, ryy, rxy, ryx,
                               rr_x_all, rr_y_all, rr_x0, rr_y0, tau):
        nc = D2_all.shape[0]
        tile_len = 6 * nc
        base_len = 3 * nc
        V = np.empty(4 * tile_len + 4 * base_len, dtype=np.float64)
        inv_tau = -1.0 / tau
        off00 = 0
        off10 = off00 + tile_len
        off00_0 = off10 + tile_len
        off10_0 = off00_0 + base_len
        off01 = off10_0 + base_len
        off11 = off01 + tile_len
        off01_0 = off11 + tile_len
        off11_0 = off01_0 + base_len

        for c in range(2):
            for n in range(nc):
                cmn = cm[n]
                for jpack in range(3):
                    idx = (c * nc + n) * 3 + jpack
                    if jpack == 0:
                        vx_seg_x = -(D2_all[n, c, 0, 0, 0] + D2_all[n, c, 0, 1, 0])
                        vy_seg_x = -(D2_all[n, c, 0, 0, 1] + D2_all[n, c, 0, 1, 1])
                        vx_seg_y = -(D2_all[n, c, 1, 0, 0] + D2_all[n, c, 1, 1, 0])
                        vy_seg_y = -(D2_all[n, c, 1, 0, 1] + D2_all[n, c, 1, 1, 1])
                    else:
                        j = jpack - 1
                        vx_seg_x = D2_all[n, c, 0, j, 0]
                        vy_seg_x = D2_all[n, c, 0, j, 1]
                        vx_seg_y = D2_all[n, c, 1, j, 0]
                        vy_seg_y = D2_all[n, c, 1, j, 1]

                    rx = rr_x_all[idx]
                    ry = rr_y_all[idx]
                    coef_x = inv_tau * p_diag[rx] * cmn
                    coef_y = inv_tau * p_diag[ry] * cmn
                    V[off00 + idx] = coef_x * (rxx[rx] * vx_seg_x + rxy[rx] * vy_seg_x)
                    V[off10 + idx] = coef_y * (ryx[ry] * vx_seg_x + ryy[ry] * vy_seg_x)
                    V[off01 + idx] = coef_x * (rxx[rx] * vx_seg_y + rxy[rx] * vy_seg_y)
                    V[off11 + idx] = coef_y * (ryx[ry] * vx_seg_y + ryy[ry] * vy_seg_y)

        for n in range(nc):
            cmn = cm[n]
            for jpack in range(3):
                idx = n * 3 + jpack
                if jpack == 0:
                    vx_0_x = 0.0
                    vy_0_x = 0.0
                    vx_0_y = 0.0
                    vy_0_y = 0.0
                    for j in range(2):
                        for c in range(2):
                            vx_0_x += D2_all[n, c, 0, j, 0]
                            vy_0_x += D2_all[n, c, 0, j, 1]
                            vx_0_y += D2_all[n, c, 1, j, 0]
                            vy_0_y += D2_all[n, c, 1, j, 1]
                else:
                    j = jpack - 1
                    vx_0_x = -(D2_all[n, 0, 0, j, 0] + D2_all[n, 1, 0, j, 0])
                    vy_0_x = -(D2_all[n, 0, 0, j, 1] + D2_all[n, 1, 0, j, 1])
                    vx_0_y = -(D2_all[n, 0, 1, j, 0] + D2_all[n, 1, 1, j, 0])
                    vy_0_y = -(D2_all[n, 0, 1, j, 1] + D2_all[n, 1, 1, j, 1])

                rx = rr_x0[idx]
                ry = rr_y0[idx]
                coef_x = inv_tau * p_diag[rx] * cmn
                coef_y = inv_tau * p_diag[ry] * cmn
                V[off00_0 + idx] = coef_x * (rxx[rx] * vx_0_x + rxy[rx] * vy_0_x)
                V[off10_0 + idx] = coef_y * (ryx[ry] * vx_0_x + ryy[ry] * vy_0_x)
                V[off01_0 + idx] = coef_x * (rxx[rx] * vx_0_y + rxy[rx] * vy_0_y)
                V[off11_0 + idx] = coef_y * (ryx[ry] * vx_0_y + ryy[ry] * vy_0_y)
        return V
else:
    _mta_c_kernel = None
    _mta_A_from_C_kernel = None
    _mta_idxi_kernel = None
    _mta_vector_assembly_kernel = None
    _mta_jac_values_kernel = None

class MetricTensorAdaptive(Monitor, Interpolater):
    def __init__(self, mesh, beta, space, config:Config):
        super().__init__(mesh, beta, space, config)
        self.config = config
        self.alpha = config.alpha
        self.tau = config.tau
        self.t_max = config.t_max
        self.maxit = config.maxit
        self.pre_steps = config.pre_steps
        self.gamma = config.gamma
        
        self.geo_core = GeometricDiscreteCore(mesh)
        self.R = self.geo_core.R_matrix()

        self.cell2cell = self.mesh.cell_to_cell()
        self.total_steps = 10
        self.t_span = self.t_max
        self.step = 10
        self.BD_projector()
        self._build_jac_pattern()
        self.tol = config.tol if config.tol is not None else self._caculate_tol()
        
    def _prepare_ivp_cache(self, X, M,theta):
        """
        预计算在一个 ODE 步进内不变的量，减少 jac/ode 回调重复开销。
        """
        E_K     = self.edge_matrix(X)           # (NC,d,d)
        E_K_inv = bm.linalg.inv(E_K)            # (NC,d,d)
        det_E_K = bm.linalg.det(E_K)            # (NC,)
        rho     = self.rho(M)                   # (NC,)
        P_diag  = self.balance(self.M_node, theta)    # (NN,)
        cache = {
            'E_K': E_K, 'E_K_inv': E_K_inv, 'det_E_K': det_E_K,
            'rho': rho, 'cm': self.cm, 'P_diag': P_diag,'theta': theta,
        }
        self._ivp_cache = cache
    
    def edge_matrix(self,X):
        return self.geo_core.edge_matrix(X)
    
    def A(self,E , E_hat , M_inv):
        cache = getattr(self, '_ivp_cache', None)
        if (self.GD == 2 and _mta_A_from_C_kernel is not None and
                cache is not None and E is cache.get('E_K') and 'C_K' in cache):
            return _mta_A_from_C_kernel(
                np.asarray(E_hat, dtype=np.float64),
                np.asarray(cache['C_K'], dtype=np.float64),
            )
        return self.geo_core.A(E , E_hat , M_inv)
    
    def rho(self,M):
        return self.geo_core.rho(M)
    
    def theta(self, M):
        return self.geo_core.theta(M)
    
    def balance(self,M_node, theta, power=None , mixed=True):
        return self.geo_core.balance(M_node, theta, power, mixed)
        
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
        d = self.GD
        gamma = self.gamma
        p = d * gamma / 2
        I  = rho/lam * ( trA**p - d**p * gamma/2 * theta**p * bm.log(detA) )
        I = bm.sum(self.cm * I)
        return I
    
    def TdA(self,trA):
        """
        T = (tr(A)^{d*gamma/2} -d^{d*gamma/2}* gamma/2 * theta^(d*gamma/2) * ln (det(A)))
        TdA = d*gamma/2 * tr(A)^{d*gamma/2 - 1} * I 
        """
        gamma = self.gamma
        d = self.GD
        p = d * gamma / 2
        pT_pA = p * (trA**(p - 1))[...,None,None] * self.I_p
        return pT_pA
    
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
    
    def lam(self , theta):
        """
        拉伸因子 lambda
        """
        d = self.GD
        gamma = self.gamma
        power = d * gamma / 2
        lam = d**(power) * theta**(power)*(1- power * bm.log(theta))
        return lam

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
        if self.GD == 2 and _mta_idxi_kernel is not None:
            return _mta_idxi_kernel(
                np.asarray(A, dtype=np.float64),
                np.asarray(g, dtype=np.float64),
                np.asarray(trA, dtype=np.float64),
                np.asarray(E_hat, dtype=np.float64),
                np.asarray(rho, dtype=np.float64),
                float(theta),
                float(self.gamma),
            )
        E_hat_inv = bm.linalg.inv(E_hat)

        TdA = self.TdA(trA)
        Tdg = self.Tdg(theta ,g)
        
        term0 = E_hat_inv @ A @ TdA
        term1 = (Tdg * g)[..., None, None] * E_hat_inv
        lam = self.lam(theta)
        Idxi_grad_part = 2/lam * rho[..., None, None] * (term0 + term1) # (NC, GD, GD)
        Idxi = self.R[None,...] @ Idxi_grad_part # (NC, GD+1, GD)
        return Idxi
    
    def BD_projector(self):
        NN = self.NN
        idx = self.Bdinnernode_idx              # 边界节点索引 (nb,)
        n = self.Bi_Lnode_normal                # (nb, 2), 每个边界节点的单位法向
        vertice_idx = self.Vertices_idx        # 角点索引 (nv,)
        projector_class = self.geo_core.bd_projector(idx , n , vertice_idx)
        self.Rxx = projector_class['Rxx']
        self.Ryy = projector_class['Ryy']
        self.Rxy = projector_class['Rxy']
        self.Ryx = projector_class['Ryx']
        self.rxx = projector_class['rxx']
        self.ryy = projector_class['ryy']
        self.rxy = projector_class['rxy']
        self.ryx = projector_class['ryx']
        
    def _build_jac_pattern(self):
        geo = self.geo_core
        geo.jac_pattern()
        self.rr_x_all = geo.rr_x_all
        self.rr_y_all = geo.rr_y_all
        self.rr_x0 = geo.rr_x0
        self.rr_y0 = geo.rr_y0
        self.I = geo.I
        self.J = geo.J
    
    def vector_construction(self, A , g ,trA , E_hat, return_local=False):
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
        if self.GD == 2 and _mta_vector_assembly_kernel is not None:
            v = _mta_vector_assembly_kernel(
                np.asarray(self.cell, dtype=np.int64),
                np.asarray(self.cm, dtype=np.float64),
                np.asarray(Idxi, dtype=np.float64),
                np.asarray(P_diag, dtype=np.float64),
                np.asarray(self.Bdinnernode_idx, dtype=np.int64),
                np.asarray(self.Bi_Lnode_normal, dtype=np.float64),
                np.asarray(self.Vertices_idx, dtype=np.int64),
                float(self.tau),
                int(self.NN),
            )
        else:
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
        if return_local:
            return v, Idxi
        return v
    
    def JAC_functional(self,A,trA , g ,E_hat,M_inv, local=None):
        d = self.GD
        NC = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同张量结构扩展"
        cache = getattr(self, '_ivp_cache', None)
        E_K     = cache['E_K']
        E_K_inv = cache['E_K_inv']
        rho     = cache['rho']
        theta   = cache['theta']

        # 差分步长（相对尺度 + 绝对下限，数值稳健）
        if local is None:
            local = self.Idxi_from_Ehat(A , g ,trA , E_hat , rho,theta)   # (NC, d+1, d)
        
        B = d * d
        k_idx, c_idx = bm.meshgrid(bm.arange(d), bm.arange(d), indexing='ij')
        k_idx = k_idx.reshape(-1)   # (B,)
        c_idx = c_idx.reshape(-1)   # (B,)
        K =bm.permute_dims(E_hat, axes=(2,1,0))  # (NC, d, d)
        K_all = K.reshape((B,NC))  # (B, NC)
        
        eps = bm.finfo(E_hat.dtype).eps
        h_mag = (K_all + bm.maximum(bm.abs(K_all), 1.0) * bm.sqrt(eps)) - K_all   # (B, NC)
        sgn   = bm.where(K_all >= 0, 1.0, -1.0)
        h_entry = sgn * bm.abs(h_mag)                 # (B, NC)
        
        basis = bm.zeros((B, d, d), **self.kwargs0)   # (B, d, d) one-hot
        basis = bm.set_at(basis, (bm.arange(B), k_idx, c_idx), 1.0)
        dE_all = h_entry[:, :, None, None] * basis[:, None, :, :]
        
        E_pos_all     = E_hat[None, ...] + dE_all                         # (B, NC, d, d)

        local_pos_list = []
        for b in range(B):
            A_pos_b = self.A(E_K, E_pos_all[b],  M_inv)
            g_pos_b = bm.linalg.det(A_pos_b)
            trA_pos_b = bm.trace(A_pos_b, axis1=1, axis2=2)
            local_pos_b = self.Idxi_from_Ehat(A_pos_b, g_pos_b, trA_pos_b, 
                                              E_pos_all[b], rho, theta)
            local_pos_list.append(local_pos_b)
        local_pos_all = bm.stack(local_pos_list, axis=0)
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
        P_diag = self._ivp_cache['P_diag']
        if d == 2 and _mta_jac_values_kernel is not None:
            V = _mta_jac_values_kernel(
                np.asarray(D2_all, dtype=np.float64),
                np.asarray(self.cm, dtype=np.float64),
                np.asarray(P_diag, dtype=np.float64),
                np.asarray(self.rxx, dtype=np.float64),
                np.asarray(self.ryy, dtype=np.float64),
                np.asarray(self.rxy, dtype=np.float64),
                np.asarray(self.ryx, dtype=np.float64),
                np.asarray(self.rr_x_all, dtype=np.int64),
                np.asarray(self.rr_y_all, dtype=np.int64),
                np.asarray(self.rr_x0, dtype=np.int64),
                np.asarray(self.rr_y0, dtype=np.int64),
                float(self.tau),
            )
            JAC = coo_matrix((V, (self.I, self.J)),shape=(2*NN, 2*NN)).tocsr()
            return JAC

        cm = self.cm
        rxx, ryy, rxy, ryx = self.rxx, self.ryy, self.rxy, self.ryx

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
        rr_x_all  = self.rr_x_all
        rr_y_all  = self.rr_y_all
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

        rr_x0  = self.rr_x0
        rr_y0  = self.rr_y0
        coef_x0 = (-1.0 / self.tau) * P_diag[rr_x0]
        coef_y0 = (-1.0 / self.tau) * P_diag[rr_y0]

        data00_0 = coef_x0 * cm_rep * ( rxx[rr_x0] * vx_0_x + rxy[rr_x0] * vy_0_x )
        data10_0 = coef_y0 * cm_rep * ( ryx[rr_y0] * vx_0_x + ryy[rr_y0] * vy_0_x )
        data01_0 = coef_x0 * cm_rep * ( rxx[rr_x0] * vx_0_y + rxy[rr_x0] * vy_0_y )
        data11_0 = coef_y0 * cm_rep * ( ryx[rr_y0] * vx_0_y + ryy[rr_y0] * vy_0_y )

        V = bm.concat([
            data00_tile, data10_tile, data00_0, data10_0,
            data01_tile, data11_tile, data01_0, data11_0
        ], axis=0)

        JAC = coo_matrix((V, (self.I, self.J)),shape=(2*NN, 2*NN)).tocsr()
        return JAC

    def sundials_integrater(self, Xi, M_inv, h, atol=1e-6, rtol=1e-4):
        """
        用 SUNDIALS/CVODE 的 BDF 变步长积分逻辑网格 ODE。
        """
        try:
            import sksundae as sun
        except ImportError as exc:
            raise ImportError(
                "method='SUNDIALS' requires scikit-SUNDAE. "
                "Install it with: conda install -c conda-forge scikit-sundae"
            ) from exc

        NN = self.NN
        GD = self.GD
        Nvar = NN * GD
        y0 = np.asarray(Xi.ravel(order='F'), dtype=np.float64)

        pattern = coo_matrix(
            (np.ones(len(self.I), dtype=np.float64),
             (np.asarray(self.I, dtype=np.int64), np.asarray(self.J, dtype=np.int64))),
            shape=(Nvar, Nvar),
        ).tocsc()
        pattern.sort_indices()

        info_cache = {'y': None, 'value': None}
        rhs_cache = {'y': None, 'value': None}
        local_cache = {'y': None, 'value': None}

        def same_state(cache, y):
            cached_y = cache['y']
            return cached_y is not None and cached_y.shape == y.shape and np.array_equal(cached_y, y)

        def info_update(y):
            if same_state(info_cache, y):
                return info_cache['value']
            Yi = y.reshape(GD, NN).T
            E_hat = self.edge_matrix(Yi)
            A = self.A(self._ivp_cache['E_K'], E_hat, M_inv)
            g = bm.linalg.det(A)
            trA = bm.trace(A, axis1=-2, axis2=-1)
            value = (A, g, trA, E_hat)
            info_cache['y'] = np.asarray(y, dtype=np.float64).copy()
            info_cache['value'] = value
            return value

        def rhsfn(t, y, yp):
            if same_state(rhs_cache, y):
                yp[:] = rhs_cache['value']
                return
            A, g, trA, E_hat = info_update(y)
            v, local = self.vector_construction(A, g, trA, E_hat, return_local=True)
            value = np.asarray(
                v.ravel(order='F'),
                dtype=np.float64,
            )
            rhs_cache['y'] = np.asarray(y, dtype=np.float64).copy()
            rhs_cache['value'] = value.copy()
            local_cache['y'] = rhs_cache['y'].copy()
            local_cache['value'] = local
            yp[:] = value

        def jacfn(t, y, yp, JJ):
            A, g, trA, E_hat = info_update(y)
            local = local_cache['value'] if same_state(local_cache, y) else None
            J = self.JAC_functional(A, trA, g, E_hat, M_inv, local=local).tocsc()
            J.sort_indices()
            if np.array_equal(J.indptr, pattern.indptr) and np.array_equal(J.indices, pattern.indices):
                JJ[:] = J.data
                return

            JJ[:] = 0.0
            for col in range(Nvar):
                p0, p1 = pattern.indptr[col], pattern.indptr[col + 1]
                j0, j1 = J.indptr[col], J.indptr[col + 1]
                if p0 == p1 or j0 == j1:
                    continue
                rows = pattern.indices[p0:p1]
                jrows = J.indices[j0:j1]
                pos = np.searchsorted(rows, jrows)
                valid = pos < rows.size
                valid[valid] &= rows[pos[valid]] == jrows[valid]
                JJ[p0 + pos[valid]] = J.data[j0:j1][valid]

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Custom sparse Jacobian approximation will be ignored.*",
                category=UserWarning,
            )
            solver = sun.cvode.CVODE(
                rhsfn,
                method='BDF',
                first_step=h,
                rtol=rtol,
                atol=atol,
                linsolver='sparse',
                sparsity=pattern,
                jacfn=jacfn,
            )
        sol = solver.solve(np.array([0.0, self.t_span], dtype=np.float64), y0)
        if not sol.success:
            raise RuntimeError(f"SUNDIALS/CVODE failed: {sol.message}")
        return sol.y[-1].reshape(GD, NN).T
        
    def linear_interpolate(self, Xi, Xi_new , X):
        """
        linear interpolation method
        
        Parameters
            moved_node: TensorLike, new node positions
        """
        node2cell = self.node2cell
        i, j = node2cell.row, node2cell.col
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
            phi = self.mesh.shape_function(lam[valid], 1)
            valid_value = bm.sum(phi[...,None] * X[self.cell[cells[valid]]], axis=1)

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
    
    def mesh_redistributor(self , total_steps=None, h = None,
                           method='BDF_LDF',return_info = False, return_timemesh = False):
        """
        逆变拉格朗日乘子法自适应网格算法
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
            h = self.tau if self.tau is not None else self.t_span/self.step
        atol = 1e-6
        rtol = atol * 100
        
        I_h = []
        cm_min = []
        I_t = []
        time_mesh = [self.mesh.node]
        global j
        j = 0
        if method == 'BDF_SMW':
            self._smw_stats = {
                'factor_count': 0,
                'factor_backend': None,
                'rank2_updates': 0,
                'rank1_updates': 0,
                'max_rank': 0,
                'line_fail': 0,
                'small_solve_fail': 0,
                'b0_solve_count': 0,
                'b0_solve_call_count': 0,
                'multi_rhs_solve_count': 0,
                'saved_w_solves': 0,
                'symbolic_factor_count': 0,
                'numeric_factor_count': 0,
                'rank_stack_updates': 0,
                'last_res': None,
            }
            self._smw_mumps_cache = {
                'ctx': None,
                'indptr': None,
                'indices': None,
                'shape': None,
            }
        for it in range(total_steps):
            self.monitor()
            self.mol_method()
            M = self.M
            M_inv = bm.linalg.inv(M)
            X = self.mesh.node
            Xi = self.logic_mesh.node
            
            theta = self.theta(M)
            self._prepare_ivp_cache(X, M,theta)
            if self.GD == 2 and _mta_c_kernel is not None:
                self._ivp_cache['C_K'] = _mta_c_kernel(
                    np.asarray(self._ivp_cache['E_K_inv'], dtype=np.float64),
                    np.asarray(M_inv, dtype=np.float64),
                )
            E_K = self._ivp_cache['E_K']

            I_base = bm.eye(self.GD, **self.kwargs0)
            self.I_p = bm.zeros_like(E_K, **self.kwargs0)
            self.I_p += I_base
            
            def info_generator(xi):
                E_hat = self.edge_matrix(xi)
                A = self.A(E_K , E_hat , M_inv)
                g = bm.linalg.det(A)
                trA = bm.trace(A, axis1=1, axis2=2)
                return E_hat , A , g , trA
            
            if method == 'BDF_LFP':
                Xinew = self.integrater(Xi, M_inv, h, atol=atol, rtol=rtol,
                                        newton_tol=1e-6, newton_maxit=20,)
            elif method == 'BDF_SMW':
                Xinew = self.integrater(Xi, M_inv, h, atol=atol, rtol=rtol,
                                        newton_tol=1e-6, newton_maxit=20,
                                        solver='smw')
            elif method in ('SUNDIALS', 'CVODE', 'SUNDIALS_CVODE'):
                Xinew = self.sundials_integrater(Xi, M_inv, h, atol=atol, rtol=rtol)
            else:
                Xinew = self.LBFGS_integrater(Xi, M_inv, h, atol=atol, rtol=rtol)
            
            Xnew = self.linear_interpolate(Xi, Xinew , X)
            
            if return_info:
                E_hat , A , g , trA = info_generator(Xinew)
                lam = self.lam(theta)
                I = self.I_func(theta, lam , trA , g, self._ivp_cache['rho'])
                I_h.append(I)
                cm_min.append(bm.min(self.cm).item())
            if return_timemesh:
                time_mesh.append(Xnew)
            
            physical_error = bm.max(bm.linalg.norm(Xnew - self.node,axis=1))
            logic_error = bm.max(bm.linalg.norm(Xinew - Xi,axis=1))
            error = min(float(physical_error), float(logic_error))
            print(
                f"step {it+1}/{self.total_steps} , "
                f"physical_error: {physical_error}, logic_error: {logic_error}"
            )
            
            self.uh = self.interpolate(Xnew)
            self._construct(Xnew)
            if error < self.tol:
                print(f"Converged at step {it+1} with error {error}")
                break

        ret = {"X": Xnew}
        if method == 'BDF_SMW':
            print("SMW stats:", self._smw_stats)
            cache = getattr(self, '_smw_mumps_cache', None)
            if cache is not None and cache.get('ctx') is not None:
                cache['ctx'].destroy()
                cache['ctx'] = None
        if return_info:
            I_h_array = bm.array(I_h , **self.kwargs0)
            I_t = (I_h_array[1:] - I_h_array[:-1])
            ret["info"] = (I_h,I_t, cm_min)
        if return_timemesh:
            ret["time_mesh"] = time_mesh
        return ret

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
    
        h = self.tau if self.tau is not None else self.t_span/self.step
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
    
    def _caculate_tol(self):
        """
        @brief caculate_tol: calculate the tolerance between logic nodes
        """
        logic_mesh = self.logic_mesh
        logic_em = logic_mesh.entity_measure('edge')
        cell2edge = logic_mesh.cell_to_edge()
        em_cell = logic_em[cell2edge]
        p = self.p
        if self.TD == 3:
            if self.g_type == "Simplexmesh" :
                logic_cm = logic_mesh.entity_measure('cell')
                mul = em_cell[:,:3]*bm.flip(em_cell[:, 3:],axis=1)
                v = 0.5*bm.sum(mul,axis=1)
                d = bm.min(bm.sqrt(v*(v-mul[:,0])*(v-mul[:,1])*(v-mul[:,2]))/(3*logic_cm))
            else:
                logic_node = logic_mesh.node
                logic_cell = logic_mesh.cell
                nocell = logic_node[logic_cell]
                lenth = bm.linalg.norm(nocell[:,0] - 
                                       nocell[:,6],axis=-1)
                d = bm.min(lenth)       
        else:
            if self.g_type == "Simplexmesh" :
                logic_cm = logic_mesh.entity_measure('cell')
                d = bm.min(bm.prod(em_cell,axis=1)/(2*logic_cm)).item()
            else:
                logic_node = logic_mesh.node
                logic_cell = logic_mesh.cell
                k = bm.arange((p+1)**2 , **self.kwargs1)
                k = k.reshape(p+1,p+1)
                con0 = logic_node[logic_cell[:,k[0,0]]]
                con1 = logic_node[logic_cell[:,k[-1,-1]]]
                con2 = logic_node[logic_cell[:,k[0,-1]]]
                con3 = logic_node[logic_cell[:,k[-1,0]]]
                e0 = bm.linalg.norm(con0 - con1,axis=1)
                e1 = bm.linalg.norm(con2 - con3,axis=1)
                d = bm.min(bm.concat([e0, e1])).item()*2
        return d*0.1/p
    
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
                                      h_min=None, h_max=None,
                                      solver='lfp'):
        """
        用隐式Euler(BDF1) + 拟牛顿 + cg 做时间积分
        """
        NN = self.NN
        GD = self.GD
        Nvar = NN * GD
        y = Xi.ravel(order='F')
        last_delta = None  # GMRES 的初始猜测
        h_max = h_max or h * 10
        h_min = h_min or h * 1e-6
        bdf_iter_stats = {
            'nonlinear_total': 0,
            'nonlinear_last': 0,
            'nonlinear_max': 0,
        }

        class _BDFStageFailure(RuntimeError):
            pass
        
        def info_update(y_new):
            Yi = y_new.reshape(GD, NN).T
            E_hat = self.edge_matrix(Yi)
            A = self.A(self._ivp_cache['E_K'], E_hat, M_inv)
            g = bm.linalg.det(A)
            trA = bm.trace(A, axis1=-2, axis2=-1)
            return A, g, trA, E_hat

        def residual_update(y_base, y_new):
            A, g, trA, E_hat = info_update(y_new)
            f = self.vector_construction(A, g, trA, E_hat).ravel(order='F')
            r = y_new - y_base - h * f
            if not np.all(np.isfinite(np.asarray(r, dtype=np.float64))):
                raise _BDFStageFailure("non-finite BDF residual")
            return r, A, g, trA, E_hat

        self_owner = self
        mumps_cache = getattr(self, '_smw_mumps_cache', None)
        local_mumps_cache = mumps_cache is None
        if mumps_cache is None:
            mumps_cache = {
                'ctx': None,
                'indptr': None,
                'indices': None,
                'shape': None,
            }
        pattern_rows = np.asarray(self.I, dtype=np.int64)
        pattern_cols = np.asarray(self.J, dtype=np.int64)
        diag_idx = np.arange(Nvar, dtype=np.int64)
        pattern = sp.coo_matrix(
            (np.zeros(pattern_rows.shape[0] + Nvar, dtype=np.float64),
             (np.concatenate((pattern_rows, diag_idx)),
              np.concatenate((pattern_cols, diag_idx)))),
            shape=(Nvar, Nvar),
        ).tocsr()
        pattern.sort_indices()
        pattern_coo = pattern.tocoo()
        pattern_rows = pattern_coo.row
        pattern_cols = pattern_coo.col
        pattern_zeros = np.zeros(pattern_rows.shape[0], dtype=np.float64)

        class _ScipyLUFactor:
            def __init__(self, A):
                self.lu = splu(A.tocsc())
                self.name = 'scipy_splu'

            def solve(self, rhs):
                return self.lu.solve(np.asarray(rhs, dtype=np.float64))

            def destroy(self):
                pass

        class _MumpsFactor:
            def __init__(self, A):
                from mumps import DMumpsContext
                A = A.tocsr()
                A.sort_indices()
                same_pattern = (
                    mumps_cache['ctx'] is not None and
                    mumps_cache['shape'] == A.shape and
                    np.array_equal(mumps_cache['indptr'], A.indptr) and
                    np.array_equal(mumps_cache['indices'], A.indices)
                )
                if not same_pattern:
                    if mumps_cache['ctx'] is not None:
                        mumps_cache['ctx'].destroy()
                    ctx = DMumpsContext(par=1, sym=0, comm=None)
                    ctx.set_silent()
                    ctx.set_centralized_sparse(A)
                    ctx.run(job=4)
                    mumps_cache['ctx'] = ctx
                    mumps_cache['indptr'] = A.indptr.copy()
                    mumps_cache['indices'] = A.indices.copy()
                    mumps_cache['shape'] = A.shape
                    smw_stats = getattr(self_owner, '_smw_stats', None)
                    if smw_stats is not None:
                        smw_stats['symbolic_factor_count'] += 1
                else:
                    ctx = mumps_cache['ctx']
                    ctx.set_centralized_assembled_values(A.data)
                    ctx.run(job=2)
                smw_stats = getattr(self_owner, '_smw_stats', None)
                if smw_stats is not None:
                    smw_stats['numeric_factor_count'] += 1
                self.ctx = mumps_cache['ctx']
                self.name = 'mumps'

            def solve(self, rhs):
                arr = np.asarray(rhs, dtype=np.float64)
                if arr.ndim == 1:
                    x = arr.copy()
                    self.ctx.set_rhs(x)
                    self.ctx.run(job=3)
                    return x
                x = np.array(arr, dtype=np.float64, order='F', copy=True)
                self.ctx.id.nrhs = x.shape[1]
                self.ctx.id.lrhs = x.shape[0]
                self.ctx._refs.update(rhs=x)
                self.ctx.id.rhs = self.ctx.cast_array(x)
                self.ctx.run(job=3)
                self.ctx.id.nrhs = 1
                self.ctx.id.lrhs = self.ctx.id.n
                return x

            def destroy(self):
                pass

        def make_direct_factor(A):
            try:
                return _MumpsFactor(A)
            except Exception as exc:
                smw_stats = getattr(self_owner, '_smw_stats', None)
                if smw_stats is not None:
                    smw_stats['mumps_factor_fail'] = smw_stats.get('mumps_factor_fail', 0) + 1
                try:
                    return _ScipyLUFactor(A)
                except Exception as scipy_exc:
                    if smw_stats is not None:
                        smw_stats['direct_factor_fail'] = smw_stats.get('direct_factor_fail', 0) + 1
                        smw_stats['last_factor_error'] = str(scipy_exc)
                    raise _BDFStageFailure(
                        f"direct factorization failed: {scipy_exc}"
                    ) from exc

        def smw_single_step(y_new, y_base, h):
            r, A, g, trA, E_hat = residual_update(y_base, y_new)
            J = self.JAC_functional(A, trA, g, E_hat, M_inv)
            if not np.all(np.isfinite(np.asarray(J.data, dtype=np.float64))):
                raise _BDFStageFailure("non-finite base tangent")
            Jcoo = J.tocoo()
            B0 = sp.coo_matrix(
                (np.concatenate((-h * Jcoo.data, pattern_zeros, np.ones(Nvar))),
                 (np.concatenate((Jcoo.row, pattern_rows, diag_idx)),
                  np.concatenate((Jcoo.col, pattern_cols, diag_idx)))),
                shape=(Nvar, Nvar),
            ).tocsr()
            B0.sort_indices()
            factor = make_direct_factor(B0)
            smw_stats = getattr(self, '_smw_stats', None)
            if smw_stats is not None:
                smw_stats['factor_count'] += 1
                smw_stats['factor_backend'] = factor.name
            max_rank = 12
            V_store = np.empty((Nvar, max_rank), dtype=np.float64)
            Z_store = np.empty((Nvar, max_rank), dtype=np.float64)
            rank = 0
            last_delta = None

            def solve_B0(rhs):
                if smw_stats is not None:
                    arr = np.asarray(rhs)
                    smw_stats['b0_solve_count'] += 1 if arr.ndim == 1 else arr.shape[1]
                    smw_stats['b0_solve_call_count'] += 1
                    if arr.ndim == 2 and arr.shape[1] > 1:
                        smw_stats['multi_rhs_solve_count'] += 1
                return factor.solve(rhs)

            def append_compact(new_V, new_Z):
                nonlocal rank
                add = len(new_V)
                if add >= max_rank:
                    keep_V = new_V[-max_rank:]
                    keep_Z = new_Z[-max_rank:]
                    for j, (v_col, z_col) in enumerate(zip(keep_V, keep_Z)):
                        V_store[:, j] = v_col
                        Z_store[:, j] = z_col
                    rank = max_rank
                else:
                    overflow = max(0, rank + add - max_rank)
                    if overflow:
                        kept = rank - overflow
                        V_store[:, :kept] = V_store[:, overflow:rank]
                        Z_store[:, :kept] = Z_store[:, overflow:rank]
                        rank = kept
                    for v_col, z_col in zip(new_V, new_Z):
                        V_store[:, rank] = v_col
                        Z_store[:, rank] = z_col
                        rank += 1
                if smw_stats is not None:
                    smw_stats['rank_stack_updates'] += 1
                    smw_stats['max_rank'] = max(smw_stats['max_rank'], rank)

            def smw_direction(r, q=None):
                if q is None:
                    q = solve_B0(r)
                if rank == 0:
                    return -q, q
                V = V_store[:, :rank]
                Z = Z_store[:, :rank]
                small = np.eye(rank) + V.T @ Z
                rhs_small = V.T @ q
                try:
                    coeff = np.linalg.solve(small, rhs_small)
                except np.linalg.LinAlgError:
                    if smw_stats is not None:
                        smw_stats['small_solve_fail'] += 1
                    return None, q
                return -q + Z @ coeff, q

            def orientation_ok(y_trial):
                Yi = y_trial.reshape(GD, NN).T
                E_hat_trial = self.edge_matrix(Yi)
                det_floor = float(getattr(self, "logic_det_floor", 1.0e-10))
                if not bool(bm.all(bm.linalg.det(E_hat_trial) > det_floor)):
                    return False
                admissible = getattr(self, "admissible_logic_node", None)
                if admissible is not None and not admissible(Yi):
                    return False
                return True

            try:
                q_cached = None
                for _ in range(newton_maxit):
                    bdf_iter_stats['nonlinear_total'] += 1
                    res_norm = float((r**2).sum()**0.5)
                    if not np.isfinite(res_norm):
                        raise _BDFStageFailure("non-finite residual norm")
                    if res_norm < newton_tol:
                        break
                    p, q = smw_direction(np.asarray(r, dtype=np.float64), q_cached)
                    q_cached = None
                    if p is None or not np.all(np.isfinite(p)):
                        break

                    lam = 1.0
                    accepted = False
                    phi0 = float(0.5 * np.dot(r, r))
                    for _ls in range(20):
                        y_trial = y_new + lam * p
                        if not orientation_ok(y_trial):
                            lam *= 0.5
                            continue
                        try:
                            r_trial, *_ = residual_update(y_base, y_trial)
                        except _BDFStageFailure:
                            lam *= 0.5
                            continue
                        phi_trial = float(0.5 * np.dot(r_trial, r_trial))
                        if not np.isfinite(phi_trial):
                            lam *= 0.5
                            continue
                        if phi_trial <= (1.0 - 1e-4 * lam) * phi0:
                            accepted = True
                            break
                        lam *= 0.5
                    if not accepted:
                        if smw_stats is not None:
                            smw_stats['line_fail'] += 1
                        break

                    s = lam * p
                    t = np.asarray(r_trial - r, dtype=np.float64)
                    w = -lam * np.asarray(r, dtype=np.float64)
                    z_w = -lam * q
                    need_next_q = float((r_trial**2).sum()**0.5) >= newton_tol
                    sn = np.linalg.norm(s)
                    tn = np.linalg.norm(t)
                    wn = np.linalg.norm(w)
                    st = float(np.dot(s, t))
                    sw = float(np.dot(s, w))
                    eps_sec = 1e-12
                    if (abs(st) >= eps_sec * max(1.0, sn * tn) and
                            abs(sw) >= eps_sec * max(1.0, sn * wn)):
                        new_V = [t / st, -w / sw]
                        if need_next_q:
                            z_and_q = solve_B0(np.column_stack((t, r_trial)))
                            new_Z = [z_and_q[:, 0], z_w]
                            q_cached = z_and_q[:, 1]
                        else:
                            new_Z = [solve_B0(t), z_w]
                        if smw_stats is not None:
                            smw_stats['rank2_updates'] += 1
                            smw_stats['saved_w_solves'] += 1
                    else:
                        ss = float(np.dot(s, s))
                        if ss <= 1e-30:
                            break
                        new_V = [s / ss]
                        if need_next_q:
                            z_and_q = solve_B0(np.column_stack((t, r_trial)))
                            new_Z = [z_and_q[:, 0] - z_w]
                            q_cached = z_and_q[:, 1]
                        else:
                            new_Z = [solve_B0(t) - z_w]
                    if any(not np.all(np.isfinite(z)) for z in new_Z):
                        raise _BDFStageFailure("non-finite SMW update")
                        if smw_stats is not None:
                            smw_stats['rank1_updates'] += 1

                    append_compact(new_V, new_Z)

                    y_new = y_trial
                    r = r_trial
                    if smw_stats is not None:
                        smw_stats['last_res'] = float((r**2).sum()**0.5)
                    last_delta = s
                return y_new, last_delta, r
            finally:
                factor.destroy()
        
        def single_step(y_new,y , h , last_delta):
            if solver == 'smw':
                return smw_single_step(y_new, y, h)
            r_pre = None
            for nit in range(newton_maxit):
                bdf_iter_stats['nonlinear_total'] += 1
                # 计算 f 和残差 r = F(y_new)
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
        bdf_step_count = 0
        bdf_accepted_count = 0
        bdf_rejected_count = 0
        bdf_nonfinite_error_count = 0
        bdf_min_h = float(h)
        bdf_last_h = float(h)
        bdf_last_scaled_error = np.nan
        bdf_max_steps = getattr(self, "bdf_max_steps", None)
        bdf_stage_failure_count = 0
        bdf_max_stage_failures = getattr(self, "bdf_max_stage_failures", 50)
        bdf_failure_reason = None

        try:
            while total_time < self.t_span:
                if bdf_max_steps is not None and bdf_step_count >= bdf_max_steps:
                    self._last_bdf_step_count = bdf_step_count
                    self._last_bdf_accepted_count = bdf_accepted_count
                    self._last_bdf_rejected_count = bdf_rejected_count
                    self._last_bdf_total_time = float(total_time)
                    raise RuntimeError(
                        f"BDF step limit exceeded: max_steps={bdf_max_steps}, "
                        f"accepted={bdf_accepted_count}, rejected={bdf_rejected_count}, "
                        f"t={total_time}, target={self.t_span}"
                    )
                bdf_step_count += 1
                if total_time + h > self.t_span:
                    h = self.t_span - total_time
                # Newton 迭代求解 F(y_new) = y_new - y - h * f(t_np1, y_new) = 0
                y_new = y.copy()
                
                nonlinear_before = bdf_iter_stats['nonlinear_total']
                try:
                    y_new , last_delta, res = single_step( y_new , y , h , last_delta)
                except _BDFStageFailure as exc:
                    bdf_rejected_count += 1
                    bdf_nonfinite_error_count += 1
                    bdf_stage_failure_count += 1
                    bdf_last_scaled_error = np.inf
                    last_delta = None
                    h = max(0.5 * float(h), float(h_min))
                    smw_stats = getattr(self, '_smw_stats', None)
                    if smw_stats is not None:
                        smw_stats['stage_fail'] = smw_stats.get('stage_fail', 0) + 1
                        smw_stats['last_stage_fail'] = str(exc)
                    if bdf_stage_failure_count >= bdf_max_stage_failures:
                        bdf_failure_reason = (
                            f"stage failure limit hit: {bdf_stage_failure_count}"
                        )
                        if smw_stats is not None:
                            smw_stats['stage_fail_limit_hit'] = True
                        break
                    if h <= float(h_min) * (1.0 + 1.0e-12):
                        bdf_failure_reason = (
                            f"stage failure at h_min={float(h_min):.6e}: {exc}"
                        )
                        break
                    continue
                nonlinear_this_step = bdf_iter_stats['nonlinear_total'] - nonlinear_before
                bdf_iter_stats['nonlinear_last'] = nonlinear_this_step
                bdf_iter_stats['nonlinear_max'] = max(
                    bdf_iter_stats['nonlinear_max'],
                    nonlinear_this_step,
                )
                
                tol_vector = atol + rtol * bm.abs(y)
                scaled_error = bm.max(bm.abs(res) / tol_vector)
                scaled_error_value = float(scaled_error)
                bdf_last_scaled_error = scaled_error_value
                bdf_last_h = float(h)
                bdf_min_h = min(bdf_min_h, float(h))

                if not np.isfinite(scaled_error_value):
                    bdf_rejected_count += 1
                    bdf_nonfinite_error_count += 1
                    h = max(0.5 * float(h), float(h_min))
                    if h <= float(h_min) * (1.0 + 1.0e-12):
                        bdf_failure_reason = (
                            f"non-finite scaled error at h_min={float(h_min):.6e}"
                        )
                        break
                    continue

                # 5. 步长自适应 (理论正确)
                if scaled_error <= 1.0:
                    # 接受步长
                    total_time += h
                    y = y_new
                    bdf_accepted_count += 1
                else:
                    bdf_rejected_count += 1
                    if h <= float(h_min) * (1.0 + 1.0e-12):
                        bdf_failure_reason = (
                            "scaled error remains too large at "
                            f"h_min={float(h_min):.6e}, "
                            f"scaled_error={scaled_error_value:.6e}"
                        )
                        break
                h = bm.clip((1.0 / scaled_error)**0.5 * h, h_min, h_max)
        finally:
            self._last_bdf_step_count = bdf_step_count
            self._last_bdf_accepted_count = bdf_accepted_count
            self._last_bdf_rejected_count = bdf_rejected_count
            self._last_bdf_total_time = float(total_time)
            self._last_bdf_h = bdf_last_h
            self._last_bdf_min_h = bdf_min_h
            self._last_bdf_scaled_error = bdf_last_scaled_error
            self._last_bdf_nonfinite_error_count = bdf_nonfinite_error_count
            self._last_bdf_stage_failure_count = bdf_stage_failure_count
            self._last_bdf_nonlinear_iteration_count = bdf_iter_stats['nonlinear_last']
            self._last_bdf_max_nonlinear_iteration_count = bdf_iter_stats['nonlinear_max']
            self._bdf_total_step_count = getattr(self, "_bdf_total_step_count", 0) + bdf_step_count
            self._bdf_total_accepted_count = getattr(self, "_bdf_total_accepted_count", 0) + bdf_accepted_count
            self._bdf_total_rejected_count = getattr(self, "_bdf_total_rejected_count", 0) + bdf_rejected_count
            self._bdf_total_nonfinite_error_count = getattr(self, "_bdf_total_nonfinite_error_count", 0) + bdf_nonfinite_error_count
            self._bdf_total_nonlinear_iteration_count = (
                getattr(self, "_bdf_total_nonlinear_iteration_count", 0)
                + bdf_iter_stats['nonlinear_total']
            )
            self._bdf_stage_count = getattr(self, "_bdf_stage_count", 0) + 1
            if local_mumps_cache and mumps_cache['ctx'] is not None:
                mumps_cache['ctx'].destroy()
                mumps_cache['ctx'] = None

        if total_time < float(self.t_span) * (1.0 - 1.0e-12):
            reason = bdf_failure_reason or "artificial time integration stopped early"
            raise RuntimeError(
                "BDF artificial time incomplete: "
                f"done={float(total_time):.6e}/{float(self.t_span):.6e}, "
                f"accepted={bdf_accepted_count}, rejected={bdf_rejected_count}, "
                f"last_h={bdf_last_h:.6e}, min_h={bdf_min_h:.6e}, "
                f"last_scaled_error={bdf_last_scaled_error:.6e}; {reason}"
            )

        Xi_new = y.reshape(GD, NN).T
        return Xi_new
    
    def preprocessor(self,fun_solver =None):
        """
        @brief preprocessor: linear transition initialization
        @param steps: fake time steps
        """
        pde = self.pde
        steps = self.pre_steps
        if fun_solver is None:
            if pde is None:
                self.uh = self.uh/steps
                for i in range(steps):
                    self.mesh_redistributor()
                    self.uh *= 1+1/(i+1)
            else:
                print("Use PDE initial solution for preprocessor.")
                for i in range(steps):
                    t = (i+1)/steps
                    self.uh = t * self.uh
                    self.mesh_redistributor()
                    self.uh = self.pspace.interpolate(pde.moving_init_solution)
        else:
            for i in range(steps):
                t = (i+1)/steps
                self.uh *= t
                self.mesh_redistributor()
                self.uh = fun_solver(self.mesh)
