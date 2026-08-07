from . import Monitor
from . import Interpolater
from .config import *
from fealpy.utils import timer
from .metric_shared import GeometricDiscreteCore
from scipy.sparse.linalg import LinearOperator,cg
from scipy.sparse import coo_matrix
import numpy as np
from .metrictensoradaptive import (
    _mta_A_from_C_kernel,
    _mta_c_kernel,
    _mta_jac_values_kernel,
    _mta_vector_assembly_kernel,
    njit,
)


if njit is not None:
    @njit(cache=True)
    def _eag_huang_idxi_kernel(A, g, trA, E_hat, rho, gamma, mu):
        nc = A.shape[0]
        local = np.empty((nc, 3, 2), dtype=np.float64)
        p = gamma
        dg_const = (2.0 ** p) * (1.0 - 2.0 * mu) * (gamma * 0.5)
        det_floor = 1.0e-14
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

            g_eff = g[n] if g[n] > det_floor else det_floor
            tr_eff = trA[n] if trA[n] > det_floor else det_floor
            td = mu * gamma * (tr_eff ** (p - 1.0))
            tgg = dg_const * (g_eff ** (gamma * 0.5))
            b00 = td * (inv00 * A[n, 0, 0] + inv01 * A[n, 1, 0]) + tgg * inv00
            b01 = td * (inv00 * A[n, 0, 1] + inv01 * A[n, 1, 1]) + tgg * inv01
            b10 = td * (inv10 * A[n, 0, 0] + inv11 * A[n, 1, 0]) + tgg * inv10
            b11 = td * (inv10 * A[n, 0, 1] + inv11 * A[n, 1, 1]) + tgg * inv11
            scale = 2.0 * rho[n]
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
else:
    _eag_huang_idxi_kernel = None

class EAGAdaptiveHuang(Monitor, Interpolater):
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
        gamma = self.gamma
        d = self.GD
        P_diag  = self.balance(self.M_node, theta,power=d*(gamma-1)/2 ,mixed=False )    # (NN,)
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

    def _det_floor(self):
        return float(getattr(self.config, "huang_det_floor", 1.0e-14))

    def _positive_det(self, g):
        return bm.maximum(g, self._det_floor())
    
    def I_func(self,trA , rho , g):
        """
        I = rho * (mu * trA^{d*gamma/2} + d^{d*gamma/2}* (1-2 * mu) * g^{gamma/2} )
        """
        d = self.GD
        gamma = self.gamma
        mu = 1/3
        g_eff = self._positive_det(g)
        trA_eff = bm.maximum(trA, self._det_floor())
        I  = rho * ( mu * trA_eff**(d*gamma/2) + d**(d*gamma/2) * (1 - 2 * mu) * g_eff**(gamma/2) )
        I = bm.sum(self.cm * I)
        return I
    
    def TdA(self, trA):
        """
        TdA =  mu * (d*gamma/2) * trA^{d*gamma/2 - 1}
        """
        d = self.GD
        gamma = self.gamma
        mu = 1/3
        trA_eff = bm.maximum(trA, self._det_floor())
        TdA = ( mu * (d * gamma / 2) * trA_eff**(d * gamma / 2 - 1))[..., None, None] *self.I_p
        return TdA
    
    def Tdg(self , g):
        """
        Tdg = rho * d^{d*gamma/2} * (1-2 * mu) * (gamma/2) * g^{gamma/2 - 1}
        """
        d = self.GD
        gamma = self.gamma
        mu = 1/3
        g_eff = self._positive_det(g)
        Tdg = d**(d * gamma / 2) * (1 - 2 * mu) * (gamma / 2) * g_eff**(gamma / 2 - 1)
        return Tdg

    def Tdg_times_g(self, g):
        d = self.GD
        gamma = self.gamma
        mu = 1/3
        g_eff = self._positive_det(g)
        return d**(d * gamma / 2) * (1 - 2 * mu) * (gamma / 2) * g_eff**(gamma / 2)
    
    def lam(self , theta):
        """
        拉伸因子 lambda
        """
        return 1.0

    def Idxi_from_Ehat(self,A ,g,trA, E_hat, rho, theta):
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
        if self.GD == 2 and _eag_huang_idxi_kernel is not None:
            return _eag_huang_idxi_kernel(
                np.asarray(A, dtype=np.float64),
                np.asarray(g, dtype=np.float64),
                np.asarray(trA, dtype=np.float64),
                np.asarray(E_hat, dtype=np.float64),
                np.asarray(rho, dtype=np.float64),
                float(self.gamma),
                float(getattr(self, "_woven_mu", 1.0 / 3.0)),
            )
        E_hat_inv = bm.linalg.inv(E_hat)
        TdA = self.TdA(trA )
        
        term0 = E_hat_inv @ A @ TdA
        term1 = self.Tdg_times_g(g)[..., None, None] * E_hat_inv

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

        Idxi = self.Idxi_from_Ehat(A , g ,trA , E_hat, rho, theta)  # (NC, GD+1, GD)
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
    
    def JAC_functional(self,A, g, trA, E_hat,M_inv, theta, local=None):
        d = self.GD
        NC = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同张量结构扩展"
        E_K = self._ivp_cache['E_K']
        rho = self._ivp_cache['rho']
        # 差分步长（相对尺度 + 绝对下限，数值稳健）
        if local is None:
            local = self.Idxi_from_Ehat(A , g ,trA, E_hat , rho , theta)   # (NC, d+1, d)
        
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
            trA_pos_b = bm.trace(A_pos_b, axis1=-2, axis2=-1)
            local_pos_b = self.Idxi_from_Ehat(A_pos_b, g_pos_b, trA_pos_b, E_pos_all[b], rho, theta) 
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
        cm = self.cm
        rxx, ryy, rxy, ryx = self.rxx, self.ryy, self.rxy, self.ryx
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

        # 一次性装配（COO -> CSR）
        V = bm.concat([
            data00_tile, data10_tile, data00_0, data10_0,
            data01_tile, data11_tile, data01_0, data11_0
        ], axis=0)
        
        JAC = coo_matrix((V, (self.I, self.J)),shape=(2*NN, 2*NN)).tocsr()
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

    def sundials_integrater(self, Xi, M_inv, h, atol=1e-6, rtol=1e-4):
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
        theta = self._ivp_cache['theta']
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
            value = np.asarray(v.ravel(order='F'), dtype=np.float64)
            rhs_cache['y'] = np.asarray(y, dtype=np.float64).copy()
            rhs_cache['value'] = value.copy()
            local_cache['y'] = rhs_cache['y'].copy()
            local_cache['value'] = local
            yp[:] = value

        def jacfn(t, y, yp, JJ):
            A, g, trA, E_hat = info_update(y)
            local = local_cache['value'] if same_state(local_cache, y) else None
            J = self.JAC_functional(A, g, trA, E_hat, M_inv, theta, local=local).tocsc()
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
    
    def mesh_redistributor(self , total_steps=None, h = None,
                           method='BDF_LBFGS',return_info = False, return_timemesh = False):
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
            
            if method == 'BDF_SMW':
                Xinew = self.integrater(Xi, M_inv, h, atol=atol, rtol=rtol,
                                        newton_tol=1e-6, newton_maxit=20,
                                        solver='smw')
            elif method in ('SUNDIALS', 'CVODE', 'SUNDIALS_CVODE'):
                Xinew = self.sundials_integrater(Xi, M_inv, h, atol=atol, rtol=rtol)
            else:
                Xinew = self.integrater(Xi, M_inv, h, atol=atol, rtol=rtol,
                                        newton_tol=1e-6, newton_maxit=20,)
            
            Xnew = self.linear_interpolate(Xi, Xinew , X)
            
            if return_info:
                E_hat , A , g , trA = info_generator(Xinew)
                lam = self.lam(theta)
                I = self.I_func(trA, self._ivp_cache['rho'],g)
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
    
    def integrater(self, Xi, M_inv, h, atol=1e-6, rtol=1e-4,
                                      newton_tol=1e-6, newton_maxit=20, 
                                      cg_tol=1e-8,
                                      h_min=None, h_max=None,
                                      solver='lfp'):
        """
        用隐式Euler(BDF1) + 拟牛顿 + cg 做时间积分
        """
        if solver == 'smw':
            from .metrictensoradaptive import MetricTensorAdaptive
            eag_jac = self.JAC_functional

            def jac_adapter(A, trA, g, E_hat, M_inv, local=None):
                theta = self._ivp_cache['theta']
                return eag_jac(A, g, trA, E_hat, M_inv, theta, local=local)

            self.JAC_functional = jac_adapter
            try:
                return MetricTensorAdaptive.integrater(
                    self, Xi, M_inv, h, atol=atol, rtol=rtol,
                    newton_tol=newton_tol, newton_maxit=newton_maxit,
                    cg_tol=cg_tol, h_min=h_min, h_max=h_max,
                    solver='smw'
                )
            finally:
                self.JAC_functional = eag_jac

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
                    theta = self._ivp_cache['theta']
                    J = self.JAC_functional(A,  g, trA, E_hat, M_inv, theta)
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
                y_new , last_delta, res = single_step( y_new , y , h , last_delta)
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
                    continue

                # 5. 步长自适应 (理论正确)
                if scaled_error <= 1.0:
                    # 接受步长
                    total_time += h
                    y = y_new
                    bdf_accepted_count += 1
                else:
                    bdf_rejected_count += 1
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

        Xi_new = y.reshape(GD, NN).T
        return Xi_new
        
