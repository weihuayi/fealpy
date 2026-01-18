from . import Monitor
from . import Interpolater
from .config import *
from ..sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.integrate import solve_ivp
from fealpy.utils import timer
time = timer()
next(time)


class EAGAdaptiveXHuang(Monitor, Interpolater):
    def __init__(self, mesh, beta, space, config:Config):
        super().__init__(mesh, beta, space, config)
        self.config = config
        self.alpha = config.alpha
        self.tau = config.tau
        self.t_max = config.t_max
        self.dt = self.tau * self.t_max
        self.tol = config.tol
        self.maxit = config.maxit
        self.pre_steps = config.pre_steps
        self.gamma = config.gamma
        self.theta = 1/3
        
        self.total_steps = 10
        self.t_span = 0.1
        self.step = 10
        self.cell2cell = self.mesh.cell_to_cell()
        self.BD_projector()
        self._last_loc_cells = None
        self._build_jac_pattern()
                
    def edge_matrix(self,X):
        """
        边矩阵 E
        E = [x_1 - x_0, x_2 - x_0, ..., x_d - x_0]
        Parameters:
            X: (NN, GD)
        Returns:
            E: (NC, GD, GD)
        """
        cell = self.cell
        X0 = X[cell[:, 0], :]
        E = X[cell[:, 1:], :] - X0[:, None, :] # (NC, GD, GD)
        E = bm.permute_dims(E, (0,2,1))  # (NC, GD, GD)
        return E
        
    def Jacobian_matrix(self,E_k,  E_hat):
        """
        雅可比矩阵 J = (F')^{-1} = E_hat * E_k^{-1}
        """
        E_k_inv = bm.linalg.inv(E_k)  # (NC, GD, GD)
        J = E_hat @ E_k_inv  # (NC, GD, GD)
        return J
    
    def rho(self,M):
        """
        计算 rho = det(M)^{1/2}
        Parameters
           M: (NC, GD, GD)
        """
        det_M = bm.linalg.det(M)
        rho = det_M ** (0.5)
        return rho
    
    def G(self):
        """
        计算 Huang's paper 中的能量密度函数 G
        G = theta * rho * tr(J M^{-1} J^T)^{dp/2} + (1 - 2*theta) * d^{dp/2} * rho^{(1-p)} * det(J)^{p}
        Parameters
           rho: (NC, )
           J: (NC, GD, GD)
           M_inv: (NC, GD, GD)
        """
        d = self.GD
        p = self.gamma
        theta = self .theta
        term1 = theta * self.rho_c * (self.trace_term ** (d * p / 2))
        term2 = (1.0 - 2.0*theta) * (d**(d * p / 2)) * self.rho_c**(1 - p) * self.det_J**(p)
        G = term1 + term2
        return G
    
    def Idx_from_EK(self, E_hat, E_K, rho, M_inv, E_K_inv):
        """
        1:d
        v = G * E_K^{-1} - E_K^{-1} GdJ E_hat E_K^{-1} - (Gdr * det(E_hat)/det(E_K)) * E_K^{-1}
            + 1/(d+1) * sum_{j=0}^{d}[ GdM @ M_{j,k}]
        """
        d = self.GD
        cell = self.cell
        theta = self.theta
        M_node = self.M_node
        p = self.gamma
        J = self.Jacobian_matrix(E_K, E_hat)
        J_T = bm.swapaxes(J , -1,-2)
        trace_term = bm.trace(J @ M_inv @ J_T, axis1=-2, axis2=-1)
        trace_term = bm.maximum(trace_term, 1e-14)
        det_J = bm.linalg.det(J)     # (NC, )
        det_J = bm.maximum(det_J, 1e-14)
        
        G_term0 = theta * rho * (trace_term ** (d * p / 2))
        G_term1 = (1.0 - 2.0*theta) * (d**(d * p / 2)) * (rho**(1 - p)) * (det_J**(p))
        G = G_term0 + G_term1
        
        power_term = trace_term ** ( (d * p / 2) - 1)
        GdJ = d * p * theta * rho[...,None,None] * power_term[...,None,None] * M_inv @ J_T
        Gdr = p * (1 - 2*theta) * d**(d * p / 2) * (rho**(1 - p)) * (det_J ** (p - 1))
        
        GdM_term1 = -0.5 * d * p * theta * rho[...,None,None] * power_term[...,None,None] * (M_inv@ J_T @ J  @ M_inv)
        GdM_term2 = 0.5 * theta * rho[...,None,None] * (trace_term ** (d * p / 2))[...,None,None] * M_inv
        GdM_term3 = 0.5 * (1 - 2*theta) * (1 - p) * d**(d * p / 2) * (rho**(1 - p))[...,None,None] *(det_J**p)[...,None,None] * M_inv
        GdM = GdM_term1 + GdM_term2 + GdM_term3
        
        Idx_term0 = - E_K_inv @ GdJ @ E_hat @ E_K_inv
        Idx_term1 = (G - Gdr * det_J)[:, None, None] * E_K_inv
        Idx_2 = Idx_term0 + Idx_term1                         # (NC, GD, GD)
        
        Idx_1 = bm.einsum('cij->cj', -Idx_2)         # (NC, GD)
        Idx_part0 = bm.concat([Idx_1[:, None, :], Idx_2], axis=1)  # (NC, GD+1, GD)
        
        M_cell_term = M_node[cell[:, 1:]] - M_node[cell[:,0]][:,None,...]  # (NC,2, GD, GD)
        tr_term = bm.einsum('cij,clij->cl', GdM, M_cell_term)  # (NC,2)
        tr_Einv = bm.einsum('clj,cl->cj', E_K_inv, tr_term)/(d + 1)  # (NC, GD)
        Idx_part1 = tr_Einv[:, None, :]  # (NC,1, GD)
        Idx = Idx_part0 + Idx_part1  # (NC, GD+1, GD)
    
        return Idx
    
    def vector_construction(self,E_hat ,E_K , rho, M_inv, E_K_inv , P_diag):
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
        cache = self._ivp_cache
        if cache is not None:
            E_hat     = cache['E_hat']
        Idx = self.Idx_from_EK(E_hat, E_K, rho, M_inv, E_K_inv)  # (NC, GD+1, GD)
        cell = self.cell
        cm = self.cm # 此处的面积可能是逻辑网格的固定面积？
        global_vector = bm.zeros((self.NN, self.GD), dtype=bm.float64)
        global_vector = bm.index_add(global_vector , cell , cm[:,None,None] * Idx)
        
        tau = self.tau
        v = -1/tau * global_vector * P_diag[:, None]  # (NN, GD)
        
        # 边界投影和角点固定
        Bi_Pnode_normal = self.Bi_Pnode_normal
        Bdinnernode_idx = self.Bdinnernode_idx
        dot = bm.sum(Bi_Pnode_normal * v[Bdinnernode_idx],axis=1)
        v = bm.set_at(v , Bdinnernode_idx ,
                                v[Bdinnernode_idx] - dot[:,None] * Bi_Pnode_normal)
        vertice_idx = self.Vertices_idx
        v = bm.set_at(v , vertice_idx , 0)
        return v
    
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
        
    def JAC_functional(self,E_K, rho, M_inv, E_K_inv, P_diag):
        d = self.GD
        NC = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同张量结构扩展"
        E_hat = self._ivp_cache['E_hat']                # (NC, d, d)
        # E_K_inv = bm.linalg.inv(E_K)                    # (NC, d, d)

        cm  = self.cm                                   # (NC,)
        local = self.Idx_from_EK(E_hat, E_K, rho, M_inv, E_K_inv)   # (NC, d+1, d)
        local_scale = cm[:, None, None] * local                     # (NC, d+1, d)

        B = d * d
        k_idx, c_idx = bm.meshgrid(bm.arange(d), bm.arange(d), indexing='ij')
        k_idx = k_idx.reshape(-1)   # (B,)
        c_idx = c_idx.reshape(-1)   # (B,)

        A = bm.permute_dims(E_K, (2, 1, 0))            # (d, d, NC)  -> (c, k, NC)
        a_all = A.reshape(B, NC)                        # (B, NC)

        eps = bm.finfo(E_K.dtype).eps
        h_mag = (a_all + bm.maximum(bm.abs(a_all), 1.0) * bm.sqrt(eps)) - a_all   # (B, NC)
        sgn   = bm.where(a_all >= 0, 1.0, -1.0)
        h_entry = sgn * bm.abs(h_mag)                 # (B, NC)

        # dE_all: (B, NC, d, d)，仅 (k,c) 处为 h_entry，其余为 0
        basis = bm.zeros((B, d, d), **self.kwargs0)   # (B, d, d) one-hot
        basis = bm.set_at(basis, (bm.arange(B), k_idx, c_idx), 1.0)
        dE_all = h_entry[:, :, None, None] * basis[:, None, :, :]     # (B, NC, d, d)

        # 批量扰动
        E_pos_all     = E_K[None, ...] + dE_all                         # (B, NC, d, d)
        E_pos_inv_all = bm.linalg.inv(E_pos_all)                        # (B, NC, d, d)
        cm_pos_all    = 0.5 * bm.abs(bm.linalg.det(E_pos_all))          # (B, NC)

        # Idx_from_EK 构造局部坐标 (B, NC, d+1, d)
        local_pos_list = []
        for b in range(B):
            local_pos_b = self.Idx_from_EK(E_hat, E_pos_all[b], rho, M_inv, E_pos_inv_all[b])  
            local_pos_list.append(local_pos_b)
        local_pos_all = bm.stack(local_pos_list, axis=0)                

        local_pos_scale_all = cm_pos_all[:, :, None, None] * local_pos_all  

        # 差分：一次性计算所有 (B, NC, d+1, d)，再写入 D2_all
        diff_all = (local_pos_scale_all - local_scale[None, ...]) / (h_entry[:, :, None, None]) 

        # 只保留 j=1..d 列 (NC, c, k, j, m) 写入
        D2_all = bm.zeros((NC, d, d, d, d), **self.kwargs0)
        diff_jm = diff_all[:, :, 1:, :]   # (B, NC, d, d) -> (B, NC, j, m)

        for b in range(B):
            c = c_idx[b]
            k = k_idx[b]
            D2_all = bm.set_at(D2_all, (slice(None), c, k, slice(None), slice(None)), diff_jm[b])

        JAC = self.JAC_assembly(D2_all, P_diag)
        return JAC
    
    def JAC_assembly(self,D2_all,P_diag):
        """
        依据局部二阶块 D2_all 装配全局雅可比
        """
        d   = self.GD
        NN  = self.NN
        NC  = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同结构扩展"
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
        rr_x_all  = self._row_off_x_tile_d % NN
        rr_y_all  = self._row_off_y_tile_d % NN
        rc_x_tile = (-1.0 / self.tau) * P_diag[rr_x_all]                    # (d*NC*(d+1),)
        rc_y_tile = (-1.0 / self.tau) * P_diag[rr_y_all]                                         # (d*NC*(d+1),)

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

        V = bm.concat([
            data00_tile, data10_tile, data00_0, data10_0,
            data01_tile, data11_tile, data01_0, data11_0
        ], axis=0)

        JAC = coo_matrix((V.astype(float), (self.I , self.J)),
                        shape=(2*NN, 2*NN)).tocsr()
        return JAC
    
    def balance(self,M_node):
        """
        计算平衡因子 P_diag
        p_i = det(M_i)^{n} , n = (p-1)/2
        Parameters
           M_node: (NN, GD, GD)
        """
        p = self.gamma
        det_M_node = bm.linalg.det(M_node)
        n = -(p - 1) /2
        P_diag =  det_M_node**n # (NN,)
        return P_diag
    
    def _construct(self,moved_node:TensorLike):
        """
        @brief construct information for the harmap method before the next iteration
        """
        self.mesh.node = moved_node
        self.node = moved_node
        self.cm = self.mesh.entity_measure('cell')
        self.sm = bm.zeros(self.NN, **self.kwargs0)
        self.sm = bm.index_add(self.sm , self.mesh.cell , self.cm[:, None])
    
    def mesh_redistributor(self):
        """
        Huang 等人的等分布与对齐自适应网格算法
        """
        for it in range(self.total_steps):
            atol = 1e-5
            rtol = atol * 100

            X = self.mesh.node
            Xi = self.logic_mesh.node
            
            self._prepare_ivp_cache(Xi)
            cache = {'y': None}
            abs_eps, rel_eps = 1e-8, 1e-5
            def same_y(y, y0):
                if y0 is None: 
                    return False
                return bm.all(bm.abs(y - y0) <= rel_eps*bm.abs(y0) + abs_eps)
            
            def ensure_state(y):
                if same_y(y, cache['y']): 
                    return
                Xc = y.reshape(self.GD, self.NN).T
                self._construct(Xc)
                self.uh = self.interpolate(Xc)
                self.monitor()
                self.mol_method()
                M = self.M; M_inv = bm.linalg.inv(M); M_node = self.M_node
                P_diag = self.balance(M_node)
                rho = self.rho(M)
                E_K = self.edge_matrix(Xc)
                E_K_inv = bm.linalg.inv(E_K)
  
                cache.update({'y': y.copy(),'M_inv': M_inv, 'P_diag': P_diag,
                              'E_K': E_K , 'E_K_inv': E_K_inv , 'rho': rho},)
                
            def ode_system(t, y):
                ensure_state(y)
                E_hat = self._ivp_cache['E_hat']
                v = self.vector_construction(E_hat, cache['E_K'], 
                                             cache['rho'], cache['M_inv'],
                                             cache['E_K_inv'], cache['P_diag'])
                time.send("ODE vector field")
                return v.ravel(order = 'F')
            
            def jac(t, y):
                ensure_state(y)
                J_y = self.JAC_functional(cache['E_K'], cache['rho'], 
                                          cache['M_inv'], cache['E_K_inv'], cache['P_diag'])       
                return J_y
            
            # ODE 求解器需要的初值
            t_span = [0,self.t_span]
            y0 = X.ravel(order = 'F')
            sol = solve_ivp(ode_system, t_span, y0, jac=jac, method='BDF',
                                        jac_sparsity=self._jac_sparsity,
                                        first_step=self.t_span/self.step,
                                        atol=atol, rtol=rtol)
            y_last = sol.y[:, -1]
            Xnew = y_last.reshape(self.GD, self.NN).T
            
            self._construct(Xnew)
  
            print(f"EAGAdaptiveXHuang: step {it+1}/{self.total_steps} completed.")
        time.send("Mesh redistribution complete")
        next(time)
        return Xnew