from . import Monitor
from . import Interpolater
from .config import *
from .tool import _compute_coef_2d, quad_equ_solver , linear_surfploter
from ..sparse import spdiags
from scipy.sparse import bmat,diags , block_diag,coo_matrix
from scipy.integrate import solve_ivp

class EAGAdaptiveHuang(Monitor, Interpolater):
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
        self.tol = self._caculate_tol()
        
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
    
    def GdJ(self,rho, J , M_inv):
        """
        计算 GdJ = dG/dJ
        Huang's paper Eq : d * p * theta * rho *tr(J M^{-1} J^T)^{(dp/2 -1)} M^{-1} J^{-1}
        Parameters
           rho: (NC, )
           J: (NC, GD, GD)
           M_inv: (NC, GD, GD)
        """
        d = self.GD
        p = self.gamma
        theta = self.theta
        J_T = bm.swapaxes(J , -1,-2)
        trace_term = bm.trace(J @ M_inv @ J_T, axis1=-2, axis2=-1)
        trace_term = bm.maximum(trace_term, 1e-14)    
        power_term = trace_term ** ( (d * p / 2) - 1)
        GdJ = d * p * theta * rho[...,None,None] * power_term[...,None,None] * M_inv @ J_T
        return GdJ
    
    def Gdr(self,rho , J):
        """
        计算 Gdr = dG/drho
        Huang's paper Eq : p*(1-theta) * d^{dp/2} * rho^{(1-p)} * det(J)^{p-1}
        Parameters
           rho: (NC, )
           J: (NC, GD, GD)
        """
        d = self.GD
        p = self.gamma
        theta = self.theta
        det_J = bm.linalg.det(J)     # (NC, )
        det_J_pos = bm.maximum(bm.abs(det_J), 1e-14)
        rho_pos = bm.maximum(bm.abs(rho), 1e-14)
        Gdr = p * (1 - 2*theta) * d**(d * p / 2) * rho_pos**(1 - p) * det_J_pos ** (p - 1)
        return Gdr
    
    def Idxi_from_Ehat(self, E_hat, E_K, rho, M_inv, E_K_inv, det_E_K):
        """
        """
        E_hat_inv = bm.linalg.inv(E_hat)               # (NC, GD, GD)
        det_E_hat = bm.linalg.det(E_hat)               # (NC,)

        J = self.Jacobian_matrix(E_K, E_hat)           # (NC, GD, GD)
        GdJ = self.GdJ(rho, J, M_inv)                  # (NC, GD, GD)
        Gdr = self.Gdr(rho, J)                         # (NC,)

        term0 = E_K_inv @ GdJ
        term1 = (Gdr * (det_E_hat / det_E_K))[..., None, None] * E_hat_inv
        Idxi_2 = term0 + term1                         # (NC, GD, GD)

        Idxi_1 = bm.einsum('cij->cj', -Idxi_2)         # (NC, GD)
        Idxi = bm.concat([Idxi_1[:, None, :], Idxi_2], axis=1)  # (NC, GD+1, GD)
        return Idxi
    
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

        Idxi = self.Idxi_from_Ehat(E_hat, E_K, rho, 
                                   M_inv,E_K_inv,det_E_K)  # (NC, GD+1, GD)
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
    
    def _prepare_ivp_cache(self, X, M, M_inv):
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
            'B': B, 'rho': rho, 'cm': self.cm, 'P_diag': P_diag,
            'rxx': self.rxx, 'ryy': self.ryy, 'rxy': self.rxy, 'ryx': self.ryx
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

        # 完成
        self._jac_pat_ready = True
    
    def JAC_functional(self,Xi,M_inv):
        """
        完全向量化的解析雅可比：无 d/c 循环，按图样一次装配 2NN×2NN。
        """
        d   = self.GD
        NN  = self.NN

        cache = self._ivp_cache
        if cache is not None:
            E_K     = cache['E_K'];     E_K_inv = cache['E_K_inv']
            det_E_K = cache['det_E_K']; B       = cache['B']
            rho     = cache['rho'];     cm      = cache['cm']
            P_diag  = cache['P_diag']
            rxx, ryy, rxy, ryx = cache['rxx'], cache['ryy'], cache['rxy'], cache['ryx']

        E_hat   = self.edge_matrix(Xi)               # (NC,d,d)
        Einv    = bm.linalg.inv(E_hat)               # (NC,d,d)
        det_Eh  = bm.linalg.det(E_hat)               # (NC,)
        J       = E_hat @ E_K_inv                    # (NC,d,d) 仅用已算 E_K_inv，避免重复 inv
        # 预量（与 GdJ/Gdr 定义严格一致）
        p      = self.gamma
        theta  = self.theta
        a      = d * p / 2.0
        c0     = d * p * theta
        detJ   = det_Eh / bm.maximum(det_E_K, 1e-14)                     # (NC,)
        C3     = p * (1.0 - 2.0*theta) * (d**(d*p/2)) \
                * (bm.maximum(rho,1e-14)**(1-p)) * (bm.maximum(detJ,1e-14)**p)  # (NC,)

        J_T    = bm.swapaxes(J, -1, -2)             # (NC,d,d)
        A0     = M_inv @ J_T                        # (NC,d,d)
        T0b    = E_K_inv @ A0                       # (NC,d,d)
        U      = J @ M_inv                          # (NC,d,d)
        S      = bm.sum(U * J, axis=(1,2))          # (NC,)
        S_a1   = bm.maximum(S, 1e-14) ** (a - 1.0)  # (NC,)
        S_a2   = bm.maximum(S, 1e-14) ** (a - 2.0)  # (NC,)
        Jinv   = E_K @ Einv                         # (NC,d,d)

        # 列空间辅助：B @ r；以及 Einv 的列/行视图
        b_all  = bm.swapaxes(Einv, -1, -2)          # (NC,d,k)  第 k 列（E_hat^{-1} e_k）

        # 预备装配常量
        cm_rep   = bm.repeat(cm, d+1)               # (NC*(d+1),)

        rc_x_tile = (-1.0 / self.tau) * P_diag[self._row_off_x_tile_d % NN]
        rc_y_tile = (-1.0 / self.tau) * P_diag[self._row_off_y_tile_d % NN]

        def pack(D2_comp):
            D1 = -bm.sum(D2_comp, axis=1, keepdims=True)
            V  = bm.concat([D1, D2_comp], axis=1)
            return V.reshape(-1)
        
        eye = bm.eye(d, **self.kwargs0) 

        # r_all: (NC,c,d)，c 轴就是 E_K_inv 的“行”（对应列扰动索引）
        r_all = E_K_inv                                    # (NC,d,d) → (NC,c,d)
        # 1 part1
        inner_all = bm.einsum('ncj,nkj->nck', r_all, U)    # (NC,c,k)
        deltaS_all = 2.0 * inner_all                       # (NC,c,k)
        scl1_all = (c0 * rho * (a - 1.0) * S_a2)[:, None, None] * deltaS_all  # (NC,c,k)
        part1_all = scl1_all[:, :, :, None, None] * T0b[:, None, None, :, :]  # (NC,c,k,d,d)

        # 2 part2：仅第 k 列非零（用单位阵把列嵌入）
        col_all = bm.einsum('nij,ncj->nci', B, r_all)      # (NC,c,d)
        s2 = (c0 * rho * S_a1)                             # (NC,)
        col2_all = col_all * s2[:, None, None]             # (NC,c,d)
        part2_all = col2_all[:, :, None, :, None] * eye[None, None, :, None, :]  # (NC,c,k,d,d)

        # 3 P1：dC3 对 k 的逐列
        t_all = bm.einsum('ncj,njk->nck', r_all, Jinv)     # (NC,c,k)
        dC3_all = (p * C3)[:, None, None] * t_all          # (NC,c,k)
        P1_all = dC3_all[:, :, :, None, None] * Einv[:, None, None, :, :]      # (NC,c,k,d,d)

        # 4 P2：d(E^-1) 的秩一外积
        row_c_Einv_all = Einv                               # (NC,c,d)
        P2_all = -(C3[:, None, None, None, None]) * (
            b_all[:, None, :, :, None] * row_c_Einv_all[:, :, None, None, :]
        )  # (NC,c,k,d,d)

        D2_all = part1_all + part2_all + P1_all + P2_all    # (NC,c,k,d,d)

        # 打包函数：把 (NC,c,d) 压成按 c 分段的 (c*NC*(d+1),)
        def pack_all(D2_comp_all):
            # D2_comp_all: (NC,c,d)
            D1 = -bm.sum(D2_comp_all, axis=2, keepdims=True)          # (NC,c,1)
            V  = bm.concat([D1, D2_comp_all], axis=2)                  # (NC,c,d+1)
            V  = bm.permute_dims(V, (1, 0, 2)).reshape(-1)             # (c*NC*(d+1),)
            return V

        # k=0 对 δx，k=1 对 δy；m=0/1 取 vx/vy 分量
        vx_seg_x_all = pack_all(D2_all[:, :, 0, :, 0])  # (d*NC*(d+1),)
        vy_seg_x_all = pack_all(D2_all[:, :, 0, :, 1])
        vx_seg_y_all = pack_all(D2_all[:, :, 1, :, 0])
        vy_seg_y_all = pack_all(D2_all[:, :, 1, :, 1])

        # 与稀疏图样对齐的行系数/投影系数（整段向量，无需切片）
        rr_x_all = self._row_off_x_tile_d % NN
        rr_y_all = self._row_off_y_tile_d % NN
        coef_x_all = rc_x_tile                              # (-1/tau)*P_diag[...]，形状已对齐 (d*NC*(d+1),)
        coef_y_all = rc_y_tile
        cm_rep_all = bm.tile(cm_rep, d)                     # (d*NC*(d+1),)

        # 直接生成四个 tile 段（替代 data**_list 拼接）
        data00_tile = coef_x_all * cm_rep_all * ( rxx[rr_x_all] * vx_seg_x_all + rxy[rr_x_all] * vy_seg_x_all )
        data10_tile = coef_y_all * cm_rep_all * ( ryx[rr_y_all] * vx_seg_x_all + ryy[rr_y_all] * vy_seg_x_all )
        data01_tile = coef_x_all * cm_rep_all * ( rxx[rr_x_all] * vx_seg_y_all + rxy[rr_x_all] * vy_seg_y_all )
        data11_tile = coef_y_all * cm_rep_all * ( ryx[rr_y_all] * vx_seg_y_all + ryy[rr_y_all] * vy_seg_y_all )

        # 处理 j=0 列（为 −sum_c）
        D2_0_x = -bm.sum(D2_all[:, :, 0, :, :], axis=1)     # (NC,d,d)
        D2_0_y = -bm.sum(D2_all[:, :, 1, :, :], axis=1)

        vx_0_x = pack(D2_0_x[:, :, 0]); vy_0_x = pack(D2_0_x[:, :, 1])
        vx_0_y = pack(D2_0_y[:, :, 0]); vy_0_y = pack(D2_0_y[:, :, 1])

        rr_x0 = self._row_off_x_0 % NN
        rr_y0 = self._row_off_y_0 % NN
        coef_x0 = (-1.0 / self.tau) * P_diag[rr_x0]
        coef_y0 = (-1.0 / self.tau) * P_diag[rr_y0]

        data00_0 = coef_x0 * cm_rep * ( rxx[rr_x0] * vx_0_x + rxy[rr_x0] * vy_0_x )
        data10_0 = coef_y0 * cm_rep * ( ryx[rr_y0] * vx_0_x + ryy[rr_y0] * vy_0_x )
        data01_0 = coef_x0 * cm_rep * ( rxx[rr_x0] * vx_0_y + rxy[rr_x0] * vy_0_y )
        data11_0 = coef_y0 * cm_rep * ( ryx[rr_y0] * vx_0_y + ryy[rr_y0] * vy_0_y )

        # 拼装稀疏矩阵（一次性 COO -> CSR）
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

        current_i, current_j = self._tri_interpolate_batch(i, j, Xnew,X, 
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
            current_i, current_j = self._tri_interpolate_batch(unique_i, unique_j,
                                                    Xnew,X, interpolated,Xi,Xi_new)
        if iteration_count >= max_iterations:
            print(f"Warning: Maximum iterations reached ({max_iterations}) without full interpolation.")

        return Xnew

    def _tri_interpolate_batch(self,nodes, cells,Xnew,X, interpolated,Xi,Xi_new):
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
    
    def balance(self,M_node):
        """
        计算平衡因子 P_diag
        p_i = det(M_i)^{n} , n = (p-1)/2
        Parameters
           M_node: (NN, GD, GD)
        """
        p = self.gamma
        det_M_node = bm.linalg.det(M_node)
        n = (p - 1) / 2
        P_diag =  det_M_node**n # (NN,)
        return P_diag
    
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
        return A, C
    
    # 这段代码被注释掉了，因为并不适合当前算法
    # def get_physical_node(self,Xinew,X,vector_field):
    #     """
    #     计算物理网格的新节点位置
    #     x_{n+1} = x_n + eta * J * vector_field
    #     J = E_K E_hat_K^{-1} 为局部雅可比矩阵将逻辑网格的位移场拉回物理网格
    #     注意上述得到的位移需要在边界处进行修正,以保持边界形状
    #     eta 步长控制,其为了了防止网格翻转,一般取 eta in (0,1]
         
    #     Parameters:
    #         vector_field(Tensor): 逻辑网格节点的速度场 (NN, GD)
    #     Return:
    #         x_new(Tensor): 物理网格的新节点位置 (NN, GD)
    #     """
    #     cell = self.cell
    #     alpha = self.alpha
    #     E = self.edge_matrix(X) # (NC, GD, GD)
    #     Xinew_0 = Xinew[cell[:,0],:] # (NC, GD)
    #     E_map = Xinew[cell[:,1:],:] - Xinew_0[:, None , :] # (NC, GD, GD)
    #     E_map = bm.permute_dims(E_map, (0,2,1))  # (NC, GD, GD)
    #     J = E @ bm.linalg.inv(E_map) # (NC, GD, GD)
    #     vf_cell = bm.mean(vector_field[self.mesh.cell], axis=1) # (NC, GD)
    #     vf_physical_cell = bm.einsum('ijk,ik->ij', J , vf_cell) # (NC, GD)
        
    #     sm = self.sm
    #     cm = self.cm
    #     vf_physical = bm.zeros_like(vector_field , **self.kwargs0) # (NN, GD)
    #     vf_physical = bm.index_add(vf_physical , self.mesh.cell , (vf_physical_cell*cm[:, None])[:, None , :])
    #     vf_physical /= sm[:, None]
        
    #     Bdinnernode_idx = self.Bdinnernode_idx
    #     dot = bm.sum(self.Bi_Pnode_normal * vf_physical[Bdinnernode_idx],axis=1)
    #     vf_physical = bm.set_at(vf_physical , Bdinnernode_idx ,
    #                             vf_physical[Bdinnernode_idx] - dot[:,None] * self.Bi_Pnode_normal)
    #     vf_physical = bm.set_at(vf_physical , self.Vertices_idx , 0)
        
    #     coef = _compute_coef_2d(vf_physical,self.AC_generator)
    #     k = quad_equ_solver(coef)
    #     positive_k = bm.where(k>0, k, 1)
    #     eta = bm.min(positive_k)
    #     Xnew = self.mesh.node + alpha * eta * vf_physical

    #     return Xnew

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
        return d*0.01/p
    
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
            self.monitor()
            self.mol_method()
            atol = 1e-6
            rtol = atol * 100
            M = self.M
            M_inv = bm.linalg.inv(M)
            X = self.mesh.node
            Xi = self.logic_mesh.node
            
            self._prepare_ivp_cache(X, M, M_inv)
            P_diag = self._ivp_cache['P_diag']
            eps = 1e-12
            s_node = bm.maximum(P_diag, eps)                     # (NN,)
            s_vec  = bm.concat([s_node, s_node], axis=0)         # (2NN,)
            s_inv  = 1.0 / s_vec

            def ode_system(t, y):
                y = (s_vec * y)
                Xi_current = y.reshape(self.GD, self.NN).T
                v = self.vector_construction(Xi_current , M_inv)
                return s_inv *v.ravel(order = 'F')
            
            def jac(t, y):
                y = (s_vec * y)
                Xi_current = y.reshape(self.GD, self.NN).T
                J_y = self.JAC_functional(Xi_current, M_inv)
                J_z = J_y.multiply(s_inv[:, None])             # 行缩放
                J_z = J_z.multiply(s_vec[None, :])
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
            Xnew = self.linear_interpolate(Xi , Xinew , X)
            self.uh = self.interpolate(Xnew)
            self._construct(Xnew)
            print(f"EAGAdaptiveHuang: step {it+1}/{self.total_steps} completed.")

        return Xnew
        