from . import Monitor
from . import Interpolater
from .config import *
from scipy.integrate import solve_ivp
from fealpy.utils import timer
from .metric_shared import GeometricDiscreteCore
from scipy.sparse.linalg import LinearOperator,cg
from scipy.sparse import coo_matrix

class EAGAdaptiveFB(Monitor, Interpolater):
    def __init__(self, mesh, beta, space, config:Config):
        super().__init__(mesh, beta, space, config)
        self.config = config
        self.alpha = config.alpha
        self.tau = config.tau
        self.t_max = config.t_max
        self.maxit = config.maxit
        self.pre_steps = config.pre_steps
        self.gamma = config.gamma
        self.dt = self.tau * self.t_max
        
        self.geo_core = GeometricDiscreteCore(mesh)
        self.R = self.geo_core.R_matrix()

        self.cell2cell = self.mesh.cell_to_cell()
        self.total_steps = 10
        self.t_span = 0.1
        self.step = 10
        self.BD_projector()
        self._build_jac_pattern()
        self.tol = self._caculate_tol()
        
    def _prepare_ivp_cache(self, X, M,theta):
        """
        预计算在一个 solve_ivp 步进内不变的量，减少 jac/ode 回调重复开销。
        """
        E_K     = self.edge_matrix(X)           # (NC,d,d)
        E_K_inv = bm.linalg.inv(E_K)            # (NC,d,d)
        det_E_K = bm.linalg.det(E_K)            # (NC,)
        rho     = self.rho(M)                   # (NC,)
        gamma = self.gamma
        d = self.GD
        P_diag  = self.balance(self.M_node, theta,power = -gamma/4 ,mixed = False )    # (NN,)
        cache = {
            'E_K': E_K, 'E_K_inv': E_K_inv, 'det_E_K': det_E_K,
            'rho': rho, 'cm': self.cm, 'P_diag': P_diag,'theta': theta,
        }
        self._ivp_cache = cache
    
    def edge_matrix(self,X):
        return self.geo_core.edge_matrix(X)
    
    def A(self,E , E_hat , M_inv):
        return self.geo_core.A(E , E_hat , M_inv)
    
    def rho(self,M):
        return self.geo_core.rho(M)
    
    def theta(self, M):
        return self.geo_core.theta(M)
    
    def balance(self,M_node, theta, power=None , mixed=True):
        return self.geo_core.balance(M_node, theta, power, mixed)
        
    def I_func(self, theta ,A , rho):
        """
        I = rho * |A - theta * I|^{2 * gamma}
        Parameters:
            theta(float): 积分全局乘子
            A(Tensor): 雅可比算子 J^{-1}M^{-1}J^-T (NC, GD, GD)
            rho(Tensor): 权重函数 rho
        Return:
            I(float): I 函数值
        """
        gamma = self.gamma
        I  = rho * (bm.sum((A - theta * self.I_p)**2, axis=(-2, -1))) ** (gamma)
        I = bm.sum(self.cm * I)
        return I
    
    def TdA(self, A  , theta):
        """
        T = |A - theta * I|^{2 * gamma}
        TdA = 2 * gamma * |A - theta * I|^{2 * gamma - 2} * (A - theta * I)
        """
        gamma = self.gamma
        p = 2 * gamma
        tildeA = A - theta * self.I_p
        FN = bm.sum(tildeA**2, axis=(-2, -1))
        eps = 1e-15
        TdA = p * ((FN+eps)**(gamma-1))[..., None, None] * tildeA
        return TdA
    
    def Tdg(self,det_A):
        """
        T = |A - theta * I|^{2 * gamma}
        Tdg = 0
        """
        Tdg = bm.zeros_like(det_A, dtype=bm.float64)
        return Tdg

    def lam(self , theta):
        """
        拉伸因子 lambda
        """
        return 1.0

    def Idxi_from_Ehat(self,A ,g,E_hat, rho, theta):
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

        TdA = self.TdA(A , theta)
        Tdg = self.Tdg(g)
        
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
    
    def vector_construction(self, A , g , E_hat):
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

        Idxi = self.Idxi_from_Ehat(A , g ,E_hat, rho, theta)  # (NC, GD+1, GD)
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
    
    def JAC_functional(self,A, g, E_hat,M_inv, theta):
        d = self.GD
        NC = self.NC
        assert d == 2, "当前实现针对 2D；3D 可按相同张量结构扩展"
        E_K = self._ivp_cache['E_K']
        rho = self.rho(self.M)
        # 差分步长（相对尺度 + 绝对下限，数值稳健）
        local = self.Idxi_from_Ehat(A , g , E_hat , rho , theta)   # (NC, d+1, d)
        local = local   # 去权重后的局部向量
        
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
            local_pos_b = self.Idxi_from_Ehat(A_pos_b, g_pos_b, E_pos_all[b], rho, theta) 
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
    
    def mesh_redistributor(self , total_steps=None, h = None,
                           method='scipy',return_info = False, return_timemesh = False):
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
            h = self.t_span/self.step
        atol = 1e-5
        rtol = atol * 100
        
        I_h = []
        cm_min = []
        I_t = []
        time_mesh = [self.mesh.node]
        
        for it in range(total_steps):
            self.monitor()
            self.mol_method()
            M = self.M
            M_inv = bm.linalg.inv(M)
            X = self.mesh.node
            Xi = self.logic_mesh.node
            
            theta = self.theta(M)
            self._prepare_ivp_cache(X, M,theta)
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
            
            if method == 'scipy':
                def ode_system(t, y):
                    Xi_current = y.reshape(self.GD, self.NN).T
                    E_hat, A , g , trA = info_generator(Xi_current)
                    v = self.vector_construction(A , g , E_hat)
                    return v.ravel(order = 'F')
                
                def jac(t, y):
                    Xi_current = y.reshape(self.GD, self.NN).T
                    E_hat, A , g , trA = info_generator(Xi_current)
                    J_y = self.JAC_functional(A ,  g , E_hat, M_inv,theta)
                    return J_y
                
                t_span = [0,self.t_span]
                y0 = Xi.ravel(order = 'F')
                sol = solve_ivp(ode_system, t_span, y0, jac=jac, method='BDF',
                                            first_step=h,
                                            atol=atol, rtol=rtol)
                y_last = sol.y[:, -1]
                Xinew = y_last.reshape(self.GD, self.NN).T
            else:
                Xinew = self.integrater(Xi, M_inv, h, atol=atol, rtol=rtol,
                                        newton_tol=1e-6, newton_maxit=20,)
            
            Xnew = self.linear_interpolate(Xi, Xinew , X)
            
            if return_info:
                E_hat , A , g , trA = info_generator(Xinew)
                lam = self.lam(theta)
                I = self.I_func(theta, A, self._ivp_cache['rho'])
                I_h.append(I)
                cm_min.append(bm.min(self.cm).item())
            if return_timemesh:
                time_mesh.append(Xnew)
            
            error = bm.max(bm.linalg.norm(Xnew - self.node,axis=1))
            print(f"step {it+1}/{self.total_steps} , error: {error}")
            
            self.uh = self.interpolate(Xnew)
            self._construct(Xnew)
            # if error < self.tol:
            #     print(f"Converged at step {it+1} with error {error}")
            #     break

        ret = {"X": Xnew}
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
                A, g, trA, E_hat = info_update(y_new)
                f = self.vector_construction(A, g,E_hat).ravel(order='F')
 
                r = y_new - y - h * f
                res_norm = (r**2).sum()**0.5
                if res_norm < newton_tol:
                    break
                lin_tol = min(0.5, bm.sqrt(res_norm)) * min(1.0, cg_tol)
                # J = d f / d y 
                if nit == 0:
                    theta = self._ivp_cache['theta']
                    J = self.JAC_functional(A,  g,E_hat, M_inv, theta)
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