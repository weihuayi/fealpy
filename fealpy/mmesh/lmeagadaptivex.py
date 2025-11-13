from . import Monitor
from . import Interpolater
from .config import *
from .tool import _compute_coef_2d, quad_equ_solver , linear_surfploter

class LMEAGAdaptiveX(Monitor, Interpolater):
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
        self.tol = self._caculate_tol()
        
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
        theta = (sigma/NC)**(-2/d)
        return theta
        
    def T(self, H):
        """
        (...)^gamma 型
        T = (1/2 * |A|^2_F + theta*(tr(A) - det(A)^{1/d}))^gamma
        
        Parameters:
            theta(float): 积分全局乘子
            A(Tensor): 雅可比算子
        Return:
            T(float): T 函数值
        """
        T = H**self.gamma
        return T

    def H(self,theta, A):
        d = self.GD
        trace_A = bm.trace(A,axis1=-2,axis2=-1)
        det_A = bm.linalg.det(A)
        A_Fnorm = bm.sum(A**2 , axis=(-2,-1))
        print("A_Fnorm:", bm.max(0.5*A_Fnorm))
        print("H:",bm.max((trace_A - d * det_A**(1/d))))
        H = 0.5 * A_Fnorm + theta * (trace_A - d * det_A**(1/d))
        return H
        
    def T_dif_A(self,theta,A):
        """
        (...)^gamma 型
        基于几何离散的 T 泛函关于算子 A 的导数
        T = (1/2 * |A|^2_F + theta*(tr(A) - det(A)^{1/d}))^r
        A = E_hat E_K^{-1}M^{-1} E_K^{-T} E_hat^T
        
        pT/pA = gamma * T^{gamma-1} *(A + theta I)
        """
        gamma = self.gamma
        H = self.H(theta, A)
        d = A.shape[-1]
        I_base = bm.eye(d, **self.kwargs0)
        I = bm.zeros_like(A , **self.kwargs0)
        I += I_base
        pT_pA = (gamma * H**(gamma-1))[...,None,None] * (A + theta * I)
        return pT_pA
    
    def T_dif_g(self, theta,A ,det_A):
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
    
    def mu(self,M):
        """
        权重函数 mu , 一般形式
        mu = (det(M))^{1/2}
        """
        mu = bm.sqrt(bm.linalg.det(M))
        return mu
    
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
    
    def balance(self,M_node,theta):
        """
        (...)^gamma 型
        平衡函数 P 为 (NN,NN) 的对角矩阵,实际组装时只需对角元
        P = diag( m * det(M)^{n})
        m = 2^gamma / (2^gamma + d^gamma*theta^{2*gamma})
        n = 1/d * (2 * gamma^2 - 2 * gamma + 3/2)
        Parameters:
            M_node(Tensor): 目标单元度量张量 (NN, GD, GD)
            theta(float): 积分全局乘子
        """
        d = self.GD
        gamma = self.gamma
        det_M_node = bm.linalg.det(M_node)
        # m = 2**gamma / (2**gamma + d**gamma * theta**(gamma))
        n = 1/d * (2 * gamma - d/2)
        P_diag =  det_M_node**n # (NN,)
        return P_diag
    
    def edge_matrix(self,mesh):
        """
        边矩阵 E
        E = [x_1 - x_0, x_2 - x_0, ..., x_d - x_0]
        """
        cell = mesh.cell
        X = mesh.entity('node')
        X0 = X[cell[:, 0], :]
        E = X[cell[:, 1:], :] - X0[:, None, :] # (NC, GD, GD)
        return E
    
    def G_dif_M(self ,mu, T ,M, M_inv, E , E_hat , TdA , Tdg , g):
        """
        基于几何离散的 G 泛函关于度量张量 M 的导数
        G = mu * T(A(M))
        pG/pM = mu* (0.5 * T * M^{-1} - M^{-1}J^{-T} TdA J^{-1} M^{-1} - Tdg * g * M)
        J^{-1} = E_hat E_K^{-1}
        """ 
        d = self.GD
        E_inv = bm.linalg.inv(E)
        J_inv = E_hat @ E_inv
        J_inv_T = bm.swapaxes(J_inv , -1,-2)
        term1 = 0.5 * T[...,None,None] * M_inv
        term2 = - M_inv @ J_inv_T @ TdA @ J_inv @ M_inv
        term3 = - (Tdg * g)[...,None,None] * M_inv
        pG_pM = mu[...,None,None] * (term1 + term2 + term3)
        return pG_pM
    
    def u(self , M_node , GdM):
        """
        辅助函数 u
        u = trace( Gdm M_node)
        Parameters:
            M_node(Tensor): 目标单元度量张量 (NN, GD, GD)
            GdM(Tensor): 基于几何离散的 G 泛函关于度量张量 M 的导数 (NC, GD, GD)
        """
        cell = self.cell
        c_M_node = M_node[cell ,...] # (NC,lv , GD, GD)
        e = bm.einsum('nik,nlij->nlkj' , GdM , c_M_node) # (NC, lv, GD, GD)
        u = bm.trace(e, axis1=-2, axis2=-1) # (NC, lv)
        return u
    
    def V(self,E):
        E_inv = bm.linalg.inv(E)
        R = self.R
        V = R @ E_inv # (NC, lv, GD)
        return V
    
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
    
    def _get_physical_node(self, move_vertor_field):
        """
        @brief calculate the physical node
        @param move_vertor_field: the move vector field of the logic node
        @param harmap: the map node after solving the harmap equation
        """
        node = self.node
        aim_field = move_vertor_field
        coef = _compute_coef_2d(aim_field,self.AC_generator)
        x = quad_equ_solver(coef)
        positive_x = bm.where(x>0, x, 1)
        eta = bm.min(positive_x)
        node = node +  self.alpha*eta* aim_field
        return node
    
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
        import matplotlib.pyplot as plt
        for i in range(maxit):
            self.monitor()
            self.mol_method()
            
            mesh = self.mesh
            logic_mesh = self.logic_mesh
            xi_n = logic_mesh.entity('node')
            
            E = self.edge_matrix(mesh) # (NC, GD, GD)
            E_inv = bm.linalg.inv(E) # (NC, GD, GD)
            E_hat = self.edge_matrix(logic_mesh) # (NC, GD, GD)
            E_hat_inv = bm.linalg.inv(E_hat) # (NC, GD, GD)
            M = self.M
            M_node = self.M_node
            M_inv = bm.linalg.inv(M) # (NC, GD, GD)
            M_node_inv = bm.linalg.inv(M_node) # (NN, GD, GD)
            
            A = self.A(E , E_hat , M_inv) # (NC, GD, GD)
            g = bm.linalg.det(A) # (NC,)
            theta = self.theta(M) # float
            mu = self.mu(M) # (NC,)
            R = self.R
            H = self.H(theta , A) # (NC,)
            T = self.T(H) # (NC,)
            T_dA = self.T_dif_A(theta , A) # (NC, GD, GD)
            T_dg = self.T_dif_g(theta , A , g) # (NC,)
                  
            sterm0_0 = -T[:,None , None] * E_inv
            sterm0_1 = 2 * E_hat_inv @ A @ T_dA @ E_hat @ E_inv
            sterm0_2 = 2 * (T_dg* g)[:, None , None] * E_inv
            sterm0 = sterm0_0 + sterm0_1 + sterm0_2 # (NC, GD, GD)
            term0 = mu[:, None , None] * R @ sterm0 # (NC, GD+1, GD)
            
            G_dM = self.G_dif_M(mu , T , M , M_inv , 
                                E , E_hat , T_dA , T_dg , g) # (NC, GD, GD)
            u = self.u(M_node , G_dM) # (NC, lv)
            V = self.V(E) # (NC, lv, GD)
            e = bm.ones((mesh.cell.shape[1],) , **self.kwargs0) # (lv,)
            term1 = -1/(self.GD+1) * bm.einsum('nk,nkd,l->nld' , u , V, e)
            
            v = term0 + term1 # (NC, lv, GD)
            grad_x = bm.zeros_like(mesh.node , **self.kwargs0) # (NN, GD)
            cm = self.cm
            grad_x = bm.index_add(grad_x , mesh.cell , cm[..., None , None] * v)
            
            P_diag = self.balance(M_node , theta) # (NN,)

            dt = self.dt
            tau = self.tau
            vector = dt / tau * grad_x # (NN, GD)
            err = bm.max(bm.linalg.norm(vector,axis=1))
            print("Iteration:", i, " Max movement:", err)
            if err < self.tol:
                print("The mesh has converged with error:", err)
                return self.mesh
            Bdinnernode_idx = self.Bdinnernode_idx
            dot = bm.sum(self.Bi_Pnode_normal * vector[Bdinnernode_idx],axis=1)
            vector = bm.set_at(vector , Bdinnernode_idx ,
                                    vector[Bdinnernode_idx] - dot[:,None] * self.Bi_Pnode_normal)
            vector = bm.set_at(vector , self.Vertices_idx , 0)
            
            node = self._get_physical_node(vector)
            I = bm.sum(cm * mu * T)
            print("The energy functional I is :", I)
            self.uh = self.interpolate(node)
            self._construct(node)

        # self.plot_cell_scalar(det_sterm0, title="Piecewise-constant T", cmap="jet", show=True)
            fig = plt.figure(figsize=(8,8),dpi=100)
            ax = fig.add_subplot(111)
            mesh.add_plot(ax)
            plt.show()

    def plot_cell_scalar(self, values, title="Cellwise T", cmap="viridis", clim=None, show=True, ax=None):
        """
        将单元分片常数标量（形如 (NC,)）在网格上用颜色显示。
        Parameters:
            values: (NC,) 分片常数（每个单元一个值）
            title:  图标题
            cmap:   颜色图名称
                clim:   (vmin, vmax) 数值范围；None 则自动
                show:   是否立即 plt.show()
                ax:     可选的 matplotlib 轴；None 则新建
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        mesh = self.mesh
        nodes = mesh.entity('node')         # (NN, 2)
        cells = mesh.cell                   # (NC, lv)
        # 转为 numpy
        try:
            vals = bm.to_numpy(values)
            pts = bm.to_numpy(nodes)
            cel = bm.to_numpy(cells)
        except Exception:
            vals = np.asarray(values)
            pts = np.asarray(nodes)
            cel = np.asarray(cells)
        # 多边形列表（每个单元的顶点坐标）
        polys = [pts[idx] for idx in cel]
        # 颜色映射
        if clim is None:
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
        else:
            vmin, vmax = clim
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        facecolors = cmap_obj(norm(vals))
         # 画图
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
            created_fig = True
        pc = PolyCollection(polys, facecolors=facecolors, edgecolors='k', linewidth=0.3)
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_title(title)
        # colorbar
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(vals)
        plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        if show and created_fig:
            plt.show()
        return ax