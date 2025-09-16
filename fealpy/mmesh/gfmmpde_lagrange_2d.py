from . import Monitor
from . import Interpolater
from .config import *
from .tool import quad_equ_solver
from ..mesh import IntervalMesh

class GFMMPDELagrange2d(Monitor, Interpolater):
    def __init__(self, mesh, beta, space, config: Config):
        super().__init__(mesh=mesh, beta=beta, space=space, config=config)
        self.config = config
        self.alpha = config.alpha
        self.tau = config.tau
        self.t_max = config.t_max
        self.tol = config.tol
        self.maxit = config.maxit
        self.pre_steps = config.pre_steps
        self.dt = self.tau * self.t_max
        self.r = config.r

        self.initialize()
        
    def initialize(self):
        """
        Initialize the GFMMPDE method
        """
        self.matrix_assemblor = self.matrix_assembly
        self.compute_coef = self._compute_coef_lagrange    

        self.SCI = ScalarConvectionIntegrator(q = self.q,method=self.assembly_method)
        self.lphi = self.lmspace.basis(self.bcs)
        self.lgphi = self.lmspace.grad_basis(self.bcs)
        self.cell2cell = self.mesh.cell_to_cell()
        self.node2cell = self.mesh.node_to_cell().tocsr()
        if self.tol == None:
            self.tol = self._caculate_tol()
        print(f"tolerance: {self.tol}")
        self.bform = BilinearForm(self.lmspace)
        self.bform.add_integrator(self.SDI,self.SCI,self.SMI)
        self.itSDI = ScalarDiffusionIntegrator(q = self.q,method= None)
        self.itSCI = ScalarConvectionIntegrator(q =self.q,method= None)
        self.itSMI = ScalarMassIntegrator(q =self.q,method= None)
        self.itSSI = ScalarSourceIntegrator(q =self.q,method= None)
        self.bc = DirichletBC(self.lmspace)

        (self.global_vnode_idx, 
        self.global_enode_idx, 
        self.global_cnode_idx) = self.global_intnode_classify()

    def matrix_assembly(self):
        """
        Assemble the matrix for the GFMMPDE method.
        Returns:
            A (bm.ndarray): The assembled matrix.
        """   
        cell2dof = self.cell2dof
        M = self.M # NC,NQ,...(GD,TD)
        M_node = self.M_node # NN,...(GD,TD)
        bcs = self.bcs
        mspace = self.mspace
        lnode_cell2dof = self.logic_mesh.node[cell2dof] # NC,LNN,GD

        gphi = mspace.grad_basis(bcs)
        lphi = self.lphi
        lgphi = self.lgphi

        J = bm.einsum('cqin,cim->cqmn',gphi,lnode_cell2dof)
        if M.ndim == 2:  # 标量情况 
            M_c_inv = 1.0 / M  # (NC, NQ)
            M_inv_cell2dof = 1.0 / M_node[cell2dof]  # (NC, LNN)
            JJ = bm.einsum('cqmn,cqkn->cqmk', J, J)  # (NC, NQ, TD, TD)
            G_xi = bm.einsum('cqid,ci->cqd', lgphi, M_inv_cell2dof)
            a = JJ * M_c_inv[..., None, None]
            b = -bm.einsum('cqmk,cqm->cqk', JJ, G_xi)
        else:  # 矩阵情况
            M_c_inv = bm.linalg.inv(M)
            M_inv_cell2dof = bm.linalg.inv(M_node)[cell2dof]
            # 原有的矩阵计算
            G_xi = bm.einsum('cqid,cimn->cqmnd', lgphi, M_inv_cell2dof)
            a = bm.einsum('cqmn,cqkl,cqnl->cqmk', J, J, M_c_inv)
            b = -bm.einsum('cqmn,cqkl,cqnlm->cqk', J, J, G_xi)

        a_diag = a[..., bm.arange(a.shape[-2]), bm.arange(a.shape[-1])]
        p = bm.sqrt(bm.sum(a_diag**2, axis=-1) + bm.sum(b**2, axis=-1))
        a /= p[...,None,None]
        b /= p[...,None]

        sm = self.sm
        d = self.d # NC,NQ
        rm = self.rm
        quad_cell = bm.einsum('cq,cqkl,q-> ckl',d*rm,a,self.ws)
        a_recover = bm.zeros((self.GDOF,self.TD,self.TD),**self.kwargs0)
        a_recover = bm.index_add(a_recover,cell2dof,quad_cell[:,None,...])
        a_recover /= sm[:,None,None]
        A_xi = bm.einsum('cqid,cidk-> cqk',lgphi,a_recover[cell2dof])
        AmB = A_xi - b
        A_value = bm.einsum('cqi,cidk-> cqdk',lphi,a_recover[cell2dof])

        SCI = self.SCI
        SDI = self.SDI
        SMI = self.SMI
        SCI.coef = self.dt * AmB
        SDI.coef = self.dt * A_value 
        SMI.coef = self.tau 
        A = self.bform.assembly()
        return A
    
    # def matrix_assembly(self):
    #     """
    #     Assemble the matrix for the GFMMPDE method.
    #     Returns:
    #         A (bm.ndarray): The assembled matrix.
    #     """   
    #     cell2dof = self.cell2dof
    #     M = self.M # NC,NQ,...(GD,TD)
    #     M_node = self.M_node # NN,...(GD,TD)
    #     bcs = self.bcs
    #     mspace = self.mspace
    #     lnode_cell2dof = self.logic_mesh.node[cell2dof] # NC,LNN,GD

    #     gphi = mspace.grad_basis(bcs)
    #     lphi = self.lphi
    #     lgphi = self.lgphi

    #     J = bm.einsum('cqin,cim->cqmn',gphi,lnode_cell2dof)
    #     if M.ndim == 2:  # 标量情况 
    #         M_c_inv = 1.0 / M  # (NC, NQ)
    #         M_inv_cell2dof = 1.0 / M_node[cell2dof]  # (NC, LNN)
    #         JJ = bm.einsum('cqmn,cqkn->cqmk', J, J)  # (NC, NQ, TD, TD)
    #         G_xi = bm.einsum('cqid,ci->cqd', lgphi, M_inv_cell2dof)
    #         a = JJ * M_c_inv[..., None, None]
    #         b = -bm.einsum('cqmk,cqm->cqk', JJ, G_xi)
    #     else:  # 矩阵情况
    #         M_c_inv = bm.linalg.inv(M)
    #         M_inv_cell2dof = bm.linalg.inv(M_node)[cell2dof]
    #         # 原有的矩阵计算
    #         G_xi = bm.einsum('cqid,cimn->cqmnd', lgphi, M_inv_cell2dof)
    #         a = bm.einsum('cqmn,cqkl,cqnl->cqmk', J, J, M_c_inv)
    #         b = -bm.einsum('cqmn,cqkl,cqnlm->cqk', J, J, G_xi)

    #     a_diag = a[..., bm.arange(a.shape[-2]), bm.arange(a.shape[-1])]
    #     p = bm.sqrt(bm.sum(a_diag**2, axis=-1) + bm.sum(b**2, axis=-1))
    #     a /= p[...,None,None]
    #     b /= p[...,None]
    #     I = bm.eye(self.TD, **self.kwargs0)
    #     I = I[None, None, ...]
    #     sm = self.sm
    #     d = self.d # NC,NQ
    #     rm = self.rm
    #     quad_cell = bm.einsum('cq,cqkl,q-> ckl',d*rm,a,self.ws)
    #     a_recover = bm.zeros((self.GDOF,self.TD,self.TD),**self.kwargs0)
    #     a_recover = bm.index_add(a_recover,cell2dof,quad_cell[:,None,...])
    #     a_recover /= sm[:,None,None]
    #     A_xi = bm.einsum('cqid,cidk-> cqk',lgphi,a_recover[cell2dof])
    #     AmB = A_xi - b
    #     A_value = bm.einsum('cqi,cidk-> cqdk',lphi,a_recover[cell2dof])

    #     SCI = self.SCI
    #     SDI = self.SDI
    #     SMI = self.SMI
    #     SCI.coef = self.dt * AmB
    #     SDI.coef = self.dt * (A_value + self.r*I)
    #     SMI.coef = self.tau 
    #     A = self.bform.assembly()
    #     return A
    
    def vector_assembly(self):
        """
        @brief vector assembly
        """
        node = self.node
        ws = self.ws
        lphi = self.lphi
        cell2dof = self.cell2dof
        b_c = self.tau*bm.einsum('cqi,cqj,cjd,q-> cid',lphi , 
                        lphi , node[cell2dof],ws)* self.logic_cm[:,None,None]

        F = bm.zeros((self.NN,self.TD),**self.kwargs0)
        F = bm.index_add(F, cell2dof , b_c)
        return F
    
    # def vector_assembly(self):
    #     """
    #     @brief vector assembly
    #     """
    #     node = self.node
    #     ws = self.ws
    #     lphi = self.lphi
    #     lgphi = self.lgphi
    #     cell2dof = self.cell2dof
    #     node = self.node
    #     grad_node = bm.einsum('cqik,cid-> cqkd', lgphi, node[cell2dof])
    #     dt = self.dt
    #     r = self.r
    #     b_c_0 = self.tau*bm.einsum('cqi,cqj,cjd,q-> cid',lphi , 
    #                     lphi , node[cell2dof],ws)* self.logic_cm[:,None,None]
    #     b_c_1 = dt*r*bm.einsum('cqid,cqkd,q-> cik',lgphi,grad_node,ws)* self.logic_cm[:,None,None]
    #     b_c = b_c_0 + b_c_1
    #     F = bm.zeros((self.NN,self.TD),**self.kwargs0)
    #     F = bm.index_add(F, cell2dof , b_c)
    #     return F
    
    def _compute_coef_lagrange(self, delta_x):
        """
        @brief compute the coefficient of the quadratic equation
        """
        J = self.mesh.jacobi_matrix(bc=self.bcs)
        gphi = self.mesh.grad_shape_function(self.bcs, p=self.p, variables='u')
        dJ = bm.einsum('...in, ...qim -> ...qnm', delta_x[self.cell2dof], gphi)
        a = bm.linalg.det(dJ) @ self.ws
        c = bm.linalg.det(J) @ self.ws
        b = (J[..., 0, 0] * dJ[..., 1, 1] + J[..., 1, 1] * dJ[..., 0, 0]
             - J[..., 0, 1] * dJ[..., 1, 0] - J[..., 1, 0] * dJ[..., 0, 1]) @ self.ws
        return [a, b, c]
    
    def _get_physical_node(self, move_vertor_field):
        """
        @brief calculate the physical node
        @param move_vertor_field: the move vector field of the logic node
        @param harmap: the map node after solving the harmap equation
        """
        node = self.node
        aim_field = move_vertor_field
        coef = self._compute_coef_lagrange(aim_field)
        x = quad_equ_solver(coef)
        positive_x = bm.where(x>0, x, 1)
        eta = bm.min(positive_x)
        node = node +  self.alpha*eta* aim_field
        return node
    
    def boundary_move(self):
        """
        Move the boundary node
        """
        M = self.M_node
        sbd_node_idx = self.sort_BdNode_idx
        node = bm.copy(self.node)
        vertices_idx = self.Vertices_idx
        circle_id = self.circle_id
        # 处理 vertices_idx 为 list 的情况
        if isinstance(vertices_idx, list):
        # 多连通区域的情况，使用 circle_id 进行分组处理
            for group_idx, vertices_group in enumerate(vertices_idx):
                # 获取当前组对应的边界节点段
                start_idx = circle_id[group_idx]
                end_idx = circle_id[group_idx + 1] if group_idx + 1 < len(circle_id) else len(sbd_node_idx)
                group_sbd_node_idx = sbd_node_idx[start_idx:end_idx]
                node = self._process_boundary_group(
                    node, M, group_sbd_node_idx, vertices_group
                )
        else:
            node = self._process_boundary_group(
                node, M, sbd_node_idx, vertices_idx
            )
        return node

    def _process_boundary_group(self, node, M, group_sbd_node_idx, vertices_group):
        """
        处理单个边界组的移动
        """
        # 构建扩展的边界节点索引（形成闭环）
        sbd_node_idx_ex = bm.concat([group_sbd_node_idx, [group_sbd_node_idx[0]]], axis=0)
        
        # 找到顶点在边界节点中的位置
        K = bm.where(group_sbd_node_idx == vertices_group[:, None])[1]
        VNN = len(vertices_group)
        v0 = node[vertices_group]
        v1 = node[bm.roll(vertices_group, shift=-1)]
        
        SDI = self.itSDI
        SMI = self.itSMI
        SCI = self.itSCI
        SSI = self.itSSI
        vector = v1 - v0
        v_length = bm.linalg.norm(vector, axis=-1)
        norm_vector = vector / v_length[:, None]

        for i in range(VNN):
            # 获取当前段的节点
            G = sbd_node_idx_ex[K[i]:K[i+1]+1 if i < VNN-1 else None]

            part_NN = G.shape[0]
            
            if M.ndim == 1:
                part_M = M[G]
            else:
                part_M = norm_vector[i] @ M[G] @ norm_vector[i]
            
            part_logic_node = bm.linspace(0, 1, num=part_NN)
            part_logic_cell = bm.stack([bm.arange(part_NN-1),
                                    bm.arange(1, part_NN)], axis=1)

            x_n = bm.linalg.norm(node[G] - v0[i], axis=-1) / v_length[i]
            pc_mea = x_n[1:] - x_n[:-1]
            sm = bm.zeros(part_NN, **self.kwargs0)
            sm = bm.index_add(sm, part_logic_cell, pc_mea[:, None])

            part_mesh = IntervalMesh(part_logic_node, part_logic_cell)
            space = LagrangeFESpace(part_mesh, p=1)
            qf = part_mesh.quadrature_formula(self.q)
            bcs, ws = qf.get_quadrature_points_and_weights()
            gphi = part_mesh.grad_shape_function(bcs, variables='x')
            phi = part_mesh.shape_function(bcs)
            
            grad_M = bm.einsum('cqid,ci-> cq', gphi, part_M[part_logic_cell])
            grad_M_node = bm.zeros(part_NN, **self.kwargs0)
            grad_M_node = bm.index_add(grad_M_node,
                                    part_logic_cell, grad_M[:, 0, None] * pc_mea[:, None])
            grad_M_node = grad_M_node / sm

            p_node = 1 / bm.sqrt(part_M**2 + grad_M_node**2)
            p = bm.einsum('cqi, ci -> cq', phi[None, ...], (p_node)[part_logic_cell])
            grad_p = bm.einsum('cqid,ci-> cqd', gphi, (p_node)[part_logic_cell])
            M_cell = bm.mean(part_M[part_logic_cell], axis=-1)

            bform = BilinearForm(space)
            bform.add_integrator(SDI, SMI, SCI)
            SDI.coef = self.dt * p * M_cell[:, None]
            SMI.coef = self.tau
            SCI.coef = self.dt * grad_p * M_cell[:, None, None]
            SDI.clear()
            SMI.clear()
            SCI.clear()
            A = bform.assembly()

            lform = LinearForm(space)
            lform.add_integrator(SSI)
            SSI.source = self.tau * space.function(x_n)
            SSI.clear()
            b = lform.assembly()
            
            bdIdx = bm.zeros(part_NN, **self.kwargs0)
            bdIdx[[0, -1]] = 1
            D0 = spdiags(1-bdIdx, 0, part_NN, part_NN)
            D1 = spdiags(bdIdx, 0, part_NN, part_NN)
            A = D0 @ A + D1
            b[0] = 0
            b[-1] = 1
            x = spsolve(A, b, 'scipy')
            node = bm.set_at(node, G, node[G[0]] + vector[i] * x[:, None])
        
        return node
    
    # def _process_boundary_group(self, node, M, group_sbd_node_idx, vertices_group):
    #     """
    #     处理单个边界组的移动
    #     """
    #     # 构建扩展的边界节点索引（形成闭环）
    #     sbd_node_idx_ex = bm.concat([group_sbd_node_idx, [group_sbd_node_idx[0]]], axis=0)
        
    #     # 找到顶点在边界节点中的位置
    #     K = bm.where(group_sbd_node_idx == vertices_group[:, None])[1]
    #     VNN = len(vertices_group)
    #     v0 = node[vertices_group]
    #     v1 = node[bm.roll(vertices_group, shift=-1)]
        
    #     SDI = self.itSDI
    #     SMI = self.itSMI
    #     SCI = self.itSCI
    #     SSI = self.itSSI
    #     vector = v1 - v0
    #     v_length = bm.linalg.norm(vector, axis=-1)
    #     norm_vector = vector / v_length[:, None]

    #     for i in range(VNN):
    #         # 获取当前段的节点
    #         G = sbd_node_idx_ex[K[i]:K[i+1]+1 if i < VNN-1 else None]

    #         part_NN = G.shape[0]
            
    #         if M.ndim == 1:
    #             part_M = M[G]
    #         else:
    #             part_M = norm_vector[i] @ M[G] @ norm_vector[i]
            
    #         part_logic_node = bm.linspace(0, 1, num=part_NN)
    #         part_logic_cell = bm.stack([bm.arange(part_NN-1),
    #                                 bm.arange(1, part_NN)], axis=1)

    #         x_n = bm.linalg.norm(node[G] - v0[i], axis=-1) / v_length[i]
    #         pc_mea = x_n[1:] - x_n[:-1]
    #         sm = bm.zeros(part_NN, **self.kwargs0)
    #         sm = bm.index_add(sm, part_logic_cell, pc_mea[:, None])

    #         part_mesh = IntervalMesh(part_logic_node, part_logic_cell)
    #         space = LagrangeFESpace(part_mesh, p=1)
    #         qf = part_mesh.quadrature_formula(self.q)
    #         bcs, ws = qf.get_quadrature_points_and_weights()
    #         gphi = part_mesh.grad_shape_function(bcs, variables='x')
    #         phi = part_mesh.shape_function(bcs)
 
    #         grad_M = bm.einsum('cqid,ci-> cq', gphi, part_M[part_logic_cell])
    #         grad_M_node = bm.zeros(part_NN, **self.kwargs0)
    #         grad_M_node = bm.index_add(grad_M_node,
    #                                 part_logic_cell, grad_M[:, 0, None] * pc_mea[:, None])
    #         grad_M_node = grad_M_node / sm

    #         p_node = 1 / bm.sqrt(part_M**2 + grad_M_node**2)
    #         p = bm.einsum('cqi, ci -> cq', phi[None, ...], (p_node)[part_logic_cell])
    #         grad_p = bm.einsum('cqid,ci-> cqd', gphi, (p_node)[part_logic_cell])
    #         M_cell = bm.mean(part_M[part_logic_cell], axis=-1)

    #         bform = BilinearForm(space)
    #         bform.add_integrator(SDI, SMI, SCI)
    #         # 扩散施加 r 光滑项
    #         SDI.coef = self.dt * (p * M_cell[:, None] + self.r)
    #         SMI.coef = self.tau
    #         SCI.coef = self.dt * grad_p * M_cell[:, None, None]
    #         SDI.clear()
    #         SMI.clear()
    #         SCI.clear()
    #         A = bform.assembly()

    #         lform = LinearForm(space)
    #         lform.add_integrator(SSI)
    #         # 扩散补偿项
    #         grad_x_n = bm.einsum('cqid,ci-> cqd', gphi, x_n[part_logic_cell])
    #         source_0 = self.dt * self.r *bm.einsum('q,cqid, cqd -> ci',ws, gphi, grad_x_n)*pc_mea[:, None]
    #         b0 = bm.zeros(part_NN, **self.kwargs0)
    #         b0 = bm.index_add(b0, part_logic_cell, source_0)
    #         SSI.source = self.tau * space.function(x_n)
    #         SSI.clear()
    #         b = lform.assembly() + b0
            
            
    #         bdIdx = bm.zeros(part_NN, **self.kwargs0)
    #         bdIdx[[0, -1]] = 1
    #         D0 = spdiags(1-bdIdx, 0, part_NN, part_NN)
    #         D1 = spdiags(bdIdx, 0, part_NN, part_NN)
    #         A = D0 @ A + D1
    #         b[0] = 0
    #         b[-1] = 1
    #         x = spsolve(A, b, 'scipy')
    #         node = bm.set_at(node, G, node[G[0]] + vector[i] * x[:, None])
        
    #     return node
    
    def func_solver(self):
        H = self.matrix_assembly()
        F = self.vector_assembly()
        TD = self.TD
        bd_condition = self.boundary_move()
        F_new = bm.zeros((self.NN,self.GD),**self.kwargs0)
        H0 = H.copy()
        for i in range(self.GD):
            self.bc.gd = bd_condition[:,i]
            H, F0 = self.bc.apply(H0, F[:,i])
            F_new = bm.set_at(F_new,(...,i), F0)

        blocks = [[None] * TD for _ in range(TD)] 
        for i in range(TD):
            blocks[i][i] = H
        H_bar = bmat(blocks, format='csr')
        F_flat = F_new.T.flatten()
        x = spsolve(H_bar,F_flat,solver='scipy')
        NN = self.NN
        x = x.reshape(TD,NN).T
        return x

    def monitor_time_smooth(self,old_M):
        M = self.M
        if old_M is None:
            self.M = M
        else:
            alpha = 0.2
            M = (1-alpha)*M + alpha*old_M
            self.M = M

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
    
    def construct(self,node):
        self.mesh.node = node
        self.node = node
        self.d = self._sqrt_det_G(self.bcs)
        self.sm = self._get_star_measure()

    def mesh_redistributor(self,maxit = 1000,pde = None):
        """
        @brief redistribute the mesh
        """
        old_M = None
        for i in range(maxit):
            self.monitor()
            self.mol_method()
            self.monitor_time_smooth(old_M)
            node = self.func_solver()
            v = node - self.node
            node = self._get_physical_node(v)
            error = bm.max(bm.linalg.norm(node - self.node,axis=1))
            print(f"iteration {i} , error: {error}")
            self.uh = self.interpolate(moved_node=node)
            old_M = self.M.copy()
            self.construct(node)
            if error < self.tol:
                break
              
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
                for i in range(steps):
                    t = (i+1)/steps
                    self.uh = t * self.uh
                    self.mesh_redistributor()
                    self.uh = pde.init_solution(self.mesh.node)
        else:
            for i in range(steps):
                t = (i+1)/steps
                self.uh *= t
                self.mesh_redistributor()
                self.uh = fun_solver(self.mesh)

    def local_intnode_classify(self,p):
        """
        Classify the local linear nodes, high-order edge nodes , high-order cell nodes
        """
        mul = bm.multi_index_matrix(p=p,dim = 2)
        local_vnode_idx = bm.where(mul // p > 0)[0]
        local_enode_idx = bm.where(bm.any((mul % p) > 0,axis=-1))[0]
        local_cnode_idx = bm.where(bm.all((mul % p) > 0,axis=-1))[0]
        return local_vnode_idx, local_enode_idx, local_cnode_idx
        
    def global_intnode_classify(self):
        """
        Classify the global linear nodes, high-order edge nodes , high-order cell nodes
        """
        mesh = self.mesh
        local_vnode_idx, local_enode_idx, local_cnode_idx = self.local_intnode_classify(p = self.p)
        cell = mesh.cell
        global_vnode_idx = cell[:,local_vnode_idx].flatten()
        global_enode_idx = cell[:,local_enode_idx].flatten()
        global_cnode_idx = cell[:,local_cnode_idx].flatten()
        global_vnode_idx = bm.unique(global_vnode_idx)
        global_enode_idx = bm.unique(global_enode_idx)
        global_cnode_idx = bm.unique(global_cnode_idx)
        return global_vnode_idx, global_enode_idx, global_cnode_idx
    
    def dof_classify(self):
        """
        Classify the degrees of freedom (DoF) for the mesh
        """
        cell2dof = self.cell2dof
        p = self.pspace.p
        local_vdof_idx, local_edof_idx, local_cdof_idx = self.local_intnode_classify(p = p)
        global_vdof_idx = cell2dof[:,local_vdof_idx].flatten()
        global_edof_idx = cell2dof[:,local_edof_idx].flatten()
        global_cdof_idx = cell2dof[:,local_cdof_idx].flatten()
        global_vdof_idx = bm.unique(global_vdof_idx)
        global_edof_idx = bm.unique(global_edof_idx)
        global_cdof_idx = bm.unique(global_cdof_idx)
        return global_vdof_idx, global_edof_idx, global_cdof_idx
    
    def matrix_split(self,A:CSRTensor):
        """
        Split the matrix into submatrices,
        A_ll is linear sub , 
        A_pp is high-order sub
        A_pl is mixed sub
        Parameter
            A: The input matrix to be split
        Returns
            A_ll: The linear submatrix
            A_pp: The high-order submatrix
            A_pl: The mixed submatrix
        """
        global_vnode_idx = self.global_vnode_idx
        global_high_order = bm.concat([self.global_enode_idx, self.global_cnode_idx])
        global_high_order = bm.unique(global_high_order)

        A_ll = A[global_vnode_idx,:][:,global_vnode_idx]
        A_pp = A[global_high_order,:][:,global_high_order]
        A_pl = A[global_high_order,:][:,global_vnode_idx]
        return A_ll, A_pp, A_pl
    
    def vector_split(self,b:TensorLike):
        """
        Split the vector into sub-vectors,
        b_l is linear sub ,
        b_p is high-order sub
        Parameter
            b: The input vector to be split
        Returns
            b_l: The linear subvector
            b_p: The high-order subvector
        """
        global_vnode_idx = self.global_vnode_idx
        global_high_order = bm.concat([self.global_enode_idx, self.global_cnode_idx])
        global_high_order = bm.unique(global_high_order)
        b_l = b[global_vnode_idx]
        b_p = b[global_high_order]
        return b_l, b_p
    