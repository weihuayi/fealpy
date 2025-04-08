from .base import *
from .tool import quad_equ_solver,cubic_equ_solver


class Harmap(MM_monitor,MM_Interpolater):
    def __init__(self,mesh,beta,vertices,r, config:Config):
        MM_monitor.__init__(self,mesh,beta,vertices,r,config)
        MM_Interpolater.__init__(self,mesh,vertices,config)
        self.alpha = config.alpha
        self.tol = config.tol
        self.maxit = config.maxit
        self.pre_steps = config.pre_steps

        self.bform = BilinearForm(self.space)
        self.bform.add_integrator(self.SDI)
        self.initialize()
    
    def initialize(self):
        """
        @brief initialize the harmap method
        """
        if self.TD == 2:
            self.equ_solver = quad_equ_solver
            self.compute_coef = self._compute_coef_2d
        elif self.TD == 3:
            self.equ_solver = cubic_equ_solver
            self.compute_coef = self._compute_coef_3d

        if self.mesh_type in ["TriangleMesh","TetrahedronMesh"]:
            self.scell = self.cell
        elif self.mesh_type == "QuadrangleMesh":
            smatrix = bm.array([[0,1,3],[1,2,0],[2,3,1],[3,0,2]],**self.kwargs1)
            self.scell = self.cell[:,smatrix].reshape(-1,3)
        elif self.mesh_type == "HexahedronMesh":
            smatrix = bm.array([[0,1,3,4],[1,2,0,5],[2,3,1,6],[3,0,2,7],
                                [4,5,7,0],[5,6,4,1],[6,7,5,2],[7,4,6,3]],**self.kwargs1)
            self.scell = self.cell[:,smatrix].reshape(-1,4)
        else:
            self.compute_coef = self._compute_coef_lagrange
        self.bdmm,self.inmm = self._mark_matrix()
        self.A,self.b = self._get_linear_constraint()
        self._get_logic_mesh()
        if self.tol == None:
            self.tol = self._caculate_tol()
        print(f"tolerance: {self.tol}")
        self._clear_unused_attributes()


    def _mark_matrix(self):
        """
        @brief construct the mask matrix (matrix partitioning, four blocks)
        """
        isBdNode = self.isBdNode
        BDNN = self.BDNN
        NN = self.NN
        INN = NN - BDNN
        kwargs0 = self.kwargs0
        kwargs1 = self.kwargs1
        Bdcol = bm.arange(NN,**kwargs1)[isBdNode]
        innercol = bm.arange(NN,**kwargs1)[~isBdNode]
        bd_extracter = CSRTensor(crow=bm.arange(BDNN+1,**kwargs1),col= Bdcol, 
                                 values = bm.ones(BDNN,**kwargs0),
                                 spshape=(BDNN,NN))
        inner_extracter = CSRTensor(crow=bm.arange(INN+1,**kwargs1),col= innercol,
                                    values = bm.ones(INN,**kwargs0),
                                    spshape=(INN,NN))
        return bd_extracter,inner_extracter

    def _clear_unused_attributes(self):
        """
        @brief clear the attributes that are not used in the following steps
        """
        attributes_to_clear = [
            'node2face',
            'sort_BdNode_idx',
            'BdFaceidx',
            'BdNodeidx',
            'logic_mesh',
            'logic_domain',
            'Vertices_idx',
            'logic_Vertices',
            'b_val0','b_val1','b_val2',  
        ]
        for attr in attributes_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)

    def _get_logic_mesh(self):
        """
        @brief construct the logical mesh
        """
        if not self.isconvex:
            if self.TD == 3:
                raise ValueError('Non-convex polyhedra cannot construct a logical mesh')
            self.logic_node = self._get_logic_node_init()
            if self.mesh_type == "LagrangeTriangleMesh":
                logic_cell = self.linermesh.cell
                linear_logic_mesh = TriangleMesh(self.logic_node,logic_cell)
                self.logic_mesh = self.mesh_class.from_triangle_mesh(linear_logic_mesh, self.p)
                self.logic_node = self.logic_mesh.node
                # self.logic_mesh = self.mesh_class.from_triangle_mesh(self.linermesh, self.p)
                # self.logic_mesh.node = self.logic_node
            elif self.mesh_type == "LagrangeQuadrangleMesh":
                logic_cell = self.linermesh.cell
                linear_logic_mesh = QuadrangleMesh(self.logic_node,logic_cell)
                self.logic_mesh = self.mesh_class.from_quadrangle_mesh(linear_logic_mesh, self.p)
                self.logic_node = self.logic_mesh.node
                # self.logic_mesh = self.mesh_class.from_quadrangle_mesh(self.linermesh, self.p)
                # self.logic_mesh.node = self.logic_node
            else:
                logic_cell = self.cell
                self.logic_mesh = self.mesh_class(self.logic_node,logic_cell)
        else:
            if self.mesh_type == "LagrangeTriangleMesh":
                self.logic_mesh = self.mesh_class.from_triangle_mesh(self.linermesh, self.p)
            elif self.mesh_type == "LagrangeQuadrangleMesh":
                self.logic_mesh = self.mesh_class.from_quadrangle_mesh(self.linermesh, self.p)
            else:
                self.logic_mesh = self.mesh_class(bm.copy(self.node),self.cell)
            self.logic_node = self.logic_mesh.node

    def _harmap_stiff(self,G:TensorLike):
        """
        @brief construct the stiffness matrix
        @param G: the inverse of monitor function
        """
        self.SDI.coef = G
        self.SDI.clear()  
        H = self.bform.assembly()
        return H
    
    def _get_linear_constraint(self):
        """
        @brief construct the linear constraint
        """
        logic_Vertices = self.logic_Vertices
        BdNodeidx = self.BdNodeidx
        Vertices_idx = self.Vertices_idx
        Bdinnernode_idx = self.Bdinnernode_idx
        Binnorm = self.Bi_Lnode_normal

        NN = self.NN
        BDNN = self.BDNN
        VNN = len(Vertices_idx)

        b = bm.zeros(NN, **self.kwargs0)
        b_val0 = self.b_val0
        b = bm.set_at(b , Bdinnernode_idx , b_val0)
        A_diag = bm.zeros((self.TD , NN)  , **self.kwargs0)
        A_diag = bm.set_at(A_diag , (...,Bdinnernode_idx) , Binnorm.T)
        A_diag = bm.set_at(A_diag , (0,Vertices_idx) , 1)
        if self.TD == 2:
            b = bm.set_at(b , Vertices_idx , logic_Vertices[:,0])
            b = bm.concat([b[BdNodeidx],logic_Vertices[:,1]])
            A_diag = A_diag[:,BdNodeidx]

            R = CSRTensor(crow=bm.arange(VNN+1,**self.kwargs1) , col=Vertices_idx, 
                          values=bm.ones(VNN,**self.kwargs0),spshape=(VNN,NN))
            R = R@self.bdmm.T
            A = bmat([[spdiags(A_diag[0], 0, BDNN, BDNN, format='csr'),
                       spdiags(A_diag[1], 0, BDNN, BDNN, format='csr')],
                      [None,R]],format='csr')

        elif self.TD == 3:
            Arrisnode_idx = self.Arrisnode_idx
            Arnnorm = self.Ar_Lnode_normal
            ArNN = len(Arrisnode_idx)
            b_val1 = self.b_val1
            b_val2 = self.b_val2
            b = bm.set_at(b , Arrisnode_idx , b_val1)
            b = bm.set_at(b , Vertices_idx , logic_Vertices[:,0])[BdNodeidx]
            b = bm.concat([b,b_val2,logic_Vertices[:,1],logic_Vertices[:,2]])       

            A_diag = bm.set_at(A_diag , (...,Arrisnode_idx) , Arnnorm[:,0,:].T)
            A_diag = A_diag[:,BdNodeidx]
            
            index1 = NN * bm.arange(self.TD) + Arrisnode_idx[:,None]
            index2 = NN * bm.arange(self.TD) + BdNodeidx[:,None]
            rol_Ar = bm.repeat(bm.arange(ArNN)[None,:],3,axis=0).flatten()
            cow_Ar = index1.T.flatten()
            data_Ar = Arnnorm[:,1,:].T.flatten()
            
            index = bm.stack([rol_Ar,cow_Ar],axis=0)
            Ar_constraint = COOTensor(index,data_Ar,spshape=(ArNN,3*NN))
            Ar_constraint = Ar_constraint.tocsr()
            Vertices_constraint1 = CSRTensor(crow=bm.arange(VNN+1) , col=Vertices_idx + NN,
                                          values=bm.ones(VNN,**self.kwargs0),spshape=(VNN,3*NN))
            Vertices_constraint2 = CSRTensor(crow=bm.arange(VNN+1) , col=Vertices_idx + 2 * NN,
                                          values=bm.ones(VNN,**self.kwargs0),spshape=(VNN,3*NN))
            
            mask_matrix = CSRTensor(crow=bm.arange(len(index2.flatten())+1), 
                                col=index2.T.flatten(), 
                                values=bm.ones(len(index2.flatten()), **self.kwargs0), 
                                spshape=(len(index2.flatten()), 3*NN))

            Ar_constraint = Ar_constraint @ mask_matrix.T
            Vertices_constraint1 = Vertices_constraint1 @ mask_matrix.T
            Vertices_constraint2 = Vertices_constraint2 @ mask_matrix.T
            A_part = hstack([spdiags(A_diag[0], 0, BDNN, BDNN, format='csr'),
                            spdiags(A_diag[1], 0, BDNN, BDNN, format='csr'),
                            spdiags(A_diag[2], 0, BDNN, BDNN, format='csr')])
            A = vstack([A_part,Ar_constraint,Vertices_constraint1,Vertices_constraint2])
        return A,b
    
    def _solve_harmap(self,G, logic_node):
        """
        @brief solve the harmap equation
        @param G: the inverse of monitor function
        @param logic_node: the logic node
        """
        isBdNode = self.isBdNode
        TD = self.TD
        BDNN = self.BDNN
        INN = self.NN - BDNN
        H = self._harmap_stiff(G)
        bdmm , inmm = self.bdmm , self.inmm
        H11 = inmm @ H @ inmm.T
        H12 = inmm @ H @ bdmm.T
        H21 = bdmm @ H @ inmm.T
        H22 = bdmm @ H @ bdmm.T
        A,b= self.A,self.b
        harmap = bm.copy(logic_node)

        F = (-H12 @ harmap[isBdNode, :]).T.flatten()
        blocks = [[None] * TD for _ in range(TD)] 
        for i in range(TD):
            blocks[i][i] = H11
        H0 = bmat(blocks, format='csr')
        move_innerlogic_node = cg(H0, F, atol=1e-8,returninfo=True)[0]
        move_innerlogic_node = move_innerlogic_node.reshape((TD, INN)).T
        harmap = bm.set_at(harmap, ~isBdNode, move_innerlogic_node)

        F = (-H21 @ move_innerlogic_node).T.flatten()
        b0 = bm.concat((F,b),axis=0)

        blocks = [[None] * TD for _ in range(TD)] 
        for i in range(TD):
            blocks[i][i] = H22
        A1 = bmat(blocks, format='csr')
        A0 = bmat([[A1,A.T],[A,None]],format='csr')
        move_bdlogic_node = spsolve(A0,b0,solver=self.solver)[:TD*BDNN]
        move_bdlogic_node = move_bdlogic_node.reshape((TD, BDNN)).T
        harmap = bm.set_at(harmap , isBdNode, move_bdlogic_node)
        move_vector_field = logic_node - harmap
        return harmap,move_vector_field
    
    def _get_physical_node(self,harmap,move_vertor_field):
        """
        @brief calculate the physical node
        @param move_vertor_field: the move vector field of the logic node
        @param harmap: the map node after solving the harmap equation
        """
        node = self.node
        cm = self.cm
        TD = self.TD
        space = self.space
        multi_index = self.multi_index 
        gphi = space.grad_basis(multi_index)
        grad_x = bm.zeros((self.NN,TD,TD),**self.kwargs0)
        pcell = self.pcell
        grad_X_incell = bm.einsum('cin, cqim -> cqnm',harmap[pcell], gphi)
        grad_x_incell = bm.linalg.inv(grad_X_incell)*cm[:,None,None,None]
        bm.index_add(grad_x , pcell , grad_x_incell)
        grad_x /= self.sm[:,None,None]
        delta_x = (grad_x @ move_vertor_field[:,:,None]).reshape(-1,TD)

        if TD == 3:
            self.Bi_Pnode_normal = self.Bi_Lnode_normal
            Ar_Pnode_normal = self.Ar_Lnode_normal
            Arrisnode_idx = self.Arrisnode_idx
            dot1 = bm.sum(Ar_Pnode_normal * delta_x[Arrisnode_idx,None],axis=-1)
            doap = dot1[:,0,None] * Ar_Pnode_normal[:,0,:] + dot1[:,1,None] * Ar_Pnode_normal[:,1,:]
            delta_x = bm.set_at(delta_x , Arrisnode_idx , delta_x[Arrisnode_idx] - doap)
        
        Bdinnernode_idx = self.Bdinnernode_idx
        dot = bm.sum(self.Bi_Pnode_normal * delta_x[Bdinnernode_idx],axis=1)
        delta_x = bm.set_at(delta_x , Bdinnernode_idx ,
                            delta_x[Bdinnernode_idx] - dot[:,None] * self.Bi_Pnode_normal)

        # physical node move distance
        coef = self.compute_coef(delta_x)
        x = self.equ_solver(coef)
        positive_x = bm.where(x>0, x, 1)
        eta = bm.min(positive_x)
        node = node +  self.alpha*eta* delta_x
        return node
    
    def _compute_coef_lagrange(self, delta_x):
        """
        @brief compute the coefficient of the quadratic equation
        """
        J = self.mesh.jacobi_matrix(bc=self.bcs)
        gphi = self.mesh.grad_shape_function(self.bcs, p=self.p, variables='u')
        dJ = bm.einsum('...in, ...qim -> ...qnm', delta_x[self.pcell], gphi)
        a = bm.linalg.det(dJ) @ self.ws
        c = bm.linalg.det(J) @ self.ws
        b = (J[..., 0, 0] * dJ[..., 1, 1] + J[..., 1, 1] * dJ[..., 0, 0]
             - J[..., 0, 1] * dJ[..., 1, 0] - J[..., 1, 0] * dJ[..., 0, 1]) @ self.ws
        return [a, b, c]
    
    def _compute_coef_2d(self, p):
        """
        Compute coefficients for 2D case.
        """
        return self._compute_general_coef(p, self._compute_coef_general_2d)

    def _compute_coef_3d(self, p):
        """
        Compute coefficients for 3D case.
        """
        return self._compute_general_coef(p, self._compute_coef_general_3d)
    
    def _compute_general_coef(self, delta_x , fun):
        """
        @brief compute the coefficient of the quadratic equation
        """
        A, C = self.AC_gererator(delta_x)
        return fun(A, C)
    
    def _compute_coef_general_2d(self,A,C):
        """
        @brief compute the coefficient of the quadratic equation
        """
        a = bm.linalg.det(C)
        c = bm.linalg.det(A)
        b = (A[:, 0, 0] * C[:, 1, 1] - 
             A[:, 0, 1] * C[:, 1, 0] + 
             C[:, 0, 0] * A[:, 1, 1] - 
             C[:, 0, 1] * A[:, 1, 0])
        return [a, b, c]

    def _compute_coef_general_3d(self,A,C):
        """
        @brief compute the coefficient of the cubic equation
        """
        a0, a1, a2 = (C[:, 1, 1] * C[:, 2, 2] - C[:, 1, 2] * C[:, 2, 1],
                      C[:, 1, 2] * C[:, 2, 0] - C[:, 1, 0] * C[:, 2, 2],
                      C[:, 1, 0] * C[:, 2, 1] - C[:, 1, 1] * C[:, 2, 0])
        b0, b1, b2 = (A[:, 1, 1] * C[:, 2, 2] - A[:, 1, 2] * C[:, 2, 1] + 
                      C[:, 1, 1] * A[:, 2, 2] - C[:, 1, 2] * A[:, 2, 1], 
                      A[:, 1, 0] * C[:, 2, 2] - A[:, 1, 2] * C[:, 2, 0] + 
                      C[:, 1, 0] * A[:, 2, 2] - C[:, 1, 2] * A[:, 2, 0], 
                      A[:, 1, 0] * C[:, 2, 1] - A[:, 1, 1] * C[:, 2, 0] + 
                     C[:, 1, 0] * A[:, 2, 1] - C[:, 1, 1] * A[:, 2, 0])
        c0, c1, c2 = (A[:, 1, 1] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 1], 
                      A[:, 1, 0] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 0],
                      A[:, 1, 0] * A[:, 2, 1] - A[:, 1, 1] * A[:, 2, 0])
        a = C[:, 0, 0] * a0 - C[:, 0, 1] * a1 + C[:, 0, 2] * a2
        ridx = bm.where(a > 1e-14)[0]
        b = (A[:, 0, 0] * a0 - A[:, 0, 1] * a1 + 
             A[:, 0, 2] * a2 + C[:, 0, 0] * b0 - 
             C[:, 0, 1] * b1 + C[:, 0, 2] * b2)
        c = (A[:, 0, 0] * b0 - A[:, 0, 1] * b1 + 
             A[:, 0, 2] * b2 + C[:, 0, 0] * c0 - 
             C[:, 0, 1] * c1 + C[:, 0, 2] * c2)
        d = A[:, 0, 0] * c0 - A[:, 0, 1] *c1 + A[:, 0, 2] * c2
        a, b, c, d = a[ridx], b[ridx], c[ridx], d[ridx]
        return [a, b, c, d]
    
    def AC_gererator(self,delta_x):
        """
        @brief generate the tensor A and C to construct the equation
        @param scell: the cell has been splited
        @param node: the physical node
        """
        node = self.node
        scell = self.scell
        A = bm.permute_dims((node[scell[:,1:]] - node[scell[:,0,None]]),axes=(0,2,1))
        C = bm.permute_dims((delta_x[scell[:,1:]] - delta_x[scell[:,0,None]]),axes=(0,2,1))
        return A,C
    
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
                d = bm.min(bm.concat([e0, e1])).item()
        return d*0.1/p
    
    def _construct(self,moved_node:TensorLike):
        """
        @brief construct information for the harmap method before the next iteration
        """
        self.interpolate(moved_node)
        self.mesh.node = moved_node
        self.space.mesh = self.mesh
        self.node = moved_node
        self.cm = self.mesh.entity_measure('cell')
        self.sm = self._get_star_measure()
        self.d = self._sqrt_det_G(self.bcs)
        self.update_matrix()
        self.mot()

    def mesh_redistributor(self):
        """
        @brief redistribute the mesh
        """
        self.mot()
        for i in range(self.maxit):
            harmap,move_vector_field = self._solve_harmap(1/self.M,self.logic_node)
            L_infty_error = bm.max(bm.linalg.norm(self.logic_node - harmap,axis=1))
            print(f'iteration: {i}, L_infty_error: {L_infty_error}')
            if L_infty_error < self.tol:
                print(f'total iteration: {i}')
                break

            moved_node = self._get_physical_node(harmap,move_vector_field)
            self._construct(moved_node)
        else:
            print('exceed the maximum iteration')

    def mp_mesh_redistributor(self):
        self.repack()
        self.mesh_redistributor()
        self.uh = self.uh.T.reshape(-1,)

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
        
    def mp_preprocessor(self):
        """
        @brief preprocessor: linear transition initialization with multi-physics
        @param steps: fake time steps
        """
        pde = self.pde
        self.repack()
        steps = self.pre_steps
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
        self.uh = self.uh.T.reshape(-1,)

    def repack(self):
        GDOF = self.space.number_of_global_dofs()
        pro_uh = bm.zeros((GDOF,self.dim),**self.kwargs0)
        for i in range(self.dim):
            pro_uh = bm.set_at(pro_uh,(...,i), segmenter(self.uh,self.dim,i))
        self.uh = pro_uh

    

