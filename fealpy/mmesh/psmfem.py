from .base import *
from .tool import quad_equ_solver,cubic_equ_solver


class PSMFEM(MM_monitor,MM_Interpolater):
    def __init__(self,mesh,beta,vertices,r, config:Config):
        MM_monitor.__init__(self,mesh,beta,vertices,r,config)
        MM_Interpolater.__init__(self,mesh,vertices,config)
        self.alpha = config.alpha
        self.tol = config.tol
        self.maxit = config.maxit
        self.pre_steps = config.pre_steps
        self.kappa = config.kappa
        self.U_bd = None

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
        self.A,self.b = self._get_linear_constraint()
        if self.tol == None:
            self.tol = self._caculate_tol()
        print(f"tolerance: {self.tol}")
        self._clear_unused_attributes()
        self._bilinear_assambly()

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
        return d*0.03
    
    def _clear_unused_attributes(self):
        """
        @brief clear the attributes that are not used in the following steps
        """
        attributes_to_clear = [
            'node2face',
            'sort_BdNode_idx',
            'BdFaceidx',
            'logic_domain',
            'Vertices_idx',
        ]
        for attr in attributes_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)

    def _get_linear_constraint(self):
        """
        @brief construct the linear constraint
        """
        BdNodeidx = self.BdNodeidx
        Vertices_idx = self.Vertices_idx
        Bdinnernode_idx = self.Bdinnernode_idx
        Binnorm = self.Bi_Lnode_normal

        NN = self.NN
        BDNN = self.BDNN
        VNN = len(Vertices_idx)

        A_diag = bm.zeros((self.TD , NN)  , **self.kwargs0)
        A_diag = bm.set_at(A_diag , (...,Bdinnernode_idx) , Binnorm.T)
        A_diag = bm.set_at(A_diag , (0,Vertices_idx) , 1)
        if self.TD == 2:
            A_diag = A_diag[:,BdNodeidx]

            R = CSRTensor(crow=bm.arange(VNN+1,**self.kwargs1) , col=Vertices_idx, 
                          values=bm.ones(VNN,**self.kwargs0),spshape=(VNN,NN))
            A = bmat([[spdiags(A_diag[0], 0, BDNN, BDNN, format='csr'),
                       spdiags(A_diag[1], 0, BDNN, BDNN, format='csr')],
                      [None,R[:,BdNodeidx]]],format='csr')
        elif self.TD == 3:
            Arrisnode_idx = self.Arrisnode_idx
            Arnnorm = self.Ar_Lnode_normal
            ArNN = len(Arrisnode_idx)    
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
        b = bm.zeros(A.shape[0],**self.kwargs0)
        return A,b
    
    def _bilinear_assambly(self):
        h = self.hmin
        r = self.r
        R = r*(1+r)
        logic_mass = self.logic_mass
        SDI = self.SDI
        bform = BilinearForm(self.lspace)
        bform.add_integrator(SDI)
        SDI.coef = h**2*R
        SDI.clear()
        A = bform.assembly()
        A += logic_mass
        K = bm.arange(self.NN,**self.kwargs1)
        BdNodeidx = self.BdNodeidx
        InnerNodeidx = K[~self.isBdNode]
        self.H00 = A[InnerNodeidx][:,InnerNodeidx]
        self.H01 = A[InnerNodeidx][:,BdNodeidx]
        self.H10 = A[BdNodeidx][:,InnerNodeidx]
        self.H11 = A[BdNodeidx][:,BdNodeidx]

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
    
    def _linear_assambly(self):
        """
        @brief calculate the b value of the quadratic equation
        G cqmm
        """
        lspace = self.lspace
        I = bm.eye(self.TD,**self.kwargs0)
        G = 1/self.M
        G = G[...,None,None]*I[None,...] # cq il
        logic_node = self.logic_mesh.node
        space = self.space
        bcs = self.bcs
        ws = self.ws
        gphi = space.grad_basis(bcs)
        pcell = self.pcell
        J = bm.einsum('cin, cqim -> cqnm',logic_node[pcell], gphi)# cq delta i
        lgphi = lspace.grad_basis(self.bcs) # cqi gamma cqj k
        stiff = (bm.linalg.det(J))**self.kappa
        J *= stiff[..., None, None]
        lgphi = lgphi@J@G
        K = bm.einsum('q,cqjl-> cjl',ws,lgphi)* self.logic_cm[:,None,None]
 
        F = bm.zeros((self.NN,2),**self.kwargs0)
        F = bm.index_add(F,pcell,K)
        b0 =  F[:,0]
        b1 =  F[:,1]
        return b0,b1
    
    def _func_solver(self,U_bd = None):
        """
        @brief solve the quadratic equation
        """
        isBdNode = self.isBdNode
        TD = self.TD
        NN = self.NN
        BDNN = self.BDNN
        H00,H01,H10,H11 = self.H00,self.H01,self.H10,self.H11
        A = self.A
        v0 = bm.zeros(A.shape[0],**self.kwargs0) 
        b0,b1 = self._linear_assambly()
        b0_inner = b0[~isBdNode]
        b1_inner = b1[~isBdNode]
        b0_bd = b0[isBdNode]
        b1_bd = b1[isBdNode]

        if U_bd is None:
            U_bd = bm.zeros((BDNN,TD),**self.kwargs0)

        U0_inner = cg(H00, b0_inner - H01@U_bd[:,0], atol=1e-6,returninfo=True)[0]
        U1_inner = cg(H00, b1_inner - H01@U_bd[:,1], atol=1e-6,returninfo=True)[0]
        U_inner = bm.concat([U0_inner[:,None],U1_inner[:,None]],axis=1)
        b_0 = -H10 @ U0_inner + b0_bd
        b_1 = -H10 @ U1_inner + b1_bd
        b = bm.concat([b_0,b_1,v0])
        blocks = [[None] * TD for _ in range(TD)] 
        for i in range(TD):
            blocks[i][i] = H11
        A1 = bmat(blocks, format='csr')
        A0 = bmat([[A1,A.T],[A,None]],format='csr')
        U_bd = spsolve(A0,b,solver='scipy')
        U_bd = U_bd[:TD*BDNN].reshape((TD,BDNN)).T
        move_vector = bm.zeros((NN,TD),**self.kwargs0)
        move_vector = bm.set_at(move_vector,~isBdNode,U_inner)
        move_vector = bm.set_at(move_vector,isBdNode,U_bd)
        return move_vector
    
    def func_solver_bd(self):
        isBdNode = self.isBdNode
        TD = self.TD
        NN = self.NN
        BDNN = self.BDNN
        INN = NN - BDNN
        H00,H01,H10,H11 = self.H00,self.H01,self.H10,self.H11
        A = self.A
        A_part0 = A[:,:BDNN]
        A_part1 = A[:,BDNN:]

        H = bmat([[ H00, H01 ,None , None , None], 
                  [H10, H11 , None , None , A_part0.T],
                  [None , None , H00 , H01 , None],
                  [None , None , H10 , H11 , A_part1.T],
                  [None , A_part0 , None , A_part1 , None]], format='csr')
        b0,b1 = self._linear_assambly()
        b0_inner = b0[~isBdNode]
        b1_inner = b1[~isBdNode]
        b0_bd = b0[isBdNode]
        b1_bd = b1[isBdNode]

        b = bm.zeros(H.shape[0],**self.kwargs0)
        b[:INN] = b0_inner
        b[INN:NN] = b0_bd
        b[NN:NN+INN] = b1_inner
        b[NN+INN:2*NN] = b1_bd

        U = cg(H,b,atol=1e-8,returninfo=True)[0]
        U = U[:NN*TD].reshape((TD,NN)).T

        move_vector = bm.zeros((NN,TD),**self.kwargs0)
        move_vector = bm.set_at(move_vector,~isBdNode,U[:INN])
        move_vector = bm.set_at(move_vector,isBdNode,U[INN:NN])
        return move_vector
    
    def _get_physical_node(self,move_vertor_field):
        """
        @brief calculate the physical node to avoid the mesh tangling
        @param move_vertor_field: the move vector field of the physical node
        """
        node = self.node
        aim_field = move_vertor_field

        coef = self.compute_coef(aim_field)
        x = self.equ_solver(coef)
        positive_x = bm.where(x>0, x, 1)
        eta = bm.min(positive_x)
        node = node +  self.alpha*eta* aim_field
        return node
    
    def _construct(self,node):

        self.interpolate(node)
        self.mesh.node = node
        self.node = node
        self.d = self._sqrt_det_G(self.bcs)
        self.update_matrix()

    def mesh_redistributor(self):
        """
        @brief redistribute the mesh
        """
        for i in range(self.maxit):
            self.arc_length()
            v = self._func_solver(U_bd=self.U_bd)
            self.U_bd = v[self.isBdNode]
            node = self._get_physical_node(v)
            error = bm.max(bm.linalg.norm(node - self.node,axis=1))
            print(f"iteration {i} , error: {error}")
            if i >= 2 and error < self.tol:
                break
            self._construct(node)
        else:
            print('exceed the maximum iteration')

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