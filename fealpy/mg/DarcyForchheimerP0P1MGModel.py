import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy.linalg import norm
from scipy.sparse.linalg import cg, inv, spsolve
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace
from ..mesh import TriangleMesh

class DarcyForchheimerP0P1MGModel:

    def __init__(self, pde, mesh, n):

        self.integrator1 = mesh.integrator(3)
        self.integrator0 = mesh.integrator(1)
        self.pde = pde
        self.uspaces = []
        self.pspaces = []
        self.IMatrix = []
        self.A = []
        self.b = []
        self.uu = [] ## 磨光之后的值
        self.pp = [] ## 磨光之后的值
        self.uuc = [] ## 每层网格的初值
        self.ppc = [] ## 每层网格的初值
        self.r = []

        mesh0 = TriangleMesh(mesh.node, mesh.ds.cell)
        uspace = VectorLagrangeFiniteElementSpace(mesh0, p=0, spacetype='D')
        self.uspaces.append(uspace)

        pspace = LagrangeFiniteElementSpace(mesh0, p=1, spacetype='C')
        self.pspaces.append(pspace)

        for i in range(n):
            I0, I1 = mesh.uniform_refine(returnim=True) # I0:NodeMatrix(u), I1:CellMatrix(p)
            self.IMatrix.append((I0[0], I1[0]))
            mesh0 = TriangleMesh(mesh.node, mesh.ds.cell)
            uspace = VectorLagrangeFiniteElementSpace(mesh0, p=0, spacetype='D')
            self.uspaces.append(uspace)
            pspace = LagrangeFiniteElementSpace(mesh0, p=1, spacetype='C')
            self.pspaces.append(pspace)

        for i in range(n+1):
            A11, A12 = self.get_linear_stiff_matrix(i)
            self.A.append((A11, A12))
            f, g = self.get_right_vector(i)
            self.b.append((f, g))

        self.uh = self.uspaces[-1].function()
        self.ph = self.pspaces[-1].function()
        self.uI = self.uspaces[-1].interpolation(pde.velocity)
        self.pI = self.pspaces[-1].interpolation(pde.pressure)

        self.nlevel = n + 1
        u,p = self.compute_initial_value()
        self.uuc.append(u)
        self.ppc.append(p)
        
    def get_linear_stiff_matrix(self, level):
        
        mesh = self.pspaces[level].mesh
        pde = self.pde
        mu = pde.mu
        rho = pde.rho

        bc = np.array([1/3,1/3,1/3], dtype=mesh.ftype)##weight
        gphi = self.pspaces[level].grad_basis(bc)
        cellmeasure = mesh.entity_measure('cell')

        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        scaledArea = mu/rho*cellmeasure

        A11 = spdiags(np.r_[scaledArea,scaledArea], 0, 2*NC, 2*NC)

        phi = self.uspaces[level].basis(bc)
        data = np.einsum('ijm, km, i->ijk', gphi, phi, cellmeasure)
        cell2dof0 = self.uspaces[level].cell_to_dof()
        ldof0 = self.uspaces[level].number_of_local_dofs()
        cell2dof1 = self.pspaces[level].cell_to_dof()
        ldof1 = self.pspaces[level].number_of_local_dofs()
		
        gdof0 = self.uspaces[level].number_of_global_dofs()
        gdof1 = self.pspaces[level].number_of_global_dofs()
        J = np.einsum('ij, k->ijk', cell2dof1, np.ones(ldof0))
        I = np.einsum('ij, k->ikj', cell2dof0, np.ones(ldof1))
        A12 = csr_matrix((data.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))
        
        return A11, A12 
        
    def get_right_vector(self, level):
        mesh = self.pspaces[level].mesh
        pde = self.pde
        mu = pde.mu
        rho = pde.rho
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        cellmeasure = mesh.entity_measure('cell')
        
        f = self.uspaces[level].source_vector(self.pde.f, self.integrator0, cellmeasure)
        b = self.pspaces[level].source_vector(self.pde.g, self.integrator1, cellmeasure)
        	
	## Neumann boundary condition
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        ec = mesh.entity_barycenter('edge')
        isBDEdge = mesh.ds.boundary_edge_flag()
        edge2node = mesh.ds.edge_to_node()
        bdEdge = edge[isBDEdge, :]
        d = np.sqrt(np.sum((node[edge2node[isBDEdge, 0], :]\
            - node[edge2node[isBDEdge, 1], :])**2, 1))
        mid = ec[isBDEdge, :]

        ii = np.tile(d*self.pde.neumann(mid)/2,(1,2))
        g = np.bincount(np.ravel(bdEdge,'F'), weights = np.ravel(ii), minlength=NN)
        g = g - b  

        return f, g    

    def compute_initial_value(self):
        mesh = self.pspaces[-1].mesh
        pde = self.pde
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()

        A11, A12 = self.A[-1]
        f, g = self.b[-1]
        
        A = bmat([(A11, A12), (A12.transpose(), None)], format='csr',dtype=np.float)
        b = np.r_[f, g]
        
        up = np.zeros(2*NC+NN, dtype=np.float)
        idx = np.arange(2*NC+NN-1)
        up[idx] = spsolve(A[idx, :][:, idx], b[idx])
        u = up[:2*NC]
        p = up[2*NC:]

        cell = mesh.entity('cell')
        cellmeasure = mesh.entity_measure('cell')
        c = np.sum(np.mean(p[cell], 1)*cellmeasure)/np.sum(cellmeasure)
        p -= c

        return u,p

    def prev_smoothing(self, u, p, level, maxN):
        mesh = self.pspaces[level].mesh
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')
        
        mu = self.pde.mu
        rho = self.pde.rho
        beta = self.pde.beta
        alpha = self.pde.alpha
        tol = self.pde.tol
        cellmeasure = mesh.entity_measure('cell')
        area = np.repeat(cellmeasure, 2)

        # get A，b on current level
        A11, A12 = self.A[level]
        A21 = A12.transpose()
        f, g =  self.b[level]

        ## P-R interation for D-F equation
        n = 0
        ru = 1
        rp = 1
        Aalpha = A11.data + area/alpha
        
        while ru+rp > tol and n < maxN:
            ## step 1: Solve the nonlinear Darcy equation
            # Knowing (u,p), explicitly compute the intermediate velocity u(n+1/2)
            F = u/alpha - (mu/rho)*u - (A12@p - f)/area
            FL = np.sqrt(F[::2]**2 + F[1::2]**2)
            gamma = 1.0/(2*alpha) + np.sqrt((1.0/alpha**2) + 4*(beta/rho)*FL)/2
            uhalf = F/np.repeat(gamma, 2)
            
            ## Step 2: Solve the linear Darcy equation
            # update RHS
            uhalfL = np.sqrt(uhalf[::2]**2 + uhalf[1::2]**2)
            fnew = f + uhalf*area/alpha\
                    - beta/rho*uhalf*np.repeat(uhalfL, 2)*area
            
            ## Direct Solver
            Aalphainv = spdiags(1/Aalpha, 0, 2*NC, 2*NC)
            Ap = A21@Aalphainv@A12
           # print('Ap',Ap.toarray())
            bp = A21@(Aalphainv@fnew) - g 
           # print('bp', bp)
            p1 = np.zeros(NN,dtype=np.float)
            p1[1:] = spsolve(Ap[1:,1:],bp[1:])
            c = np.sum(np.mean(p1[cell],1)*cellmeasure)/np.sum(cellmeasure)
            p1 -= c
            u1 = Aalphainv@(fnew - A12@p1)
            
            ## Updated residual and error of consective iterations
            n = n + 1
            uLength = np.sqrt(u1[::2]**2 + u1[1::2]**2)
            Lu = A11@u1 + (beta/rho)*np.repeat(uLength*cellmeasure, 2)*u1 + A12@p1
            ru = norm(f - Lu)/norm(f)
            if norm(g) == 0:
                rp = norm(g - A21@u1)
            else:
                rp = norm(g - A21@u1)/norm(g)
            eu = np.max(abs(u1 - u))
            ep = np.max(abs(p1 - p))

            u[:] = u1
            p[:] = p1
                                
        
        return u, p, ru, rp

    def post_smoothing(self, u, p, level):
        mesh = self.pspaces[level].mesh
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        mu = self.pde.mu
        rho = self.pde.rho
        beta = self.pde.beta
        alpha = self.pde.alpha
        tol = self.tol
        maxN = self.pde.mg_maxN
        cellmeasure = mesh.entity_measure('cell')
        area = np.repeat(cellmeasure,2)
        
        # get A，b on current level
        A11, A12 = self.A[level]
        A21 = A12.transpose()
        f, g =  self.b[level]
        
        ## P-R interation for D-F equations

        n = 0
        ru = 1
        rp = 1
        uhalf[:] = uh
        Aalpha = A11.data + area/alpha
        Aalphainv = spdiags(1/Aalpha, 0, 2*NC, 2*NC)
        Ap = A21@Aalphainv@A12
        while ru+rp > tol and n < maxN:
            ## step 2: Solve the linear Darcy equation
            # update RHS
            uhalfL = np.sqrt(uhalf[::2]**2 + uhalf[1::2]**2)
            fnew = f + uhalf*area/alpha\
                    - beta/rho*uhalf*np.repeat(uhalfL, 2)*area
            
            ## Direct Solver
           # print('Ap',Ap.toarray())
            bp = A21@(Aalphainv@fnew) - g
           # print('bp', bp)
            p1 = np.zeros(NN,dtype=np.float)
            p1[1:] = spsolve(Ap[1:,1:],bp[1:])
            c = np.sum(np.mean(p1[cell],1)*cellmeasure)/np.sum(cellmeasure)
            p1 -= c
            u1 = Aalphainv@(fnew - A12@p1)
            
            ## step 1: Solve the nonlinear Darcy equation
            # Knowing (u,p), explicitly compute the intermediate velocity u(n+1/2)
            F = u/alpha - (mu/rho)*u - (A12@p - f)/area
            FL = np.sqrt(F[::2]**2 + F[1::2]**2)
            gamma = 1.0/(2*alpha) + np.sqrt((1.0/alpha**2) + 4*(beta/rho)*FL)/2
            uhalf = F/np.repeat(gamma, 2)
            
            ## Updated residual and error of consective iterations
            n = n + 1
            uLength = np.sqrt(u1[::2]**2 + u1[1::2]**2)
            Lu = A11@u1 + (beta/rho)*np.repeat(uLength*cellmeasure, 2)*u1 + A12@p1
            ru = norm(f - Lu)/norm(f)
            if norm(g) == 0:
                rp = norm(g - A21@u1)
            else:
                rp = norm(g - A21@u1)/norm(g)
            eu = np.max(abs(u1 - u))
            ep = np.max(abs(p1 - p))

            u[:] = u1
            p[:] = p1
            
        return u, p, ru, rp

    def fas(self, level):       
        mesh = self.pspaces[level].mesh
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        cellmeasure = mesh.entity_measure('cell')
        
        mu = self.pde.mu
        rho = self.pde.rho
        beta = self.pde.beta
        alpha = self.pde.alpha
        tol = self.pde.tol
        J = self.nlevel
        # coarsest level: exact solve
        if level == 1:##
            u,p,ru,rp = self.prev_smoothing(self.uuc[J-level-1], self.ppc[J-level-1], level, self.pde.maxN)
            rn = ru[end] + rp[end]
            uu.append(u)
            pp.append(p)
            return u,p
            
        ## Presmoothing
        print(J-level-1)
        u,p,ru, rp = self.prev_smoothing(self.uuc[J-level-1], self.ppc[J-level-1], level, self.pde.mg_maxN)
        self.uu.append(u)
        self.pp.append(p)
        # form residual on the fine grid
        # get A，b on current level
        A11, A12 = self.A[level]
        A21 = A12.transpose()
        f, g =  self.b[level]
        
        uLength = np.sqrt(u[::2]**2 + u[1::2]**2)
        Lu = A11@u + (beta/rho)*np.repeat(uLength*cellmeasure, 2)*u + A12@p
        r = f - Lu
        
        # restrict residual to the coarse grid
        I0, I1 = self.IMatrix[level]
        rc = I0.T@r
        uc = (I0.T@u)/4## ????要不要除以4
        pc = p[:Nc]
        self.uuc.append(uc)                            
        self.ppc.append(pc)
        self.r.append(rc)
        ## ????
        
    def mg(self):    
        mu = self.pde.mu
        rho = self.pde.rho
        beta = self.pde.beta
        alpha = self.pde.alpha
        tol = self.pde.tol
        level = self.nlevel
        
        for i in range(level-1,0,-1):
            self.fas(i) ###???需要循环结束再做提升
        
        for i in range(level-1, 0, -1):
            mesh = self.pspaces[level].mesh
            cellmeasure = mesh.entity_measure('cell')
            I0, I1 = self.IMatrix[i]
            u = self.uu[level-i-1]
            eu = I0@(u - self.uuc[level-i-1])
            
            # project eu back to the div free subspace
            # get A，b on current level
            A11, A12 = self.A[i]
            A21 = A12.transpose()
            f, g =  self.b[i]
            uLength = np.sqrt(u[::2]**2 + u[1::2]**2)
            Au = A11.data + beta*rho*np.repeat(uLength*area, 2)
            Auinv = spdiags(1/Au, 0, 2*NC, 2*NC)
            Atuta = A21@Auinv@A12
            bp = A21@eu
            
            theta = np.zeros(bp.shape())
            theta[1:] = spsolve(Atuta[1:, 1:], bp[1:])
            c = np.sum(np.mean(p[cell], 1)*cellmeasure)/np.sum(cellmeasure)
            theta -= c
            delta = Auinv@(A12@theta)
            u = u + eu - delta
            epc = self.pp[level-i-1] - self.ppc[level-i-1]
            HB = self.uniformcoarsenred(i)
            p = p + self.nodeinterpolate(epc, HB)
            
            ## Postsmoothing
            u, p, ru, rp = self.post_smoothing(u, p, i)
            
            
    def uniformcoarsenred(self,level):
        mesh = self.pspaces[level].mesh
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')

        HB = np.zeros((NN,3),dtype=np.int)
        if NC%4 == 0:
            NCc = NC//4 # number of triangles in the coarse grid

        else:
            return

        ## Find nodes
        t1 = np.arange(NCc)
        t2 = t1 + NCc
        t3 = t2 + NCc
        t4 = t3 + NCc

        if any(cell[t1,1] != cell[t2,0]) or any(cell[t1,2] != cell[t3,0]) or\
                any(cell[t4,0] != cell[t2,2]) or any(cell[t4,1] != cell[t3,0]):
            return
        p1 = cell[t1, 0]
        p2 = cell[t2, 1]
        p3 = cell[t3, 2]
        p4 = cell[t4, 0]
        p5 = cell[t1, 2]
        p6 = cell[t1, 1]

        ## Update and remove triangles
        cell[t1,:] = np.c_[p1,p2,p3]
        cell = cell[t1,:]

        ## Record HB
        HB[p6,:] = np.c_[p6, p1, p2]
        HB[p4,:] = np.c_[p4, p2, p3]
        HB[p5,:] = np.c_[p5, p1, p3]
        
        Nc = np.max(cell)
        HB = HB[Nc:,:]

        ## Update boundary edges
        bdFlag = np.arange(0)

        return HB
        
    def nodeinterpolate(self,u,HB):
        oldN = u.shape[0]
        newN = max(HB.shape[0],HB.shape[1])
        if oldN >= newN:
            idx = (HB == 0)
            u[idx,:] = np.arange(0)
        else:
            u[newN,:] = 0
            if min(HB[:,0]) > oldN:
                u[HB[:,0],:] = (u[HB[:,1],:] + u[HB[:,2],:])/2
            else:
                while oldN < newN:
                    newNode = np.arange(oldN, newN)
                    firstNewNode = newNode[(HB[newNode,1] <= oldN) and (HB[newNode,2] <= oldN)]
                    u[HB[firstNewNode,0]] = (u[HB[firstNewNode,1]] + u[HB[firstNewNode,2]])/2

        return u
           
            
            
            
            
               
               
