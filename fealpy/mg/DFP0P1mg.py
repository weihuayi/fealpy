import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, bmat, spdiags

from numpy.linalg import norm
from scipy.sparse.linalg import cg, inv, spsolve
from ..mg.DarcyP0P1 import DarcyP0P1

class DFP0P1mg():
    def __init__(self, pde, integrator0, integrator1):
        self.pde = pde
        self.integrator0 = integrator0
        self.integrator1 = integrator1
        self.level = pde.level
        
        mesh = pde.init_mesh(self.level)
        NC = mesh.number_of_cells()
        mfem = DarcyP0P1(pde,mesh,1,integrator1)
        self.uh0,self.ph0 = mfem.solve()
        A = mfem.get_left_matrix()
        A11 = A[:2*NC, :2*NC]
        A12 = A[:2*NC, 2*NC:]
        A21 = A[2*NC:, :2*NC]
        
        b = mfem.get_right_vector()
        f = b[:2*NC]
        g = b[2*NC:]
        
        cellmeasure = mesh.entity_measure('cell')
        area = np.repeat(cellmeasure, 2)
        alpha = pde.alpha
        Aalpha = A11 + spdiags(area/alpha, 0, 2*NC, 2*NC)
        Aalphainv = spdiags(1/Aalpha.data, 0, 2*NC, 2*NC)
        Ap = A21@Aalphainv@A12
        
        self.A = A11
        self.G = A12
        self.Gt = A21
        self.f = f
        self.g = g
        self.Ap = Ap

    def DarcyForchP0P1smoothing12(self,level,maxN, u, p, A, G, Gt, f, g, Ap):
        mesh =self.pde.init_mesh(level)
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')

        
        mu = self.pde.mu
        rho = self.pde.rho
        beta = self.pde.beta
        alpha = self.pde.alpha
        tol = self.pde.tol
        
        cellmeasure = mesh.entity_measure('cell')
        area = np.repeat(cellmeasure,2)


        ## P-R interation for D-F equation
        n = 0
        ru1 = np.ones(maxN+1, dtype=np.float)
        rp1 = np.ones(maxN+1, dtype=np.float)
        Aalpha = A + spdiags(area/alpha, 0, 2*NC, 2*NC)
        Aalphainv = spdiags(1/Aalpha.data, 0, 2*NC, 2*NC)
        print('f',f.shape)

        while ru1[n]+rp1[n] > tol and n < maxN:
            ## step 1: Solve the nonlinear Darcy equation
            # Knowing (u,p), explicitly compute the intermediate velocity u(n+1/2)
            F = u/alpha - (mu/rho)*u - (G@p - f)/area
            FL = np.sqrt(F[:NC]**2 + F[NC:]**2)
            gamma = 1/(2*alpha) + 1/2*np.sqrt(1/alpha**2 + 4*FL*beta/rho)
            uhalf = F/np.repeat(gamma, 2)

            ## Step 2: Solve the linear Darcy equation
            # update RHS
            uhalfL = np.sqrt(uhalf[:NC]**2 + uhalf[NC:]**2)
            fnew = f + uhalf*area/alpha - beta/rho*uhalf*np.repeat(uhalfL, 2)
            bp = Gt@(Aalphainv@fnew) - g

            p = np.zeros(NN, dtype=np.float)
            p[1:] = spsolve(Ap[1:, 1:], bp[1:])
            c = np.sum(np.mean(p[cell], 1)*cellmeasure)/np.sum(cellmeasure)
            p = p - c
            u = Aalphainv@(fnew - G@p)
            #print('NC',NC)

            ## Update residual and error of consective iterations
            n = n+1
            uLength = np.sqrt(u[:NC]**2 + u[NC:]**2)
            Lu = A@u + (beta/rho)*np.repeat(uLength,2)*u + G@p
            ru1[n] = norm(f - Lu)/norm(f)
            if norm(g) == 0:
                rp1[n] = norm(g - Gt@u)
            else:
                rp1[n] = norm(g - Gt@u)/norm(g)
        ru = np.zeros(n,dtype=np.float)
        rp = np.zeros(n,dtype=np.float)
        ru[:] = ru1[:n]
        rp[:] = rp1[:n]

        return u, p, ru, rp
        
    def uniformcoarsenred(self,level):
        mesh = self.pde.init_mesh(self.n+level)
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
        
    def DarcyForchP0P1smoothing21(self, u, A, G, Gt, f, g, Ap):
#        J = self.pde.J
#        uh0, ph0, A, G, Gt,f,g,Ap, mesh = self.init_data(J)
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        mu = self.pde.mu
        rho = self.pde.rho
        beta = self.pde.beta
        alpha = self.pde.alpha
        tol = self.pde.tol
        maxN = self.pde.mg_maxN
        cellmeasure = mesh.entity_measure('cell')
        area = np.repeat(cellmeasure, 2)
        ## P-R interation for D-F equations

        n = 0
        ru1 = np.ones(maxN,)
        rp1 = np.ones(maxN,)
        uhalf[:] = u
    
        Aalphainv = A + spdiags(area/alpha, 0, 2*NC, 2*NC)

        while ru1[n]+rp1[n] > tol and n < maxN:
            ## step 2: Solve the linear Darcy equation
            # update RHS

            uhalfL = np.sqrt(uhalf[:NC]**2 + uhalf[NC:]**2)
            fnew = f + uhalf*area/alpha - beta/rho*uhalf*np.repeat(uhalfL, 2)
            bp = Gt@(Aalphainv@fnew) - g

            p = np.zeros(NN, dtype=np.float)
            p[1:] = spsolve(Ap[1:, 1:], bp[1:])
            c = np.sum(np.mean(p[cell],1)*cellmeasure)/np.sum(cellmeasure)
            p = p - c
            u = Aalphainv@(fnew - G@p)

            ## step 1:Solve the nonlinear Darcy equation
            # Knowing(u,p), explicitly compute the intermediate velocity
            # u(n+1/2)

            F = u/alpha - (mu/rho)*u - (G@p - f)*area
            FL = np.sqrt(F[:NC]**2 + F[NC:]**2)
            gamma = 1/(2*alpha) + 1/2*np.sqrt((1/alpha**2)+4*(beta/rho)*FL)
            uhalf = F/np.repeat(gamma, 2)

            ## Update residual and error of consective iterations
            n = n + 1
            uLength = np.sqrt(u[:NC]**2 + u[NC:]**2)
            Lu = A@u + (beta/rho)*np.repeat(uLength, 2)*u + G@p
            ru1[n] = norm(f - Lu)/norm(f)
            if norm(g) == 0:
                rp1[n] = norm[g - Gt@u]
            else:
                rp1[n] = norm(g - Gt@u)/norm(g)
        ru[:] = ru1[1:n]
        rp[:] = rp1[1:n]

        return u,p,ru,rp
        
        
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





    def DarcyForchheimersmoothing(self,level, u, p, A, G, Gt, f, g, Ap):
        pde = self.pde
        mesh = pde.init_mesh(level)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        rho = pde.rho
        beta = pde.beta
        alpha = pde.alpha
        tol = pde.tol
        J = level - 3
        print('level',level)
        print('J',J)
        
        u_c = {}
        p_c = {}
        Pro = {}
        u_Length = {}

        # coarsest level: exact solve
        if J == 1:
            u,p,ru,rp = self.DarcyForchP0P1smoothing12(level,pde.maxN, u, p, A, G, Gt, f, g, Ap)
            rn = ru[-1] + rp[-1]

            return u,p,rn

        ## Presmoothing
        u,p,ru,rp = self.DarcyForchP0P1smoothing12(level,pde.mg_maxN, u, p, A, G, Gt, f, g, Ap)

        if ru.any() <= tol:
            rn = ru[-1] + rp[-1]

            return u,p,rn

        ##Transfer to coarse grid
        # coarsen grid and form transfer operators
        meshc = pde.init_mesh(level - 1)
        NNc = meshc.number_of_nodes()
        NCc = meshc.number_of_cells()

        # get restrict and promote operators
        data = np.ones(4*NCc, dtype=np.int)
        I = np.arange(4*NCc)
        J = np.repeat(np.arange(NCc),4)
        Pro_u = coo_matrix((data, (I,J)),dtype=np.int)
        Pro_u = bmat([(Pro_u, coo_matrix(Pro_u.shape,dtype=np.int)),(None, Pro_u)], format='csr',dtype=np.int)

        Res_u = coo_matrix((data,(J,I)),dtype=np.int)
        Res_u = bmat([(Res_u,coo_matrix(Res_u.shape,dtype=np.int)), (None,Res_u)],format='csr',dtype=np.int)
        # get residual on the fine grid
        uLength = np.sqrt(u[:NC]**2 + u[NC:]**2)
        Lu = A@u + (beta/rho)*np.repeat(uLength, 2)*u + G@p
        r = f - Lu

        # restrict residual to the coarse grid
        rc = Res_u@r
        uc = (Res_u@u)/4 #because fine grid = 4(coarse)
        pc = p[:NNc]
        u_c[level] = uc
        p_c[level] = pc
        Pro[level] = Pro_u
        u_Length[level] = uLength
        self.u_c = u_c
        self.p_c = p_c
        self.Pro = Pro
        self.u_Length = u_Length

        ## Coarse grid correction

        # get the matrix on the coarse grid
        fem = DarcyP0P1(pde, meshc, 1, self.integrator1)
        fem.solve()
        A = fem.get_left_matrix()###???????
        Ac = A[:2*NCc, :2*NCc]
        Gc = A[:2*NCc, 2*NCc:]
        Gct = A[2*NCc:, :2*NCc]
        cellmeasure1 = meshc.entity_measure('cell')
        ucLength = np.sqrt(uc[:NCc]**2 + uc[NCc:]**2)
        Lcuc = Ac@uc + beta/rho*np.repeat(ucLength, 2)*uc + Gc@pc
        fc = rc + Lcuc
        gc = Gct@uc
        areac = np.repeat(cellmeasure1, 2)

        Aalphac = Ac + spdiags(areac/alpha, 0, 2*NCc, 2*NCc)
        Aalphainvc = spdiags(1/Aalphac.data, 0, 2*NCc, 2*NCc)
        Apc = Gct@Aalphainvc@Gc

        return uc, pc, Ac, Gc, Gct, fc, gc, Apc
        
    def solve(self):
        level = self.pde.level
        u = self.uh0
        p = self.ph0
        A = self.A
        G = self.G
        Gt = self.Gt
        f = self.f
        g = self.g
        Ap = self.Ap
        
        for i in range(level, 3, -1):
            if i > 4:
                u, p, A, G, Gt, f, g, Ap = self.DarcyForchheimersmoothing(i, u, p, A, G, Gt, f, g, Ap)
                print('i',i)
            else:
                v,q,rn = self.DarcyForchheimersmoothing(i, u, p, A, G, Gt, f, g, Ap)
        print('Pro',self.Pro)
        for i in range(3, level+1):
            Pro_u = self.Pro[i] 
            uLength = self.u_Length[i]
            uc = self.u_c[i]
            pc = self.p_c[i]           
            ## Prolongate correction to fine space
            eu = Pro_u@(v - uc)
            # project eu back to the div free subspace
            Au = A + spdiags((beta/rho)*np.repeat(uLength*area,2), 0, 2*NC, 2*NC)
            Auinv = spdiags(1/Au.data, 0, 2*NC, 2*NC)
            Atuta = Gt@Auinv@G
            bp = Gt@eu

            theta = np.zeros(NN,)
            theta[1:] = spsolve(Atuta[1:,1:],bp[1:])
            c = np.sum(np.mean(p[cell],1)*cellmeasure)/np.sum(cellmeasure)
            theta = theta - c
            delta = Auinv@(G@theta)
            u = u + eu - theta
            epc = q - pc
            p = p + nodeinterpolate(epc,HB)

            ## Postsmoothing 
            u,p,ru,rp = DarcyForchP0P1smoothing21(u, A, G, Gt, f, g, Ap)
            rn = ru[-1] + rp[-1]

        return u,p,rn

            
            


