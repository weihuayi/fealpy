import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy.linalg import norm
from scipy.sparse.linalg import cg, inv, spsolve
from ..mg.DarcyP0P1 import DarcyP0P1

class DarcyForchP0P1mg():
    def __init__(self, n, J, pde, integrator0, integrator1):
        self.n = n
        self.pde = pde
        self.integrator0 = integrator0 
        self.integrator1 = integrator1
        mesh = pde.init_mesh(n+J+2)
        mesh1 = pde.init_mesh(n+J+1)
        self.mesh = mesh


    def init_data(self):
        pde = self.pde
        J = pde.J
        mesh = pde.init_mesh(self.n + J +2)
        NC = mesh.number_of_cells()
        mfem = DarcyP0P1(pde, mesh, 1, self.integrator1)
        uh0,ph0 = mfem.solve()
        A = mfem.get_left_matrix()
        A11 = A[:2*NC, :2*NC]
        A12 = A[:2*NC, 2*NC:]
        A21 = A[2*NC:, :2*NC]

        b = mfem.get_right_vector()
        f = b[:2*NC]
        g = b[2*NC:]

        cellmeasure = mesh.entity_measure('cell')
        area = np.tile(cellmeasure,2)
        alpha = pde.alpha
        Aalpha = A11 + spdiags(area/alpha, 0, 2*NC, 2*NC)
        Aalphainv = inv(Aalpha)
        Ap = A21@Aalphainv@A12
        return uh0, ph0, A11, A12, A21, f, g, Ap

    def DarcyForchP0P1smoothing12(self,uh0,ph0,A,G,Gt,f,g,Ap,J,maxN):
        mesh =self.pde.init_mesh(self.n+J+2)
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')
        mu = self.pde.mu
        rho = self.pde.rho
        beta = self.pde.beta
        alpha = self.pde.alpha
        tol = self.pde.tol
        cellmeasure = mesh.entity_measure('cell')
        area = np.tile(cellmeasure,2)


        ## P-R interation for D-F equation
        n = 0
        ru1 = np.ones(maxN+1, dtype=np.float)
        rp1 = np.ones(maxN+1, dtype=np.float)
        Aalphainv = A + spdiags(area/alpha, 0, 2*NC, 2*NC)
        print('f',f.shape)

        while ru1[n]+rp1[n] > tol and n < maxN:
            ## step 1: Solve the nonlinear Darcy equation
            # Knowing (u,p), explicitly compute the intermediate velocity u(n+1/2)
            F = uh0/alpha - (mu/rho)*uh0 - (G@ph0 - f)/area
            FL = np.sqrt(F[:NC]**2 + F[NC:]**2)
            gamma = 1/(2*alpha) + 1/2*np.sqrt(1/alpha**2 + 4*FL*beta/rho)
            uhalf = F/np.r_[gamma,gamma]

            ## Step 2: Solve the linear Darcy equation
            # update RHS
            uhalfL = np.sqrt(uhalf[:NC]**2 + uhalf[NC:]**2)
            fnew = f + uhalf*area/alpha - beta/rho*uhalf*np.r_[uhalfL,uhalfL]
            bp = Gt@(Aalphainv*fnew) - g

            p = np.zeros(NN, dtype=np.float)
            p[1:] = spsolve(Ap[1:,1:],bp[1:])
            c = np.sum(np.mean(p[cell],1)*cellmeasure)/np.sum(cellmeasure)
            p = p-c
            u = Aalphainv@(fnew - G@p)

            ## Update residual and error of consective iterations
            n = n+1
            uLength = np.sqrt(u[:NC]**2 + u[NC:]**2)
            Lu = A@u + (beta/rho)*np.r_[uLength,uLength]*u + G@p
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

    def uniformcoarsenred(self,J):
        mesh = self.pde.init_mesh(self.n+J+2)
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')
        print('NC',NC)
        print('NN',NN)

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
        print('vv',np.c_[p6, p1, p2].shape)
        print('p6',p6.shape)
        print('HB',HB.shape)


        ## Record HB
        HB[p6,:] = np.c_[p6, p1, p2]
        HB[p4,:] = np.c_[p4, p2, p3]
        HB[p5,:] = np.c_[p5, p1, p3]
        
        Nc = np.max(cell)
        HB = HB[Nc:,:]

        ## Update boundary edges
        bdFlag = np.arange(0)

        return HB

    def DarcyForchP0P1smoothing21(self,uh):
        J = self.pde.J
        uh0, ph0, A, G, Gt,f,g,Ap, mesh = self.init_data(J)
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        mu = self.pde.mu
        rho = self.pde.rho
        beta = self.pde.beta
        alpha = self.pde.alpha
        tol = self.tol
        maxN = self.pde.mg_maxN
        cellmeasure = mesh.entity_measure('cell')
        area = np.r_[cellmeasure, cellmeasure]
        ## P-R interation for D-F equations

        n = 0
        ru1 = np.ones(maxN,)
        rp1 = np.ones(maxN,)
        uhalf[:] = uh
    
        Aalphainv = A11 + spdiags(area/alpha, 0, 2*NC, 2*NC)

        while ru1[n]+rp1[n] > tol and n < maxN:
            ## step 2: Solve the linear Darcy equation
            # update RHS

            uhalfL = np.sqrt(uhalf[:NC]**2 + uhalf[NC:]**2)
            fnew = f + uhalf*area/alpha - beta/rho*uhalf*np.r_[uhalfL,uhalfL]
            bp = Gt@(Aalphainv*fnew) - g

            p = np.zeros(NN, dtype=np.float)
            p[1:] = spsolve(Ap[1:,1:],bp[1:])
            c = np.sum(np.mean(p[cell],1)*cellmeasure)/np.sum(cellmeasure)
            p = p-c
            u = Aalphainv@(fnew - G@p)

            ## step 1:Solve the nonlinear Darcy equation
            # Knowing(u,p), explicitly compute the intermediate velocity
            # u(n+1/2)

            F = u/alpha - (mu/rho)*u - (G@p - f)*area
            FL = np.sqrt(F[:NC]**2 + F[NC:]**2)
            gamma = 1/(2*alpha) + 1/2*np.sqrt((1/alpha**2)+4*(beta/rho)*FL)
            uhalf = F/np.r_[gamma,gamma]

            ## Update residual and error of consective iterations
            n = n+1
            uLength = np.sqrt(u[:NC]**2 + u[NC:]**2)
            Lu = A@u + (beta/rho)*np.r_[uLength,uLength]*u + G@p
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



    def DarcyForchsmoothing(self,uh0,ph0,i,A,G,Gt,f,g,Ap):
        pde = self.pde
        mesh = pde.init_mesh(self.n+i+2)
        mesh1 = pde.init_mesh(self.n+i+1)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        rho = pde.rho
        beta = pde.beta
        alpha = pde.alpha
        tol = self.pde.tol

       

        # coarsest level: exact solve
        if i == 1:
            u,p,ru,rp = self.DarcyForchP0P1smoothing12(uh0,ph0,A,G,Gt,f,g,Ap,i,pde.maxN)
            rn = ru[end] + rp[end]
            return u,p

        ## Presmoothing
        u,p,ru, rp = self.DarcyForchP0P1smoothing12(uh0,ph0,A,G,Gt,f,g,Ap,i,pde.mg_maxN)

        if ru.any() <= tol:
            rn = ru[end] + rp[end]
            return u,p

        ## Transter to coarse grid
        # corsen grid and form transfer operators
        HB = self.uniformcoarsenred(i)
        elemc = mesh1.entity('cell')
        Nc = np.max(elemc)
        NTc = NC//4
        #if not np.empty (HB).all():# Delete this judgment
        Pro_u = np.tile(np.eye(elemc.shape[0]),(4,1))
        top = np.column_stack((Pro_u,np.zeros(Pro_u.shape)))
        bot = np.column_stack((np.zeros(Pro_u.shape),Pro_u))
        Pro_u = np.row_stack((top,bot))
        Res_u = Pro_u.T

        # form residual on the fine grid
        uLength = np.sqrt(uh0[:NC]**2 + uh0[NC:]**2)
        Lu = A@uh0 + (beta/rho)*np.tile(uLength,2)*uh0 + G@ph0
        r = f - Lu
        print('r',r.dtype)
        print('r1',r.shape)
        print('Res_u',Res_u)


        # restrict residual to the coarse grid
        rc = Res_u@r
        uc = (Res_u@uh0)/4
        pc = ph0[:Nc]
        ## Coarse grid correction
        # form the matrix on the coarse grid

        fem = DarcyP0P1(pde, mesh1, 1, self.integrator1)
        NC1 = mesh1.number_of_cells()
        fem.solve()
        A1 = fem.get_left_matrix()
        Ac = A1[:2*NC1, :2*NC1]
        Gc = A1[:2*NC1, 2*NC1:]
        Gct = A1[2*NC1:, :2*NC1]
        cellmeasure1 = mesh1.entity_measure('cell')
        ucLength = np.sqrt(uc[:NC1]**2 + uc[NC1:]**2)
        Lcuc = Ac@uc + beta/rho*np.tile(ucLength,2)*uc + Gc@pc
        fc = rc + Lcuc
        gc = Gct@uc
        area2 = np.r_[cellmeasure1,cellmeasure1]
        
        Aalphac = Ac + spdiags(area2/alpha, 0, 2*NC1, 2*NC1)
        Aalphainvc = inv(Aalphac)
        Apc = Gct@Aalphainvc@Gc
        return Pro_u,uLength,uc,pc,HB,Ac,Gc,Gct,fc,gc,Apc



    def solve(self):
        J = self.pde.J

        for i in range(J,0,-1):
            if i == J and i != 1:
                uh0, ph0, A, G, Gt, f, g, Ap = self.init_data()
                Pro_u,uLength,uc,pc,HB,Ac,Gc,Gct,fc,gc,Apc = self.DarcyForchsmoothing(uh0,ph0,i,A,G,Gt,f,g,Ap)
                
            elif i > 1:
                Pro_u,uLength,uc,pc,HB,AC,Gc,Gct,fc,gc,Apc = self.DarcyForchsmoothing(uc,pc,i,Ac,Gc,Gct,fc,gc,Apc)

            else:
                v,q = self.DarcyForchsmoothing(uc,pc,i,Ac,Gc,Gct,fc,gc,Apc)


        ## Prolongate correction to fine space
        eu = Pro_u@(v - uc)
        # project eu back to the div free subspace
        Au = A + spdiags((beta/rho)*np.tile(uLength*area,(2,1)), 0, 2*NC, 2*NC)
        Auinv = inv(Au)
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
        DarcyForchP0P1smoothing21(u)
        rn = ru[-1] + rp[-1]

        return u,p,rn


