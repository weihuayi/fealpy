import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy.linalg import norm
from scipy.sparse.linalg import cg, inv, spsolve
from ..mg.DarcyP0P1 import DarcyP0P1

class DarcyForchP0P1mg():
    def __init__(self, n, J, pde, integrator0, integrator1):
        self.pde = pde
        mesh = pde.init_mesh(n+J+2)
        mesh1 = pde.init_mesh(n+J+1)
        self.mesh = mesh
        self.mesh1 = mesh1
        NC = mesh.number_of_cells()
        mfem = DarcyP0P1(pde, mesh, 1, integrator1)
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

        self.A = A11
        self.Ap = Ap
        self.G = A12
        self.Gt = A21
        self.uh0 = uh0
        self.ph0 = ph0
        self.J = J
        self.cellmeasure = cellmeasure
        self.f = f
        self.g = g

    def DarcyForchP0P1smoothing12(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        mu = self.pde.mu
        rho = self.pde.rho
        beta = self.pde.beta
        alpha = self.pde.alpha
        maxN = self.pde.maxN
        cellmeasure = self.cellmeasure
        area = np.tile(cellmeasure,2)

        A = self.A
        G = self.G
        Gt = self.Gt
        f = self.f
        g = self.g

        ## P-R interation for D-F equation
        n = 0
        ru = ones(maxN, dtype=np.float)
        rp = ones(maxN, dtype=np.float)
        Aalphadiag = np.arange(0,dtype=np.float)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i == j:
                    Aalphadiag = np.append(Aalphadiag,A[i,j])
        Aalphadiag = Aalphadiag + area/alpha
        Aalphainv = spdiags(1/Aalphadiag, 0, 2*NC, 2*NC)

        while ru[n]+rp[n] > tol and n < maxN:
            ## step 1: Solve the nonlinear Darcy equation
            # Knowing (u,p), explicitly compute the intermediate velocity u(n+1/2)
            F = self.uh0/alpha - (mu/rho)*self.uh0 - (G*self.ph0 - f)/area
            FL = np.sqrt(F[:2*NC]**2 + F[2*NC:]**2)
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
            ru[n] = norm(f - Lu)/norm(f)
            if norm(g) == 0:
                rp[n] = norm[g - Gt@u]
            else:
                rp[n] = norm(g - Gt@u)/norm(g)
        ru = ru[1:n]
        rp = rp[1:n]
        self.uh0[:] = u
        self.ph0[:] = p

        return ru, rp

    def uniformcoarsenred(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cell = mesh.ds.cell()

        HB = np.arange(0)
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



    def solve(self):
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        rho = self.pde.rho
        alpha = self.pde.alpha
        J = self.pde.J
        tol = self.pde.tol

        A = self.A
        G = self.G
        f = self.f
        

        # coarsest level: exact solve
        if J == 1:
            ru,rp = self.DarcyForchP0P1smoothing12()
            rn = ru[end] + rp[end]
            return

        ## Presmoothing
        ru, rp = self.DarcyForchP0P1smoothing12()

        if ru <= tol:
            rn = ru[end] + rp[end]
            return

        ## Transter to coarse grid
        # corsen grid and form transfer operators
        HB = self.uniformcoarsenred()
        elemc = self.mesh1.number_of_cells()
        Nc = np.max(elemc)
        NTc = NC//4
        if not np.empty (HB).all():
            Pro_u = np.tile(eye(elemc.shape[0]),(4,1))
            top = np.column_stack((Pro_u,np.zeros(Pro_u.shape)))
            bot = np.column_stack((np.zeros(Pro_u.shape),Pro_u))
            Pro_u = np.row_stack((top,bot))
            Res_u = Pro_u.T

        # form residual on the fine grid
        uLength = np.sqrt(self.uh0[:NC]**2 + self.uh0[NC:]**2)
        Lu = A@self.uh0 + (beta/rho)*np.tile(uLength,(2,1))*self.uh0 + G@self.ph0
        r = f - Lu

        # restrict residual to the coarse grid
        rc = Res_u@r
        uc = (Res_u@self.uh0)/4
        pc = self.ph0[:Nc]

        ## Coarse grid correction
        # form the matrix on the coarse grid

        fem = DarcyP0P1(self.pde, self.mesh1, 1, self.integrator1)
        NC1 = mesh1.number_of_cells()
        fem.solve()
        A1 = fem.get_left_matrix()
        Ac = A1[:2*NC1, :2*NC1]
        Gc = A1[:2*NC1, 2*NC1:]
        Gct = A1[2*NC1:, :2*NC1]
        cellmeasure1 = mesh1.entity_measure('cell')
        ucLength = np.sqrt(uc[:NC1]**2 + uc[NC1:]**2)
        Lcuc = Ac@uc + beta/rho*np.tile(ucLength,(2,1))*uc + Gc@pc
        fc = rc + Lcuc
        gc = Gct@uc
        area2 = np.r_[cellmeasure1,cellmeasure1]
        
        Aalphac = Ac + spdiags(area2/alpha, 0, 2*NC1, 2*NC1)
        Aalphainvc = inv(Aalphac)
        Apc = Gct@Aalphainvc@Gc
        print(J)

        return DarcyForchP0P1mg(n,J-1,pde,self.integrator0,integrator1)
