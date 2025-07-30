from typing import Optional
from fealpy.backend import backend_manager as bm
from fealpy.sparse import spdiags,csr_matrix
from fealpy.solver import GAMGSolver


from typing import Optional
from ..backend import TensorLike

class DarcyForchheimerTPDv:
    """
    Simplified TPDv solver for Darcy-Forchheimer problems.

    Usage:
        solver = DarcyForchheimerTPDv(B, f, g, u_bform, Mu, pde, u0, p0, ...)
        uh, ph, resu, resp = solver.solve()
    """

    def __init__(self,
                 B: TensorLike,  # Matrix B
                 f: TensorLike,  # RHS vector f
                 g: TensorLike,  # RHS vector g
                 u_bform,  # Bilinear form for velocity mass
                 Mu,  # Scalar mass integrator
                 pde,  # PDE object, contains physical coefficients and terms
                 u0,  # Initial guess for velocity (TensorFunction )
                 p0,  # Initial guess for pressure (TensorFunction)
                 maxIt: int = 100,  # Maximum number of iterations
                 tol: float = 1e-6,  # Tolerance for convergence
                 gamma0: float = 2.0,  # Relaxation parameter for Schur preconditioner
                 stepsize: float = 0.4,  # Step size for updates
                 scaleu: float = 0.8,  # Scaling factor for velocity update
                 update_frequency: int = 5  # Frequency of updating AMG preconditioner
                 ):
        """
        Initialization for Darcy-Forchheimer TPDv solver.

        Parameters:
            B (TensorLike): Matrix B that couples velocity and pressure.
            f (TensorLike): Right-hand side vector for the velocity equation.
            g (TensorLike): Right-hand side vector for the pressure equation.
            u_bform (BilinearForm): Bilinear form used to assemble the velocity mass matrix.
            Mu (ScalarMassIntegrator): Scalar mass integrator used to compute the mass matrix.
            pde (PDE): PDE object containing coefficients like beta and mu.
            u0 (TensorLike): Initial guess for the velocity field.
            p0 (TensorLike): Initial guess for the pressure field.
            maxIt (int): Maximum number of iterations for the solver.
            tol (float): Convergence tolerance for residuals.
            gamma0 (float): Relaxation parameter for updating the Schur preconditioner.
            stepsize (float): Step size for updating velocity and pressure fields.
            scaleu (float): Scaling factor for velocity updates.
            amg_update_frequency (int): Frequency of updating the AMG preconditioner.
        """
        
        # data
        self.B = B        
        self.f = f
        self.g = g
        self.u_bform = u_bform  # BilinearForm for velocity mass (contains integrators)
        self.Mu = Mu            # ScalarMassIntegrator object whose .coef will be set
        self.pde = pde
        
        # prepare sizes
        self.Nu = int(self.B.shape[1])
        self.Np = int(self.B.shape[0])
        self.Nt = self.Nu // 2  # Number of elements (2D vector field)

        # prepare initial guesses
        self.u0 = u0
        self.p0 = p0

        # algorithmic params
        self.maxIt = int(maxIt)
        self.tol = float(tol)
        self.gamma0 = float(gamma0)
        self.stepsize = float(stepsize)
        self.scaleu = float(scaleu)
        self.update_frequency = int(update_frequency)

        # free pressure indices 
        self.freep = bm.arange(self.Np - 1)
        self.Bff = self.B[self.freep, :]
        self.Bfft = self.Bff.T

        # storage for last computed matrices
        self.M = None
        self.S0 = None
        self.mg = None
        
        self.resu = None
        self.resp = None


    def _update_mass(self):

        self.Mu.coef = lambda bcs, index: self.pde.beta * bm.sqrt(
            bm.sum(self.u0(bcs, index) ** 2, axis=-1)
        )
        self.M = self.u_bform.assembly()  # re-assemble mass matrix with updated coef

    
    def _update_S(self,type = 'classic'):
        if type == 'classic':
            mdiags = self.M.diags().values
            Minv = spdiags(1.0 / mdiags, 0, self.Nu, self.Nu)
        elif type == 'Jainv':
            Minv =self.MJinv()
        
        if self.S0 is None:
            self.S0 = self.Bff @ (Minv @ self.Bfft)
        else:
            S = self.Bff @ (Minv @ self.Bfft)
                # relaxation / smoothing of S0
            self.S0 = (self.S0 + self.gamma0 * self.stepsize * S) / (1.0 + self.gamma0 * self.stepsize)
        self.S0 = self.S0.sum_duplicates()
        self.mg = GAMGSolver()
        self.mg.setup(self.S0)
        
        
    def Jacobian(self):

        u = self.u0[:].reshape(self.Nt, 2)        # (Nt,2)
        r = bm.sqrt(u[:, 0]**2 + u[:, 1]**2)   # (Nt,)

        s = self.pde.mu + self.pde.beta * r      # (Nt,)
        a11 = s + self.pde.beta * (u[:, 0]**2) / r
        a22 = s + self.pde.beta * (u[:, 1]**2) / r
        a12 = self.pde.beta * (u[:, 0] * u[:, 1]) / r
        i = bm.arange(self.Nt, dtype=bm.int64)
        rows = bm.concatenate([2*i, 2*i, 2*i+1, 2*i+1])
        cols = bm.concatenate([2*i, 2*i+1, 2*i, 2*i+1])
        data = bm.concatenate([a11, a12, a12, a22])
        J = csr_matrix((data, (rows, cols)), shape=(self.Nu, self.Nu))
        return J

    def Jinv(self):

        ux = self.u0[0::2]    # shape (Nt,)
        uy = self.u0[1::2]    # shape (Nt,)

        r = bm.sqrt(ux * ux + uy * uy)    # (Nt,)
        s = self.pde.mu + self.pde.beta * r                     # (Nt,)

        a11 = s + self.pde.beta * ux * ux / r           # (Nt,)
        a22 = s + self.pde.beta * uy * uy / r           # (Nt,)
        a12 = self.pde.beta * ux * uy / r               # (Nt,)

        det = a11 * a22 - a12 * a12
        detinv = 1.0 / det

        d11 = a22 * detinv    # (Nt,)
        d22 = a11 * detinv    # (Nt,)
        d12 = -a12 * detinv   # (Nt,)
        idx = bm.arange(self.Nt, dtype=bm.int64)   # (Nt,)

        rows = bm.concatenate([2 * idx, 2 * idx, 2 * idx+1, 2 * idx+1])
        cols = bm.concatenate([2 * idx, 2 * idx+1, 2 * idx, 2 * idx+1])
        data = bm.concatenate([d11, d12, d12, d22])

        Jinv = csr_matrix((data, (rows, cols)), shape=(self.Nu, self.Nu))

        return Jinv
    
    def MJinv(self):
        Jinv = self.Jinv()
        mesh = self.u_bform.space.mesh
        area = mesh.entity_measure('cell')
        area_elem = bm.repeat(area, 2)
        M0inv = spdiags(1.0 / area_elem, 0, self.Nu, self.Nu)
        MJinv = M0inv @ Jinv
        return MJinv

    def TPDv(self):

        # initial global arrays
        poldAll = self.p0.copy()
        # enforce pressure reference
        poldAll[-1] = 0.0
        uoldAll = self.u0.copy()

        resu = bm.zeros(self.maxIt)
        resp = bm.zeros(self.maxIt)
        self._update_mass()
        self._update_S()
        # initially M, S0, mg already prepared in __init__
        for ite in range(self.maxIt):
            
            self.u0[:] = uoldAll
            # explicit/transport step (prediction)
            Bp = self.Bfft @ poldAll[self.freep]       # B^T p on velocity space
            du_tmp = self.M @ uoldAll + Bp - self.f
            du = du_tmp / self.M.diags().values
            unew_tmp = uoldAll - du / self.scaleu

            # Schur right-hand side for pressure update
            dp_rhs = self.g[self.freep] - self.Bff @ unew_tmp

            # occasionally update S0 and AMG preconditioner
            if ite % self.update_frequency == 0:
                self._update_S()

            # solve Schur system approximately via AMG
            dpAll, _ = self.mg.solve(dp_rhs)
            pnew = poldAll[self.freep] - (self.stepsize / self.scaleu) * dpAll
            pnewAll = poldAll.copy()
            pnewAll[self.freep] = pnew

            # velocity update (explicit convex combination)
            unewAll = (1.0 - self.stepsize) * uoldAll + self.stepsize * unew_tmp
            self._update_mass()
            # record residuals
            
            resu[ite] = bm.linalg.norm(self.M @ unewAll + (self.Bfft @ pnew) - self.f)
            resp[ite] = bm.linalg.norm(self.Bff @ unewAll - self.g[self.freep])

            # convergence check (relative to f norm)
            if ite > 0 and (resu[ite] / bm.linalg.norm(self.f)) < self.tol:
                # update stored iterates and break
                uoldAll, poldAll = unewAll, pnewAll
                resu = resu[:ite + 1]
                resp = resp[:ite + 1]
                print("TPDv converged at iteration", ite)
                break

            # prepare next iter
            uoldAll, poldAll = unewAll, pnewAll

        # store histories
        self.resu = resu
        self.resp = resp

        return uoldAll, poldAll, resu, resp
    
    
    def TPDv_IMEX(self):

        # initial global arrays
        poldAll = self.p0.copy()
        # enforce pressure reference
        poldAll[-1] = 0.0
        uoldAll = self.u0.copy()

        resu = bm.zeros(self.maxIt)
        resp = bm.zeros(self.maxIt)
        self._update_mass()
        self._update_S()
        mesh = self.u_bform.space.mesh
        Nt = mesh.number_of_cells()
        area = mesh.entity_measure('cell')   # shape (Nt,)

        for ite in range(self.maxIt):

            self.u0[:] = uoldAll
            u_sigma = self.u0[:].reshape(Nt, 2)
            u_sigma_norm = bm.linalg.norm(u_sigma, axis=1)
            sigma_elem = self.pde.beta * u_sigma_norm + self.pde.mu

            Bp = self.Bfft @ poldAll[self.freep]
            du_tmp = self.M @ uoldAll + Bp - self.f
            du = du_tmp/self.M.diags().values
            unew_tmp = uoldAll - du / self.scaleu

            dp_rhs = self.g[self.freep] - self.Bff @ unew_tmp

            # 每5步更新 S0 & AMG
            if ite % 5 == 0:
                self._update_S()

            # AMG 解 dp
            dpAll, _ = self.mg.solve(dp_rhs)
            pnew = poldAll[self.freep] - (self.stepsize / self.scaleu) * dpAll
            pnewAll = poldAll.copy()
            pnewAll[self.freep] = pnew

            # 按单元组织的速度、Bp、f（形状 (Nt,2)）
            u_elem = uoldAll.reshape(Nt, 2)        # 旧速度，按单元两分量
            Bp = self.Bfft @ pnewAll[self.freep]
            Bp_elem = (Bp).reshape(Nt, 2)          # 由 Bfft @ pold 得到的向量 (Nu,)
            f_elem = self.f.reshape(Nt, 2)              # 右端 f，按单元两分量
            area_elem = mesh.entity_measure('cell').reshape(Nt)   # (Nt,)
            alphau = self.stepsize / self.scaleu


            coeff = (sigma_elem / alphau - self.pde.mu).reshape(Nt, 1)   # (Nt,1)
            F_half_elem = coeff * u_elem - (Bp_elem / area_elem.reshape(Nt, 1)) + (f_elem / area_elem.reshape(Nt, 1))  # (Nt,2)
            F_abs = bm.linalg.norm(F_half_elem, axis=1)   # (Nt,) 
            term_under_sqrt = (sigma_elem**2) / (alphau**2) + 4.0 * (self.pde.beta) * F_abs  # (Nt,)
            gamma = 0.5 * (sigma_elem / alphau) + 0.5 * bm.sqrt(term_under_sqrt)   # (Nt,)

            uhalf_elem = F_half_elem / gamma.reshape(Nt, 1)    # (Nt,2)

            unewAll = uhalf_elem.reshape(self.Nu)

            self.u0[:] = unewAll
            self._update_mass()
    
            resu[ite] = bm.linalg.norm(self.M @ unewAll + (self.Bfft @ pnew) - self.f)
            resp[ite] = bm.linalg.norm(self.Bff @ unewAll - self.g[self.freep])

            if ite > 0 and resu[ite] / bm.linalg.norm(self.f) < self.tol:
                print("Converged at iteration", ite)
                uoldAll, poldAll = unewAll, pnewAll
                break

            uoldAll, poldAll = unewAll, pnewAll

        return uoldAll, poldAll, resu, resp
    
    
    def TPDv_Newton(self):
               
        poldAll = self.p0.copy()
        poldAll[-1] = 0.0
        uoldAll = self.u0.copy()

        resu = bm.zeros(self.maxIt)
        resp = bm.zeros(self.maxIt)
        
        self._update_mass()
        self._update_S(type='Jainv')
        
        for ite in range(self.maxIt):
            
            Bp = self.Bfft @ poldAll[self.freep]
            ru = self.M @ uoldAll + Bp - self.f
            rp = -(self.Bff @ uoldAll - self.g[self.freep])
            MJinv = self.MJinv()
            rp = self.Bff @ (MJinv @ ru) + rp
            dpAll,_ = self.mg.solve(rp)
            du = MJinv @ (ru - self.Bfft @ dpAll[self.freep])
            unewAll = uoldAll - self.stepsize * du
            pnewAll = poldAll.copy()
            pnewAll[self.freep] = poldAll[self.freep] - self.stepsize * dpAll[self.freep]
            
            self.u0[:] = unewAll
            self._update_mass()
            self._update_S(type='Jainv')
            # record residuals
            
            resu[ite] = bm.linalg.norm(self.M @ unewAll + (self.Bfft @ pnewAll) - self.f)
            resp[ite] = bm.linalg.norm(self.Bff @ unewAll - self.g[self.freep])

            # convergence check (relative to f norm)
            if ite > 0 and (resu[ite] / bm.linalg.norm(self.f)) < self.tol:
                # update stored iterates and break
                uoldAll, poldAll = unewAll, pnewAll
                resu = resu[:ite + 1]
                resp = resp[:ite + 1]
                print("TPDv converged at iteration", ite)
                break

            # prepare next iter
            uoldAll, poldAll = unewAll, pnewAll

        # store histories
        self.resu = resu
        self.resp = resp

        return uoldAll, poldAll, resu, resp
    
    def TPDv_PDHG(self):
        # initial global arrays
        poldAll = self.p0.copy()
        # enforce pressure reference
        poldAll[-1] = 0.0
        uoldAll = self.u0.copy()

        resu = bm.zeros(self.maxIt)
        resp = bm.zeros(self.maxIt)
        self._update_mass()
        self._update_S()
        mesh = self.u_bform.space.mesh
        Nt = mesh.number_of_cells()
        area = mesh.entity_measure('cell')   # shape (Nt,)
        
        alphau = 0.5
        alphap = 0.5
        
        for ite in range(self.maxIt):
            
            Bp = self.Bfft @ poldAll[self.freep]
            F_half = uoldAll - alphau * Bp/bm.repeat(area, 2) + alphau * self.f/bm.repeat(area, 2)
            F_abs = bm.linalg.norm(F_half.reshape(Nt, 2), axis=1)
            gamma = 0.5*(1 + alphau) + 0.5*bm.sqrt((1+alphau)**2 + 4*alphau*self.pde.beta*F_abs)
            uhalf = F_half/bm.repeat(gamma, 2)
            
            dp_temp = self.g[self.freep] - self.Bff @ uhalf
            pnew = poldAll[self.freep] - alphap*dp_temp[self.freep]
            pnewAll = poldAll.copy()
            pnewAll[self.freep] = pnew
            
            unewAll = uhalf + 0.5*(uoldAll - uhalf)
            
            self.u0[:] = unewAll
            self._update_mass()
    
            resu[ite] = bm.linalg.norm(self.M @ unewAll + (self.Bfft @ pnew) - self.f)
            resp[ite] = bm.linalg.norm(self.Bff @ unewAll - self.g[self.freep])

            if ite > 0 and resu[ite] / bm.linalg.norm(self.f) < self.tol:
                print("Converged at iteration", ite)
                uoldAll, poldAll = unewAll, pnewAll
                break

            uoldAll, poldAll = unewAll, pnewAll
            theta = 1/bm.sqrt(1 + 2*self.gamma0*alphau)
            alphau = theta*alphau
            alphap = alphap/theta
            
        return uoldAll, poldAll, resu, resp
        
        
            