from ..backend import backend_manager as bm
from ..sparse import csr_matrix, spdiags
from ..operator import LinearOperator
from ..solver import gmres, minres, GAMGSolver, DirectSolverManager
from ..fem import ScalarMassIntegrator, BilinearForm
from ..functionspace import LagrangeFESpace

class LinearElasticityHZFEMFastSolve:
    
    def __init__(self, A_full, F, vspace, solver: str = 'gmres', rtol: float = 1e-8, restart: int = 20, maxit: int = None):
        '''
        Solve the mixed linear elasticity system given full matrix A and RHS F.

        System: [M, B^T; B, 0] [x0; x1] = [F0; F1]

        Parameters
        ----------
        A_full : csr_matrix
            Block matrix of shape (ndof, ndof) partitioned as [M, B^T; B, 0].
        F : TensorLike
            Right-hand side vector concatenated as [F0; F1].
        vspace : LagrangeFESpace
            Finite element space for displacement, used to build AMG.
        solver : {'gmres', 'minres'}
            Krylov solver to use.
        tol : float
            Convergence tolerance.
        restart : int
            Restart parameter for GMRES.
        maxiter : int or None
            Maximum number of iterations.
        '''
        self.A_full = A_full
        self.F = F
        self.vspace = vspace
        self.solver = solver.lower()
        self.rtol = rtol
        self.restart = restart
        self.maxit = maxit

        # Determine block sizes: total dofs and stress dofs
        ndof = A_full.shape[0]
        mesh = vspace.mesh
        gdim = mesh.geo_dimension()
        vgdof = vspace.number_of_global_dofs() * gdim
        tgdof = ndof - vgdof
        self.tgdof = tgdof
        self.vgdof = vgdof
        self.gdim = gdim

        # Extract sub-blocks M and B
        # A_full = [M, B^T; B, 0]
        self.M = self.A_full[:tgdof, :tgdof]
        B_block = self.A_full[tgdof:, :tgdof]
        self.B = B_block  # B maps stress->displacement

        # Build Schur complement-like matrix S
        D = self.M.diags().values
        self.D = D
        S = self.B @ spdiags(1 / D, 0, tgdof, tgdof) @ self.B.T
        self.S = S

        # Setup AMG for coarse correction
        cspace = LagrangeFESpace(mesh, 1)
        bform = BilinearForm(cspace)
        bform.add_integrator(ScalarMassIntegrator(q=3))
        S_coarse = bform.assembly()
        self.ml = GAMGSolver(isolver='MG', ptype='V', sstep=3, theta=0.25)
        self.ml.setup(S_coarse)

        # Build interpolation matrix PI
        NC = mesh.number_of_cells()
        bc = vspace.dof.multiIndex / vspace.p
        entries = bm.tile(bc, (NC, 1))
        fldof = vspace.number_of_local_dofs()
        cldof = cspace.number_of_local_dofs()
        fgdof = vspace.number_of_global_dofs()
        cgdof = cspace.number_of_global_dofs()
        I = bm.broadcast_to(vspace.cell_to_dof()[:, :, None], (NC, fldof, gdim+1))
        J = bm.broadcast_to(cspace.cell_to_dof()[:, None, :], (NC, fldof, cldof))
        self.PI = csr_matrix((entries.ravel(), (I.ravel(), J.ravel())), shape=(fgdof, cgdof))

    def linear_operator(self, b):
        '''Apply full operator A_full to vector b.'''
        return self.A_full@b

    def gmres_preconditioner(self, r):
        '''GMRES preconditioner: block L, AMG V-cycle, block U.'''
        m, gdim = self.tgdof, self.gdim
        r1 = r[m:]

        # Block L solve
        u0 = r[:m] / self.D
        u1 = bm.zeros_like(r1)
        r1 -= self.B @ u0
        dsm = DirectSolverManager()
        dsm.set_matrix(self.S.tril(), matrix_type='L')
        for _ in range(10):
            u1 += dsm.solve(r1 - self.S @ u1)

        # AMG V-cycle
        r2 = r1 - self.S @ u1
        for i in range(gdim):
            crm = self.PI.T @ r2[i::gdim]
            delta, _ = self.ml.solve(crm)
            u1[i::gdim] += self.PI @ delta

        # Block U solve
        dsm.set_matrix(self.S.triu(), matrix_type='U')
        for _ in range(10):
            u1 += dsm.solve(r1 - self.S @ u1)

        return bm.concatenate([u0 + (self.B.T @ u1) / self.D, -u1])

    def minres_preconditioner(self, r):
        '''MINRES preconditioner: block L, AMG V-cycle, block U.'''
        m, gdim = self.tgdof, self.gdim
        r1 = r[m:]

        # Block L solve
        u0 = r[:m] / self.D
        u1 = bm.zeros_like(r1)
        dsm = DirectSolverManager()
        dsm.set_matrix(self.S.tril(), matrix_type='L')
        for _ in range(30):
            u1[:] += dsm.solve(r1 - self.S @ u1)

        # AMG V-cycle
        r2 = r1 - self.S @ u1
        for i in range(gdim):
            crm = self.PI.T @ r2[i::gdim]
            delta, _ = self.ml.solve(crm)
            u1[i::gdim] += self.PI @ delta

        # Block U solve
        dsm.set_matrix(self.S.triu(), matrix_type='U')
        for _ in range(30):
            u1[:] += dsm.solve(r1 - self.S @ u1)

        return bm.concatenate([u0, u1])

    def solve(self):
        '''Solve the system using selected iterative solver.'''
        ndof = self.A_full.shape[0]
        A_op = LinearOperator((ndof, ndof), matvec=self.linear_operator)

        if self.solver == 'gmres':
            P_op = LinearOperator((ndof, ndof), matvec=self.gmres_preconditioner)
            func, opts = gmres, dict(restart=self.restart, rtol=self.rtol, maxit=self.maxit)
        elif self.solver == 'minres':
            P_op = LinearOperator((ndof, ndof), matvec=self.minres_preconditioner)
            func, opts = minres, dict(rtol=self.rtol, maxit=self.maxit)
        else:
            raise ValueError(f"Unknown solver '{self.solver}'")

        solution, info = func(A_op, self.F, M=P_op, **opts)

        return solution, info
