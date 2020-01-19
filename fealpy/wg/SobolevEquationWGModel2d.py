import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, bmat

from ..functionspace import WeakGalerkinSpace2d

class SobolevEquationWGModel2d:
    """
    Solve Sobolev equation by weak Galerkin method.

    """
    def __init__(self, pde, mesh, p, q=None):
        self.pde = pde
        self.mesh = mesh
        self.space = WeakGalerkinSpace2d(mesh, p=p, q=q)
        self.construct_marix()

    def init_solution(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = self.space.function(dim=NL)
        uh[:, 0] = self.space.project(lambda x:pde.solution(x, 0.0))

        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            self.space.set_dirichlet_bc(uh[:, i], lambda x: self.pde.dirichlet(x, t))
        return uh

    def construct_marix(self):
        """
        构造 Soblove 方程对应的所有矩阵
        """
        gdof = self.space.number_of_global_dofs()
        cell2dof, cell2dofLocation = self.space.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        H0 = self.space.H0
        R = self.space.R
        def f0(i):
            R0 = R[0][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
            R1 = R[1][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
            return R0.T@H0[i]@R0, R1.T@H0[i]@R1, R0.T@H0[i]@R1

        NC = self.mesh.number_of_cells()
        M = list(map(f0, range(NC)))

        idx = list(map(np.meshgrid, cd, cd))
        I = np.concatenate(list(map(lambda x: x[1].flat, idx)))
        J = np.concatenate(list(map(lambda x: x[0].flat, idx)))

        val = np.concatenate(list(map(lambda x: x[0].flat, M)))
        M00 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(lambda x: x[1].flat, M)))
        M11 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(lambda x: x[2].flat, M)))
        M01 = csr_matrix((val, (I, J)), shape=(gdof, gdof))


        self.D = bmat([[M00, M01], [M01.T, M11]], format='csr') # weak divergence matrix
        self.G = M00 + M11 # weak gradient matrix

        self.S = self.space.stabilizer_matrix()
        self.M = self.space.mass_matrix()

        self.A1 = self.D + bmat([[self.S, None], [None, self.S]], format='csr') + bmat([[self.M, None], [None, self.M]], format='csr')/self.pde.mu
        self.A2 = self.G + self.S

    def project(self, u, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = self.space.function(dim=NL)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            uh[:, i] = self.space.project(lambda x:u(x, t))
        return uh

    def get_current_left_matrix(self, timeline):
        return self.A2

    def get_current_right_vector(self, data, timeline):
        mu = self.pde.mu
        epsilon = self.pde.epsilon

        uh = data[0]
        fh = data[1]
        solver = data[2]
        i = timeline.current

        cell2dof, cell2dofLocation = self.space.cell_to_dof(doftype='all')
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])

        NC = self.mesh.number_of_cells()
        gdof = self.space.number_of_global_dofs()
        c2d = self.space.cell_to_dof(doftype='cell') # only get the dofs in cell
        # (\nabla u, \bfq)
        F0 = np.zeros((gdof, 2), dtype=self.space.ftype)
        f00 = lambda j: self.R[0][:, cell2dofLocation[j]:cell2dofLocation[j+1]]@uh[cd[j], i]
        f01 = lambda j: self.R[1][:, cell2dofLocation[j]:cell2dofLocation[j+1]]@uh[cd[j], i]
        F0[c2d.flat, 0] = np.concatenate(list(map(f00, range(NC))))
        F0[c2d.flat, 1] = np.concatenate(list(map(f01, range(NC))))

        # -(f, \nabla\cdot \bfq)
        F1 = np.zeros((gdof, 2), dtype=self.space.ftype)
        f10 = lambda j: fh[c2d[j, :], i]@self.R[0][:, cell2dofLocation[j]:cell2dofLocation[j+1]]
        f11 = lambda j: fh[c2d[j, :], i]@self.R[1][:, cell2dofLocation[j]:cell2dofLocation[j+1]]
        """
        F1[cell2dof.flat, 0] = np.concatenate(
            list(map(f10, range(NC)))
            )
        F1[c2d.flat, 1] = np.concatenate(
            list(map(f11, range(NC)))
            )
        """

        F1[cell2dof, 0] = list(map(f10, range(NC)))
        F1[cell2dof, 1] = list(map(f11, range(NC)))

        F = epsilon/mu*F0 - F1
        ph = solver(self.A1, F.T.flat).reshape(-1, 2, order='F')

        dt = timeline.current_time_step_length()
        t0 = timeline.current_time_level()
        F = (1- dt*epsilon/mu)*(self.A2@uh[:, i])

        f20 = lambda j: ph[c2d[j, :], 0]@self.R[0][:, cell2dofLocation[j]:cell2dofLocation[j+1]]
        f22 = lambda j: ph[c2d[j, :], 1]@self.R[1][:, cell2dofLocation[j]:cell2dofLocation[j+1]]
        f = lambda j: f20(j) + f22(j)
        F[cell2dof] +=  list(map(f, range(NC)))
        return F

    def solve(self, data, A, b, solver, timeline):
        uh = data[0]
        i = timeline.current
        uh[:, i] = solver(A, b)

    def apply_boundary_condition(self, A, F, timeline):
        t1 = timeline.next_time_level()
        A, F = self.space.apply_dirichlet_bc(lambda x: self.pde.dirichlet(x, t1), A, F)
        return A, F
