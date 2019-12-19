import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, bmat

from ..functionspace import WeakGalerkinSpace2d

class SobolevEquationWGModel2d:
    def __init__(self, pde, mesh, p, q=None):
        self.pde = pde
        self.mesh = mesh
        self.space = WeakGalerkinSpace2d(mesh, p=p, q=q)
        self.construct_marix()

    def init_solution(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = self.space.function(dim=NL)
        return uh

    def construct_marix(self):
        """
        构造 Soblove 方程对应的所有矩阵
        """
        gdof = self.space.number_of_global_dofs()
        cell2dof, cell2dofLocation = self.space.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = np.hsplit(self.space.R0, cell2dofLocation[1:-1])
        R1 = np.hsplit(self.space.R1, cell2dofLocation[1:-1])

        H0 = self.space.H0

        f0 = lambda x: x[0].T@x[1]@x[2]
        M00 = list(map(f0, zip(R0, H0, R0)))
        M01 = list(map(f0, zip(R0, H0, R1)))
        M11 = list(map(f0, zip(R1, H0, R1)))

        f1 = lambda x: np.repeat(x, x.shape[0])
        f2 = lambda x: np.tile(x, x.shape[0])
        f3 = lambda x: x.flat

        I = np.concatenate(list(map(f1, cd)))
        J = np.concatenate(list(map(f2, cd)))

        val = np.concatenate(list(map(f3, M00)))
        M00 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(f3, M01)))
        M01 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(f3, M11)))
        M11 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        self.D = bmat([[M00, M01], [M01.T, M11]], format='csr') # weak divergence matrix
        self.G = M00 + M11 # weak gradient matrix


        self.S = self.space.stabilizer_matrix()
        self.M = self.space.mass_matrix()

        self.A1 = self.D + bmat([[self.S, None], [None, self.S]], format='csr') + bmat([[self.M, None], [None, self.M]], format='csr')/self.pde.mu
        self.A2 = self.G + self.S

    def solution_projection(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = self.space.function(dim=NL)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            uh[:, i] = self.space.projection(lambda x:self.pde.solution(x, t))
        return uh

    def source_projection(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        fh = self.space.function(dim=NL)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            fh[:, i] = self.space.projection(lambda x:self.pde.source(x, t))
        return fh

    def get_current_left_matrix(self, timeline):
        return self.A2

    def get_current_right_vector(self, data, timeline):
        mu = self.pde.mu
        epsilon = self.pde.epsilon

        uh = data[0]
        fh = data[1]
        solver = data[2]
        i = timeline.current

        cell2dof, cell2dofLocation = self.cell_to_dof(doftype='all')
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = self.space.R0
        R1 = self.space.R1

        c2d = self.space.cell_to_dof(doftype='cell') # only get the dofs in cell
        # epsilon/mu(\nabla u, \bfq)
        F0 = np.zeros((gdof, 2), dtype=self.space.ftype)
        f00 = lambda j: R0[:, cell2dofLocation[j]:cell2dofLocation[j+1]]@uh[cd[j], i]
        f01 = lambda j: R1[:, cell2dofLocation[j]:cell2dofLocation[j+1]]@uh[cd[j], i]
        F0[c2d.flat, 0] = np.concatenate(list(map(f00, range(NC))))
        F0[c2d.flat, 1] = np.concatenate(list(map(f01, range(NC))))

        # -(f, \nabla\cdot \bfq)
        F1 = np.zeros((gdof, 2), dtype=self.space.ftype)
        f10 = lambda j: fh[c2d[j, :], i]@R0[:, cell2dofLocation[j]:cell2dofLocation[j+1]]
        f11 = lambda j: fh[c2d[j, :], i]@R1[:, cell2dofLocation[j]:cell2dofLocation[j+1]]
        F1[cell2dof.flat, 0] = np.concatenate(
            list(map(f10, range(NC)))
            )
        F1[cell2dof.flat, 1] = np.concatenate(
            list(map(f11, range(NC)))
            )

        F = epsilon/mu*F0 - F1
        ph = solver(self.A1, F.reshape(-1, order='F')).reshape(-1, 2, order='F')

        dt = timeline.current_time_step_length()
        t0 = timeline.current_time_level()
        F = (1- dt*epsilon/mu)*(self.A2@uh[:, i])

        f20 = lambda j: qh[c2d[j, :], 0]@R0[:, cell2dofLocation[j]:cell2dofLocation[j+1]]
        f22 = lambda j: qh[c2d[j, :], 1]@R1[:, cell2dofLocation[j]:cell2dofLocation[j+1]]
        f = lambda j: f20(j) + f22(j)
        F[cell2dof] +=  np.concatenate(list(map(f, range(NC))))
        return F

    def apply_boundary_condition(self, A, F, timeline):
        t1 = timeline.next_time_level()
        A, F = self.space.apply_dirichlet_bc(lambda x: self.pde.dirichlet(x, t1), A, F)
        return A, F
