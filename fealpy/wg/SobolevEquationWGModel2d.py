import numpy as np
from scipy.sparse import coo_matrix, bmat

from ..functionspace import WeakGalerkinSpace2d

class SobolevEquationWGModel2d:
    def __init__(self, pde, mesh, p, q=None):
        self.pde = pde
        self.mesh = mesh
        self.space = WeakGalerkinSpace2d(mesh, p=p, q=q)

    def init_solution(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = self.space.function(dim=NL)
        return uh

    def construct_marix(self):
        gdof = self.space.number_of_global_dofs()
        cell2dof, cell2dofLocation = self.cell_to_dof()
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
        M00 = coo_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(f3, M01)))
        M01 = coo_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(f3, M11)))
        M11 = coo_matrix((val, (I, J)), shape=(gdof, gdof))

        D = bmat([[M00, M01], [M01.T, M11]], format='csr')
        return M

    def grad_matrix(self):
       pass 

    def projection(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uI = self.space.function(dim=NL)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            uI[:, i] = self.space.projection(lambda x:self.pde.solution(x, t)) 
        return uI

    def get_current_left_matrix(self, timeline):
        dt = timeline.current_time_step_length()
        return self.M + 0.5*dt*self.A

    def get_current_right_vector(self, uh, timeline):
        dt = timeline.current_time_step_length()
        t0 = timeline.current_time_level()
        t1 = timeline.next_time_level()
        f0 = lambda x: self.pde.source(x, t0) + self.pde.source(x, t1)
        #f0 = lambda x: self.pde.source(x, t1)
        F = self.space.source_vector(f0)
        return self.M@uh - 0.5*dt*(self.A@uh - F)

    def apply_boundary_condition(self, A, b, timeline):
        t1 = timeline.next_time_level()
        bc = DirichletBC(self.space, lambda x:self.pde.dirichlet(x, t1))
        A, b = bc.apply(A, b)
        return A, b
