import numpy as np

import matplotlib.pyplot as plt
from heatequation_2d import CosCosExpData
from fealpy.timeintegratoralg.timeline_new import UniformTimeLine
from fealpy.boundarycondition import BoundaryCondition
from scipy.sparse.linalg import spsolve
from fealpy.tools.show import showmultirate, show_error_table

class ParabolicFEMModel():
    def __init__(self, pde, mesh, p=1, q=6):
        from fealpy.functionspace import LagrangeFiniteElementSpace
        from fealpy.boundarycondition import BoundaryCondition
        self.space = LagrangeFiniteElementSpace(mesh, p)
        self.mesh = self.space.mesh
        self.pde = pde

        self.ftype = self.mesh.ftype
        self.itype = self.mesh.itype

        self.M = self.space.mass_matrix()
        self.A = self.space.stiff_matrix()

    def init_solution(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = np.zeros((gdof, NL), dtype=self.mesh.ftype)
        uh[:, 0] = self.space.interpolation(lambda x:self.pde.solution(x, 0.0))
        return uh

    def interpolation(self, u, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        ps = self.space.interpolation_points()
        uI = np.zeros((gdof, NL), dtype=self.mesh.ftype)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            uI[..., i] = u(ps, t)
        return uI

    def get_current_left_matrix(self, timeline):
        dt = timeline.current_time_step_length()
        return self.M + 0.5*dt*self.A

    def get_current_right_vector(self, uh, timeline):
        i = timeline.current
        dt = timeline.current_time_step_length()
        t0 = timeline.current_time_level()
        t1 = timeline.next_time_level()
        f = lambda x: self.pde.source(x, t0) + self.pde.source(x, t1)
        F = self.space.source_vector(f)
        return self.M@uh[:, i] - 0.5*dt*self.A@uh[:, i]

    def apply_boundary_condition(self, A, b, timeline):
        t1 = timeline.next_time_level()
        bc = BoundaryCondition(self.space, neuamn = lambda x:self.pde.neuman(x, t1))
        b = bc.apply_neuman_bc(b)
        return A, b

    def solve(self, uh, A, b, solver, timeline):
        i = timeline.current
        uh[:, i+1] = solver(A, b)

class TimeIntegratorAlgTest():
    def __init__(self):
        pass

    def test_ParabolicFEMModel_time(self, maxit=4):
        pde = CosCosExpData(0.1)
        mesh = pde.init_mesh(n=0)
        timeline = UniformTimeLine(0, 1, 2)

        errorType = ['$|| u - u_h ||_\infty$', '$||u-u_h||_0$']
        errorMatrix = np.zeros((len(errorType), maxit), dtype=mesh.ftype)
        Ndof = np.zeros(maxit, dtype=mesh.itype)
        for i in range(maxit):
            print(i)
            model = ParabolicFEMModel(pde, mesh, p=1)
            Ndof[i] = model.space.number_of_global_dofs()

            uh = model.init_solution(timeline)

            timeline.time_integration(uh, model, spsolve)
            uI = model.interpolation(pde.solution, timeline)
            errorMatrix[0, i] = np.max(np.abs(uI - uh))

            u = lambda x:model.pde.solution(x, 1.0)
            uh = model.space.function(array=uh[:, -1])
            errorMatrix[1, i] = model.space.integralalg.L2_error(u, uh)
            print(errorMatrix)

            timeline.uniform_refine()
            mesh.uniform_refine(surface=pde.surface)

        show_error_table(Ndof, errorType, errorMatrix)
        showmultirate(plt, 0, Ndof, errorMatrix, errorType)
        plt.show()

test = TimeIntegratorAlgTest()
test.test_ParabolicFEMModel_time()
