import numpy as np
from fealpy.timeintegratoralg.timeline_new import UniformTimeLine
from fealpy.timeintegratoralg.timeline_new import ChebyshevTimeLine
from fealpy.boundarycondition import DirichletBC
from fealpy.solver import MatlabSolver


class ParabolicFEMModel():
    def __init__(self, pde, mesh, p=1, q=6):
        from fealpy.functionspace import LagrangeFiniteElementSpace
        from fealpy.boundarycondition import DirichletBC
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
        return uh

    def interpolation(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        ps = self.space.interpolation_points()
        uI = np.zeros((gdof, NL), dtype=self.mesh.ftype)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            uI[:, i] = self.pde.solution(ps, t)
        return uI

    def get_current_left_matrix(self, timeline):
        dt = timeline.current_time_step_length()
        return self.M + 0.5*dt*self.A

    def get_current_right_vector(self, uh, timeline):
        dt = timeline.current_time_step_length()
        t0 = timeline.current_time_level()
        t1 = timeline.next_time_level()
        f0 = lambda x: self.pde.source(x, t0) + self.pde.source(x, t1)
        F = self.space.source_vector(f0)
        return self.M@uh - 0.5*dt*(self.A@uh -F)

    def apply_boundary_condition(self, A, b, timeline):
        t1 = timeline.next_time_level()
        bc = DirichletBC(self.space, lambda x:self.pde.dirichlet(x, t1))
        A, b = bc.apply(A, b)
        return A, b

class SurfaceParabolicFEMModel():
    def __init__(self):
        pass

class TimeIntegratorAlgTest():
    def __init__(self):
        self.solver = MatlabSolver()

    def test_ParabolicFEMModel_time(self, maxit=4):
        from fealpy.pde.parabolic_model_2d import SinSinExpData
        pde = SinSinExpData()
        mesh = pde.init_mesh(10)
        timeline = pde.time_mesh(0, 1, 10)
        dmodel = ParabolicFEMModel(pde, mesh)
        error = np.zeros(maxit, dtype=mesh.ftype)
        for i in range(maxit):
            print(i)
            uh = dmodel.init_solution(timeline)
            uI = dmodel.interpolation(timeline)
            uh[:, 0] = uI[:, 0]
            timeline.time_integration(uh, dmodel, self.solver.divide)
            error[i] = np.max(np.abs(uh - uI))
            timeline.uniform_refine()

        print(error[:-1]/error[1:])
        print(error)



test = TimeIntegratorAlgTest()
test.test_ParabolicFEMModel_time()

 
