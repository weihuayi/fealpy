import numpy as np
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.timeintegratoralg.timeline_new import UniformTimeLine
from fealpy.timeintegratoralg.timeline_new import ChebyshevTimeLine
from fealpy.boundarycondition import DirichletBC
from fealpy.solver import MatlabSolver
from scipy.sparse.linalg import spsolve


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
        uh[:, 0] = self.space.interpolation(lambda x:self.pde.solution(x, 0.0))
        return uh

    def interpolation(self, u, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        ps = self.space.interpolation_points()
        uI = np.zeros((gdof, NL), dtype=self.mesh.ftype)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            uI[:, i] = u(ps, t)
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
        return self.M@uh[:, i] - 0.5*dt*(self.A@uh[:, i] - F)

    def apply_boundary_condition(self, A, b, timeline):
        t1 = timeline.next_time_level()
        bc = DirichletBC(self.space, lambda x:self.pde.dirichlet(x, t1))
        A, b = bc.apply(A, b)
        return A, b

    def solve(self, uh, A, b, solver, timeline):
        i = timeline.current
        uh[:, i+1] = solver(A, b)

class SurfaceParabolicFEMModel():
    def __init__(self, pde, mesh, p=1, q=6, p0=None):
        from fealpy.functionspace import SurfaceLagrangeFiniteElementSpace
        from fealpy.boundarycondition import DirichletBC
        self.space = SurfaceLagrangeFiniteElementSpace(mesh, pde.surface, p=p,
                p0=p0, q=q)
        self.mesh = self.space.mesh
        self.surface = pde.surface
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

    def init_source(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        F = np.zeros((gdof, NL), dtype=self.mesh.ftype)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            F[:, i] = self.space.source_vector(lambda x: self.pde.source(x, t))
        return F
    
    def init_delta(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        delta = np.zeros((gdof, NL), dtype=self.mesh.ftype)
        return delta

    def interpolation(self, u, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        ps = self.space.interpolation_points()
        uI = np.zeros((gdof, NL), dtype=self.mesh.ftype)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            uI[:, i] = u(ps, t)
        return uI

    def get_current_left_matrix(self, timeline):
        dt = timeline.current_time_step_length()
        return self.M + 0.5*dt*self.A

    def get_current_right_vector(self, uh, timeline, returnF=True):
        dt = timeline.current_time_step_length()
        i = timeline.current
        t0 = timeline.current_time_level()
        t1 = timeline.next_time_level()
        f = lambda x: self.pde.source(x, t0) + self.pde.source(x, t1)
        F = self.space.source_vector(f)
        if returnF is True:
            return self.M@uh[:, i] - 0.5*dt*(self.A@uh[:, i] + F)
        else:
            return self.M@uh[:, i] - 0.5*dt*self.A@uh[:, i]
    
    def get_residual(self, uh, timeline):
        A = self.space.stiff_matrix()
        F = self.init_source(timeline)
        return -A@uh + F

    def get_error(self, data, timeline):
        uh = data[0]
        intq = data[1]
        M = self.space.mass_matrix()
        return uh[:, [0]] + spsolve(M, intq) - uh

    def get_error_right_vector(self, data, timeline):
        delta = data[0]
        uh = data[1]
        d = data[2]
        M = self.space.mass_matrix()
        i = timeline.current
        dt = timeline.current_time_step_length()
        return self.get_current_right_vector(delta, timeline, returnF=False) + dt*M@d[:, i+1]


    def apply_boundary_condition(self, A, b, timeline, returnu=True):
        if returnu is True:
            t1 = timeline.next_time_level()
            bc = DirichletBC(self.space, lambda x:self.pde.solution(x, t1), self.is_boundary_dof)
            A, b = bc.apply(A, b)
            return A, b
        else:
            bc = DirichletBC(self.space, 0, self.is_boundary_dof)
            A, b = bc.apply(A, b)
            return A, b

    def is_boundary_dof(self, p):
        isBdDof = np.zeros(p.shape[0], dtype=np.bool)
        isBdDof[0] = True
        return isBdDof

    def solve(self, uh, A, b, solver, timeline):
        i = timeline.current
        uh[:, i+1] = solver(A, b)

class TimeIntegratorAlgTest():
    def __init__(self):
        self.solver = MatlabSolver()

    def test_ParabolicFEMModel_time(self, maxit=4):
        from fealpy.pde.parabolic_model_2d import SinSinExpData
        pde = SinSinExpData()
        domain = pde.domain()
        mesh = triangle(domain, h=0.01)
        timeline = pde.time_mesh(0, 1, 2)
        error = np.zeros(maxit, dtype=mesh.ftype)
        for i in range(maxit):
            print(i)
            dmodel = ParabolicFEMModel(pde, mesh)
            uh = dmodel.init_solution(timeline)

            timeline.time_integration(uh, dmodel, self.solver.divide)

            uI = dmodel.interpolation(timeline)

            error[i] = np.max(np.abs(uh - uI))

            timeline.uniform_refine()
            mesh.uniform_refine()

        print(error[:-1]/error[1:])
        print(error)

    def test_SurfaceParabolicFEMModel_time(self, maxit=4):
        from fealpy.pde.surface_parabolic_model_3d import SinSinSinExpDataSphere
        pde = SinSinSinExpDataSphere()
        mesh = pde.init_mesh(n=5)
        timeline = pde.time_mesh(0, 1, 2)
        error = np.zeros(maxit, dtype=mesh.ftype)
        for i in range(maxit):
            print(i)
            dmodel = SurfaceParabolicFEMModel(pde, mesh)
            uh = dmodel.init_solution(timeline)

            timeline.time_integration(uh, dmodel, self.solver.divide)

            uI = dmodel.interpolation(pde.solution, timeline)
            error[i] = np.max(np.abs(uh - uI))

            timeline.uniform_refine()
            mesh.uniform_refine(surface=pde.surface)

        print(error[:-1]/error[1:])
        print(error)

    def test_SurfaceParabolicFEMModel_sdc_time(self, maxit=1):
        from fealpy.pde.surface_parabolic_model_3d import SinSinSinExpDataSphere
        pde = SinSinSinExpDataSphere()
        mesh = pde.init_mesh(n=5)
        timeline = pde.time_mesh(0, 1, 2, timeline='chebyshev')
        error = np.zeros(maxit, dtype=mesh.ftype)
        for i in range(maxit):
            print(i)
            dmodel = SurfaceParabolicFEMModel(pde, mesh)

            uh = dmodel.init_solution(timeline)

            timeline.time_integration(uh, dmodel, self.solver.divide)

            intq = timeline.residual_integration(uh, dmodel)

            d = timeline.error([uh, intq], dmodel)
            delta = dmodel.init_delta(timeline) 
            timeline.error_integration([delta, uh, d], dmodel, self.solver.divide)


            uI = dmodel.interpolation(pde.solution, timeline)
            error[i] = np.max(np.abs(uh - uI))

            timeline.uniform_refine()
            mesh.uniform_refine(surface=pde.surface)

        print(error[:-1]/error[1:])
        print(error)



test = TimeIntegratorAlgTest()
test.test_SurfaceParabolicFEMModel_sdc_time()

 
