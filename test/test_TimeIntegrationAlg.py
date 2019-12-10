from fealpy.timeintegratoralg.timeline_new import UniformTimeLine,
from fealpy.timeintegratoralg.timeline_new import ChebyshevTimeLine
from fealpy.timeintegratoralg.TimeIntegrationAlg import TimeIntegrationAlg


class ParabolicFEMModel():
    def __init__(self, mesh, pde, timeline, p=1, q=6):
        from fealpy.functionspace import LagrangeFiniteElementSpace
        from fealpy.boundarycondition import DirichletBC
        self.space = LagrangeFiniteElementSpace(mesh, p, q)
        self.mesh = self.space.mesh
        self.pde = pde
        self.timeline = timeline
        NL = self.timeline.get_number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        self.uh = np.zeros((gdof, NL), dtype=mesh.ftype)
        self.uI = np.zeros((gdof, NL), dtype=mesh.ftype)
        self.M = self.space.mass_matrix()
        self.A = self.space.stiff_matrix()

    def get_current_left_matrix(self):
        dt = self.timeline.get_current_time_step_length()
        return self.M + 0.5*dt*self.A

    def get_current_right_vector(self):
        current = self.timeline.current
        dt = self.timeline.get_current_time_step_length()
        F = self.space.source_vector(
                lambda x:self.pde.source(x, self.timeline.get_next_time_level())
                )
        return ( self.M@self.uh[:, current] -
                0.5*dt*self.A@self.uh[:, current] +
                dt*F
                )

class SurfaceParabolicFEMModel():
    def __init__(self):
        pass

class TimeIntegrationAlgTest():
    def __init__(self):
        pass
