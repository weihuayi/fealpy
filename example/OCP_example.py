
import numpy as np


class Model():
    def __init__(self, pde, mesh, timeline):
        self.pde = pde
        self.mesh = mesh
        self.uspace = RaviartThomasFiniteElementSpace2d(mesh, p=0)
        self.pspace = self.uspace.smspace 

        self.timeline = timeline
        NL = timeline.number_of_time_levels()

        # state variable
        self.yh = self.pspace.function(dim=NL)
        self.uh = self.pspace.function(dim=NL)
        self.tph = self.uspace.function(dim=NL)
        self.ph = self.uspace.function()

        # costate variable

        self.A = self.uspace.stiff_matrix()
        self.D = self.uspace.div_matrix() # TODO: 确定符号
        self.M = self.pspace.cell_mass_matrix() # 每个单元上是 1 x 1 的矩阵

    def get_state_current_right_vector(self):
        pass

    def get_costate_current_right_vector(self):
        pass

    def state_one_step_solve(self):
        pass

    def costate_one_step_solve(self):
        pass

    def state_solve(self):
        timeline = self.timeline
        timeline.reset()
        while not timeline.stop():
            self.state_solve()
            timeline.current += 1
        timeline.reset()

    def costate_solve(self):
        timeline = self.timeline
        timeline.reset()
        while not timeline.stop():
            self.state_solve()
            timeline.current += 1
        timeline.reset()
        pass

    def nonlinear_solve(self, maxit=1000):




mesh = ...
timeline = ....
state = StateModel(pde, mesh, timeline)




