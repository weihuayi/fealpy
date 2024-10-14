import numpy as np
from ..solver import MatlabSolver


class TimeIntegrationAlg:
    def __init__(self):
        """
        Parameter
        ---------
        """
        self.solver = MatlabSolver()

    def run(self, uh, dmodel, timeline):
        timeline.reset()
        NL = timeline.number_of_time_levels()
        for i in range(NL-1):
            A = dmodel.get_current_left_matrix(timeline)
            b = dmodel.get_current_right_vector(uh[:, i], timeline)
            AD, bd = dmodel.apply_boundary_condition()
            uh[:, i+1] = self.solver.divide(AD, bd)
            timeline.advance()
        timeline.reset()
