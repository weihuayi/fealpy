import numpy as np
from ..solver import MatlabSolver


class TimeIntegrationAlg:
    def __init__(self):
        """
        Parameter
        ---------
        """
        self.solver = MatlabSolver()

    def run(self, dmodel):
        dmodel.timeline.reset()
        while not dmodel.timeline.stop():
            A = dmodel.get_current_left_matrix()
            b = dmodel.get_current_right_vector()
            AD, bd = dmodel.apply_boundary_condition()
            uh = self.solver.divide(AD, bd)
            dmodel.update_solution(uh)
            dmodel.timeline.advance()
