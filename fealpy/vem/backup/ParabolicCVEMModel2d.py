import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace import ConformingVirtualElementSpace2d

class ParabolicCVEMModel2d():

    def __init__(self, pde, mesh, p=1, q=3):
        self.space = ConformingVirtualElementSpace2d(mesh, p, q)
        self.mesh = self.space.mesh
        self.pde = pde

    def solve(self, t0, t1, NT):
        """
        Parameters
        ----------
        t0 : the start time
        t1 : the stop time
        NT : the number of segments on [t0, t1]
        """
        A = self.space.stiff_matrix()
        M = self.space.mass_matrix()
        timeline = self.pde.time_mesh(t0, t1, NT)
        dt = timeline.get_current_time_step_length()
