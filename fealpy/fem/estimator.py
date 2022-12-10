import numpy as np
from ..functionspace import LagrangeFiniteElementSpace



class MaxwellNedelecFEMResidualEstimator():
    def __init__(self, uh, pde):
        self.uh = uh
        self.space = uh.space
        self.mesh = self.space.mesh
        self.pde = pde

    def estimate(self, q=1):

        lspace = LagrangeFiniteElementSpace()
        qf = mesh.integrator(q)
        bcs, ws = qf.get_quadrature_points_and_weights()


