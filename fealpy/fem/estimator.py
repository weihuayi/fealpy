import numpy as np
from ..functionspace import LagrangeFiniteElementSpace



class MaxwellNedelecFEMResidualEstimator2d():
    def __init__(self, uh, pde):
        self.uh = uh
        self.space = uh.space
        self.mesh = self.space.mesh
        self.pde = pde

    def estimate(self, q=1):
    
        p = 2
        lspace = LagrangeFiniteElementSpace(self.mesh, p=2, spacetype='D')

        luh = lspace.function()

        qf = mesh.integrator(q)
        bcs, ws = qf.get_quadrature_points_and_weights()

        # 0. R_K^1 

        # 0.0 构造一个 L2 的函数 

        ib = self.mesh.multi_index_matrix(p)/p
        ps = self.mesh.bc_to_point(ibcs)
        luh[:] = self.uh.curl_value(ibcs).flat/pde.mu(ps).flat



        


        # R_K^2 


        # J_e^1

        # J_e^2


class MaxwellNedelecFEMResidualEstimator3d():
    def __init__(self, uh, pde):
        self.uh = uh
        self.space = uh.space
        self.mesh = self.space.mesh
        self.pde = pde

    def estimate(self, q=1):

        lspace = LagrangeFiniteElementSpace(self.mesh, p=1, spacetype='D')

        qf = mesh.integrator(q)
        bcs, ws = qf.get_quadrature_points_and_weights()

