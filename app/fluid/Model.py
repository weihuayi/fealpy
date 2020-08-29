import numpy as np

from fealpy.mesh import LagrangeQuadrangleMesh
from fealpy.mesh import MeshFactory


class TriplePointShockInteractionModel:

    def __init__(self):
        self.domain = [0, 7, 0, 3]

    def init_mesh(self, n, p):
        mf = MeshFactory()
        mesh = mf.boxmesh2d(self.domain, nx=70, ny =30, p=p) 
        mesh.uniform_refine(n)
        return mesh

    def init_rho(self, p):
        x = p[..., 0]
        y = p[..., 1]

        rho = np.zeros(p.shape[:-1], dtype=p.dtype)
        rho[x < 1] = 1.0
        rho[(x > 1) & (y < 1.5)] = 1
        rho[(x > 1) & (y > 1.5)] = 0.125
        return val

    def init_velocity(self, p):
        val = np.array([0.0, 0.0], dtype=p.dtype)
        shape = (len(p.shape) - 1)*(1, ) + (-1, )
        return val.reshape(shape)
        

    def init_pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape[:-1], dtype=p.dtype)
        val[x < 1] = 1.0
        val[(x > 1) & (y < 1.5)] = 0.1
        val[(x > 1) & (y > 1.5)] = 0.1 
        return val

    def adiabatic_inex(self, p):
        """
        Notes
        -----
        绝热指数
        """
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape[:-1], dtype=p.dtype)
        val[x < 1] = 1.5
        val[(x > 1) & (y < 1.5)] = 1.4 
        val[(x > 1) & (y > 1.5)] = 1.5 
        return val



