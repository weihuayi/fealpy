import numpy as np

import matplotlib.pyplot as plt
from fealpy.mesh import PolygonMesh
from fealpy.vem.LinearElasticityVEMModel import LinearElasticityVEMModel


class LinearElasticityModel():
    def __init__(self, lam, mu):
        self.lam = lam
        self.mu = mu

    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = np.exp(x)
        val[..., 1] = np.exp(y)
        return val

node = np.array([ (0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float)
cell = np.array([0, 1, 2, 3], dtype=np.int)
cellLocation = np.array([0, 4], dtype=np.int)

mesh = PolygonMesh(node, cell, cellLocation)

pde = LinearElasticityModel(1, 1)
p = 2
q = 5
vem = LinearElasticityVEMModel(pde, mesh, p, q)

G = vem.matrix_G()
print(G)

#B = vem.matrix_B()
#print(B)
