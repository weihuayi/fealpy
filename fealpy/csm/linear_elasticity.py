import numpy as np

class LinearElasticity():
    def __init__(self, mesh, model):
        self.lam = model.lam
        self.mu = model.mu
        
        self.mesh = mesh

    def tangent_matrix(self):
        mu = self.mu
        lam = self.lam
        mesh = self.mesh
        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        n = GD*(GD+1)//2
        D = np.zeros((NC, n, n), dtype=np.float_)
        for i in range(n):
            D[:, i, i] += mu
        for i in range(GD):
            for j in range(i, GD):
                if i == j:
                    D[:, i, i] += mu+lam
                else:
                    D[:, i, j] += lam
                    D[:, j, i] += lam
        return D


