import numpy as np
from scipy.sparse import triu, tril


def tri_odt_opt(mesh):
    pass


class OptAlg():
    def __init__(self, mesh, quality):
        self.mesh = mesh
        self.quality = quality 

    def run(self, maxit=10):
        i = 0
        mesh = self.mesh
        quality = self.quality
        isFreeNode = ~(mesh.ds.boundary_point_flag())
        qmax = np.max(quality.quality(mesh.point, mesh.ds.cell))
        while i < maxit:
            try:
                i += 1
                alpha = 1
                F, gradF = quality.objective_function(mesh.point, mesh.ds.cell)
                newPoint = mesh.point.copy()
                newPoint[isFreeNode] = mesh.point[isFreeNode] - alpha*gradF[isFreeNode] 
                while ~quality.is_valid(newPoint, mesh.ds.cell) or np.max(quality.quality(newPoint, mesh.ds.cell)) > qmax: 
                    alpha /= 2
                    newPoint[isFreeNode] = mesh.point[isFreeNode] - alpha*gradF[isFreeNode] 
                print(alpha)
                mesh.point = newPoint
            except StopIteration:
                break



