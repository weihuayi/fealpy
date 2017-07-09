import numpy as np
from scipy.sparse import triu, tril


class OptAlg():

    def __init__(self, mesh, quality):
        self.mesh = mesh
        self.quality = quality 

    def run(self, maxit=10):
        i = 0
        while i < maxit:
            try:
                i += 1
                self.jacobi()
            except StopIteration:
                break

    def jacobi(self):
        mesh = self.mesh
        quality = self.quality

        point = mesh.point
        N = mesh.number_of_points()
        F, gradF, A, B = quality.objective_function(mesh)
        C = -(triu(A, 1) + tril(A, -1))
        D = A.diagonal()
        X = (C@point[:, 0] + B@point[:, 1])/D
        Y = (C@point[:, 1] - B@point[:, 0])/D
        isBdPoint = mesh.ds.boundary_point_flag()
        mesh.point[~isBdPoint, 0] = X[~isBdPoint]
        mesh.point[~isBdPoint, 1] = Y[~isBdPoint]
        print("Value of Obj function:", F)
        print("The norm of gradient:", np.linalg.norm(gradF)) 



