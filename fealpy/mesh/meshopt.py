import numpy as np


class OptAlg():

    def __init__(self, mesh, quality):
        self.mesh = mesh
        self.quality = quality 

    def run(self, maxit=10):
        i = 0
        while i < maxit:
            try:
            except StopIteration:
                break

    def jacobi(self):
        mesh = self.mesh
        quality = self.quality

        point = mesh.point
        N = mesh.number_of_points()


