import numpy as np
from fealpy.mesh.TetrahedronMesh import TetrahedronMesh


class EigenLShape3d:
    """
    example 4.4

    -\Delta u = \lambda u in \Omega
    u = 0 on \partial \Omega
    """

    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tet'):
        node = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]], dtype=np.float)

        cell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int)
        mesh = TetrahedronMesh(node, cell)
        mesh.uniform_bisect(n)
        mesh.delete_cell(lambda bc: (bc[:, 0] < 0.5) & (bc[:, 1] < 0.5))
        return mesh

    def solution(self, p):
        return 0

    def Dirichlet(self, p):
        return self.solution(p)


class EigenHarmonicOscillator3d:
    """
    example 4.5

    -1/2*\Delta u + 1/2*|x|^2 u = \lambda u  in R^3

    lambda = 1.5

    u = c exp(-|x|^2/2)
    """
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tet'):
        """L-shape domain
        \Omega = [-5.5, 5.5]*[-5.5, 5.5]*[-5.5, 5.5]\[-5.5, 0)*[-5.5, 0)*[-5.5,
        5.5]
        """
        node = np.array([
            [-5.5, -5.5, -5.5],
            [5.5, -5.5, -5.5],
            [5.5, 5.5, -5.5],
            [-5.5, 5.5, -5.5],
            [-5.5, -5.5, 5.5],
            [5.5, -5.5, 5.5],
            [5.5, 5.5, 5.5],
            [-5.5, 5.5, 5.5]], dtype=np.float)

        cell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int)
        mesh = TetrahedronMesh(node, cell)
        mesh.uniform_bisect(n)
        return mesh

    def diffusion_coefficient(self, p):
        return 0.5

    def reaction_coefficient(self, p):
        return 0.5*np.sum(p**2, axis=-1)

    def solution(self, p):
        val = np.exp(-np.sum(p**2, axis=-1)/2)
        return val

    def smallest_eignvalue(self):
        return 1.5

    def Dirichlet(self, p):
        return self.solution(p)


class EigenSchrodinger3d:
    """
    example  4.6

    \init_R^3  |u|^2 dx = 1
    \lambda_n = -1/(2*n*n)
    """
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tet'):
        node = np.array([
            [-20, -20, -20],
            [20, -20, -20],
            [20, 20, -20],
            [-20, 20, -20],
            [-20, -20, 20],
            [20, -20, 20],
            [20, 20, 20],
            [-20, 20, 20]], dtype=np.float)

        cell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int)
        mesh = TetrahedronMesh(node, cell)
        mesh.uniform_bisect(n)
        return mesh

    def solution(self, p):
        return 0

    def diffusion_coefficient(self, p):
        return 0.5

    def reaction_coefficient(self, p):
        return -1/np.sqrt(np.sum(p**2, axis=-1))

    def Dirichlet(self, p):
        return self.solution(p)
