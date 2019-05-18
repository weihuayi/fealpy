import numpy as np
from fealpy.mesh.TetrahedronMesh import TetrahedronMesh


class CosCosCosData:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tet'):
        node = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]], dtype=np.float)

        cell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int)
        mesh = TetrahedronMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    def solution(self, p):
        """ the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
        return u

    def gradient(self, p):
        """ The gradient of the exact solution
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = -pi*sin(pi*x)*cos(pi*y)*cos(pi*z)
        val[..., 1] = -pi*cos(pi*x)*sin(pi*y)*cos(pi*z)
        val[..., 2] = -pi*cos(pi*x)*cos(pi*y)*sin(pi*z)
        return val

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = 3*np.pi**2*np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
        return val

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)


class LShapeRSinData:
    def __init__(self):
        pass

    def init_mesh(self, n=2, meshtype='tet'):
        node = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]], dtype=np.float)

        cell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int)
        mesh = TetrahedronMesh(node, cell)
        for i in range(n):
            mesh.bisect()

        NN = mesh.number_of_nodes()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        bc = mesh.entity_barycenter('cell')
        isDelCell = ((bc[:, 0] > 0) & (bc[:, 1] < 0))
        cell = cell[~isDelCell]
        isValidNode = np.zeros(NN, dtype=np.bool)
        isValidNode[cell] = True
        node = node[isValidNode]

        idxMap = np.zeros(NN, dtype=mesh.itype)
        idxMap[isValidNode] = range(isValidNode.sum())
        cell = idxMap[cell]
        mesh = TetrahedronMesh(node, cell)

        return mesh

    def solution(self, p):
        """ the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta[theta < 0] += 2*pi
        u = r**(2/3)*np.sin(2*theta/3)
        return u

    def gradient(self, p):
        """ The gradient of the exact solution
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos

        x = p[..., 0]
        y = p[..., 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta + 2*pi)

        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = (
                2/3*r**(-1/3)*sin(2*theta/3)*x/r -
                2/3*r**(2/3)*cos(2*theta/3)*y/r**2
                )
        val[..., 1] = (
                2/3*r**(-1/3)*sin(2*theta/3)*y/r +
                2/3*r**(2/3)*cos(2*theta/3)*x/r**2
                )
        return val

    def source(self, p):
        val = np.zeros(p.shape[:-1], dtype=p.dtype)
        return val

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)
