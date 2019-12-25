import numpy as np

from ..mesh.PolygonMesh import PolygonMesh
from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.TriangleMesh import TriangleMesh, TriangleMeshWithInfinityNode

class SinSinData:
    def __init__(self, box, alpha, nu):
        self.alpha = alpha
        self.nu = nu
        self.box = box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype is 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'quad':
            nx = 4
            ny = 4
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'Poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 

    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros((len(x),2), dtype=np.float)
        val[:,0] = np.sin(x)*np.sin(y)
        val[:,1] = np.cos(x)*np.cos(y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 2*self.alpha*(np.cos(x)*np.sin(y) - np.sin(1)*(1 - np.cos(1)))
        return val

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros((len(x),2), dtype=np.float)
        val[:,0] = (2*self.nu - 2*self.alpha)*np.sin(x)*np.sin(y)
        val[:,1] = (2*self.nu + 2*self.alpha)*np.cos(x)*np.cos(y)
        return val

    def dirichlet(self, p):
        return self.solution(p)

class CosSinData:
    def __init__(self, box):
        self.box = box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype is 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'quad':
            nx = 4
            ny = 4
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'Poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 

    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        cos = np.cos
        sin = np.sin
        val = np.zeros((len(x),2), dtype=np.float)
        val[:,0] = -1/2*cos(x)*cos(x)*cos(y)*sin(y)
        val[:,1] = 1/2*cos(x)*sin(x)*cos(y)*cos(y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.sin(x) - np.sin(y)
        return val

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = np.sin
        cos = np.cos
        val = np.zeros((len(x),2), dtype=np.float)
        val[:,0] = -3*cos(x)*cos(x)*sin(y)*cos(y) \
                 + sin(x)*sin(x)*sin(y)*cos(y) - cos(x) + sin(y)
        val[:,1] = 3*sin(x)*cos(x)*cos(y)*cos(y) \
                 - cos(x)*sin(x)*sin(y)*sin(y) - sin(x) + cos(y)
        return val

    def dirichlet(self, p):
        return self.solution(p)

class PolyData:
    def __init__(self, box):
        self.box = box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype is 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'quad':
            nx = 4
            ny = 4
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'Poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 

    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros((len(x),2), dtype=np.float)
        val[:,0] = y**4 + 1
        val[:,1] = x**4 + 1
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x**3 - y**3
        return val

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros((len(x),2), dtype=np.float)
        val[:,0] = -12*y**2 - 3*x**2
        val[:,1] = -12*x**2 + 3*y**2
        return val

    def dirichlet(self, p):
        return self.solution(p)
 
