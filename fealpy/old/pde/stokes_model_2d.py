import numpy as np

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import PolygonMesh
from fealpy.mesh import StructureQuadMesh, QuadrangleMesh
from fealpy.mesh import TriangleMesh, TriangleMeshWithInfinityNode

class StokesModelData_0:
    """
    [0, 1]^2
    u(x, y) = (sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(piy))
    p = 1/(y**2 + 1) - pi/4
    """
    def __init__(self):
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 2
            ny = 2
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = sin(pi*x)*cos(pi*y) 
        val[..., 1] = -cos(pi*x)*sin(pi*y) 
        return val

    @cartesian
    def strain(self, p):
        """
        (nabla u + nabla u^T)/2
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        val[..., 0, 0] = pi*cos(pi*x)*cos(pi*y)  
        val[..., 1, 1] = -pi*cos(pi*x)*cos(pi*y) 
        return val


    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 1/(y**2 + 1) - pi/4 
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = pi**2*sin(pi*x)*cos(pi*y)
        val[..., 1] = 2*y/(y**2 + 1)**2 - pi**2*sin(pi*y)*cos(pi*x)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class StokesModelData_1:
    def __init__(self):
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 2
            ny = 2
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = -cos(x)*cos(x)*cos(y)*sin(y)/2
        val[..., 1] = cos(x)*sin(x)*cos(y)*cos(y)/2
        return val

    @cartesian
    def strain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        val[..., 0, 0] = sin(x)*sin(y)*cos(x)*cos(y) 
        val[..., 0, 1] = -sin(x)**2*cos(y)**2/4 + sin(y)**2*cos(x)**2/4 
        val[..., 1, 0] = val[..., 0, 1] 
        val[..., 1, 1] = -sin(x)*sin(y)*cos(x)*cos(y) 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.sin(x) - np.sin(y)
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = -2*sin(y)*cos(y)*cos(x)**2 - cos(x) + cos(y)*sin(y)/2
        val[..., 1] =  2*sin(x)*cos(x)*cos(y)**2 + cos(y) - cos(x)*sin(x)/2
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class StokesModelData_2:
    def __init__(self, nu=1):
        self.nu = 1
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 2
            ny = 2
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2*pi*cos(pi*y)*sin(pi*x)**2*sin(pi*y) 
        val[..., 1] =-2*pi*cos(pi*x)*sin(pi*x)*sin(pi*y)**2 
        return val

    @cartesian
    def strain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        val[..., 0, 0] = 4*pi**2*sin(pi*x)*sin(pi*y)*cos(pi*x)*cos(pi*y)  
        val[..., 0, 1] = pi**2*sin(pi*x)**2*cos(pi*y)**2 - pi**2*sin(pi*y)**2*cos(pi*x)**2 
        val[..., 1, 0] = val[..., 0, 1] 
        val[..., 1, 1] =-4*pi**2*sin(pi*x)*sin(pi*y)*cos(pi*x)*cos(pi*y)  
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 0 
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 6*pi**3*sin(pi*x)**2*sin(pi*y)*cos(pi*y) - 2*pi**3*sin(pi*y)*cos(pi*x)**2*cos(pi*y)
        val[..., 1] =-6*pi**3*sin(pi*x)*sin(pi*y)**2*cos(pi*x) + 2*pi**3*sin(pi*x)*cos(pi*x)*cos(pi*y)**2
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class StokesModelData_3:
    def __init__(self, nu=1):
        self.nu = 1
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 2
            ny = 2
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2*pi*cos(pi*y)*sin(pi*x)**2*sin(pi*y) 
        val[..., 1] =-2*pi*cos(pi*x)*sin(pi*x)*sin(pi*y)**2 
        return val

    @cartesian
    def strain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        val[..., 0, 0] = 4*pi**2*sin(pi*x)*sin(pi*y)*cos(pi*x)*cos(pi*y)  
        val[..., 0, 1] = pi**2*sin(pi*x)**2*cos(pi*y)**2 - pi**2*sin(pi*y)**2*cos(pi*x)**2 
        val[..., 1, 0] = val[..., 0, 1] 
        val[..., 1, 1] =-4*pi**2*sin(pi*x)*sin(pi*y)*cos(pi*x)*cos(pi*y)  
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = np.sin
        val = sin(x) - sin(y) 
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 6*pi**3*sin(pi*x)**2*sin(pi*y)*cos(pi*y) - 2*pi**3*sin(pi*y)*cos(pi*x)**2*cos(pi*y) - cos(x) 
        val[..., 1] =-6*pi**3*sin(pi*x)*sin(pi*y)**2*cos(pi*x) + 2*pi**3*sin(pi*x)*cos(pi*x)*cos(pi*y)**2 + cos(y)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class StokesModelData_4:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-1, -1),
            ( 0, -1),
            (-1,  0),
            ( 0,  0),
            ( 1,  0),
            (-1,  1),
            ( 0,  1),
            ( 1,  1)], dtype=np.float)
        if meshtype == 'tri':
            cell = np.array([
                (2, 0, 3),
                (1, 3, 0),
                (5, 2, 6),
                (3, 6, 2),
                (6, 3, 7),
                (4, 7, 3)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            cell = np.array([
                (0, 1, 3, 2),
                (2, 3, 6, 5),
                (3, 4, 7, 6)], dtype=np.int)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (2, 0, 3),
                (1, 3, 0),
                (5, 2, 6),
                (3, 6, 2),
                (6, 3, 7),
                (4, 7, 3)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2*(x**3 - x)**2*(3*y**2 - 1)*(y**3 - y) 
        val[..., 1] = (3*x**2 - 1)*(-2*x**3 + 2*x)*(y**3 - y)**2 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 1/(x**2 + 1) - pi/4
        return val

    @cartesian
    def strain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        val[..., 0, 0] = 2*(6*x**2 - 2)*(x**3 - x)*(3*y**2 - 1)*(y**3 - y)   
        val[..., 0, 1] += 3*x*(-2*x**3 + 2*x)*(y**3 - y)**2 
        val[..., 0, 1] += 6*y*(x**3 - x)**2*(y**3 - y) 
        val[..., 0, 1] += (2 - 6*x**2)*(3*x**2 - 1)*(y**3 - y)**2/2 
        val[..., 0, 1] += (x**3 - x)**2*(3*y**2 - 1)**2 
        val[..., 1, 0] = val[..., 0, 1] 
        val[..., 1, 1] = (3*x**2 - 1)*(-2*x**3 + 2*x)*(6*y**2 - 2)*(y**3 - y)  
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)

        val[..., 0] -= 3*x*(-2*x**3 + 2*x)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] -= 12*x*(x**3 - x)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] += 2*x/(x**2 + 1)**2 
        val[..., 0] -= 18*y*(x**3 - x)**2*(3*y**2 - 1) 
        val[..., 0] -= (2 - 6*x**2)*(3*x**2 - 1)*(6*y**2 - 2)*(y**3 - y)/2 
        val[..., 0] -= (3*x**2 - 1)*(6*x**2 - 2)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] -= 6*(x**3 - x)**2*(y**3 - y) 

        val[..., 1] -= 6*x*(2 - 6*x**2)*(y**3 - y)**2 
        val[..., 1] += 6*x*(3*x**2 - 1)*(y**3 - y)**2 
        val[..., 1] -= 12*y*(3*x**2 - 1)*(-2*x**3 + 2*x)*(y**3 - y) 
        val[..., 1] -= 6*y*(6*x**2 - 2)*(x**3 - x)*(y**3 - y) 
        val[..., 1] -= (3*x**2 - 1)*(-2*x**3 + 2*x)*(3*y**2 - 1)*(6*y**2 - 2) 
        val[..., 1] -= (6*x**2 - 2)*(x**3 - x)*(3*y**2 - 1)**2 
        val[..., 1] -= 3*(-2*x**3 + 2*x)*(y**3 - y)**2
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class StokesModelData_5:
    def __init__(self, nu=1):
        self.nu = 1
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 2
            ny = 2
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2*(x**3 - x)**2*(3*y**2 - 1)*(y**3 - y) 
        val[..., 1] = (3*x**2 - 1)*(-2*x**3 + 2*x)*(y**3 - y)**2 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 1/(x**2 + 1) - pi/4
        return val

    @cartesian
    def strain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        val[..., 0, 0] = 2*(6*x**2 - 2)*(x**3 - x)*(3*y**2 - 1)*(y**3 - y)   
        val[..., 0, 1] += 3*x*(-2*x**3 + 2*x)*(y**3 - y)**2 
        val[..., 0, 1] += 6*y*(x**3 - x)**2*(y**3 - y) 
        val[..., 0, 1] += (2 - 6*x**2)*(3*x**2 - 1)*(y**3 - y)**2/2 
        val[..., 0, 1] += (x**3 - x)**2*(3*y**2 - 1)**2 
        val[..., 1, 0] = val[..., 0, 1] 
        val[..., 1, 1] = (3*x**2 - 1)*(-2*x**3 + 2*x)*(6*y**2 - 2)*(y**3 - y)  
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)

        val[..., 0] -= 3*x*(-2*x**3 + 2*x)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] -= 12*x*(x**3 - x)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] += 2*x/(x**2 + 1)**2 
        val[..., 0] -= 18*y*(x**3 - x)**2*(3*y**2 - 1) 
        val[..., 0] -= (2 - 6*x**2)*(3*x**2 - 1)*(6*y**2 - 2)*(y**3 - y)/2 
        val[..., 0] -= (3*x**2 - 1)*(6*x**2 - 2)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] -= 6*(x**3 - x)**2*(y**3 - y) 

        val[..., 1] -= 6*x*(2 - 6*x**2)*(y**3 - y)**2 
        val[..., 1] += 6*x*(3*x**2 - 1)*(y**3 - y)**2 
        val[..., 1] -= 12*y*(3*x**2 - 1)*(-2*x**3 + 2*x)*(y**3 - y) 
        val[..., 1] -= 6*y*(6*x**2 - 2)*(x**3 - x)*(y**3 - y) 
        val[..., 1] -= (3*x**2 - 1)*(-2*x**3 + 2*x)*(3*y**2 - 1)*(6*y**2 - 2) 
        val[..., 1] -= (6*x**2 - 2)*(x**3 - x)*(3*y**2 - 1)**2 
        val[..., 1] -= 3*(-2*x**3 + 2*x)*(y**3 - y)**2
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class StokesModelData_6:
    def __init__(self, nu=1):
        self.nu = 1

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-1, -1),
            ( 0, -1),
            (-1,  0),
            ( 0,  0),
            ( 1,  0),
            (-1,  1),
            ( 0,  1),
            ( 1,  1)], dtype=np.float)
        if meshtype == 'tri':
            cell = np.array([
                (2, 0, 3),
                (1, 3, 0),
                (5, 2, 6),
                (3, 6, 2),
                (6, 3, 7),
                (4, 7, 3)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            cell = np.array([
                (0, 1, 3, 2),
                (2, 3, 6, 5),
                (3, 4, 7, 6)], dtype=np.int)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (2, 0, 3),
                (1, 3, 0),
                (5, 2, 6),
                (3, 6, 2),
                (6, 3, 7),
                (4, 7, 3)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2*(x**3 - x)**2*(3*y**2 - 1)*(y**3 - y) 
        val[..., 1] = (3*x**2 - 1)*(-2*x**3 + 2*x)*(y**3 - y)**2 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 0 
        return val

    @cartesian
    def strain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        val[..., 0, 0] = 2*(6*x**2 - 2)*(x**3 - x)*(3*y**2 - 1)*(y**3 - y)   
        val[..., 0, 1] += 3*x*(-2*x**3 + 2*x)*(y**3 - y)**2 
        val[..., 0, 1] += 6*y*(x**3 - x)**2*(y**3 - y) 
        val[..., 0, 1] += (2 - 6*x**2)*(3*x**2 - 1)*(y**3 - y)**2/2 
        val[..., 0, 1] += (x**3 - x)**2*(3*y**2 - 1)**2 
        val[..., 1, 0] = val[..., 0, 1] 
        val[..., 1, 1] = (3*x**2 - 1)*(-2*x**3 + 2*x)*(6*y**2 - 2)*(y**3 - y)  
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)

        val[..., 0] -= 3*x*(-2*x**3 + 2*x)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] -= 12*x*(x**3 - x)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] -= 18*y*(x**3 - x)**2*(3*y**2 - 1) 
        val[..., 0] -= (2 - 6*x**2)*(3*x**2 - 1)*(6*y**2 - 2)*(y**3 - y)/2 
        val[..., 0] -= (3*x**2 - 1)*(6*x**2 - 2)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] -= 6*(x**3 - x)**2*(y**3 - y) 
        val[..., 1] -= 6*x*(2 - 6*x**2)*(y**3 - y)**2 
        val[..., 1] += 6*x*(3*x**2 - 1)*(y**3 - y)**2 
        val[..., 1] -= 12*y*(3*x**2 - 1)*(-2*x**3 + 2*x)*(y**3 - y) 
        val[..., 1] -= 6*y*(6*x**2 - 2)*(x**3 - x)*(y**3 - y) 
        val[..., 1] -= (3*x**2 - 1)*(-2*x**3 + 2*x)*(3*y**2 - 1)*(6*y**2 - 2) 
        val[..., 1] -= (6*x**2 - 2)*(x**3 - x)*(3*y**2 - 1)**2 
        val[..., 1] -= 3*(-2*x**3 + 2*x)*(y**3 - y)**2
        return val

    def dirichlet(self, p):
        return self.velocity(p)

class StokesModelData_7:
    def __init__(self, nu=1):
        self.nu = 1
        self.box = [-1, 1, -1, 1]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 2
            ny = 2
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = y**2
        val[..., 1] = x**2 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        shape = len(x.shape)*(1, )
        val = np.zeros(shape, dtype=np.float)
        return val

    @cartesian
    def strain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        val[..., 0, 1] = x + y 
        val[..., 1, 0] = val[..., 0, 1] 
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        shape = len(x.shape)*(1, ) + (2, )
        val = -np.ones(shape, dtype=np.float)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class StokesModelData_8:
    """
    [0, 1] \times [-0.25,0]
    u(x, y) = (sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(piy))
    p = 1/(y**2 + 1) - pi/4
    """
    def __init__(self):
        self.box = [0, 1, -0.25, 0]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 2
            ny = 2
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = x**2*y**2+np.exp(-y) 
        val[..., 1] = -2/3*x*y**3+2-pi*sin(pi*x) 
        return val

    @cartesian
    def strain(self, p):
        """
        (nabla u + nabla u^T)/2
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        val[..., 0, 0] = pi*cos(pi*x)*cos(pi*y)  
        val[..., 1, 1] = -pi*cos(pi*x)*cos(pi*y) 
        return val


    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = -(2-pi*np.sin(pi*x))*np.cos(2*pi*y) 
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        mu=1
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = mu*(-2*x**2-2*y**2-np.exp(-y))+pi**2*cos(pi*x)*cos(2*pi*y)
        val[..., 1] = mu*(4*x*y-pi**3*sin(pi*x))+2*pi*(2-pi*sin(pi*x))*sin(2*pi*y)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)


## 下面的例子还没有测试

class StokesModelData_9:
    """
    u(x, y) = (y**3, x**3)
    p(x, y) = x**2 - 1/3
    """
    def __init__(self):
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 4
            ny = 4
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = y**3 
        val[..., 1] = x**3 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x**3 - 1/3 
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 0 #TODO
        val[..., 1] = 0 
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class ModelData_1:
    """

    """
    def __init__(self):
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 4
            ny = 4
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        r = (x + 1)**2 + (y + 1)**2
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = (x + 1)/r
        val[..., 1] = (y + 1)/r 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = (x**2+y**2)**(2/3) - 0 #TODO:
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 0 #TODO
        val[..., 1] = 0 
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class PolySquareData:
    def __init__(self):
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 4
            ny = 4
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = y**2 
        val[..., 1] = -x**2 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = (x**2 + y**2)**(1/3) - 0 #TODO:  
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 0 
        val[..., 1] = 0
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class PolySquareData:
    def __init__(self):
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 4
            ny = 4
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = y**2 
        val[..., 1] = -x**2 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = (x**2 + y**2)**(1/3) - 0 #TODO:  
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 0 
        val[..., 1] = 0
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class LShapeSingularData:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-1, -1),
            ( 0, -1),
            ( 1, -1),
            (-1,  0),
            ( 0,  0),
            ( 1,  0),
            (-1,  1),
            ( 0,  1)], dtype=np.float)
        if meshtype == 'tri':
            cell = np.array([
                (3, 0, 4),
                (1, 4, 0),
                (4, 1, 5),
                (2, 5, 1),
                (6, 3, 7),
                (4, 7, 3)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (3, 0, 4),
                (1, 4, 0),
                (4, 1, 5),
                (2, 5, 1),
                (6, 3, 7),
                (4, 7, 3)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        r = (x - 1)**2 + (y - 1)**2
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = (x - 1)/r
        val[..., 1] = (y - 1)/r 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x + 1/6 
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 0 
        val[..., 1] = 0
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)




class PolyData:
    def __init__(self, box):
        self.box = box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 4
            ny = 4
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'Poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = y**4 + 1
        val[..., 1] = x**4 + 1
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x**3 - y**3
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[:,0] = -12*y**2 - 3*x**2
        val[:,1] = -12*x**2 + 3*y**2
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)


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

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 4
            ny = 4
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'Poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[:,0] = np.sin(x)*np.sin(y)
        val[:,1] = np.cos(x)*np.cos(y)
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 2*self.alpha*(np.cos(x)*np.sin(y) - np.sin(1)*(1 - np.cos(1)))
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[:,0] = (2*self.nu - 2*self.alpha)*np.sin(x)*np.sin(y)
        val[:,1] = (2*self.nu + 2*self.alpha)*np.cos(x)*np.cos(y)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

 
class PolyY2X2Data:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 4
            ny = 4
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh
 
    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = y**2
        val[..., 1] = x**2
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(x.shape, dtype=np.float) 
        return val


    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = -1
        val[..., 1] = -1
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)



class StokesModelRTData:
    """
    [0, 1]^2
    u(x, y) = 0
    p = Ra*(y^3 - y^2/2 + y - 7/12)
    """
    def __init__(self, Ra):
        self.box = [0, 1, 0, 1]
        self.Ra = Ra

    def domain(self):
        return self.box

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'tri':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quad':
            nx = 2
            ny = 2
            mesh = StructureQuadMesh(self.box, nx, ny)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'poly':
            cell = np.array([
                (1, 2, 0),
                (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            nmesh = TriangleMeshWithInfinityNode(mesh)
            pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
            pmesh = PolygonMesh(pnode, pcell, pcellLocation)
            return pmesh

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 0
        val[..., 1] = 0
        return val

    @cartesian
    def strain(self, p):
        """
        (nabla u + nabla u^T)/2
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        return val


    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = self.Ra*(y**3 - y**2/2 + y - 7/12)
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 0
        val[..., 1] = -self.Ra*(1 - y + 3*y**2)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

