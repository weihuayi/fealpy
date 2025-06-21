import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian, barycentric

class pLaplaceData():
    def __init__(self, n=3):
        self.n = n

    @cartesian
    def init_solution(self, p):
        n = self.n
        x = p[..., 0]
        y = p[..., 1]
        return np.sin(np.pi*x)*np.sin(np.pi*y)

    def init_mesh(self, n=0):
        box = [0, 1, 0, 1]
        mesh = TriangleMesh.from_box(box, nx=3, ny=3)
        mesh.uniform_refine(n)
        return mesh

    def dirichlet(self, p):
        z = np.zeros(p.shape[:-1], dtype=np.float_)
        return z

    def a(self, u):
        n = self.n

        @barycentric
        def fff(p):
            gu = u.grad_value(p)
            lgu = np.linalg.norm(gu[:], axis=-1)
            return lgu**(n-2)
        return fff

    def a_u(self, u):
        n = self.n
        @barycentric
        def fff(p):
            gu = u.grad_value(p) #(NQ, NC, GD)
            lgu = np.linalg.norm(gu[:], axis=-1) #(NQ, NC)
            return (n-2)*(lgu[..., None]**(n-4))*gu
        return fff

class meanCurvatureData():
    def __init__(self):
        pass

    @cartesian
    def init_solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return 10*np.sin(np.pi*x)*np.sin(np.pi*y)

    def init_mesh(self, n=0):
        box = [0, 1, 0, 1]
        mesh = MeshFactory.boxmesh2d(box, nx=3, ny=3, meshtype='tri')
        mesh.uniform_refine(n)
        return mesh

    def dirichlet(self, p):
        z = np.zeros(p.shape[:-1], dtype=np.float_)
        return z

    def a(self, u):
        @barycentric
        def fff(p):
            gu = u.grad_value(p)
            lgu = np.linalg.norm(gu[:], axis=-1)
            return 1/np.sqrt(1+lgu**2)
        return fff

    def JU(self, u):
        '''@
        @note : 有错误
        '''
        n = self.n

        @barycentric
        def fff(p):
            gu = u.grad_value(p)
            lgu = np.linalg.norm(gu[:], axis=-1)
            return (n-2)*(lgu**(n-4))*gu
        return fff


