from typing import Optional
from ..backend import backend_manager as bm
from ..backend import TensorLike
from .geometry_base import GeometryBase
from .functional import project

class CircleCurve():
    def __init__(self, center=bm.array([0.0, 0.0]), radius=1.0):
        self.center = center
        self.radius = radius
        self.box = [-1.5, 1.5, -1.5, 1.5]

    def init_mesh(self, n):
        from ..mesh import IntervalMesh

        dt = 2*bm.pi/n
        theta  = bm.arange(0, 2*bm.pi, dt)

        node = bm.zeros((n, 2), dtype = bm.float64)
        cell = bm.zeros((n, 2), dtype = bm.int32)

        node[:, 0] = self.radius*bm.cos(theta)
        node[:, 1] = self.radius*bm.sin(theta)
        node += self.center

        cell[:, 0] = bm.arange(n)
        cell[:, 1][:-1] = bm.arange(1,n)

        mesh = IntervalMesh(node, cell)

        return mesh 

    def __call__(self, p):
        return bm.sqrt(bm.sum((p - self.center)**2, axis=-1))-self.radius

    def value(self, p):
        return self(p)

    def gradient(self, p):
        l = bm.sqrt(bm.sum((p - self.center)**2, axis=-1))
        n = (p - self.center)/l[..., None]
        return n

    def distvalue(self, p):
        p, d, n= project(self, p, maxit=200, tol=1e-8, returngrad=True, returnd=True)
        return d, n

    def project(self, p):
        """
        @brief 把曲线附近的点投影到曲线上
        """
        p, d = project(self, p, maxit=200, tol=1e-8, returnd=True)
        return p, d 
