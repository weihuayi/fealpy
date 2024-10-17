from typing import Optional
from ..backend import backend_manager as bm
from ..backend import TensorLike
from .geometry_base import GeometryBase
from .functional import project

class SphereSurface():
    def __init__(self, center=[0.0, 0.0, 0.0], radius=1.0):
        self.center = center
        self.radius = radius
        r = radius + radius/10
        x = center[0]
        y = center[1]
        z = center[2]
        self.box = [x-r, x+r, y-r, y+r, z-r, z+r]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 or (X, Y, Z)")

        cx, cy, cz = self.center
        r = self.radius
        return bm.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) - r 

    def gradient(self, p):
        l = bm.sqrt(bm.sum((p - self.center)**2, axis=-1))
        n = (p - self.center)/l[..., None]
        return n

    def unit_normal(self, p):
        return self.gradient(p)

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        shape = p.shape[:-1]+(3, 3)
        H = bm.zeros(shape, dtype=bm.float64)
        L = bm.sqrt(bm.sum(p*p, axis=-1))
        L3 = L**3
        H[..., 0, 0] = 1/L-x**2/L3
        H[..., 0, 1] = -x*y/L3
        H[..., 1, 0] = H[..., 0, 1]
        H[..., 0, 2] = - x*z/L3
        H[..., 2, 0] = H[..., 0, 2]
        H[..., 1, 1] = 1/L - y**2/L3
        H[..., 1, 2] = -y*z/L3
        H[..., 2, 1] = H[..., 1, 2]
        H[..., 2, 2] = 1/L - z**2/L3
        return H

    def jacobi_matrix(self, p):
        H = self.hessian(p)
        n = self.unit_normal(p)
        p[:], d = self.project(p)

        J = -(d[..., None, None]*H + bm.einsum('...ij, ...ik->...ijk', n, n))
        J[..., range(3), range(3)] += 1
        return J

    def tangent_operator(self, p):
        pass

    def project(self, p, maxit=200, tol=1e-8):
        d = self(p)
        p = p - d[..., None]*self.unit_normal(p)
        return p, d

    def init_mesh(self, meshtype='tri', returnnc=False, p=None):
        if meshtype == 'tri':
            t = (bm.sqrt(5) - 1)/2
            node = bm.array([
                [ 0, 1, t],
                [ 0, 1,-t],
                [ 1, t, 0],
                [ 1,-t, 0],
                [ 0,-1,-t],
                [ 0,-1, t],
                [ t, 0, 1],
                [-t, 0, 1],
                [ t, 0,-1],
                [-t, 0,-1],
                [-1, t, 0],
                [-1,-t, 0]], dtype=bm.float64)
            cell = bm.array([
                [6, 2, 0],
                [3, 2, 6],
                [5, 3, 6],
                [5, 6, 7],
                [6, 0, 7],
                [3, 8, 2],
                [2, 8, 1],
                [2, 1, 0],
                [0, 1,10],
                [1, 9,10],
                [8, 9, 1],
                [4, 8, 3],
                [4, 3, 5],
                [4, 5,11],
                [7,10,11],
                [0,10, 7],
                [4,11, 9],
                [8, 4, 9],
                [5, 7,11],
                [10,9,11]], dtype=bm.int32)
            node, d = self.project(node)
            if returnnc:
                return node, cell
            else:
                if p is None:
                    from fealpy.mesh.triangle_mesh import TriangleMesh
                    return TriangleMesh(node, cell) 
                else:
                    from fealpy.old.mesh.backup import LagrangeTriangleMesh
                    return LagrangeTriangleMesh(node, cell, p=p, surface=self) 

        elif meshtype == 'quad':
            node = bm.array([
                (-1, -1, -1),
                (-1, -1, 1),
                (-1, 1, -1),
                (-1, 1, 1),
                (1, -1, -1),
                (1, -1, 1),
                (1, 1, -1),
                (1, 1, 1)], dtype=bm.float64)
            cell = bm.array([
                (0, 1, 4, 5),
                (6, 7, 2, 3),
                (2, 3, 0, 1),
                (4, 5, 6, 7),
                (1, 3, 5, 7),
                (2, 0, 6, 4)], dtype=bm.int32)
            node, d = self.project(node)
            if returnnc:
                return node, cell
            else:
                if p is None:
                    from fealpy.mesh.quadrangle_mesh import QuadrangleMesh 
                    return QuadrangleMesh(node, cell) 
                else:
                    from fealpy.old.mesh.backup import LagrangeQuadrangleMesh 
                    return LagrangeQuadrangleMesh(node, cell, p=p, surface=self)

