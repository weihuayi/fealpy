from ..typing import  TensorLike
from ..decorator import cartesian
from ..backend import backend_manager as bm

class SphereSurface():
    def __init__(self, center=[0.0, 0.0, 0.0], radius=1.0):
        self.center = center
        self.radius = radius
        r = radius + radius/10
        x, y, z = center
        self.box = [x-r, x+r, y-r, y+r, z-r, z+r]

    @cartesian
    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x, y, z = p[..., 0], p[..., 1], p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 or (X, Y, Z)")

        cx, cy, cz = self.center
        r = self.radius
        return bm.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) - r 
    
    @cartesian
    def project(self, p:TensorLike, maxit=200, tol=1e-8) -> TensorLike:
        d = self(p)
        p = p - d[..., None]*self.unit_normal(p)
        return p, d
    
    @cartesian
    def gradient(self, p:TensorLike) -> TensorLike:
        center = bm.tensor(self.center, dtype=p.dtype)
        l = bm.sqrt(bm.sum((p - center)**2, axis=-1))
        n = (p - center)/l[..., None]
        return n
    
    @cartesian
    def unit_normal(self, p:TensorLike) -> TensorLike:
        return self.gradient(p)
    
    @cartesian
    def hessian(self, p:TensorLike) -> TensorLike:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]

        L = bm.sqrt(bm.sum(p*p, axis=-1))
        L3 = L**3

        H00 = 1/L - x**2/L3
        H01 = -x*y/L3
        H02 = -x*z/L3
        H11 = 1/L - y**2/L3
        H12 = -y*z/L3
        H22 = 1/L - z**2/L3
        
        row0 = bm.stack([H00, H01, H02], axis=-1)
        row1 = bm.stack([H01, H11, H12], axis=-1)
        row2 = bm.stack([H02, H12, H22], axis=-1)
        
        H = bm.stack([row0, row1, row2], axis=-2)
        return H
    
    @cartesian
    def jacobi_matrix(self, p:TensorLike) -> TensorLike:
        H = self.hessian(p)
        n = self.unit_normal(p)
        p[:], d = self.project(p)

        J = -(d[..., None, None]*H + bm.einsum('...ij, ...ik->...ijk', n, n))
        J[..., range(3), range(3)] += 1
        return J
    
    @cartesian
    def tangent_operator(self, p):
        pass

    @cartesian
    def init_mesh(self, mtype='tri', returnnc=False, p=None):
        if mtype == 'tri':
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
                    from fealpy.mesh import TriangleMesh
                    return TriangleMesh(node, cell) 
                else:
                    from fealpy.mesh import LagrangeTriangleMesh
                    return LagrangeTriangleMesh(node, cell, p=p, surface=self) 

        elif mtype == 'quad':
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
                    from fealpy.mesh import QuadrangleMesh 
                    return QuadrangleMesh(node, cell) 
                else:
                    from fealpy.mesh import LagrangeQuadrangleMesh 
                    return LagrangeQuadrangleMesh(node, cell, p=p, surface=self)

