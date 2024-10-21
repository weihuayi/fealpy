import numpy as np
from .geoalg import project

class ScaledSurface():

    def __init__(self, surface, scale):
        self.surface = surface
        self.scale = scale

    def __call__(self, p):
        p0 = p/self.scale
        return self.surface(p0)

    def unit_normal(self, p):
        p0 = p/self.scale
        return self.surface.unit_normal(p0)

    def project(self, p, maxit=200, tol=1e-8):
        p0 = p/self.scale
        p1, d = self.surface.project(p0, maxit=200, tol=1e-8)
        return p1*self.scale, d*self.scale 

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
        return np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) - r 

    def gradient(self, p):
        l = np.sqrt(np.sum((p - self.center)**2, axis=-1))
        n = (p - self.center)/l[..., np.newaxis]
        return n

    def unit_normal(self, p):
        return self.gradient(p)

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        shape = p.shape[:-1]+(3, 3)
        H = np.zeros(shape, dtype=np.float)
        L = np.sqrt(np.sum(p*p, axis=-1))
        L3=L**3
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

        J = -(d[..., np.newaxis, np.newaxis]*H + np.einsum('...ij, ...ik->...ijk', n, n))
        J[..., range(3), range(3)] += 1
        return J

    def tangent_operator(self, p):
        pass

    def project(self, p, maxit=200, tol=1e-8):
        d = self(p)
        p = p - d[..., np.newaxis]*self.unit_normal(p)
        return p, d

    def init_mesh(self, meshtype='tri', returnnc=False, p=None):
        if meshtype == 'tri':
            t = (np.sqrt(5) - 1)/2
            node = np.array([
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
                [-1,-t, 0]], dtype=np.float64)
            cell = np.array([
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
                [10,9,11]], dtype=np.int_)
            node, d = self.project(node)
            if returnnc:
                return node, cell
            else:
                if p is None:
                    from fealpy.mesh.backup import TriangleMesh
                    return TriangleMesh(node, cell) 
                else:
                    from fealpy.mesh.backup import LagrangeTriangleMesh
                    return LagrangeTriangleMesh(node, cell, p=p, surface=self) 

        elif meshtype == 'quad':
            node = np.array([
                (-1, -1, -1),
                (-1, -1, 1),
                (-1, 1, -1),
                (-1, 1, 1),
                (1, -1, -1),
                (1, -1, 1),
                (1, 1, -1),
                (1, 1, 1)], dtype=np.float64)
            cell = np.array([
                (0, 1, 4, 5),
                (6, 7, 2, 3),
                (2, 3, 0, 1),
                (4, 5, 6, 7),
                (1, 3, 5, 7),
                (2, 0, 6, 4)], dtype=np.int_)
            node, d = self.project(node)
            if returnnc:
                return node, cell
            else:
                if p is None:
                    from fealpy.mesh import QuadrangleMesh 
                    return QuadrangleMesh(node, cell) 
                else:
                    from fealpy.mesh.backup import LagrangeQuadrangleMesh 
                    return LagrangeQuadrangleMesh(node, cell, p=p, surface=self) 


class TwelveSpheres:
    def __init__(self, r=0.7):
        self.center = np.array([
            [ 1.0,  0.0,               0.0],
            [-1.0,  0.0,               0.0],
            [ 0.5,  0.866025403784439, 0.0],
            [-0.5,  0.866025403784439, 0.0],
            [ 0.5, -0.866025403784439, 0.0],
            [-0.5, -0.866025403784439, 0.0],
            [ 2.0,  0.0,               0.0],
            [ 1.0,  1.73205080756888,  0.0],
            [-1.0,  1.73205080756888,  0.0],
            [-2.0,  0.0,               0.0],
            [-1.0, -1.73205080756888,  0.0],
            [ 1.0, -1.73205080756888,  0.0]], dtype=np.float)
        self.radius = r*np.ones(12)
        self.box = [-3.2, 3.2, -3.2, 3.2, -3.2, 3.2]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            c = self.center
            r = self.radius
            d  = np.sqrt(np.sum((p - c.reshape(-1, 1, 3))**2, axis=2)) - r.reshape(-1, 1)
            return np.min(d, axis=0)  
        elif len(args) == 3:
            X, Y, Z = args
            dim = np.prod(X.shape)
            p = np.zeros((dim, 3), dtype=np.float)
            p[:, 0] = X.flatten()
            p[:, 1] = Y.flatten()
            p[:, 2] = Z.flatten()
            c = self.center
            r = self.radius
            d  = np.sqrt(np.sum((p - c.reshape(-1, 1, 3))**2, axis=2)) - r.reshape(-1, 1)
            return np.min(d, axis=0).reshape(X.shape)
        else:
            raise ValueError("the args must be a N*3 or X, Y, Z")

    def project(self, p, maxit=200, tol=1e-8):
        eps = np.finfo(float).eps
        deps = np.sqrt(eps)
        d = self(p)
        depsx = np.array([deps, 0, 0])
        depsy = np.array([0, deps, 0])
        depsz = np.array([0, 0, deps])
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[:, 0] = (self(p + depsx) - d)/deps
        grad[:, 1] = (self(p + depsy) - d)/deps
        grad[:, 2] = (self(p + depsz) - d)/deps
        p -= d.reshape(-1, 1)*grad
        return p, d

    def gradient(self, p):
        eps = np.finfo(float).eps
        deps = np.sqrt(eps)
        d = self(p)
        depsx = np.array([deps, 0, 0])
        depsy = np.array([0, deps, 0])
        depsz = np.array([0, 0, deps])
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[:, 0] = (self(p + depsx) - d)/deps
        grad[:, 1] = (self(p + depsy) - d)/deps
        grad[:, 2] = (self(p + depsz) - d)/deps
        return grad

    def unit_normal(self, p):
        return self.gradient(p)

    def init_mesh(self):
        pass

class HeartSurface:
    def __init__(self):
        self.box = [-2, 2, -2, 2, -2, 2]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        return (x - z**2)**2 + y**2 + z**2 - 1.0

    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = 2*(x - z**2)
        grad[..., 1] = 2*y
        grad[..., 2] = -4*(x - z**2)*z + 2*z
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=-1, keepdims=True))
        n = grad/l
        return n

    def div_unit_normal(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        t1 = -2*x**3+10*x**2*z**2+2*x**2-2*x*y**2-14*x*z**4-8*x*z**2+6*y**2*z**2+2*y**2+6*z**6+6*z**4+2*z**2
        t2 = (4*x**2*z**2+x**2-8*x*z**4-6*x*z**2+y**2+4*z**6+5*z**4 +z**2)**(3/2)
        div = t1/t2
        return div

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        shape = p.shape[0:-1]+(3, 3)
        H = np.zeros(shape, dtype=np.float)
        
        H[..., 0, 0] = 2.0 
        H[..., 0, 1] = 0.0
        H[..., 1, 0] = 0.0
        H[..., 0, 2] = -4*z
        H[..., 2, 0] = -4*z
        H[..., 1, 1] = 2
        H[..., 1, 2] = 0.0
        H[..., 2, 1] = 0.0
        H[..., 2, 2] = -4*x + 12*z**2 + 2
        return H
    
    def jacobi_matrix(self, p):
        H = self.hessian(p)
        n = self.unit_normal(p)
        p[:], d = self.project(p)

        J = -(d[..., np.newaxis, np.newaxis]*H + np.einsum('...ij, ...ik->...ijk', n, n))
        J[..., range(3), range(3)] += 1
        return J


    def init_mesh(self, meshdata=None):
        import scipy.io as sio
        from fealpy.mesh import TriangleMesh
        if meshdata is None:
            data = sio.loadmat('../fealpy/meshdata/heart.mat')
        else:
            data = sio.loadmat(meshdata)
        node = data['node']
        cell = np.array(data['elem'] - 1, dtype=np.int64)
        return TriangleMesh(node, cell)


class EllipsoidSurface:
    def __init__(self, c=[9, 3, 1]):
        m = np.max(c)
        self.box = [-m, m, -m, m, -m, m]
        self.c = c

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        a, b, c = self.c
        return x**2/a**2 + y**2/b**2 + z**2/c**2 - 1 
 
    def project(self, p, maxit=200, tol=1e-8):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        a, b, c = self.c
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = 2*x/a**2 
        grad[..., 1] = 2*y/b**2 
        grad[..., 2] = 2*z/c**2 
        return grad

    def hessian(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        a, b, c = self.c
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        shape = p.shape[0:-1]+(3, 3)
        H = np.zeros(shape, dtype=np.float)
        S = a**4*b**4*z**2+a**4*c**4*y**2+b**4*c**4*x**2
        T = np.sqrt(S/a**4*b**4*c**4)
        
        H[..., 0, 0] = (b**4*z**2+c**4*y**2)*a**2/(S*T)  
        H[..., 0, 1] = -a**2*c**4*x*y/(S*T)
        H[..., 1, 0] = -b**2*c**4*x*y/(S*T)
        H[..., 0, 2] = -a**2*b**4*x*z/(S*T)
        H[..., 2, 0] = -b**4*c**2*x*z/(S*T)
        H[..., 1, 1] = b**2*(a**4*z**2+c**4*x**2)/(S*T)
        H[..., 1, 2] = -a**4*b**2*y*z/(S*T)
        H[..., 2, 1] = -a**4*c**2*y*z/(S*T)
        H[..., 2, 2] = c**2*(a**4*y**2+b**4*x**2)/(S*T)
        return H

    def jacobi_matrix(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        H = self.hessian(p)
        n = self.unit_normal(p)
        p[:], d = self.project(p)

        J = -(d[..., np.newaxis, np.newaxis]*H + np.einsum('...ij, ...ik->...ijk', n, n))
        J[..., range(3), range(3)] += 1
        return J

    def unit_normal(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=1, keepdims=True))
        n = grad/l
        return n

    def init_mesh(self, meshdata=None):
        import scipy.io as sio
        from fealpy.mesh import TriangleMesh
        if meshdata is None:
            data = sio.loadmat('../fealpy/meshdata/ellipsoid.mat')
        else:
            data = sio.loadmat(meshdata)
        node = data['node']
        cell = np.array(data['elem'] - 1, dtype=np.int64)
        return TriangleMesh(node, cell)

class TorusSurface:
    def __init__(self):
        self.box = [-6, 6, -6, 6, -6, 6]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        return np.sqrt(x**2 + y**2 + z**2 + 16 - 8*np.sqrt(x**2 + y**2)) - 1 
    
    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        s1 = np.sqrt(x**2 + y**2)
        s2 = np.sqrt(s1**2 + z**2 + 16 - 8*s1)
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = (s1 - 4)*x/(s1*s2)
        grad[..., 1] = (s1 - 4)*y/(s1*s2)
        grad[..., 2] = z/s2
        return grad

    def hessian(self, p):
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]
        shape = p.shape[0:-1]+(3, 3)
        H = np.zeros(shape, dtype=np.float)
        s = x**2 + y**2
        s1 = np.sqrt(x**2 + y**2)
        s2 = np.sqrt(s1**2 + z**2 + 16 - 8*s1)

        H[..., 0, 0] = -x**2*s*(s1-4)**2+(4*x**2*s1-4*s1**3+s**2)*s2**2/(s**2*s2**3)
        H[..., 0, 1] = x*y*(-s1**3*(s1-4)**2+4*s*s2**2)/(s1**5*s2**3)
        H[..., 1, 0] = H[..., 0, 1]
        H[..., 0, 2] = x*z*(s1-4)/(s1*s2**3)
        H[..., 2, 0] = H[..., 0, 2]
        H[..., 1, 1] = -y**2*s*(s1-4)**2+(4*y**2*s1-4*s1**3+s**2)*s2**2/(s**2*s2**3)
        H[..., 1, 2] = y*z*(s1-4)/(s1*s2**3)
        H[..., 2, 1] = H[..., 1, 2]
        H[..., 2, 2] = s-8*s1+16/s2**3
        return H

    def jacobi_matrix(self, p):
        H = self.hessian(p)
        n = self.unit_normal(p)
        p[:], d = self.project(p)

        J = -(d[..., np.newaxis, np.newaxis]*H + np.einsum('...ij, ...ik->...ijk', n, n))
        J[..., range(3), range(3)] += 1
        return J

    def unit_normal(self, p):
        return self.gradient(p)

    def div_unit_normal(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        s = x**2 + y**2
        s1 = np.sqrt(x**2 + y**2)
        s2 = np.sqrt(s1**2 + z**2 + 16 - 8*s1)
        div = -z**2*s**2-s**2*(s1-4)**2+s**2*s2**2+2*(2*x**2*s1+2*y**2*s1-4*s1**3+s**2)*s2**2/(s**2*s2**3) 
        return div

    def init_mesh(self, meshdata=None):
        import scipy.io as sio
        from fealpy.mesh import TriangleMesh
        if meshdata is None:
            data = sio.loadmat('../fealpy/meshdata/torus.mat')
        else:
            data = sio.loadmat(meshdata)
        node = data['node']
        cell = np.array(data['elem'] - 1, dtype=np.int64)
        return TriangleMesh(node, cell)



class OrthocircleSurface:
    def __init__(self, c=[0.075, 3]):
        self.box = [-2, 2, -2, 2, -2, 2]
        self.c = c

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., :, 0]
            y = p[..., :, 1]
            z = p[..., :, 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        x2 = x**2
        y2 = y**2
        z2 = z**2
        d1 = (x2 + y2 - 1)**2 + z2
        d2 = (y2 + z2 - 1)**2 + x2
        d3 = (z2 + x2 - 1)**2 + y2
        r2 = x2 + y2 + z2
        c = self.c

        return d1*d2*d3 - c[0]**2*(1 + c[1]*r2) 

    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        c = self.c
        x, y, z = p[..., :, 0], p[..., :, 1], p[..., :, 2]
        x2, y2, z2 = x**2, y**2, z**2
        d1 = (x2 + y2 - 1)**2 + z2
        d2 = (y2 + z2 - 1)**2 + x2
        d3 = (z2 + x2 - 1)**2 + y2
        d11 = d1**2 + z2
        d22 = d2**2 + x2
        d33 = d3**2 + y2
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., :, 0] = 4*d1*x*d22*d33 + 2*d11*x*d33 + 4*d11*d22*d3*x \
                - 2*c[0]**2*c[1]*x 
        grad[..., :, 1] = 4*d1*y*d22*d33 + 4*d11*d2*y*d33 + 2*d11*d22*y \
                -2 * c[0]**2 * c[1] * y
        grad[..., :, 2] = 2*z*d22*d33 + 4*d11*d2*z*d33 + 4*d11*d22*d3*z \
                - 2* c[0]**2 * c[1] * z
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=1, keepdims=True))
        return grad/l

    def init_mesh(self, meshdata=None):
        import scipy.io as sio
        from fealpy.mesh import TriangleMesh
        if meshdata is None:
            data = sio.loadmat('../fealpy/meshdata/orthocircle.mat')
        else:
            data = sio.loadmat(meshdata)
        node = data['node']
        cell = np.array(data['elem'] - 1, dtype=np.int64)
        return TriangleMesh(node, cell)


#class QuarticsSurface:
#    def __init__(self, r=1.05):
#        self.box = [-2, 2, -2, 2, -2, 2]
#        self.r = r
#
#    def __call__(self, *args):
#        if len(args) == 1:
#            p, = args
#            x = p[..., :, 0]
#            y = p[..., :, 1]
#            z = p[..., :, 2]
#        elif len(args) == 3:
#            x, y, z = args
#        else:
#            raise ValueError("the args must be a N*3 array or x, y, z")
#        x2 = x**2
#        y2 = y**2
#        z2 = z**2
#        r = self.r
#        return  (x2 - 1)**2 + (y2 - 1)**2 + (z2 - 1)**2 - r
#
#    def project(self, p, maxit=200, tol=1e-8):
#        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
#        return p0, d
#
#    def gradient(self, p):
#        x, y, z = p[..., :, 0], p[..., :, 1], p[..., :, 2]
#        x2, y2, z2 = x**2, y**2, z**2
#        grad = np.zeros(p.shape, dtype=p.dtype)
#        grad[..., :, 0] = 4*(x2 - 1)*x  
#        grad[..., :, 1] = 4*(y2 - 1)*x 
#        grad[..., :, 2] = 4*(z2 - 1)*x 
#        return grad
#
#    def unit_normal(self, p):
#        grad = self.gradient(p)
#        l = np.sqrt(np.sum(grad**2, axis=1, keepdims=True))
#        return grad/l
#
#    def init_mesh(self, meshdata=None):
#        import scipy.io as sio
#        from fealpy.mesh import TriangleMesh
#        if meshdata is None:
#            data = sio.loadmat('../fealpy/meshdata/quartics.mat')
#        else:
#            data = sio.loadmat(meshdata)
#        node = data['node']
#        cell = np.array(data['elem'] - 1, dtype=np.int64)
#        return TriangleMesh(node, cell)


class ImplicitSurface:
    def __init__(self, expression):
        self.expression = expression

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
            z = p[:, 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")


class ParabolicSurface:
    def __init__(self):
        self.box = [-2, 2, -2, 2, -2, 2]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        return x**2 + y**2 - z

    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = 2*x
        grad[..., 1] = 2*y
        grad[..., 2] = -1
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=-1, keepdims=True))
        n = grad/l
        return n

    def div_unit_normal(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        t1 = 4*(2*x**2 + 2*y**2 + 1)
        t2 = (4*x**2 + 4*y**2 + 1)**(3/2)
        div = t1/t2
        return div

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        shape = p.shape[0:-1]+(3, 3)
        H = np.zeros(shape, dtype=np.float)
        
        H[..., 0, 0] = 2.0 
        H[..., 0, 1] = 0.0
        H[..., 1, 0] = 0.0
        H[..., 0, 2] = 0.0
        H[..., 2, 0] = 0.0
        H[..., 1, 1] = 2
        H[..., 1, 2] = 0.0
        H[..., 2, 1] = 0.0
        H[..., 2, 2] = 0.0
        return H
    
    def jacobi_matrix(self, p):
        pass

    def init_mesh(self, meshdata=None):
        import scipy.io as sio
        from fealpy.mesh import TriangleMesh
        if meshdata is None:
            data = sio.loadmat('../fealpy/meshdata/parabolic.mat')
        else:
            data = sio.loadmat(meshdata)
        node = data['node']
        cell = np.array(data['elem'], dtype=np.int64)
        return TriangleMesh(node, cell)


class QuarticsSurface:
    def __init__(self):
        self.box = [-2, 2, -2, 2, -2, 2]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        return (x**2 - 1)**2 + (y**2 - 1)**2 + (z**2 - 1)**2 - 1.05

    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = 4*x*(x**2 - 1)
        grad[..., 1] = 4*y*(y**2 - 1)
        grad[..., 2] = 4*z*(z**2 - 1)
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=-1, keepdims=True))
        n = grad/l
        return n

    def div_unit_normal(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        x2, y2, z2 = x**2, y**2, z**2

        t1 =-x2*(x2-1)**2*(3*x2-1)-y2*(y2-1)**2*(3*y2-1)-z2*(z2-1)**2*(3*z2-1) + 3*(x2*(x2-1)**2+y2*(y2-1)**2+z2*(z2-1)**2)*(x2+y2+z2-1)
        t2 = (x2*(x2-1)**2+y2*(y2-1)**2+z2*(z2-1)**2)**(3/2)
        div = t1/t2
        return div

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        shape = p.shape[0:-1]+(3, 3)
        H = np.zeros(shape, dtype=np.float)
        
        H[..., 0, 0] = 12*x**2 - 4
        H[..., 0, 1] = 0.0
        H[..., 1, 0] = 0.0
        H[..., 0, 2] = 0.0
        H[..., 2, 0] = 0.0
        H[..., 1, 1] = 12*y**2 - 4
        H[..., 1, 2] = 0.0
        H[..., 2, 1] = 0.0
        H[..., 2, 2] = 12*z**2 - 4
        return H
    
    def jacobi_matrix(self, p):
        pass
    
    def init_mesh(self, meshdata=None):
        import scipy.io as sio
        from fealpy.mesh import TriangleMesh
        if meshdata is None:
            data = sio.loadmat('../fealpy/meshdata/quartics.mat')
        else:
            data = sio.loadmat(meshdata)
        node = data['node']
        cell = np.array(data['elem'] - 1, dtype=np.int64)
        return TriangleMesh(node, cell)

class SquaredSurface:
    def __init__(self):
        self.box = [-2, 2, -2, 2, -2, 2]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        return x**8 + y**8 + z**8 - 1

    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = 8*x**7
        grad[..., 1] = 8*y**7
        grad[..., 2] = 8*z**7
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=-1, keepdims=True))
        n = grad/l
        return n

    def div_unit_normal(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        s = 64*x**14 + 64*y**14 + 64*z**14
        t1 =3584*(x**20 + y**20 + z**20)/s**(3/2)
        t2 = 56*(x**6 + y**6 + z**6)/s**(1/2)
        div = -t1 + t2
        return div

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        shape = p.shape[0:-1]+(3, 3)
        H = np.zeros(shape, dtype=np.float)
        
        H[..., 0, 0] = 56*x**6
        H[..., 0, 1] = 0.0
        H[..., 1, 0] = 0.0
        H[..., 0, 2] = 0.0
        H[..., 2, 0] = 0.0
        H[..., 1, 1] = 56*y**6
        H[..., 1, 2] = 0.0
        H[..., 2, 1] = 0.0
        H[..., 2, 2] = 56*z**6
        return H
    
    def jacobi_matrix(self, p):
        pass
    
    def init_mesh(self, meshdata=None):
        import scipy.io as sio
        from fealpy.mesh import TriangleMesh
        if meshdata is None:
            data = sio.loadmat('../fealpy/meshdata/squared.mat')
        else:
            data = sio.loadmat(meshdata)
        node = data['node']
        cell = np.array(data['elem'] - 1, dtype=np.int64)
        return TriangleMesh(node, cell)


class SaddleSurface:
    def __init__(self):
        self.box = [-2, 2, -2, 2, -2, 2]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        return x**2 - y**2 - z

    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = 2*x
        grad[..., 1] = -2*y
        grad[..., 2] = -1
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=-1, keepdims=True))
        n = grad/l
        return n

    def div_unit_normal(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        t1 = -8*(x**2 - y**2)
        t2 = (4*x**2 + 4*y**2 + 1)**(3/2)
        div = t1/t2
        return div

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        shape = p.shape[0:-1]+(3, 3)
        H = np.zeros(shape, dtype=np.float)
        
        H[..., 0, 0] = 2.0 
        H[..., 0, 1] = 0.0
        H[..., 1, 0] = 0.0
        H[..., 0, 2] = 0.0
        H[..., 2, 0] = 0.0
        H[..., 1, 1] = -2
        H[..., 1, 2] = 0.0
        H[..., 2, 1] = 0.0
        H[..., 2, 2] = 0.0
        return H
    
    def jacobi_matrix(self, p):
        pass

    def init_mesh(self, meshdata=None):
        import scipy.io as sio
        from fealpy.mesh import TriangleMesh
        if meshdata is None:
            data = sio.loadmat('../fealpy/meshdata/saddle.mat')
        else:
            data = sio.loadmat(meshdata)
        node = data['node']
        cell = np.array(data['elem'], dtype=np.int64)
        return TriangleMesh(node, cell)
   
class Fisher_Koch_S:
    def __init__(self, l=3):
        self.box = [-3, 3, -3, 3, -3, 3]
        self.l = l

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        l = self.l
        return np.cos((4 * np.pi * x) / l) * np.sin((2 * np.pi * y) / l) * np.cos((2 * np.pi * z) / l) \
            + np.cos((4 * np.pi * y) / l) * np.sin((2 * np.pi * z) / l) * np.cos((2 * np.pi * x) / l) \
            + np.cos((4 * np.pi * z) / l) * np.sin((2 * np.pi * x) / l) * np.cos((2 * np.pi * y) / l)

    def project(self, p, maxit=200, tol=1e-8):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        l = self.l
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = -2 * np.pi * np.sin(2 * np.pi * x / l) * np.sin(2 * np.pi * z / l) * np.cos(4 * np.pi * y / l) / l \
                       - 4 * np.pi * np.sin(4 * np.pi * x / l) * np.sin(2 * np.pi * y / l) * np.cos(2 * np.pi * z / l) / l \
                       + 2 * np.pi * np.cos(2 * np.pi * x / l) * np.cos(2 * np.pi * y / l) * np.cos(4 * np.pi * z / l) / l
        grad[..., 1] = -2 * np.pi * np.sin(2 * np.pi * x / l) * np.sin(2 * np.pi * y / l) * np.cos(4 * np.pi * z / l) / l \
                       - 4 * np.pi * np.sin(4 * np.pi * y / l) * np.sin(2 * np.pi * z / l) * np.cos(2 * np.pi * x / l) / l \
                       + 2 * np.pi * np.cos(4 * np.pi * x / l) * np.cos(2 * np.pi * y / l) * np.cos(2 * np.pi * z / l) / l
        grad[..., 2] = -4 * np.pi * np.sin(2 * np.pi * x / l) * np.sin(4 * np.pi * z / l) * np.cos(2 * np.pi * y / l) / l \
                       - 2 * np.pi * np.sin(2 * np.pi * y / l) * np.sin(2 * np.pi * z / l) * np.cos(4 * np.pi * x / l) / l \
                       + 2 * np.pi * np.cos(2 * np.pi * x / l) * np.cos(4 * np.pi * y / l) * np.cos(2 * np.pi * z / l) / l
        return grad

    def hessian(self, p):
        raise NotImplementedError('NotImplemented')

    def jacobi_matrix(self, p):
        raise NotImplementedError('NotImplemented')

    def unit_normal(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad ** 2, axis=1, keepdims=True))
        n = grad / l
        return n

    def init_mesh(self, meshdata=None):
        raise NotImplementedError('NotImplemented')


class F_RD:
    def __init__(self, l=3):
        self.box = [-3, 3, -3, 3, -3, 3]
        self.l = l

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        l = self.l
        return 4 * np.cos((2 * np.pi * x) / l) * np.cos((2 * np.pi * y) / l) * np.cos((2 * np.pi * z) / l) \
            - np.cos((4 * np.pi * x) / l) * np.cos((4 * np.pi * y) / l) \
            - np.cos((4 * np.pi * y) / l) * np.cos((4 * np.pi * z) / l) \
            - np.cos((4 * np.pi * z) / l) * np.cos((4 * np.pi * x) / l)

    def project(self, p, maxit=200, tol=1e-8):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        l = self.l
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = -8 * np.pi * np.sin(2 * np.pi * x / l) * np.cos(2 * np.pi * y / l) * np.cos(2 * np.pi * z / l) / l \
                       + 4 * np.pi * np.sin(4 * np.pi * x / l) * np.cos(4 * np.pi * y / l) / l \
                       + 4 * np.pi * np.sin(4 * np.pi * x / l) * np.cos(4 * np.pi * z / l) / l
        grad[..., 1] = -8 * np.pi * np.sin(2 * np.pi * y / l) * np.cos(2 * np.pi * x / l) * np.cos(2 * np.pi * z / l) / l \
                       + 4 * np.pi * np.sin(4 * np.pi * y / l) * np.cos(4 * np.pi * x / l) / l \
                       + 4 * np.pi * np.sin(4 * np.pi * y / l) * np.cos(4 * np.pi * z / l) / l
        grad[..., 2] = -8 * np.pi * np.sin(2 * np.pi * z / l) * np.cos(2 * np.pi * x / l) * np.cos(2 * np.pi * y / l) / l \
                       + 4 * np.pi * np.sin(4 * np.pi * z / l) * np.cos(4 * np.pi * x / l) / l \
                       + 4 * np.pi * np.sin(4 * np.pi * z / l) * np.cos(4 * np.pi * y / l) / l
        return grad

    def hessian(self, p):
        raise NotImplementedError('NotImplemented')

    def jacobi_matrix(self, p):
        raise NotImplementedError('NotImplemented')

    def unit_normal(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad ** 2, axis=1, keepdims=True))
        n = grad / l
        return n

    def init_mesh(self, meshdata=None):
        raise NotImplementedError('NotImplemented')


class Gyroid:
    def __init__(self, l=3):
        self.box = [-3, 3, -3, 3, -3, 3]
        self.l = l

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        l = self.l
        return np.sin((2 * np.pi * x) / l) * np.cos((2 * np.pi * y) / l) \
            + np.sin((2 * np.pi * y) / l) * np.cos((2 * np.pi * z) / l) \
            + np.sin((2 * np.pi * z) / l) * np.cos((2 * np.pi * x) / l)

    def project(self, p, maxit=200, tol=1e-8):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        l = self.l
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = -2*np.pi*np.sin(2*np.pi*x/l)*np.sin(2*np.pi*z/l)/l \
                       + 2*np.pi*np.cos(2*np.pi*x/l)*np.cos(2*np.pi*y/l)/l
        grad[..., 1] = -2*np.pi*np.sin(2*np.pi*x/l)*np.sin(2*np.pi*y/l)/l \
                       + 2*np.pi*np.cos(2*np.pi*y/l)*np.cos(2*np.pi*z/l)/l
        grad[..., 2] = -2*np.pi*np.sin(2*np.pi*y/l)*np.sin(2*np.pi*z/l)/l \
                       + 2*np.pi*np.cos(2*np.pi*x/l)*np.cos(2*np.pi*z/l)/l
        return grad

    def hessian(self, p):
        raise NotImplementedError('NotImplemented')

    def jacobi_matrix(self, p):
        raise NotImplementedError('NotImplemented')

    def unit_normal(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad ** 2, axis=1, keepdims=True))
        n = grad / l
        return n

    def init_mesh(self, meshdata=None):
        raise NotImplementedError('NotImplemented')


class Neovius:
    def __init__(self, l=3):
        self.box = [-3, 3, -3, 3, -3, 3]
        self.l = l

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        l = self.l
        return 3 * (np.cos((2 * np.pi * x) / l) + np.cos((2 * np.pi * y) / l) + np.cos((2 * np.pi * z) / l)) \
            + 4 * np.cos((2 * np.pi * x) / l) * np.cos((2 * np.pi * y) / l) * np.cos((2 * np.pi * z) / l)

    def project(self, p, maxit=200, tol=1e-8):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        l = self.l
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = -8*np.pi*np.sin(2*np.pi*x/l)*np.cos(2*np.pi*y/l)*np.cos(2*np.pi*z/l)/l \
                       - 6*np.pi*np.sin(2*np.pi*x/l)/l
        grad[..., 1] = -8*np.pi*np.sin(2*np.pi*y/l)*np.cos(2*np.pi*x/l)*np.cos(2*np.pi*z/l)/l \
                       - 6*np.pi*np.sin(2*np.pi*y/l)/l
        grad[..., 2] = -8*np.pi*np.sin(2*np.pi*z/l)*np.cos(2*np.pi*x/l)*np.cos(2*np.pi*y/l)/l \
                       - 6*np.pi*np.sin(2*np.pi*z/l)/l
        return grad

    def hessian(self, p):
        raise NotImplementedError('NotImplemented')

    def jacobi_matrix(self, p):
        raise NotImplementedError('NotImplemented')

    def unit_normal(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad ** 2, axis=1, keepdims=True))
        n = grad / l
        return n

    def init_mesh(self, meshdata=None):
        raise NotImplementedError('NotImplemented')


class Schwarz_D:
    def __init__(self, l=3):
        self.box = [-3, 3, -3, 3, -3, 3]
        self.l = l

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        l = self.l
        return np.sin((2 * np.pi * x) / l) * np.sin((2 * np.pi * y) / l) * np.sin((2 * np.pi * z) / l) \
            + np.sin((2 * np.pi * x) / l) * np.cos((2 * np.pi * y) / l) * np.cos((2 * np.pi * z) / l) \
            + np.cos((2 * np.pi * x) / l) * np.sin((2 * np.pi * y) / l) * np.cos((2 * np.pi * z) / l) \
            + np.cos((2 * np.pi * x) / l) * np.cos((2 * np.pi * y) / l) * np.sin((2 * np.pi * z) / l)

    def project(self, p, maxit=200, tol=1e-8):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        l = self.l
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = -2*np.pi*np.sin(2*np.pi*x/l)*np.sin(2*np.pi*y/l)*np.cos(2*np.pi*z/l)/l \
                       - 2*np.pi*np.sin(2*np.pi*x/l)*np.sin(2*np.pi*z/l)*np.cos(2*np.pi*y/l)/l \
                       + 2*np.pi*np.sin(2*np.pi*y/l)*np.sin(2*np.pi*z/l)*np.cos(2*np.pi*x/l)/l \
                       + 2*np.pi*np.cos(2*np.pi*x/l)*np.cos(2*np.pi*y/l)*np.cos(2*np.pi*z/l)/l
        grad[..., 1] = -2*np.pi*np.sin(2*np.pi*x/l)*np.sin(2*np.pi*y/l)*np.cos(2*np.pi*z/l)/l \
                       + 2*np.pi*np.sin(2*np.pi*x/l)*np.sin(2*np.pi*z/l)*np.cos(2*np.pi*y/l)/l \
                       - 2*np.pi*np.sin(2*np.pi*y/l)*np.sin(2*np.pi*z/l)*np.cos(2*np.pi*x/l)/l \
                       + 2*np.pi*np.cos(2*np.pi*x/l)*np.cos(2*np.pi*y/l)*np.cos(2*np.pi*z/l)/l
        grad[..., 2] = 2*np.pi*np.sin(2*np.pi*x/l)*np.sin(2*np.pi*y/l)*np.cos(2*np.pi*z/l)/l \
                       - 2*np.pi*np.sin(2*np.pi*x/l)*np.sin(2*np.pi*z/l)*np.cos(2*np.pi*y/l)/l \
                       - 2*np.pi*np.sin(2*np.pi*y/l)*np.sin(2*np.pi*z/l)*np.cos(2*np.pi*x/l)/l \
                       + 2*np.pi*np.cos(2*np.pi*x/l)*np.cos(2*np.pi*y/l)*np.cos(2*np.pi*z/l)/l
        return grad

    def hessian(self, p):
        raise NotImplementedError('NotImplemented')

    def jacobi_matrix(self, p):
        raise NotImplementedError('NotImplemented')

    def unit_normal(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad ** 2, axis=1, keepdims=True))
        n = grad / l
        return n

    def init_mesh(self, meshdata=None):
        raise NotImplementedError('NotImplemented')


class Schwarz_P:
    def __init__(self, l=3):
        self.box = [-3, 3, -3, 3, -3, 3]
        self.l = l

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        l = self.l
        return np.cos((2 * np.pi * x) / l) + np.cos((2 * np.pi * y) / l) + np.cos((2 * np.pi * z) / l)

    def project(self, p, maxit=200, tol=1e-8):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        l = self.l
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = -2*np.pi*np.sin(2*np.pi*x/l)/l
        grad[..., 1] = -2*np.pi*np.sin(2*np.pi*y/l)/l
        grad[..., 2] = -2*np.pi*np.sin(2*np.pi*z/l)/l
        return grad

    def hessian(self, p):
        raise NotImplementedError('NotImplemented')

    def jacobi_matrix(self, p):
        raise NotImplementedError('NotImplemented')

    def unit_normal(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad ** 2, axis=1, keepdims=True))
        n = grad / l
        return n

    def init_mesh(self, meshdata=None):
        raise NotImplementedError('NotImplemented')


class C_I_2_Y:
    def __init__(self, l=3):
        self.box = [-3, 3, -3, 3, -3, 3]
        self.l = l

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 array or x, y, z")

        l = self.l
        return 2 * np.sin((4 * np.pi * x) / l) * np.cos((2 * np.pi * y) / l) * np.sin((2 * np.pi * z) / l) \
            + 2 * np.sin((4 * np.pi * y) / l) * np.cos((2 * np.pi * z) / l) * np.sin((2 * np.pi * x) / l) \
            + 2 * np.sin((4 * np.pi * z) / l) * np.cos((2 * np.pi * x) / l) * np.sin((2 * np.pi * y) / l) \
            + np.cos((4 * np.pi * x) / l) * np.cos((4 * np.pi * y) / l) \
            + np.cos((4 * np.pi * y) / l) * np.cos((4 * np.pi * z) / l) \
            + np.cos((4 * np.pi * z) / l) * np.cos((4 * np.pi * x) / l)

    def project(self, p, maxit=200, tol=1e-8):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        p0, d = project(self, p, maxit=maxit, tol=tol, returngrad=False, returnd=True)
        return p0, d

    def gradient(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        l = self.l
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[..., 0] = -4*np.pi*np.sin(2*np.pi*x/l)*np.sin(2*np.pi*y/l)*np.sin(4*np.pi*z/l)/l \
                       - 4*np.pi*np.sin(4*np.pi*x/l)*np.cos(4*np.pi*y/l)/l \
                       - 4*np.pi*np.sin(4*np.pi*x/l)*np.cos(4*np.pi*z/l)/l \
                       + 4*np.pi*np.sin(4*np.pi*y/l)*np.cos(2*np.pi*x/l)*np.cos(2*np.pi*z/l)/l \
                       + 8*np.pi*np.sin(2*np.pi*z/l)*np.cos(4*np.pi*x/l)*np.cos(2*np.pi*y/l)/l
        grad[..., 1] = 8*np.pi*np.sin(2*np.pi*x/l)*np.cos(4*np.pi*y/l)*np.cos(2*np.pi*z/l)/l \
                       - 4*np.pi*np.sin(4*np.pi*x/l)*np.sin(2*np.pi*y/l)*np.sin(2*np.pi*z/l)/l \
                       - 4*np.pi*np.sin(4*np.pi*y/l)*np.cos(4*np.pi*x/l)/l \
                       - 4*np.pi*np.sin(4*np.pi*y/l)*np.cos(4*np.pi*z/l)/l \
                       + 4*np.pi*np.sin(4*np.pi*z/l)*np.cos(2*np.pi*x/l)*np.cos(2*np.pi*y/l)/l
        grad[..., 2] = -4*np.pi*np.sin(2*np.pi*x/l)*np.sin(4*np.pi*y/l)*np.sin(2*np.pi*z/l)/l \
                       + 4*np.pi*np.sin(4*np.pi*x/l)*np.cos(2*np.pi*y/l)*np.cos(2*np.pi*z/l)/l \
                       + 8*np.pi*np.sin(2*np.pi*y/l)*np.cos(2*np.pi*x/l)*np.cos(4*np.pi*z/l)/l \
                       - 4*np.pi*np.sin(4*np.pi*z/l)*np.cos(4*np.pi*x/l)/l \
                       - 4*np.pi*np.sin(4*np.pi*z/l)*np.cos(4*np.pi*y/l)/l
        return grad

    def hessian(self, p):
        raise NotImplementedError('NotImplemented')

    def jacobi_matrix(self, p):
        raise NotImplementedError('NotImplemented')

    def unit_normal(self, p):
        if p.ndim == 1:
            p = p[np.newaxis, :]
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad ** 2, axis=1, keepdims=True))
        n = grad / l
        return n

    def init_mesh(self, meshdata=None):
        raise NotImplementedError('NotImplemented')
