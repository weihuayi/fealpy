import numpy as np

class DistDomain2d():
    def __init__(self, fd, fh, bbox, pfix=None, *args):
        self.params = fd, fh, bbox, pfix, args

class DistDomain3d():
    def __init__(self, fd, fh, bbox, pfix=None, *args):
        self.params = fd, fh, bbox, pfix, args

def dcircle(p, cxy, r):
    x = p[:, 0]
    y = p[:, 1]
    return np.sqrt((x - cxy[0])**2 + (y - cxy[1])**2) - r

def dsine(p,cxy,r):
    x = p[:,0]
    y = p[:,1]
    return (y - cxy[1]) - r*np.sin(x-cxy[0])

def dparabolic(p,cxy,r):
    x = p[:,0]
    y = p[:,1]
    return (y - cxy[1])**2 - 2*r*x

def drectangle(p, box):
    return -dmin(
            dmin(dmin(p[:,1] - box[2], box[3]-p[:,1]), p[:,0] - box[0]),
            box[1] - p[:,0])  
        

def dpoly(p, poly):
    pass

def ddiff(d0, d1):
    return dmax(d0, -d1)

def dmin(d0, d1):
    dd = np.concatenate((d0.reshape((-1,1)), d1.reshape((-1,1))), axis=1)
    return dd.min(axis=1)

def dmax(d0, d1):
    dd = np.concatenate((d0.reshape((-1,1)), d1.reshape((-1,1))), axis=1)
    return dd.max(axis=1)


# 3D surface 
class Sphere(object):
    def __init__(self, center=np.array([0.0, 0.0, 0.0]), radius=1.0):
        self.center = center
        self.radius = radius
        self.box = [-1.2, 1.2, -1.2, 1.2, -1.2, 1.2]

    def __call__(self, *args):
        if len(args) == 1:
            p, = args
            x = p[:, 0]
            y = p[:, 1]
            z = p[:, 2]
        elif len(args) == 3:
            x, y, z = args
        else:
            raise ValueError("the args must be a N*3 or (X, Y, Z)")

        cx, cy, cz = self.center
        r = self.radius
        return np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) - r 

    def gradient(self, p):
        l = np.sqrt(np.sum((p - self.center)**2, axis=1))
        n = (p - self.center)/l.reshape(-1, 1)
        return n

    def unit_normal(self, p):
        return self.gradient(p)

    def hessian(self, p):
        pass

    def tangent_operator(self, p):
        pass

    def project(self, p):
        d= self(p)
        p = p - d.reshape(-1, 1)*self.gradient(p)
        return p, d

    def init_mesh(self):
        from .TriangleMesh import TriangleMesh
        t = (np.sqrt(6) - 1)/2
        point = np.array([
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
            [-1,-t, 0]], dtype=np.float)
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
            [10,9,11]], dtype=np.int)
        point, d = self.project(point)
        return TriangleMesh(point, cell) 

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

    def project(self, p):
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



def project(surface, p, maxit=200, tol=1e-8):
    eps = np.finfo(float).eps
    p0 = p
    k = 0
    while k < maxit:
        value = surface(p0)
        s = np.sign(value)
        grad = surface.gradient(p0)
        lg = np.sum(grad**2, axis=1)
        n = grad/np.sqrt(lg).reshape(-1, 1)
        p0 = p0 - value.reshape(-1, 1)*n
        v = s.reshape(-1, 1)*(p - p0)
        lv = np.sqrt(np.sum(v**2, axis=1))
        isNotOK = (lv > np.sqrt(eps))
        ev = n[isNotOK] - v[isNotOK]/(lv.reshape(-1, 1)[isNotOK])
        e = np.max(np.sqrt(value[isNotOK]**2/(lg.reshape(-1, 1)[isNotOK]) + np.sum(ev**2, axis=1)))
        if e > tol:
            break
        else:
            k += 1

    d = s*lv
    return p0, d

class EllipsoidSurface:
    def __init__(self, c=[5, 4, 3]):
        m = np.max(c)
        self.box = [-m, m, -m, m, -m, m]
        self.c = c

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

        a, b, c = self.c
        return x**2/a**2 + y**2/b**2 + z**2/c**2 - 1 

    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol)
        return p0, d

    def gradient(self, p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        a, b, c = self.c
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[:, 0] = 2*x/a**2 
        grad[:, 1] = 2*y/b**2 
        grad[:, 2] = 2*z/c**2 
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=1, keepdims=True))
        return grad/l

    def init_mesh(self):
        pass

class TorusSurface:
    def __init__(self):
        self.box = [-6, 6, -6, 6, -6, 6]

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

        return np.sqrt(x**2 + y**2 + z**2 + 16 - 8*np.sqrt(x**2 + y**2)) - 1 

    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol)
        return p0, d

    def gradient(self, p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        s1 = np.sqrt(x**2 + y**2)
        s2 = np.sqrt(s1**2 + z**2 + 16 - 8*s1)
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[:, 0] = -(4 - s1)*x/(s1*s2) 
        grad[:, 1] = -(4 - s1)*y/(s1*s2) 
        grad[:, 2] = z/s2 
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=1, keepdims=True))
        return grad/l

    def init_mesh(self):
        pass

class HeartSurface:
    def __init__(self):
        self.box = [-2, 2, -2, 2, -2, 2]

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

        return (x - z**2)**2 + y**2 + z**2 - 1.0 

    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol)
        return p0, d

    def gradient(self, p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[:, 0] = 2*(x - z**2)
        grad[:, 1] = 2*y
        grad[:, 2] = -4*(x - z**2)*z + 2*z
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=1, keepdims=True))
        return grad/l

    def init_mesh(self):
        pass

class OrthocircleSurface:
    def __init__(self, c=[0.075, 3]):
        self.box = [-2, 2, -2, 2, -2, 2]
        self.c = c

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
        p0, d = project(self, p, maxit=maxit, tol=tol)
        return p0, d

    def gradient(self, p):
        c = self.c
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        x2, y2, z2 = x**2, y**2, z**2
        d1 = (x2 + y2 - 1)**2 + z2
        d2 = (y2 + z2 - 1)**2 + x2
        d3 = (z2 + x2 - 1)**2 + y2
        d11 = d1**2 + z2
        d22 = d2**2 + x2
        d33 = d3**2 + y2
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[:, 0] = 4*d1*x*d22*d33 + 2*d11*x*d33 + 4*d11*d22*d3*x \
                - 2*c[0]**2*c[1]*x 
        grad[:, 1] = 4*d1*y*d22*d33 + 4*d11*d2*y*d33 + 2*d11*d22*y \
                -2 * c[0]**2 * c[1] * y
        grad[:, 2] = 2*z*d22*d33 + 4*d11*d2*z*d33 + 4*d11*d22*d3*z \
                - 2* c[0]**2 * c[1] * z
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=1, keepdims=True))
        return grad/l

    def init_mesh(self):
        pass

class QuarticsSurface:
    def __init__(self, r=1.05):
        self.box = [-2, 2, -2, 2, -2, 2]
        self.r = r

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
        x2 = x**2
        y2 = y**2
        z2 = z**2
        r = self.r
        return  (x2 - 1)**2 + (y2 - 1)**2 + (z2 - 1)**2 - r

    def project(self, p, maxit=200, tol=1e-8):
        p0, d = project(self, p, maxit=maxit, tol=tol)
        return p0, d

    def gradient(self, p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        x2, y2, z2 = x**2, y**2, z**2
        grad = np.zeros(p.shape, dtype=p.dtype)
        grad[:, 0] = 4*(x2 - 1)*x  
        grad[:, 1] = 4*(y2 - 1)*x 
        grad[:, 2] = 4*(z2 - 1)*x 
        return grad

    def unit_normal(self, p):
        grad = self.gradient(p)
        l = np.sqrt(np.sum(grad**2, axis=1, keepdims=True))
        return grad/l

    def init_mesh(self):
        pass

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
