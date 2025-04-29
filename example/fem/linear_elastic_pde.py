import numpy as np
from sympy import symbols, sin, cos, Matrix, lambdify
from sympy import derive_by_array, eye, tensorcontraction

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian

class LinearElasticPDE():
    def __init__(self, u, lambda0, lambda1):
        x, y = symbols('x y')
        self.u = u
        self.lambda0 = lambda0
        self.lambda1 = lambda1

        grad_u = Matrix([[0, 0], [0, 0]])
        grad_u[0, 0] = u[0].diff(x)
        grad_u[0, 1] = u[0].diff(y)
        grad_u[1, 0] = u[1].diff(x)
        grad_u[1, 1] = u[1].diff(y)

        epsilon = (grad_u + grad_u.T) / 2     # 对称化操作

        trepsilon = tensorcontraction(epsilon, (0, 1))

        c0 = 1/lambda0
        c1 = lambda1/(lambda0 - 2*lambda1)
        sigma = c0*(c1*trepsilon*eye(2) + epsilon) 

        f = [-sigma[0, 0].diff(x) - sigma[0, 1].diff(y), 
             -sigma[1, 0].diff(x) - sigma[1, 1].diff(y)]

        self.sigmaxx = lambdify((x, y), sigma[0, 0], 'numpy')
        self.sigmayy = lambdify((x, y), sigma[1, 1], 'numpy')
        self.sigmaxy = lambdify((x, y), sigma[0, 1], 'numpy')

        self.fx = lambdify((x, y), f[0], 'numpy')
        self.fy = lambdify((x, y), f[1], 'numpy')

        self.ux = lambdify((x, y), u[0], 'numpy')
        self.uy = lambdify((x, y), u[1], 'numpy')

    def stress(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sigma = bm.zeros(p.shape[:-1] + (3, ), dtype=bm.float64)
        sigma[..., 0] = self.sigmaxx(x, y)
        sigma[..., 1] = self.sigmaxy(x, y)
        sigma[..., 2] = self.sigmayy(x, y)
        return sigma

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        f = bm.zeros(p.shape[:-1] + (2, ), dtype=bm.float64)
        f[..., 0] = self.fx(x, y)
        f[..., 1] = self.fy(x, y)
        return f

    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        u = bm.zeros(p.shape[:-1] + (2, ), dtype=bm.float64)
        u[..., 0] = self.ux(x, y)
        u[..., 1] = self.uy(x, y)
        return u

    def boundart_displacement(self, p):
        return self.displacement(p) 

    def boundart_stress(self, p, n):
        sigma = self.stress(p)
        bs = bm.zeros(p.shape, dtype=bm.float64)
        bs[..., 0] = sigma[..., 0]*n[..., 0] + sigma[..., 1]*n[..., 1]
        bs[..., 1] = sigma[..., 2]*n[..., 0] + sigma[..., 1]*n[..., 1]
        return bs

class LinearElasticPDE3d():
    def __init__(self, u, lambda0, lambda1):
        x, y, z = symbols('x y z')
        self.u = u
        self.lambda0 = lambda0
        self.lambda1 = lambda1

        variables = [x, y, z]

        grad_u = Matrix([[u[i].diff(variables[j]) for j in range(3)] for i in range(3)])
        epsilon = (grad_u + grad_u.T) / 2     # 对称化操作

        trepsilon = tensorcontraction(epsilon, (0, 1))

        c0 = 1/lambda0
        c1 = lambda1/(lambda0 - 3*lambda1)
        sigma = c0*(c1*trepsilon*eye(3) + epsilon) 

        f = [-sigma[0, 0].diff(x) - sigma[0, 1].diff(y) - sigma[0, 2].diff(z),
             -sigma[1, 0].diff(x) - sigma[1, 1].diff(y) - sigma[1, 2].diff(z),
             -sigma[2, 0].diff(x) - sigma[2, 1].diff(y) - sigma[2, 2].diff(z)]

        self.sigmaxx = lambdify((x, y, z), sigma[0, 0], 'numpy')
        self.sigmaxy = lambdify((x, y, z), sigma[0, 1], 'numpy')
        self.sigmaxz = lambdify((x, y, z), sigma[0, 2], 'numpy')
        self.sigmayy = lambdify((x, y, z), sigma[1, 1], 'numpy')
        self.sigmayz = lambdify((x, y, z), sigma[1, 2], 'numpy')
        self.sigmazz = lambdify((x, y, z), sigma[2, 2], 'numpy')

        self.fx = lambdify((x, y, z), f[0], 'numpy')
        self.fy = lambdify((x, y, z), f[1], 'numpy')
        self.fz = lambdify((x, y, z), f[2], 'numpy')

        self.ux = lambdify((x, y, z), u[0], 'numpy')
        self.uy = lambdify((x, y, z), u[1], 'numpy')
        self.uz = lambdify((x, y, z), u[2], 'numpy')

    def stress(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sigma = bm.zeros(p.shape[:-1] + (6, ), dtype=bm.float64)
        sigma[..., 0] = self.sigmaxx(x, y, z)
        sigma[..., 1] = self.sigmaxy(x, y, z)
        sigma[..., 2] = self.sigmaxz(x, y, z)
        sigma[..., 3] = self.sigmayy(x, y, z)
        sigma[..., 4] = self.sigmayz(x, y, z)
        sigma[..., 5] = self.sigmazz(x, y, z)
        return sigma

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        f = bm.zeros(p.shape[:-1] + (3, ), dtype=bm.float64)
        f[..., 0] = self.fx(x, y, z)
        f[..., 1] = self.fy(x, y, z)
        f[..., 2] = self.fz(x, y, z)
        return f

    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = bm.zeros(p.shape[:-1] + (3, ), dtype=bm.float64)
        u[..., 0] = self.ux(x, y, z)
        u[..., 1] = self.uy(x, y, z)
        u[..., 2] = self.uz(x, y, z)
        return u

    def boundart_displacement(self, p):
        return self.displacement(p) 

    def boundart_stress(self, p, n):
        symidx = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        sigma = self.stress(p)
        bs = bm.zeros(p.shape, dtype=bm.float64)
        bs[..., 0] = bm.sum(sigma[..., symidx[0]]*n, axis=-1) 
        bs[..., 1] = bm.sum(sigma[..., symidx[1]]*n, axis=-1)
        bs[..., 2] = bm.sum(sigma[..., symidx[2]]*n, axis=-1)
        return bs


if __name__ == '__main__':
    x, y = symbols('x y')
    u0 = sin(x)
    u1 = cos(y)

    u = [u0, u1]
    pde = LinearElasticPDE(u, 1, 1)


    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=10, ny=10)
    points = mesh.entity('node')
    cell = mesh.entity('cell')

    a = pde.displacement(points)
    print(a)






















