
import sympy as sp
from sympy.abc import x, y, z
from ..functionspace.femdof import multi_index_matrix1d
from ..functionspace.femdof import multi_index_matrix2d
from ..functionspace.femdof import multi_index_matrix3d

class MonomialSpace3d:
    def __init__(self, p, domain=[-1, 1, -1, 1]):
        self.p = p
        self.domain = [-1, 1, -1, 1]
        self.x, self.y, self.z = sp.symbols('x, y, z', real=True)
        self.h = 2
        self.phi = self.basis()

    def number_of_dofs(self, p=None):
        p = self.p if p is None else p
        return (p+1)*(p+2)//2

    def basis(self):
        index = multi_index_matrix3d(self.p)
        return phi

    def integrate(self):
        phi = self.phi
        n = self.number_of_dofs()
        H = sp.zeros(n, n)
        for i in range(n):
            for j in range(i, n):
                H[j, i] = H[i, j] = sp.integrate(phi[i]*phi[j], (self.x, -1, 1), (self.y, -1, 1))
        return H

    def jacobian(self):
        return self.phi.jacobian([self.x, self.y])

    def construct_G(self):
        phi = self.phi
        J = self.jacobian()
        print(J)
        n = self.number_of_dofs()
        T00 = sp.zeros(n, n)
        T10 = sp.zeros(n, n)
        T11 = sp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                T00[i, j] = sp.integrate(J[i, 0]*J[j, 0], (self.x, -1, 1), (self.y, -1, 1))
                T11[i, j] = sp.integrate(J[i, 1]*J[j, 1], (self.x, -1, 1), (self.y, -1, 1))
                T10[i, j] = sp.integrate(J[i, 0]*J[j, 1], (self.x, -1, 1), (self.y, -1, 1))

        G00 = T00 + 1/2*T11
        G10 = 1/2*T10
        G11 = T11 + 1/2*T00

        for i in range(n):
            for j in range(n):
                t0 = sp.integrate(J[i, 0], (self.x, -1, 1), (self.y, -1, 1))
                t1 = sp.integrate(J[i, 1], (self.x, -1, 1), (self.y, -1, 1))
                t2 = sp.integrate(J[j, 0], (self.x, -1, 1), (self.y, -1, 1))
                t3 = sp.integrate(J[j, 1], (self.x, -1, 1), (self.y, -1, 1))
                G00[i, j] += t1*t3/16
                G10[i, j] -= t0*t3/16
                G11[i, j] += t0*t2/16

        for i in range(n):
            for j in range(n):
                t0 = sp.integrate(phi[i], (self.x, -1, 1), (self.y, -1, 1))
                t1 = sp.integrate(phi[j], (self.x, -1, 1), (self.y, -1, 1))
                G00[i, j] += t0*t1/16
                G11[i, j] += t0*t1/16

        p = self.p
        n1 = self.number_of_dofs(p-1)
        B0 = sp.zeros(n, n1)
        B1 = sp.zeros(n, n1)
        for i in range(n):
            for j in range(n1):
                B0[i, j] = sp.integrate(J[i, 0]*phi[j], (self.x, -1, 1), (self.y, -1, 1))
                B1[i, j] = sp.integrate(J[i, 1]*phi[j], (self.x, -1, 1), (self.y, -1, 1))

        return (G00, G11, G10.T), (B0, B1)

