import sympy as sp
from sympy.abc import x, y

from ..functionspace.femdof import multi_index_matrix1d
from ..functionspace.femdof import multi_index_matrix2d
from ..functionspace.femdof import multi_index_matrix3d


class MonomialSpace2d:
    def __init__(self, p, domain=sp.Matrix([-1, 1, -1, 1])):
        self.p = p
        self.domain = domain 
        self.barycenter =  sp.Matrix([(domain[0] + domain[1])/2, (domain[2] + domain[3])/2])
        self.x, self.y = sp.symbols('x, y', real=True)
        self.h = domain[1] - domain[0] 
        self.phi = self.basis()

    def number_of_dofs(self, p=None):
        p = self.p if p is None else p
        return (p+1)*(p+2)//2

    def basis(self, p=None):
        p = self.p if p is None else p
        dofs = self.number_of_dofs(p=p)
        phi = sp.ones(1, dofs)
        phi[1] = (self.x - self.barycenter[0])/self.h
        phi[2] = (self.y - self.barycenter[1])/self.h
        if p > 1:
            start = 3
            for i in range(2, p+1):
                for j, k in zip(range(start, start+i), range(start-i, start)):
                    phi[j] = phi[k]*phi[1]
                phi[start+i] = phi[start-1]*phi[2]
                start += i+1
        return phi

    def mass_matrix(self, p=None):
        p = self.p if p is None else p
        dofs = self.number_of_dofs(p=p)
        domain = self.domain
        phi = self.basis(p=p)
        n = self.number_of_dofs(p=p)
        H = sp.zeros(n, n)
        for i in range(n):
            for j in range(i, n):
                H[j, i] = H[i, j] = sp.integrate(phi[i]*phi[j], (self.x,
                    domain[0], domain[1]), (self.y, domain[2], domain[3]))
        return H

    def basis_jacobian(self, p=None):
        phi = self.basis(p=p)
        return phi.jacobian([self.x, self.y])

    def vector_basis(self, p=None):
        phi = self.basis(p=p)
        zero = sp.ZeroMatrix(1, phi.cols) 
        return sp.BlockMatrix([[phi, zero], [zero, phi]])

    def sym_grad_vector_basis(self, p=None):
        phi = self.basis(p=p)
        zero = sp.ZeroMatrix(1, phi.cols) 
        A00 = sp.BlockMatrix([[phi.diff(self.x), zero]])
        A11 = sp.BlockMatrix([[zero, phi.diff(self.y)]])
        A01 = sp.BlockMatrix([[phi.diff(self.y)/2, phi.diff(self.x)/2]])
        return A00, A11, A01 

    def div_sym_grad_vector_basis(self, p=None):
        phi = self.basis(p=p)
        xx = phi.diff(self.x, 2)
        yy = phi.diff(self.y, 2)
        xy = phi.diff(self.x).diff(self.y)
        return sp.BlockMatrix([[xx + yy/2, xy/2], [xy/2, xx/2 + yy]])

    def grad_space_basis(self, p=None):
        p = self.p if p is None else p
        phi = self.basis(p=p+1)
        M = self.h*phi.jacobian()
        return M

    def perp_grad_space_basis(self, p=None):
        p = self.p if p is None else p
        if p > 1:
            phi = self.basis(p=p-1)
            return sp.BlockMatrix([phi[2]*phi, -phi[1]*phi])
        else:
            return sp.ZeroMatrix(2, 1)

    def curl_vector_basis(self, p=None):
        phi = self.basis(p=p)
        return sp.BlockMatrix([[phi.diff(self.y), -phi.diff(self.x)]])

    def div_vector_basis(self, p=None):
        phi = self.basis(p=p)
        return sp.BlockMatrix([[phi.diff(self.x), phi.diff(self.y)]])

    def split_vector_basis(self, p=None):
        p = self.p if p is None else p
        m = multi_index_matrix2d(p)
        print(m)


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

