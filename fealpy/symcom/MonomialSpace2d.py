import sympy as sp
from sympy.abc import x, y
from ..functionspace.femdof import multi_index_matrix1d
from ..functionspace.femdof import multi_index_matrix2d
from ..functionspace.femdof import multi_index_matrix3d


class MonomialSpace2d:
    def __init__(self, p, domain=[-1, 1, -1, 1]):
        self.p = p
        self.domain = [-1, 1, -1, 1]
        self.x, self.y = sp.symbols('x, y', real=True)
        self.phi = self.basis()

    def number_of_dofs(self):
        p = self.p
        return (p+1)*(p+2)//2

    def basis(self):
        dofs = self.number_of_dofs()
        phi = sp.ones(1, dofs)
        p = self.p
        phi[1] = self.x
        phi[2] = self.y
        if p > 1:
            start = 3
            for i in range(2, p+1):
                for j, k in zip(range(start, start+i), range(start-i, start)):
                    phi[j] = phi[k]*phi[1]
                phi[start+i] = phi[start-1]*phi[2]
                start += i+1
        return phi

    def integrate(self):
        phi = self.phi
        n = self.number_of_dofs()
        H = sp.zeros(n, n)
        for i in range(n):
            for j in range(i, n):
                print(phi[i])
                H[j, i] = H[i, j] = sp.integrate(phi[i]*phi[j], (self.x, -1, 1), (self.y, -1, 1))
        return H


