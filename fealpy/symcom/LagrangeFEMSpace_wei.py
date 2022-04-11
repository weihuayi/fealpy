
import numpy as np
import sympy as sp

class LagrangeFEMSpace2d:
    def __init__(self, N=20):
        self.l = sp.symbols('lambda_0, lambda_1, lambda_2', real=True)

        s = 'n_0'
        for i in range(1, N):
            s = s +', n_%d'%(i)
        self.n = sp.symbols(s, real=True)

        s = 'c_0'
        for i in range(1, N):
            s = s + ', c_%d'%(i)
        self.c = sp.symbols(s, real=True)

        self.n2num = {}
        val = 1 
        i = 1
        for s in self.n:
            self.n2num[s] = val
            val *= i
            i += 1

        self.c2num = {}
        i = 0
        for s in self.c:
            self.c2num[s] = i
            i += 1


    def multi_index_matrix(self, p):
        ldof = (p+1)*(p+2)//2
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 3), dtype=np.int_)
        multiIndex[:, 2] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 1] = idx0 - multiIndex[:,2]
        multiIndex[:, 0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        return multiIndex

    def number_of_dofs(self, p):
        return (p+1)*(p+2)//2

    def basis(self, p):
        c = self.c
        l = self.l
        ldof = self.number_of_dofs(p)
        A = sp.ones(p+1, 3)
        A[1, 0] = p*l[0]
        A[1, 1] = p*l[1]
        A[1, 2] = p*l[2]
        for i in range(2, p+1):
            for j in range(3):
                A[i, j] = (p*l[j] - c[i-1])*A[i-1, j]*(1/sp.factorial(i))
        mi = self.multi_index_matrix(p)
        phi = sp.zeros(1, ldof)
        for i in range(ldof):
            phi[i] = A[mi[i, 0], 0]*A[mi[i, 1], 1]*A[mi[i, 2], 2]
        return phi

    def mass_matrix(self, p=1):
        ldof = self.number_of_dofs(p)
        M = sp.zeros(ldof, ldof)
        n = self.n
        mi = self.multi_index_matrix(p)
        phi = self.basis(p)
        for i in range(ldof):
            for j in range(ldof):
                M[i, j] = self.integrate(phi[i]*phi[j], p)
                M[i, j] = M[i, j].subs(self.c2num).subs(self.n2num)
        return M

    def integrate(self, f, p):
        f = f.expand()
        n = self.n
        r = 0
        #print('f=', f, 'f.as_coeff_add() = ', f.as_coeff_add())
        for m in f.as_coeff_add()[1]:
            c = m.as_coeff_mul()[0]
            coef, a = self.multi_index(m, p)
            #print("m:\n", m, c, coef, a)
            r += c*coef*n[a[0]]*n[a[1]]*n[a[2]]/n[sum(a)+2]

        return n[2]*r

    def multi_index(self, monoial, p):
        """
        @brief 返回单项式的系数和幂指数多重指标
        """
        l = self.l
        c = self.c
        d = monoial.as_powers_dict()
        a0 = int(d.get(l[0]) or 0)
        a1 = int(d.get(l[1]) or 0)
        a2 = int(d.get(l[2]) or 0)

        coef = 1
        for i in range(1, p):
            val = d.get(c[i])
            if val:
                coef = coef*c[i]**val
        return coef, (a0, a1, a2)


if __name__ == "__main__":
    p=1
    space = LagrangeFEMSpace2d()
    M = space.mass_matrix(p)
    print(M)
