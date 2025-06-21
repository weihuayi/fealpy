
import numpy as np
import sympy as sp

class LagrangeFEMSpace:
    def __init__(self, GD, N=20):
        self.GD = GD

        s = 'lambda_0'
        for i in range(1, GD+1):
            s = s + ', lambda_%d'%(i)
        self.l = sp.symbols(s, real=True)

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
        GD = self.GD
        ldof = self.number_of_dofs(p)
        if GD == 1:
            multiIndex = np.zeros((ldof, 2), dtype=np.int_)
            multiIndex[:, 0] = np.arange(p, -1, -1)
            multiIndex[:, 1] = p - multiIndex[:, 0] 
        elif GD == 2:
            idx = np.arange(0, ldof)
            idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
            multiIndex = np.zeros((ldof, 3), dtype=np.int_)
            multiIndex[:, 2] = idx - idx0*(idx0 + 1)/2
            multiIndex[:, 1] = idx0 - multiIndex[:,2]
            multiIndex[:, 0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        elif GD == 3: 
            idx = np.arange(1, ldof)
            idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
            idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
            idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
            idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
            multiIndex = np.zeros((ldof, 4), dtype=np.int_)
            multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
            multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
            multiIndex[1:, 1] = idx0 - idx2
            multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
        return multiIndex

    def number_of_dofs(self, p):
        GD = self.GD
        if GD == 1:
            ldof = p + 1
        elif GD == 2:
            ldof = (p+1)*(p+2)//2
        elif GD == 3:
            ldof = (p+1)*(p+2)*(p+3)//6
        return ldof 

    def basis(self, p):
        GD = self.GD
        n = self.n
        c = self.c
        l = self.l
        ldof = self.number_of_dofs(p)
        A = sp.ones(p+1, GD+1)
        for i in range(GD+1):
            A[1, i] = p*l[i]

        for i in range(2, p+1):
            for j in range(GD+1):
                A[i, j] = (p*l[j] - c[i-1])*A[i-1, j]/n[i]
        mi = self.multi_index_matrix(p)
        phi = sp.ones(1, ldof)
        for i in range(ldof):
            for j in range(GD+1):
                phi[i] *= A[mi[i, j], j]
        return phi

    def mass_matrix(self, p=1):
        ldof = self.number_of_dofs(p)
        M = sp.zeros(ldof, ldof)
        n = self.n
        phi = self.basis(p)
        for i in range(ldof):
            for j in range(ldof):
                M[i, j] = self.integrate(phi[i]*phi[j], p)
                print(i, j , M[i, j])
                M[i, j] = M[i, j].subs(self.c2num).subs(self.n2num)
                print(i, j , M[i, j])
        return M

    def integrate(self, f, p):
        """

        """
        GD = self.GD
        n = self.n
        f = f.expand()
        r = 0
        for m in f.as_coeff_add()[1]:
            c = m.as_coeff_mul()[0]
            coef, a = self.multi_index(m, p)
            c *= coef
            for i in range(GD+1):
                c *= n[a[i]]
            r += c/n[sum(a) + GD]
        return n[GD]*r + f.as_coeff_add()[0]

    def multi_index(self, monoial, p):
        """
        @brief 返回单项式的系数和幂指数多重指标
        """
        GD = self.GD
        l = self.l
        c = self.c
        d = monoial.as_powers_dict()
        a = np.zeros(GD+1, dtype=np.int_) #返回幂指标
        for i in range(GD+1):
            a[i] = int(d.get(l[i]) or 0)

        coef = 1
        for i in range(1, p):
            val = d.get(c[i])
            if val:
                coef = coef*c[i]**val
        return coef, a 

    def stiff_matrix(self, p):
        GD = self.GD
        l = self.l
        ldof1 = self.number_of_dofs(p1)
        ldof2 = self.number_of_dofs(p2)
        phi1 = self.basis(p1)
        phi2 = self.basis(p2)
        S = np.zeros(shape = (ldof1,ldof2,d+1,d+1))
        p = max(p1, p2)
        for i in range(ldof1):
            for j in range(ldof2):
                for m in range(GD + 1):
                    for n in range(GD+1):
                        temp= sp.diff(phi1[i],l[m])*sp.diff(phi2[j],l[n])
                        S[i,j,m,n] = self.integrate(temp) 
        return S

if __name__ == "__main__":
    from sympy import *
    p = 3
    GD = 3
    space = LagrangeFEMSpace(GD)
    M = space.mass_matrix(p)
    print(latex(1120*M))
