
import numpy as np
import sympy as sp

class SimplexElementBasis:
    def __init__(self, GD, btype='lagrange'):
        self.GD = int(GD)
        lam = 'l0'
        glam = 'gl0'
        for i in range(1, GD+1):
            lam = lam + ', l%d'%(i)   
            glam = glam + ', gl%d'%(i)   
        self.l = sp.symbols(lam, real=True)
        self.gl = sp.symbols(glam, real=True)
        print(self.gl)

        if btype in ['lagrange', 'l']:
            self.basis = self.lagrange_basis
        elif btype in ['bernstein', 'b']:
            self.basis = self.bernstein_basis
    
    def number_of_dofs(self, p): 
        GD = self.GD
        val = 1
        for i in range(1, GD+1):
            val *= (i+p)/i 
        return int(val) 

    def multi_index_matrix(self, p):
        ldof = self.number_of_dofs(p)
        GD = self.GD
        if GD==1:
            multiIndex = np.zeros((ldof, 2), dtype=np.int_)
            multiIndex[:, 0] = np.arange(p, -1, -1)
            multiIndex[:, 1] = p - multiIndex[:, 0] 
        elif GD==2:
            idx = np.arange(0, ldof)
            idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
            multiIndex = np.zeros((ldof, 3), dtype=np.int_)
            multiIndex[:, 2] = idx - idx0*(idx0 + 1)/2
            multiIndex[:, 1] = idx0 - multiIndex[:,2]
            multiIndex[:, 0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        elif GD==3: 
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


    def lagrange_basis(self,p):
        l = self.l
        GD = self.GD
        ldof = self.number_of_dofs(p)
        A = sp.ones(p+1, GD+1)
        for i in range(1, p+1):
            for j in range(GD+1):
                A[i, j] = (p*l[j] - (i-1))*A[i-1, j]
        for i in range(1,p+1):
            A[i,:] /= sp.factorial(i)
        mi = self.multi_index_matrix(p)
        phi = sp.ones(1, ldof)
        for i in range(ldof):
            for j in range(GD+1):
                phi[i] *= A[mi[i, j], j]
        return phi

    def bernstein_basis(self,p):
        l = self.l
        GD = self.GD
        ldof = self.number_of_dofs(p)
        A = sp.ones(p+1, GD+1)
        for i in range(1, p+1):
            for j in range(GD+1):
                A[i, j] = l[j]*A[i-1, j]

        P = [i for i in range(p+1)]
        P[0] = 1
        for i in range(p):
            P[i+1] *= P[i]
            for j in range(GD+1):
                A[i+1, j] /= P[i+1]

        mi = self.multi_index_matrix(p)
        phi = sp.ones(1, ldof)
        for i in range(ldof):
            for j in range(GD+1):
                phi[i] *= A[mi[i, j], j]
        for i in range(ldof):
            phi[i] *= P[p]
        return phi

    def grad_barnstein_basis(self, p, phi=None):
        if phi is None:
            phi = self.bernstein_basis(p)

        gphi = []
        for ph in phi:
            gph = []
            for i in range(self.GD+1):
                gph.append(sp.diff(ph, self.l[i])*self.gl[i])
            gphi.append(gph)
        return gphi

    def hermite_basis(self):
        """
        @brief 基于 4 次 Bernstein 基来实现顶点 C_1 连续的 Hermite 元 
        """
        p = 4


    def multi_index(self, monoial):
        """
        @brief 幂指数多重指标
        """
        l = self.l
        GD = self.GD
        m = monoial.as_powers_dict()
        a = np.zeros(GD+1, dtype=np.int_) #返回幂指标
        for i in range(GD+1):
            a[i] = int(m.get(l[i]) or 0)
        return a


    def integrate(self, f):
        GD = self.GD
        f = f.expand()
        r = 0    #积分值
        for m in f.as_coeff_add()[1]:
            c = m.as_coeff_mul()[0] #返回系数
            a = self.multi_index(m) #返回单项式的幂指标
            temp = 1
            for i in range(GD+1):
                temp *= sp.factorial(a[i])
            r += sp.factorial(GD)*c*temp/sp.factorial(sum(a)+GD)
        return r + f.as_coeff_add()[0]
        
    
    def phi_phi_matrix(self, p1, p2, p3=None):
        ldof1 = self.number_of_dofs(p1)
        ldof2 = self.number_of_dofs(p2)
        phi1 = self.basis(p1)
        phi2 = self.basis(p2)
        M = sp.tensor.array.MutableDenseNDimArray(sp.zeros(ldof1*ldof2),(1,ldof1,ldof2))
        for i in range(ldof1):
            for j in range(ldof2):
                M[0,i, j] = self.integrate(phi1[i]*phi2[j])
        return M

    def gphi_gphi_matrix(self, p1, p2):
        l = self.l
        GD = self.GD
        ldof1 = self.number_of_dofs(p1)
        ldof2 = self.number_of_dofs(p2)
        phi1 = self.basis(p1)
        phi2 = self.basis(p2)
        S =sp.tensor.array.MutableDenseNDimArray(sp.zeros(ldof1*ldof2*(GD+1))*(GD+1)\
                , (ldof1,ldof2,GD+1,GD+1))
        for i in range(ldof1):
            for j in range(ldof2):
                for m in range(GD + 1):
                    for n in range(GD + 1):
                        temp= sp.diff(phi1[i],l[m])*sp.diff(phi2[j],l[n])
                        S[i,j,m,n] = self.integrate(temp) 
        return S


    def gphi_phi_matrix(self, p1, p2):
        l = self.l
        GD = self.GD
        ldof1 = self.number_of_dofs(p1)
        ldof2 = self.number_of_dofs(p2)
        phi1 = self.basis(p1)
        phi2 = self.basis(p2)
        #S = np.zeros(shape = (ldof1, ldof2, GD+1))
        S =sp.tensor.array.MutableDenseNDimArray(sp.zeros(ldof1*ldof2*(GD+1))\
                ,(ldof1, ldof2 ,GD+1))
        for i in range(ldof1):
            for j in range(ldof2):
                for n in range(GD + 1):
                    temp= sp.diff(phi1[i],l[n])*phi2[j]
                    S[i,j,n] = self.integrate(temp) 
        return S
    
    def phi_gphi_phi_matrix(self, p1, p2, p3):
        l = self.l
        GD = self.GD
        ldof1 = self.number_of_dofs(p1)
        ldof2 = self.number_of_dofs(p2)
        ldof3 = self.number_of_dofs(p3)
        phi1 = self.basis(p1)
        phi2 = self.basis(p2)
        phi3 = self.basis(p3)
        #S = np.zeros(shape = (ldof1, ldof2, ldof3, GD+1))
        S =sp.tensor.array.MutableDenseNDimArray(sp.zeros(ldof1*ldof2*ldof3*(GD+1))\
                ,(ldof1, ldof2 ,ldof3, GD+1))
        for i in range(ldof1):
            for j in range(ldof2):
                for k in range(ldof3):
                    for n in range(GD + 1):
                        temp= phi1[i]*sp.diff(phi2[j],l[n])*phi3[k]
                        S[i, j, k, n] = self.integrate(temp) 
        return S


    def phi_phi_phi_matrix(self, p1, p2, p3):
        l = self.l
        GD = self.GD
        ldof1 = self.number_of_dofs(p1)
        ldof2 = self.number_of_dofs(p2)
        ldof3 = self.number_of_dofs(p3)
        phi1 = self.basis(p1)
        phi2 = self.basis(p2)
        phi3 = self.basis(p3)
        #S = np.zeros(shape = (ldof1, ldof2, ldof3))
        S = sp.tensor.array.MutableDenseNDimArray(sp.zeros(ldof1*ldof2*ldof3)\
                ,(ldof1, ldof2 ,ldof3))
        for i in range(ldof1):
            for j in range(ldof2):
                for k in range(ldof3):
                        temp= phi1[i]*phi2[j]*phi3[k]
                        S[i, j, k] = self.integrate(temp) 
        return S

    def gphi_gphi_phi_matrix(self, p1, p2, p3):
        l = self.l
        GD = self.GD
        ldof1 = self.number_of_dofs(p1)
        ldof2 = self.number_of_dofs(p2)
        ldof3 = self.number_of_dofs(p3)
        phi1 = self.basis(p1)
        phi2 = self.basis(p2)
        phi3 = self.basis(p3)
        S=sp.tensor.array.MutableDenseNDimArray(sp.zeros(ldof3*ldof1*ldof2*(GD+1)*(GD+1))\
                , (ldof1,ldof2,ldof3,GD+1,GD+1))
        for i in range(ldof1):
            for j in range(ldof2):
                for k in range(ldof3):
                    for m in range(GD + 1):
                        for n in range(GD + 1):
                            temp = sp.diff(phi1[i],l[m])*sp.diff(phi2[j],l[n])*phi3[k]
                            S[i,j,k,m,n] = self.integrate(temp) 
        return S

    def sp_to_np_function(self, f_sp):
        GD = self.GD
        l = ['l'+str(i) for i in range(GD+1)]
        return sp.lambdify(l, f_sp, "numpy")

if __name__ == "__main__":
    from sympy import *
    space = SimplexElementBasis(2)
    M = space.gphi_gphi_phi_matrix(2, 2, 2)
    print(M)
