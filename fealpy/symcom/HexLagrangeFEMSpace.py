import numpy as np
import sympy as sp

class HexLagrangeFEMSpace2d:
    def __init__(self, N=20):

        # 参考单元变量
        self.bu = sp.symbols('u, v, w', real=True)
        self.bx = sp.symbols('x, y, z', real=True)

        # 一个四面体单元的顶点
        self.bv = sp.ones(8, 3)
        for i in range(0, 8):
            self.bv[i, 0] = sp.symbols('x_%d'%(i), real=True)
            self.bv[i, 1] = sp.symbols('y_%d'%(i), real=True)
            self.bv[i, 2] = sp.symbols('z_%d'%(i), real=True)

        # 
        self.l = sp.ones(3, 2)
        for i in range(3):
            self.l[i, 0] = 1 - self.bu[i]
            self.l[i, 1] = self.bu[i]


    def basis(self, idx=None):
        l = self.l
        phi = sp.ones(1, 8)

        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    phi[c] = l[2, k]*l[1, j]*l[0, i] 
                    c += 1
        if idx:
            phi0 = sp.ones(1, 8)
            for i in range(8):
                phi0[i] = phi[idx[i]]
            return phi0
        else:
            return phi

    def jacobian(self, idx=None):
        """
        [[x/u x/v x/w]
         [y/u y/v y/w]
         [z/u z/v z/w]]
        """
        bv = self.bv
        gphi = self.grad_basis(idx=idx)
        J = sp.zeros(3, 3)
        for k in range(8):
            for i in range(3):
                for j in range(3):
                    J[i, j] = J[i, j] +gphi[k, i]*bv[k, j]
        return J


    def grad_basis(self, idx=None):
        """
        [[phi_0/u  phi_0/v phi_0/w]
         [phi_1/u, phi_1/v phi_1/w]
         .
         .
         .]
        """
        phi = self.basis(idx=idx)
        J = phi.jacobian(space.bu)
        return J


if __name__ == "__main__":
    idx = [0, 4, 6, 2, 1, 5, 7, 3]
    space = HexLagrangeFEMSpace2d()
    phi = space.basis(idx)
    u, v, w = space.bu
    bu2num = {u:1/2, v:1/2, w:1/2}
    print(phi.subs(bu2num))
    print(phi.jacobian(space.bu).subs(bu2num))

    J = space.jacobian(idx)
    print(J)

