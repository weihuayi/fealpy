
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace


class ContinummDFModel2d:

    def __init__(self, 
            ka=2.7e-3,  # kN/mm，临界能量释放率
            l0=1.33e-2, # mm, 尺度系数
            la=121.15,  # kN/mm^{-2}， 拉梅第一参数
            mu=80.77    # kN/mm^{-2}，拉梅第二参数
            )
        self.ka = ka
        self.l0 = l0
        self.la = la
        self.mu = mu

        self.index = np.array([
            (0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 1, 1),
            (0, 2, 0, 0, 0, 1),
            (1, 0, 1, 1, 0, 0),
            (1, 1, 1, 1, 1, 1),
            (1, 2, 1, 1, 0, 1),
            (2, 0, 0, 1, 0, 0),
            (2, 1, 0, 1, 1, 1),
            (2, 2, 0, 1, 0, 1)], dtype=np.int_)

    def init_mesh(self, n=4):
        node = np.array([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]], dtype=np.float64)

        cell = np.array([
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5]], dtype=np.int_)


        mesh = TriangleMesh(node, cell)
        mesh.uniform_bisect(n=n)
        return mesh

    def strain(self, uh):
        """
        @brief 给定一个位移，计算相应的应变，这里假设是线性元
        """
        mesh = uh.space.mesh
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        gphi = mesh.grad_lambda() # NC x 3 x 2

        s = np.zeros((NC, 2, 2), dtype=np.float64)
        s[:, 0, 0] = np.sum(uh[:, 0][cell]*gphi[:, :, 0], axis=-1) 
        s[:, 1, 1] = np.sum(uh[:, 1][cell]*gphi[:, :, 1], axis=-1)

        val = np.sum(uh[:, 0][cell]*gphi[:, :, 1], axis=-1)
        val += np.sum(uh[:, 1][cell]*gphi[:, :, 0], axis=-1)
        val /=2.0 
        s[:, 0, 1] = val
        s[:, 1, 0] = val 
        return s

    def strain_pm_eig_decomposition(self, s):
        """
        @brief 应变的正负特征分解
        """
        w, v = np.linalg.eigh(s)
        p, m = self.macaulay_operation(w)

        sp = np.zeros_like(s)
        sm = np.zeros_like(s)

        for i in range(2)
            n0 = v[:, :, i] # (NC, 2)

            n1 = p[:, i, None]*n0 # (NC, 2)
            sp += = n1[:, :, None]*n0[:, None, :]
            
            n1 = m[:, i, None]*n0
            sm += = n1[:, :, None]*n0[:, None, :]

        return sp, sm


    def strain_energy_density_decomposition(self, s):
        """
        @brief 应变能密度的分解
        """

        
        la = self.la
        mu = self.mu

        sp, sm = self.strain_pm_eig_decomposition(s)

        tp, tm = macaulay_operation(np.trace(s))
        tsp = np.sum(sp**2, axis=(1, 2)) 
        tsm = np.sum(sm**2, axis=(1, 2))

        phi_p = la*tp**2/2.0 + mu*tsp
        phi_m = la*tm**2/2.0 + mu*tsm
        return phi_p, phi_m

    def macaulay_operation(self, alpha):
        """
        @brief 麦考利运算
        """
        val = np.abs(alpha)
        p = (alpha + val)/2.0
        m = (alpha - val)/2.0
        return p, m 

    def strain_eigs(self, s):
        """
        @brief 给定每个单元上的应变，进行特征值分解
        """

        w, v = np.linalg.eigh(s)
        return w, v

    def heaviside(self, x, k=1):
        """
        @brief 
        """
        val = 1.0/(1.0 + np.exp(-2*k*x))
        return val


    def dsigma_depsilon(self, bcs, phi, s):
        """
        @brief 计算应力关于应变的导数矩阵
        """

        eps = 1e-10 

        NQ = len(bcs)
        NC = len(s)
        D = np.zeros((NQ, NC, 3, 3), dtype=np.float64)

        w, v = self.strain_eigs(s)
        hwp = self.heaviside(w)
        hwm = self.heaviside(-w)

        flag = np.abs(w[:, 0] - w[:, 1]) < eps
        c0 = np.abs(w[:, 0]) - np.abs(w[:, 1])
        c1 = w[:, 0] - w[:, 1]

        flag = np.abs(c1) < eps
        cp = np.zeros_like(c0)
        cm = np.zeros_like(c0)

        cp[flag] = (hwp[flag, 0] + hwp[flag, 1])/2.0
        cp[~flag] = (1 + c0[~flag]/c1[~flag])/4.0

        cp[flag] = (hwm[flag, 0] + hwm[flag, 1])/2.0
        cp[~flag] = (1 - c0[~flag]/c1[~flag])/4.0

        ts = np.trace(s)
        la = self.la
        mu = self.mu

        val0 = (1 - phi(bcs))**2 + eps

        val1 = val0*self.heaviside(ts) # (NQ, NC) * (NC, )
        val1 += self.heaviside(-ts)
        val1 *= la

        D[:, :, 0, 0] = val1
        D[:, :, 0, 1] = val1
        D[:, :, 1, 0] = val1
        D[:, :, 1, 1] = val1


        for m, n, i, j, k, l in self.index:
            D[:, :, m, n] +=  





p = 1
n = 0

space = LagrangeFiniteElementSpace(mesh, p=p)

uh0 = space.function(dim=2)
phi0 = space.funciion()
H0 = space.function()

uh1 = space.function(dim=2)
phi1 = space.function()
H1 = space.function()

fig, axes = plt.subplots()
mesh.add_plot(axes)
#mesh.find_node(axes, showindex=True)
plt.show()

