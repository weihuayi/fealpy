import numpy as np

class SpectralDecomposition():
    def __init__(self, mesh, lam=121.15, mu=80.77, Gc=2.7e-3, l0=0.015):
        self.lam = lam
        self.mu = mu
        self.Gc = Gc
        self.l0 = l0
        
        self.mesh = mesh
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

    def strain(self, uh):
        """
        @brief 给定一个位移，计算相应的应变，这里假设是线性元
        """
        mesh = self.mesh
        mesh = self.mesh
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        gphi = mesh.grad_lambda()  # NC x 3 x 2

        s = np.zeros((NC, 2, 2), dtype=np.float64)
        if uh.space.doforder == 'sdofs':
            uh = uh.T
        s[:, 0, 0] = np.sum(uh[:, 0][cell] * gphi[:, :, 0], axis=-1)
        s[:, 1, 1] = np.sum(uh[:, 1][cell] * gphi[:, :, 1], axis=-1)

        val = np.sum(uh[:, 0][cell] * gphi[:, :, 1], axis=-1)
        val += np.sum(uh[:, 1][cell] * gphi[:, :, 0], axis=-1)
        val /= 2.0
        s[:, 0, 1] = val
        s[:, 1, 0] = val
        return s
    
    def macaulay_operation(self, alpha):
        """
        @brief 麦考利运算
        """
        val = np.abs(alpha)
        p = (alpha + val) / 2.0
        m = (alpha - val) / 2.0
        return p, m

    def strain_pm_eig_decomposition(self, s):
        """
        @brief 应变的正负特征分解
        @param[in] s 单元应变数组，（NC, 2, 2）
        """
        w, v = np.linalg.eigh(s) # w 特征值, v 特征向量
        p, m = self.macaulay_operation(w)

        sp = np.zeros_like(s)
        sm = np.zeros_like(s)

        for i in range(2):
            n0 = v[:, :, i]  # (NC, 2)
            n1 = p[:, i, None] * n0  # (NC, 2)
            sp += n1[:, :, None] * n0[:, None, :]

            n1 = m[:, i, None] * n0
            sm += n1[:, :, None] * n0[:, None, :]

        return sp, sm

    def strain_energy_density_decomposition(self, s):
        """
        @brief 应变能密度的分解
        """

        lam = self.lam
        mu = self.mu

        # 应变正负分解
        sp, sm = self.strain_pm_eig_decomposition(s)

        ts = np.trace(s, axis1=1, axis2=2)
        tp, tm = self.macaulay_operation(ts)
        tsp = np.trace(sp**2, axis1=1, axis2=2)
        tsm = np.trace(sm**2, axis1=1, axis2=2)

        phi_p = lam * tp ** 2 / 2.0 + mu * tsp
        phi_m = lam * tm ** 2 / 2.0 + mu * tsm
        return phi_p, phi_m

    def strain_eigs(self, s):
        """
        @brief 给定每个单元上的应变，进行特征值分解
        """

        w, v = np.linalg.eig(s)
        return w, v

    def heaviside(self, x, k=1):
        """
        @brief
        """
        val = np.zeros_like(x)
        val[x > 1e-13] = 1
        val[np.abs(x) < 1e-13] = 0.5
        val[x < -1e-13] = 0
        return val

    def dsigma_depsilon(self, phi, uh):
        """
        @brief 计算应力关于应变的导数矩阵
        @param phi 单元重心处的相场函数值, (NC, )
        @param uh 位移
        @return D 单元刚度系数矩阵
        """

        eps = 1e-10
        lam = self.lam
        mu = self.mu
        s = self.strain(uh)

        NC = len(s)
        D = np.zeros((NC, 3, 3), dtype=np.float64)

        ts = np.trace(s, axis1=1, axis2=2)
        w, v = self.strain_eigs(s)
        hwp = self.heaviside(w)
        hwm = self.heaviside(-w)

        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        c0 = (1 - phi(bc)) ** 2 + eps
        c1 = np.zeros_like(c0)
        c2 = np.zeros_like(c0)

        flag = (w[:, 0] == w[:, 1])
        c1[flag] = hwp[flag, 0] / 2.0
        c2[flag] = hwm[flag, 0] / 2.0

        r = np.sum(w[~flag], axis=-1) / np.sum(np.abs(w[~flag]), axis=-1)
        c1[~flag] = (1 + r) / 4.0 
        c2[~flag] = (1 - r) / 4.0


        d0 = 2 * mu * (c0 * hwp[:, 0] + hwm[:, 0])
        d1 = 2 * mu * (c0 * hwp[:, 1] + hwm[:, 1])
        d2 = 2 * mu * (c0 * c1 + c2)

        val = lam * (c0 * self.heaviside(ts) + self.heaviside(-ts))
        D[:, 0, 0] = val
        D[:, 0, 1] = val
        D[:, 1, 0] = val
        D[:, 1, 1] = val
        
        for m, n, i, j, k, l in self.index:
            D[:, m, n] += d0 * v[:, i, 0] * v[:, j, 0] * v[:, k, 0] * v[:, l, 0]
            D[:, m, n] += d1 * v[:, i, 1] * v[:, j, 1] * v[:, k, 1] * v[:, l, 1]
            val = v[:, i, 0] * v[:, k, 0] * v[:, j, 1] * v[:, l, 1]
            val += v[:, i, 0] * v[:, l, 0] * v[:, j, 1] * v[:, k, 1]
            val += v[:, i, 1] * v[:, k, 1] * v[:, j, 0] * v[:, l, 0]
            val += v[:, i, 1] * v[:, l, 1] * v[:, j, 0] * v[:, k, 0]
            D[:, m, n] += d2 * val
        D = (D + D.swapaxes(1,2))/2
        return D
    
    def get_dissipated_energy(self, d):

        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        mesh = self.mesh
        cm = mesh.entity_measure('cell')
        g = d.grad_value(bc)

        val = self.Gc/2/self.l0*(d(bc)**2+self.l0**2*np.sum(g*g, axis=1))
        dissipated = np.dot(val, cm)
        return dissipated

    
    def get_stored_energy(self, psi_s, d):
        eps = 1e-10

        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        c0 = (1 - d(bc)) ** 2 + eps
        mesh = self.mesh
        cm = mesh.entity_measure('cell')
        val = c0*psi_s
        stored = np.dot(val, cm)
        return stored

class VolumeBiasStrainDecomposition():
    def __init__(self, mesh, lam=121.15, mu=80.77, Gc=2.7e-3, l0=0.015):
        self.lam = lam
        self.mu = mu
        self.Gc = Gc
        self.l0 = l0
        
        self.mesh = mesh
        




