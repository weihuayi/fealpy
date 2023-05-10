
import numpy as np

class ProvidesSymmetricTangentOperatorIntegrator:
    def __init__(self, lam, mu, uh, d, H):
        self.lam = lam
        self.mu = mu
        self.uh = uh
        self.d = d
        self.kappa = self.lam + 2 * self.mu /3 # 压缩模量
        self.H = H

    def strain(self, uh):
        """
        @brief 给定一个位移，计算相应的应变，这里假设是线性元
        """
        mesh = self.mesh
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        gphi = mesh.grad_lambda()  # NC x 3 x 2

        s = np.zeros((NC, 2, 2), dtype=np.float64)
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
        n = (alpha + val) / 2.0
        p = (alpha - val) / 2.0
        return n, p

    def deviator(self, val):
        """
        @brief 计算偏差
        """
        '''
        diff = val - np.mean(val, axis=(1, 2))
        abs_val = np.abs(diff[:, ...])
        dev = np.mean(abs_val)
        '''
        I = np.eye(2)
        tr = np.trace(val)
        dev = val - (1/3) * tr * I
        return dev

    def disp_tangent_operator(self):
        uh = self.uh
        kappa = self.kappa
        mu = self.mu
        d = self.d
        H = self.H

        s = self.strain(uh) # 计算应变
        trs = np.trace(s, axis1=1, axis2=2) # 应变的迹
        tp = np.maximum(trs, 0) # 正应变的迹
        tn = trs - tp # 负应变的迹
        trp, trn = self.macaulay_operation(trs) # 迹的麦考利运算
        dev = self.deviator(s) # 应变的偏差
        val = np.einsum('ijk, ijk -> i', dev, dev)
        tsp = kappa * trp**2/2.0 + mu * val 
#        H = max(tsp, H) # 更新历史函数
       
        # 计算应力
        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        g_d = (1-d(bc))**2 + 1e-10
        sigma = np.einsum('i, ijk -> ijk', 2*mu*g_d, dev)
        val = g_d * trp + trn
        sigma[:, 0, 0] += val * kappa
        sigma[:, 1, 1] += val * kappa

        # 计算应力关于应变的偏导

        return sigma



    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None, out=None):
        self.space = space[0]
        self.mesh = space[0].mesh
        a = self.disp_tangent_operator()
        print(a)


    def assembly_cell_matrix_fast(self, space0, _, index=np.s_[:], cellmeasure=None):
        pass


    def assembly_cell_matrix_ref(self, space0, _, index=np.s_[:], cellmeasure=None):
        pass
