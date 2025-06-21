import numpy as np
from ..functionspace import LagrangeFiniteElementSpace


class MaxwellNedelecFEMResidualEstimator2d():
    def __init__(self, uh, dtype=None):
        self.uh = uh
        self.space = uh.space
        self.mesh = self.space.mesh
        self.dtype = np.float64 if dtype == None else dtype

    def estimate(self, eps=None, mu=None, source=None, kappa=1):
        cellmeasure = self.mesh.entity_measure('cell')
        ch = np.sqrt(cellmeasure)

        R1 = self.cell_error_one(source, mu, kappa, eps)
        R2 = self.cell_error_two(source, kappa, eps)
        J1 = self.edge_one_error(mu)
        J2 = self.edge_two_error(eps, source, kappa)
        eta = ch ** 2 * (R1 + R2) + ch * (J1 + J2)
        return eta

    def cell_error_one(self, source=None, mu=None, kappa=1, eps=None):
        p = 2
        lspace = LagrangeFiniteElementSpace(self.mesh, p=p, spacetype='D')

        luh = lspace.function()

        qf = self.mesh.integrator(q=8)
        bcs, ws = qf.get_quadrature_points_and_weights()
        point = self.mesh.bc_to_point(bcs)
        cellmeasure = self.mesh.entity_measure('cell')

        NC = self.mesh.number_of_cells()
        NQ = len(bcs)
        val = np.zeros(shape=(NQ, NC, 2))  # 被积函数

        '''
        1. 估计单元内的后验误差
        R₁ = ▽ × (1/μ ▽ × Eₕ) - k²𝜀Eₕ - F

        1.1 计算 1/μ ▽ × Eₕ , 并将其插值到 Lagrange 空间
        '''

        # 0.0 构造一个 L2 的函数

        ib = self.mesh.multi_index_matrix(p) / p
        ps = self.mesh.bc_to_point(ib)

        # 1/μ ▽ × Eₕ
        if mu == None:
            luh[:] = self.uh.curl_value(ib).flatten()
        else:
            luh[:] = self.uh.curl_value(ib).flatten() / mu(ps).flatten()

        grad_luh = lspace.grad_value(luh, bcs)
        p_x = grad_luh[..., 0]  # (NQ, NC)
        p_y = grad_luh[..., 1]

        '''
        若 u 是一个标量, 那么 ▽ × u = (∂ᵧu, -∂ₓu)ᵀ
        '''
        # cuh 存储了 (∂ᵧu, -∂ₓu)ᵀ
        val[..., 0] = p_y
        val[..., 1] = -p_x

        k2 = kappa ** 2
        eh = self.space.value(self.uh, bcs)

        if eps == None:
            val[:] -= k2 * eh
        else:
            eps = eps(point)
            val[:] -= k2 * np.einsum('qcij, qci -> qcj', eps, eh)

        if source == None:
            val = 0 - val
        else:
            f = source(point)
            val = f - val

        '''
        下面是对 R1 在每个单元上做积分
        '''
        val = np.power(val, 2)
        intR1 = np.einsum('i, ija, j -> j', ws, val, cellmeasure)
        return intR1

    def cell_error_two(self, source=None, kappa=1, eps=None):
        p = 2
        lspace = LagrangeFiniteElementSpace(self.mesh, p=p, spacetype='D')

        qf = self.mesh.integrator(q=8)
        bcs, ws = qf.get_quadrature_points_and_weights()
        point = self.mesh.bc_to_point(bcs)
        cellmeasure = self.mesh.entity_measure('cell')

        NC = self.mesh.number_of_cells()
        NQ = len(bcs)
        val = np.zeros(shape=(NQ, NC))  # 被积函数

        aval = lspace.function(dim=2)
        ib = self.mesh.multi_index_matrix(p) / p
        ps = self.mesh.bc_to_point(ib)

        k2 = kappa ** 2

        if eps == None:
            qq = k2 * self.uh.value(ib)
        else:
            eps = self.pde.eps(ps)
            qq = k2 * np.einsum('ijbw, ijw -> ijw', eps, self.uh.value(ib))

        if source is not None:
            qq = source(ps) + qq

        '''
        eps : (NQ, NC, 2, 2)
        uh : (NQ, NC, 2)
        '''

        # (NQ, NC) ==> gdof
        aval[:] = qq.reshape(-1, 2)
        val[:] = lspace.div_value(aval, bcs)

        '''
        下面是对 R2 在每个单元上做积分
        '''
        val = np.power(val, 2)
        intR2 = np.einsum('i, qj, j -> j', ws, val, cellmeasure)
        return intR2

    def edge_one_error(self, mu=None):
        bc = np.array([1 / 3, 1 / 3, 1 / 3])
        ps = self.mesh.bc_to_point(bc)
        edgemeasure = self.mesh.entity_measure('edge')

        curl_val = self.space.curl_value(self.uh, bc)  # (NC, )

        if mu == None:
            curl = curl_val[:, None]
        elif np.ndim(mu(ps)) == 1:
            mu = (mu(ps) ** -1)[:, None]  # (NC, 2)
            curl = np.einsum('ci, c -> ci', mu, curl_val)
        else:
            mu = mu(ps) ** -1
            curl = np.einsum('ci, c -> ci', mu, curl_val)

        n = self.mesh.face_unit_normal()  # (NC, 2)
        nn = np.zeros_like(n)
        nn[..., 0] = n[..., 1]
        nn[..., 1] = -n[..., 0]

        face2cell = self.mesh.ds.face_to_cell()

        J = edgemeasure * np.sum((curl[face2cell[:, 0]] - curl[face2cell[:, 1]]) * nn, axis=-1) ** 2

        NC = self.mesh.number_of_cells()
        J1 = np.zeros(NC, dtype=self.dtype)
        np.add.at(J1, face2cell[:, 0], J)
        np.add.at(J1, face2cell[:, 1], J)
        return J1

    def edge_two_error(self, eps=None, source=None, kappa=1):
        bc = np.array([1 / 3, 1 / 3, 1 / 3])
        ps = self.mesh.bc_to_point(bc)

        val = self.space.value(self.uh, bc)  # (NC, )
        if eps == None:
            NC = self.mesh.number_of_cells()
            eps = np.zeros(shape=(NC, 2, 2), dtype=self.dtype)
            eps[..., 0, 0] = 1
            eps[..., 1, 1] = 1
        else:
            eps = eps(ps)  # (NC, 2, 2)


        nu = self.mesh.face_unit_normal()  # 单位法向
        face2cell = self.mesh.ds.face_to_cell()

        mass = kappa ** 2 * np.einsum('cii, ci -> ci', eps, val)

        if source is not None:
            a = mass + source(ps)
        else:
            a = mass

        edgemeasure = self.mesh.entity_measure('edge')
        J = edgemeasure * np.sum((a[face2cell[:, 0]] - a[face2cell[:, 1]]) * nu, axis=-1) ** 2

        NC = self.mesh.number_of_cells()
        J2 = np.zeros(NC, dtype=self.dtype)
        np.add.at(J2, face2cell[:, 0], J)
        np.add.at(J2, face2cell[:, 1], J)
        return J2


class MaxwellNedelecFEMResidualEstimator3d():
    def __init__(self, uh, pde):
        self.uh = uh
        self.space = uh.space
        self.mesh = self.space.mesh
        self.pde = pde

    def estimate(self, q=1):
        lspace = LagrangeFiniteElementSpace(self.mesh, p=1, spacetype='D')

        qf = mesh.integrator(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
