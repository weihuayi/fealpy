import numpy as np
from ..functionspace import LagrangeFiniteElementSpace



class MaxwellNedelecFEMResidualEstimator2d():
    def __init__(self, uh, pde):
        self.uh = uh
        self.space = uh.space
        self.mesh = self.space.mesh
        self.pde = pde

    def estimate(self):
        cellmeasure = self.mesh.entity_measure('cell')
        ch = np.sqrt(cellmeasure)

        R1 = self.cell_error_one() ** 2
        R2 = self.cell_error_two() ** 2
        J1 = self.edge_one_error() ** 2
        J2 = self.edge_two_error() ** 2
        eta = ch ** 2 * (R1 + R2) * ch * (J1 + J2)
        eta = ch * J2
        return eta ** 0.5


    def cell_error_one(self, q=1):
        p = 2
        lspace = LagrangeFiniteElementSpace(self.mesh, p=2, spacetype='D')

        luh = lspace.function()

        qf = self.mesh.integrator(q=8)
        bcs, ws = qf.get_quadrature_points_and_weights()

        '''
        1. 估计单元内的后验误差
        R₁ = ▽ × (1/μ ▽ × Eₕ) - k²𝜀Eₕ - F

        1.1 计算 1/μ ▽ × Eₕ , 并将其插值到 Lagrange 空间
        '''

        # 0.0 构造一个 L2 的函数

        ib = self.mesh.multi_index_matrix(p) / p
        ps = self.mesh.bc_to_point(ib)

        # 1/μ ▽ × Eₕ
        luh[:] = self.uh.curl_value(ib).flatten() / self.pde.mu(ps).flatten()

        '''
        若 u 是一个标量, 那么 ▽ × u = (∂ᵧu, -∂ₓu)ᵀ
        '''
        grad_luh = lspace.grad_value(luh, ib)
        p_x = grad_luh[..., 0]  # (NQ, NC)
        p_y = grad_luh[..., 1]

        # cuh 存储了 (∂ᵧu, -∂ₓu)ᵀ
        cuh = np.zeros_like(grad_luh)  # (NQ, NC, dim=2)
        cuh[..., 0] = p_y
        cuh[..., 1] = -p_x

        '''
        计算 mass = k²𝜀Eₕ - F
        '''
        k2 = self.pde.kappa ** 2
        eps = self.pde.eps(ps)  # (NQ, NC, 2, 2)
        eh = self.uh(ib)  # (NQ, NC, 2)
        f = self.pde.source(ib)
        mass = k2 * np.einsum('qcij, qci -> qcj', eps, eh)
        R1 = (cuh - mass) ** 2

        '''
        下面是对 R1 在每个单元上做积分
        '''
        cellmeasure = self.mesh.entity_measure('cell')
        intR1 = np.einsum('i, qja, j -> j', ws, R1, cellmeasure)
        return np.sqrt(intR1)

    def cell_error_two(self):
        p = 2
        lspace = LagrangeFiniteElementSpace(self.mesh, p=p, spacetype='D')

        qf = self.mesh.integrator(q=8)
        bcs, ws = qf.get_quadrature_points_and_weights()

        aval = lspace.function(dim=2)
        ib = self.mesh.multi_index_matrix(p) / p
        ps = self.mesh.bc_to_point(ib)

        k2 = self.pde.kappa ** 2
        F = self.pde.source(ps)
        eps = self.pde.eps(ps)
        q = np.einsum('ijbq, ijw -> ijq', eps, self.uh.value(ib))
        aval[:] = (k2 * q + F).reshape(-1, 2)

        R2 = lspace.div_value(aval, ib) ** 2
        '''
        下面是对 R2 在每个单元上做积分
        '''
        cellmeasure = self.mesh.entity_measure('cell')
        intR2 = np.einsum('i, qj, j -> j', ws, R2, cellmeasure)
        return np.sqrt(intR2)

    def edge_one_error(self):
        bc = np.array([1 / 3, 1 / 3, 1 / 3])
        ps = self.mesh.bc_to_point(bc)
        edgemeasure = self.mesh.entity_measure('edge')

        curl_val = self.space.curl_value(self.uh, bc)  # (NC, )
        mu = self.pde.mu(ps) ** -1  # (NC, 2)

        n = self.mesh.face_unit_normal()  # (NC, 2)
        nn = np.zeros_like(n)
        nn[..., 0] = n[..., 1]
        nn[..., 1] = -n[..., 0]

        face2cell = self.mesh.ds.face_to_cell()
        curl = np.einsum('ci, c -> ci', mu, curl_val)
        J = edgemeasure * np.sum((curl[face2cell[:, 0]] - curl[face2cell[:, 1]]) * nn, axis=-1) ** 2

        NC = self.mesh.number_of_cells()
        J1 = np.zeros(NC)
        np.add.at(J1, face2cell[:, 0], J)
        np.add.at(J1, face2cell[:, 1], J)
        return np.sqrt(J1)

    def edge_two_error(self):
        bc = np.array([1 / 3, 1 / 3, 1 / 3])
        ps = self.mesh.bc_to_point(bc)

        val = self.space.value(self.uh, bc)  # (NC, )
        eps = self.pde.eps(ps)  # (NC, 2, 2)
        F = self.pde.source(ps)

        nu = self.mesh.face_unit_normal()  # 单位法向
        face2cell = self.mesh.ds.face_to_cell()

        mass = self.pde.kappa ** 2 * np.einsum('cii, ci -> ci', eps, val)
        a = mass + F

        edgemeasure = self.mesh.entity_measure('edge')
        J = edgemeasure *  np.sum((a[face2cell[:, 0]] - a[face2cell[:, 1]]) * nu, axis=-1) ** 2


        NC = self.mesh.number_of_cells()
        J2 = np.zeros(NC)
        np.add.at(J2, face2cell[:, 0], J)
        np.add.at(J2, face2cell[:, 1], J)
        return np.sqrt(J2)



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

