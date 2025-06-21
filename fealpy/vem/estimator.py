import numpy as np

from ..quadrature import GaussLegendreQuadrature


class PoissonCVEMEstimator:
    def __init__(self, space, M, PI1):
        self.space = space
        self.M = M
        self.PI1 = PI1

    def residual_estimate(self, uh, f):
        """
        @brief 
        """
        space = self.space
        smspace = space.smspace
        mesh = space.mesh
        cm = mesh.entity_measure('cell')
        p = space.p


        #phi = smspace.basis
        #def u(x, index):
        #    return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
        #bb = mesh.integral(u, q=p+3, celltype=True)
        #g = lambda x: inv(x[0])@x[1]
        #bb = np.concatenate(list(map(g, zip(self.M, bb))))
        #fh = smspace.function(array=bb)

        sh = space.project_to_smspace(uh, self.PI1) 
        def term0(x, index):
            return (f(x) + smspace.laplace_value(sh, x, index))**2

        e0 = mesh.integral(term0, q=p+3, celltype=True)
        e0 *= cm**2

        NE = mesh.number_of_edges()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])

        n = mesh.edge_unit_normal()
        t = mesh.edge_unit_tangent()
        em = mesh.entity_measure('edge')


        # 获取区间积分公式
        qf = GaussLegendreQuadrature(p+3) 
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.edge_bc_to_point(bcs)

        # 内部边上的积分
        lgrad = sh.grad_value(ps, index=edge2cell[:, 0])
        rgrad = sh.grad_value(ps, index=edge2cell[:, 1])
        e1 = np.zeros(NE, dtype=mesh.ftype)
        e2 = np.zeros(NE, dtype=mesh.ftype)

        t0 = np.einsum(
            'ijm, jm->j',
            lgrad[:, ~isBdEdge] - rgrad[:, ~isBdEdge],
            n[~isBdEdge])
        
        t1 = np.einsum('ijm, jm->j', lgrad[:, ~isBdEdge] - rgrad[:, ~isBdEdge], 
                t[~isBdEdge])

        e1[~isBdEdge] = (t0*em[~isBdEdge])**2
        e2[~isBdEdge] = (t1*em[~isBdEdge])**2

        np.add.at(e0, edge2cell[:, 0], e1)
        np.add.at(e0, edge2cell[~isBdEdge, 1], e1[~isBdEdge])

        np.add.at(e0, edge2cell[:, 0], e2)
        np.add.at(e0, edge2cell[~isBdEdge, 1], e2[~isBdEdge])
        return e0





class PoissonNCVEMEstimator:
    def __init__(self, space):
        self.space = space
