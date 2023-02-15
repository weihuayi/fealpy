import numpy as np
from .GaussLobattoQuadrature import GaussLobattoQuadrature
from .GaussLegendreQuadrature import GaussLegendreQuadrature

class PolygonMeshIntegralAlg():
    def __init__(self, mesh, q, cellmeasure=None, cellbarycenter=None):
        self.mesh = mesh

        self.integrator = mesh.integrator(q)
        self.cellintegrator = self.integrator 
        self.cellbarycenter = cellbarycenter if cellbarycenter is not None \
                else mesh.entity_barycenter('cell')
        self.cellmeasure = cellmeasure if cellmeasure is not None \
                else mesh.entity_measure('cell')

        self.edgemeasure = mesh.entity_measure('edge')
        self.edgebarycenter = mesh.entity_barycenter('edge')
        self.edgeintegrator = GaussLegendreQuadrature(q)

        self.facemeasure = self.edgemeasure
        self.facebarycenter = self.edgebarycenter
        self.faceintegrator = self.edgeintegrator

    def triangle_measure(self, tri):
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        area = np.cross(v1, v2)/2
        return area

    def edge_integral(self, u, q=None, index=None):
        """
        Note:

        edgetype 参数要去掉， 函数名字意味着是逐个实体上的积分

        """
        mesh = self.mesh
        NE = mesh.number_of_edges()
        node = mesh.entity('node')
        edge = mesh.entity('edge')

        qf = self.edgeintegrator
        bcs, ws = qf.quadpts, qf.weights

        index = index or np.s_[:]
        ps = mesh.edge_bc_to_point(bcs, index=index)
        val = u(ps) # TODO: 这里默认为空间坐标, 是否存在重心坐标的形式?
        e = np.einsum('q, qe..., e->e...', ws, val, self.edgemeasure[index])
        return e

    def face_integral(self, u, q=None, index=None):
        """
        """
        return self.edge_integral(u, facetype, q, index)

    def cell_integral(self, u,  q=None):
        """
        TODO:
            引入 power 参数
        """
        return self.integral(u, celltype=True, q=q)

    def integral(self, u, celltype=False, q=None):
        mesh = self.mesh
        node = mesh.node
        bc = self.cellbarycenter

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        NC = mesh.number_of_cells()

        qf = self.cellintegrator if q is None else self.mesh.integrator(q)
        bcs, ws = qf.quadpts, qf.weights

        tri = [bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]]
        a = self.triangle_measure(tri)
        pp = np.einsum('ij, jkm->ikm', bcs, tri, optimize=True)
        val = u(pp, edge2cell[:, 0])

        shape = (NC, ) + val.shape[2:]
        e = np.zeros(shape, dtype=np.float64)

        ee = np.einsum('i, ij..., j->j...', ws, val, a, optimize=True)
        np.add.at(e, edge2cell[:, 0], ee)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if np.sum(isInEdge) > 0:
            tri = [
                    bc[edge2cell[isInEdge, 1]],
                    node[edge[isInEdge, 1]],
                    node[edge[isInEdge, 0]]
                    ]
            a = self.triangle_measure(tri)
            pp = np.einsum('ij, jkm->ikm', bcs, tri, optimize=True)
            val = u(pp, edge2cell[isInEdge, 1])
            ee = np.einsum('i, ij..., j->j...', ws, val, a, optimize=True)
            np.add.at(e, edge2cell[isInEdge, 1], ee)

        if celltype is True:
            return e
        else:
            return e.sum(axis=0)

    def fun_integral(self, f, celltype=False, q=None):
        def u(x, index):
            return f(x)
        return self.integral(u, celltype=celltype, q=q)

    def error(self, u, v, celltype=False, power=2, q=None):
        """

        Notes
        -----
        给定两个函数，计算两个函数的之间的差，默认计算 L2 差（power=2)

        power 的取值可以是任意的 p。

        TODO
        ----
        1. 考虑无穷范数的情形
        """
        def efun(x, index):
            return np.abs(u(x) - v(x, index))**power

        e = self.integral(efun, celltype=celltype, q=q)
        if isinstance(e, np.ndarray):
            n = len(e.shape) - 1
            if n > 0:
                for i in range(n):
                    e = e.sum(axis=-1)

        if celltype == False:
            e = np.power(np.sum(e), 1/power)
        else:
            e = np.power(np.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e

    def L1_error(self, u, uh, celltype=False, q=None):
        def f(x, index):
            return np.abs(u(x) - uh(x, index))
        e = self.integral(f, celltype=celltype, q=q)
        return e

    def L2_error(self, u, uh, celltype=False, q=None):
        #TODO: deal with u is a discrete Function 
        def f(x, index):
            return (u(x) - uh(x, index))**2
        e = self.integral(f, celltype=celltype, q=q)
        if isinstance(e, np.ndarray):
            n = len(e.shape) - 1
            if n > 0:
                for i in range(n):
                    e = e.sum(axis=-1)
        if celltype is False:
            e = e.sum()

        return np.sqrt(e)

    def L2_error_1(self, u, uh, celltype=False, q=None):
        def f(x, index):
            return (u(x, index) - uh(x, index))**2
        e = self.integral(f, celltype=celltype, q=q)
        if isinstance(e, np.ndarray):
            n = len(e.shape) - 1
            if n > 0:
                for i in range(n):
                    e = e.sum(axis=-1)
        if celltype is False:
            e = e.sum()

        return np.sqrt(e)

    def edge_L2_error(self, u, uh, celltype=False, q=None):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        p = uh.space.p

        qf = self.edgeintegrator if q is None else GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights

        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        val = u(ps) - uh.edge_value(bcs)
        e = np.sqrt(np.sum(
                np.einsum(
                    'i, ij..., ij..., j->...', ws, val, val, self.edgemeasure
                    )/NE)
                )
        return e

    def Lp_error(self, u, uh, p, celltype=False, q=None):
        def f(x, index):
            return np.abs(u(x) - uh(x, index))**p
        e = self.integral(f, celltype=celltype, q=q)
        return e**(1/p)
