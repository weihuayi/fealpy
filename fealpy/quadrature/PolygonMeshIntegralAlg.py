import numpy as np
from .GaussLobattoQuadrature import GaussLobattoQuadrature
from .GaussLegendreQuadrature import GaussLegendreQuadrature

class PolygonMeshIntegralAlg():
    def __init__(self, pmesh, q, cellmeasure=None, cellbarycenter=None):
        self.pmesh = pmesh

        self.cellmeasure = cellmeasure if cellmeasure is not None \
                else pmesh.entity_measure('cell')
        self.cellbarycenter = cellbarycenter if cellbarycenter is not None \
                else pmesh.entity_barycenter('cell')
        self.cellintegrator = pmesh.integrator(q)

        self.edgemeasure = pmesh.entity_measure('edge')
        self.edgebarycenter = pmesh.entity_barycenter('edge')
        self.edgeintegrator = GaussLegendreQuadrature(q)

    def triangle_measure(self, tri):
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        area = np.cross(v1, v2)/2
        return area

    def edge_integral(self, u, edgetype=False, q=None):
        mesh = self.pmesh
        NE = mesh.number_of_edges()
        node = mesh.entity('node')
        edge = mesh.entity('edge')

        qf = self.edgeintegrator
        bcs, ws = qf.quadpts, qf.weights

        ps = mesh.edge_bc_to_point(bcs)
        val = u(ps)
        if edgetype is True:
            e = np.einsum('i, ij..., j->j...', ws, val, self.edgemeasure)
        else:
            e = np.einsum('i, ij..., j->...', ws, val, self.edgemeasure)
        return e

    def integral(self, u, celltype=False, q=None):
        pmesh = self.pmesh
        node = pmesh.node
        bc = self.cellbarycenter

        edge = pmesh.entity('edge')
        edge2cell = pmesh.ds.edge_to_cell()

        NC = pmesh.number_of_cells()

        qf = self.cellintegrator if q is None else self.pmesh.integrator(q)
        bcs, ws = qf.quadpts, qf.weights

        tri = [bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]]
        a = self.triangle_measure(tri)
        pp = np.einsum('ij, jkm->ikm', bcs, tri)
        val = u(pp, edge2cell[:, 0])

        shape = (NC, ) + val.shape[2:]
        e = np.zeros(shape, dtype=np.float)

        ee = np.einsum('i, ij..., j->j...', ws, val, a)
        np.add.at(e, edge2cell[:, 0], ee)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if np.sum(isInEdge) > 0:
            tri = [
                    bc[edge2cell[isInEdge, 1]],
                    node[edge[isInEdge, 1]],
                    node[edge[isInEdge, 0]]
                    ]
            a = self.triangle_measure(tri)
            pp = np.einsum('ij, jkm->ikm', bcs, tri)
            val = u(pp, edge2cell[isInEdge, 1])
            ee = np.einsum('i, ij..., j->j...', ws, val, a)
            np.add.at(e, edge2cell[isInEdge, 1], ee)

        if celltype is True:
            return e
        else:
            return e.sum(axis=0)

    def fun_integral(self, f, celltype=False, q=None):
        def u(x, index):
            return f(x)
        return self.integral(u, celltype=celltype, q=q)

    def error(self, efun, celltype=False, power=None, q=None):
        e = self.integral(efun, celltype=celltype, q=q)
        if isinstance(e, np.ndarray):
            n = len(e.shape) - 1
            if n > 0:
                for i in range(n):
                    e = e.sum(axis=-1)
        if celltype is False:
            e = e.sum()

        if power is not None:
            return power(e)
        else:
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

    def edge_L2_error(self, u, uh, celltype=False, q=None):
        mesh = self.pmesh
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
