import numpy as np
import types

class PolygonMeshIntegralAlg():
    def __init__(self, integrator, pmesh, area=None, barycenter=None):
        self.pmesh = pmesh
        self.integrator = integrator
        if area is None:
            self.area = pmesh.entity_measure('cell')
        else:
            self.area = area

        if barycenter is None:
            self.barycenter = pmesh.entity_barycenter('cell')
        else:
            self.barycenter = barycenter

    def triangle_area(self, tri):
        v1 = tri[1] - tri[0] 
        v2 = tri[2] - tri[0] 
        area = np.cross(v1, v2)/2
        return area

    def integral(self, u, celltype=False):
        pmesh = self.pmesh
        node = pmesh.node
        bc = self.barycenter
        #print('bc', bc.shape)

        edge = pmesh.ds.edge
        edge2cell = pmesh.ds.edge2cell

        NC = pmesh.number_of_cells()
        #print('NC', NC)

        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights
        #print('bcs',bcs)


        tri = [bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]]
        a = self.triangle_area(tri)
        pp = np.einsum('ij, jkm->ikm', bcs, tri)
        val = u(pp, edge2cell[:, 0])

        shape = (NC, ) + val.shape[2:]
        e = np.zeros(shape, dtype=np.float)

        ee = np.einsum('i, ij..., j->j...', ws, val, a)
        np.add.at(e, edge2cell[:, 0], ee)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if np.sum(isInEdge) > 0:
            tri = [bc[edge2cell[isInEdge, 1]], node[edge[isInEdge, 1]], node[edge[isInEdge, 0]]]
            a = self.triangle_area(tri)
            pp = np.einsum('ij, jkm->ikm', bcs, tri)
            val = u(pp, edge2cell[isInEdge, 1])
            ee = np.einsum('i, ij..., j->j...', ws, val, a)
            np.add.at(e, edge2cell[isInEdge, 1], ee)

        if celltype is True:
            return e
        else:
            return e.sum(axis=0) 

    def fun_integral(self, f, celltype=False):
        def u(x, cellidx):
            return f(x)
        return self.integral(u, celltype)

    def L1_error(self, u, uh, celltype=False):
        def f(x, cellidx):
            return np.abs(u(x) - uh(x, cellidx))
        e = self.integral(f, celltype=celltype)
        return e

    def L2_error(self, u, uh, celltype=False):
        def f(x, cellidx):
            #print(u(x).shape, uh(x, cellidx))
            return (u(x) - uh(x, cellidx))**2
        #print('uh',uh)
        #print('u',u)
        e = self.integral(f, celltype=celltype)
        if isinstance(e, np.ndarray):
            e = e.sum()
        return np.sqrt(e)

    def Lp_error(self, u, uh, p, celltype=False):
        def f(x, cellidx):
            return np.abs(u(x) - uh(x, cellidx))**p
        e = self.integral(f, celltype=celltype)
        return e**(1/p)

        

