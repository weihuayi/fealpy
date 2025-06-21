import numpy as np

from ..geometry import huniform 

class TrussMesher2d:

    def __init__(self, domain, hmin, 
            fh=huniform, dptol=0.001, ttol=0.1, fscale=1.2
            ):

        self.domain = domain
        self.hmin = hmin
        self.fh = fh
        self.dptol = dptol
        self.ttol = ttol
        self.fscale = fscale

        eps = np.finfo(float).eps
        self.geps = 0.001*h
        self.deps = np.sqrt(eps)*h
        self.dt = 0.2

        self.maxmove = float('inf')

        self.time_elapsed = 0
        self.count = 0

    def set_init_mesh(self):
        """
        @brief 生成初始网格
        """
        domain = self.domian
        bbox = domain.bbox
        hmin = self.hmin

        xh = bbox[1] - bbox[0]
        yh = bbox[3] - bbox[2]
        N = int(xh/hmin)+1
        M = int(yh/(hmin*np.sqrt(3)/2))+1

        mg = np.mgrid[
                bbox[2]:bbox[3]:complex(0, M), 
                bbox[0]:bbox[1]:complex(0, N)
                ]
        x = mg[1, :, :]
        y = mg[0, :, :]
        x[1::2, :] = x[1::2, :] + h/2
        p = np.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), axis=1)
        p = p[domain(p) < -self.geps, :]
        r0 = 1/fh(p)**2
        p = p[np.random.random((p.shape[0],)) < r0/np.max(r0),:]

        pfix = domain.facet(0)

        if pfix is not None:
            p = np.concatenate((pfix, p), axis=0)

        t = self.delaunay(p)
        self.mesh = TriangleMesh(p, t)

    def construct_truss_mesh(self, ):
        """ Construct edge and edge2cell from cell
        """
        NC = self.NC
        NEC = self.NEC

        totalEdge = self.total_edge()
        _, i0, j = np.unique(np.sort(totalEdge, axis=-1),
                return_index=True,
                return_inverse=True,
                axis=0)
        NE = i0.shape[0]
        self.NE = NE

        self.edge2cell = np.zeros((NE, 4), dtype=self.itype)

        i1 = np.zeros(NE, dtype=self.itype)
        i1[j] = range(NEC*NC)

        self.edge2cell[:, 0] = i0//NEC
        self.edge2cell[:, 1] = i1//NEC
        self.edge2cell[:, 2] = i0%NEC
        self.edge2cell[:, 3] = i1%NEC

        self.edge = totalEdge[i0, :]



