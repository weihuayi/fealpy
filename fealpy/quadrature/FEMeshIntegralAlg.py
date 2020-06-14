import numpy as np

class FEMeshIntegralAlg():
    def __init__(self, mesh, q, cellmeasure=None):
        self.mesh = mesh
        self.integrator = mesh.integrator(q, 'cell')

        self.cellintegrator = self.integrator
        self.cellbarycenter =  mesh.entity_barycenter('cell')
        self.cellmeasure = cellmeasure if cellmeasure is not None \
                else mesh.entity_measure('cell')

        self.edgemeasure = mesh.entity_measure('edge')
        self.edgebarycenter = mesh.entity_barycenter('edge')
        self.edgeintegrator = mesh.integrator(q, 'edge')

        GD = mesh.geo_dimension()
        if GD == 3:
            self.facemeasure = mesh.entity_measure('face')
            self.facebarycenter = mesh.entity_measure('face')
            self.faceintegrator = mesh.integrator(q, 'face') 
        else:
            self.facemeasure = self.edgemeasure
            self.facebarycenter = self.edgebarycenter
            self.faceintegrator = self.edgeintegrator

    def construct_matrix(self, basis0, basis1=None,  
            cell2dof0=None, gdof0=None, 
            cell2dof1=None, gdof1=None, 
            barycenter=True, q=None):

        mesh = self.mesh
        qf = self.integrator if q is None else mesh.integrator(q, 'cell')
        bcs, ws = qf.quadpts, qf.weights

        phi0 = basis0(bcs)
        phi1 = phi0 if basis1 is None else basis1(bcs)

        M = np.einsum('i, ijk..., ijm..., j->jkm', ws, phi0, phi1,
                self.cellmeasure, optimize=True)

        if cell2dof0 is not None: # construct the global matrix
            gdof0 = gdof0 or cell2dof0.max()
            ldof0 = cell2dof0.shape[1] 

            if cell2dof1 is not None:
                gdof1 = gdof1 or cell2dof1.max()
                ldof1 = cell2dof1.shape[1]
                I = np.einsum('k, ij->ijk', np.ones(ldof1), cell2dof0)
                J = np.einsum('k, ij->ikj', np.ones(ldof0), cell2dof1)
            else:
                gdof1 = gdof0
                ldof1 = ldof0
                I = np.einsum('k, ij->ijk', np.ones(ldof0), cell2dof0)
                J = I.swapaxes(-1, -2)

            M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))

        return M

    def construct_vector(self, f, basis, cell2dof, gdof=None, dim=None, barycenter=False):
        mesh = self.mesh
        qf = self.integrator if q is None else mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if barycenter:
            val = f(bcs)
        else:
            ps = mesh.bc_to_point(bcs, etype='cell')
            val = f(ps)
        phi = basis(bcs)
        bb = np.einsum('m, mi..., mik, i->ik...',
                ws, fval, phi, self.cellmeasure)
        gdof = gdof or cell2dof.max()
        shape = (gdof, ) if dim is None else (gdof, dim)
        b = np.zeros(shape, dtype=phi.dtype)
        if dim is None:
            np.add.at(b, cell2dof, bb)
        else:
            np.add.at(b, (cell2dof, np.s_[:]), bb)
        return b

    def construct_boundary_vector(self, u, basis, face2dof, threshold=None):
        mesh = self.mesh
        measure = self.facemeasure

        qf = self.edgeintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]
        ps = mesh.bc_to_point(bcs, etype='face', index=index)
        en = mesh.edge_unit_normal(index=index)
        val = u(ps)

        face2cell = mesh.ds.face_to_cell()
        index = face2cell[index, 0]
        phi = basis(ps, barycenter=False, index=index)


    def edge_integral(self, u, edgetype=False, q=None, barycenter=True):
        mesh = self.mesh

        qf = self.edgeintegrator if q is None else mesh.integrator(q, 'edge')
        bcs, ws = qf.quadpts, qf.weights

        if barycenter:
            val = u(bcs)
        else:
            ps = mesh.bc_to_point(bcs, etype='edge')
            val = u(ps)

        if edgetype is True:
            e = np.einsum('i, ij..., j->j...', ws, val, self.edgemeasure)
        else:
            e = np.einsum('i, ij..., j->...', ws, val, self.edgemeasure)
        return e

    def face_integral(self, u, facetype=False, q=None, barycenter=True):
        mesh = self.mesh

        qf = self.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights

        if barycenter:
            val = u(bcs)
        else:
            ps = mesh.bc_to_point(bcs, etype='face')
            val = u(ps)

        dim = len(ws.shape)
        s0 = 'abcde'
        s1 = '{}, {}j..., j->j...'.format(s0[0:dim], s0[0:dim])
        if facetype is True:
            e = np.einsum(s1, ws, val, self.facemeasure)
        else:
            e = np.einsum(s1, ws, val, self.facemeasure)
        return e

    def cell_integral(self, u, celltype=False, q=None, barycenter=True):
        mesh = self.mesh

        qf = self.integrator if q is None else mesh.integrator(q, 'cell')
        bcs, ws = qf.quadpts, qf.weights

        if barycenter:
            val = u(bcs)
        else:
            ps = mesh.bc_to_point(bcs, etype='cell')
            val = u(ps)
        dim = len(ws.shape)
        s0 = 'abcde'
        s1 = '{}, {}j..., j->j...'.format(s0[0:dim], s0[0:dim])
        if celltype is True:
            e = np.einsum(s1, ws, val, self.cellmeasure)
        else:
            e = np.einsum(s1, ws, val, self.cellmeasure)
        return e

    def integral(self, u, celltype=False, barycenter=True):
        qf = self.integrator
        bcs = qf.quadpts # 积分点 (NQ, 3)
        ws = qf.weights # 积分点对应的权重 (NQ, )
        if barycenter:
            val = u(bcs)
        else:
            ps = self.mesh.bc_to_point(bcs) # (NQ, NC, 2)
            val = u(ps)
        dim = len(ws.shape)
        s0 = 'abcde'
        s1 = '{}, {}j..., j->j...'.format(s0[0:dim], s0[0:dim])
        e = np.einsum(s1, ws, val, self.cellmeasure)
        if celltype is True:
            return e
        else:
            return e.sum()

    def L2_norm(self, uh, celltype=False):
        def f(x):
            return uh(x)**2
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return np.sqrt(e.sum())
        else:
            return np.sqrt(e)

    def L2_norm_1(self, uh, celltype=False):
        def f(x):
            return np.sum(uh**2, axis=-1)*self.cellmeasure

        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return np.sqrt(e.sum())
        else:
            return np.sqrt(e)

    def L1_error(self, u, uh, celltype=False):
        def f(x):
            xx = self.mesh.bc_to_point(x)
            return np.abs(u(xx) - uh(x))
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return e.sum()
        else:
            return e
        return

    def L2_error(self, u, uh, celltype=False):
        def f(bc):
            xx = self.mesh.bc_to_point(bc)
            return (u(xx) - uh(bc))**2
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return np.sqrt(e.sum())
        else:
            return np.sqrt(e)
        return 

    def L2_error_uI_uh(self, uI, uh, celltype=False):
        def f(x):
            return (uI(x) - uh(x))**2
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return np.sqrt(e.sum())
        else:
            return np.sqrt(e)
        return 

    def Lp_error(self, u, uh, p, celltype=False):
        def f(x):
            xx = self.mesh.bc_to_point(x)
            return np.abs(u(xx) - uh(x))**p
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return e.sum()**(1/p)
        else:
            return e**(1/p)
        return 
