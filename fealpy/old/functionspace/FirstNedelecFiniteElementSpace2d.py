import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import spsolve, cg

from scipy.sparse import csr_matrix, coo_matrix
from ..decorator import barycentric
from .Function import Function
from ..quadrature import FEMeshIntegralAlg
from ..decorator import timer


class NDof2d:
    def __init__(self, mesh):
        """
        Parameters
        ----------
        mesh : TriangleMesh object
        spacetype : the space type, 'first' or 'second' 

        Notes
        -----

        Reference
        ---------
        """
        self.mesh = mesh
        self.cell2dof = self.cell_to_dof()  # 默认的自由度数组

    def boundary_dof(self, threshold=None):
        """
        """
        idx = self.mesh.ds.boundary_edge_index()
        if threshold is not None:  # TODO: threshold 可以是一个指标数组
            bc = self.mesh.entity_barycenter('edge', index=idx)
            flag = threshold(bc)
            idx = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def is_boundary_dof(self, threshold=None):
        """
        """
        idx = self.mesh.ds.boundary_edge_index()
        if threshold is not None:  # TODO: threshold 可以是一个指标数组
            bc = self.mesh.entity_barycenter('edge', index=idx)
            flag = threshold(bc)
            idx = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def edge_to_dof(self, index=np.s_[:]):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edof = self.number_of_local_dofs('edge')
        edge2dof = np.arange(NE * edof).reshape(NE, edof)
        return edge2dof[index]

    def cell_to_dof(self):
        """
        """
        mesh = self.mesh

        cell2edge = mesh.ds.cell_to_edge()
        return cell2edge

    def number_of_local_dofs(self, doftype='all'):
        if doftype == 'all':  # number of all dofs on a cell
            return 3
        elif doftype in {'cell', 2}:  # number of dofs inside the cell
            return 0
        elif doftype in {'face', 'edge', 1}:  # number of dofs on a edge
            return 1
        elif doftype in {'node', 0}:  # number of dofs on a node
            return 0

    def number_of_global_dofs(self):
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge')
        gdof = NE * edof
        return gdof


class FirstNedelecFiniteElementSpace2d:
    def __init__(self, mesh, p=0, q=None, dof=None):
        """
        Parameters
        ----------
        mesh : TriangleMesh
        spacetype : the space type, 'first' or 'second'
        q : the index of quadrature fromula
        dof : the object for degree of freedom

        Note
        ----

        """
        self.mesh = mesh
        self.p = p

        if dof is None:
            self.dof = NDof2d(mesh)
        else:
            self.dof = dof

        self.integralalg = FEMeshIntegralAlg(self.mesh, p+2)
        self.integrator = self.integralalg.integrator

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

    def boundary_dof(self):
        return self.dof.boundary_dof()

    @barycentric
    def face_basis(self, bc, index=None, barycenter=True):
        return self.edge_basis(bc, index, barycenter)

    @barycentric
    def basis(self, bc, index=np.s_[:]):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(3,)` or `(NQ, 3)`
        Returns
        -------
        phi : numpy.ndarray
            the shape of 'phi' can be `(NC, ldof, 2)` or `(NQ, NC, ldof, 2)`

        See Also
        --------

        Notes
        -----
        """

                # 每个单元上的全部自由度个数
        ldof = self.number_of_local_dofs(doftype='all')
        edof = self.number_of_local_dofs(doftype='edge')

        mesh = self.mesh
        glambda = mesh.grad_lambda()  # (NC, 3, 2)
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()

        shape = bc.shape[:-1] + (NC, ldof, GD)
        phi = np.zeros(shape, dtype=self.ftype)  # (NQ, NC, ldof, 2)

        cell2edgesign = mesh.ds.cell_to_edge_sign()  # (NC, 3)

        phi[:, :, 0, :] = bc[..., 1, None, None] * glambda[:, 2, :] - bc[..., 2, None, None] * glambda[:, 1, :]
        phi[:, :, 1, :] = bc[..., 2, None, None] * glambda[:, 0, :] - bc[..., 0, None, None] * glambda[:, 2, :]
        phi[:, :, 2, :] = bc[..., 0, None, None] * glambda[:, 1, :] - bc[..., 1, None, None] * glambda[:, 0, :]

        phi[..., ~cell2edgesign[:, 0], 0, :] *= -1
        phi[..., ~cell2edgesign[:, 1], 1, :] *= -1
        phi[..., ~cell2edgesign[:, 2], 2, :] *= -1

        return phi

    @barycentric
    def rot_basis(self, bc, index=np.s_[:]):
        return self.curl_basis(bc, index, barycenter)

    @barycentric
    def curl_basis(self, bc, index=np.s_[:]):
        """

        Parameters
        ----------

        Notes
        -----
        curl [v_0, v_1] = \partial v_1/\partial x - \partial v_0/\partial y

        """

        # 每个单元上的全部自由度个数
        ldof = self.number_of_local_dofs(doftype='all')
        edof = self.number_of_local_dofs(doftype='edge')

        mesh = self.mesh
        glambda = mesh.grad_lambda()  # (NC, 3, 2)
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        shape = bc.shape[:-1] + (NC, ldof)
        phi = np.zeros(shape, dtype=self.ftype)  # (NQ, NC, ldof, 2)

        cell2edgesign = mesh.ds.cell_to_edge_sign()  # (NC, 3)

        phi[..., 0] = 2 * (glambda[..., 1, 0] * glambda[..., 2, 1] - glambda[..., 1, 1] * glambda[..., 2, 0])
        phi[..., 1] = 2 * (glambda[..., 2, 0] * glambda[..., 0, 1] - glambda[..., 2, 1] * glambda[..., 0, 0])
        phi[..., 2] = 2 * (glambda[..., 0, 0] * glambda[..., 1, 1] - glambda[..., 0, 1] * glambda[..., 1, 0])

        phi[..., ~cell2edgesign[:, 0], 0] *= -1
        phi[..., ~cell2edgesign[:, 1], 1] *= -1
        phi[..., ~cell2edgesign[:, 2], 2] *= -1
        
        return phi

    @barycentric
    def grad_basis(self, bc, index=np.s_[:]):
        """

        Parameters
        ----------

        Notes
        -----

        """
        stype = self.spacetype

        # 每个单元上的全部自由度个数
        ldof = self.number_of_local_dofs(doftype='all')
        edof = self.number_of_local_dofs(doftype='edge')

        mesh = self.mesh
        glambda = mesh.grad_lambda()  # (NC, 3, 2)
        GD = mesh.geo_dimension()

        shape = bc.shape[:-1] + (ldof, GD, GD)
        phi = np.zeros(shape, dtype=self.ftype)  # (NQ, NC, ldof, 2, 2)

        cell2edgesign = mesh.ds.cell_to_edge_sign()  # (NC, 3)

        phi[..., 0, 0, 1] = glambda[..., 1, 1] * glambda[..., 2, 0] - glambda[..., 1, 0] * glambda[..., 2, 1]
        phi[..., 0, 1, 0] = -1 * phi[..., 0, 0, 1]

        phi[..., 1, 0, 1] = glambda[..., 2, 1] * glambda[..., 0, 0] - glambda[..., 2, 0] * glambda[..., 0, 1]
        phi[..., 1, 1, 0] = -1 * phi[..., 1, 0, 1]

        phi[..., 2, 0, 1] = glambda[..., 0, 1] * glambda[..., 1, 0] - glambda[..., 0, 0] * glambda[..., 1, 1]
        phi[..., 2, 1, 0] = -1 * phi[..., 2, 0, 1]

        phi[..., ~cell2edgesign[:, 0], 0, :, :] *= -1
        phi[..., ~cell2edgesign[:, 1], 1, :, :] *= -1
        phi[..., ~cell2edgesign[:, 2], 2, :, :] *= -1
        return phi

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        phi = self.basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum('qclg, cl->qcg', phi, uh[cell2dof[index]])
        print(">>> , ", val.shape)
        return val

    @barycentric
    def rot_value(self, uh, bc, index=np.s_[:]):
        return self.curl_value(uh, bc, index)

    @barycentric
    def curl_value(self, uh, bc, index=np.s_[:]):
        cphi = self.curl_basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, cphi, uh[cell2dof[index]])
        return val

    @barycentric
    def edge_value(self, uh, bc, index=np.s_[:], left=True):
        phi = self.edge_basis(bc, index=index, left=left)
        edge2dof = self.dof.edge_to_dof()[index]
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[edge2dof])
        return val

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        pass

    def function(self, dim=None, array=None, dtype=np.float64):
        return Function(self, dim=dim, array=array, coordtype='barycentric',
                        dtype=dtype)

    def project(self, u):
        A = self.mass_matrix()
        b = self.source_vector(u)
        up = self.function()
        up[:] = spsolve(A, b)
        return up

    def interpolation(self, u):
        p = self.p
        mesh = self.mesh

        uh = self.function()
        edge2dof = self.dof.edge_to_dof()
        t = mesh.edge_unit_tangent()

        @barycentric
        def f0(bc):
            ps = mesh.bc_to_point(bc)
            return np.einsum('ijk, jk, ijm->ijm', u(ps), t, self.smspace.edge_basis(ps))

        uh[edge2dof] = self.integralalg.edge_integral(f0)

        if p >= 1:
            NE = mesh.number_of_edges()
            NC = mesh.number_of_cells()
            edof = self.number_of_local_dofs('edge')
            idof = self.number_of_local_dofs('cell')  # dofs inside the cell
            cell2dof = NE * edof + np.arange(NC * idof).reshape(NC, idof)

            @barycentric
            def f1(bc):  # TODO: check here
                ps = mesh.bc_to_point(bc)
                return np.einsum('ijk, ijm->ijkm', u(ps), self.smspace.basis(ps, p=p - 1))

            val = self.integralalg.cell_integral(f1)
            uh[cell2dof[:, 0:idof // 2]] = val[:, 0, :]
            uh[cell2dof[:, idof // 2:]] = val[:, 1, :]
        return uh

    def mass_matrix(self, c=None, q=None):
        """
        """
        mesh = self.mesh
        cellmeasure = mesh.entity_measure('cell')
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = self.basis(bcs)

        if c is None:
            M = np.einsum('i, ijkd, ijmd, j->jkm', ws, phi, phi, cellmeasure, optimize=True)
        else:
            if callable(c):
                if c.coordtype == 'barycentric':
                    c = c(bcs)
                elif c.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bcs)
                    c = c(ps)

            if isinstance(c, (int, float)):
                M = np.einsum('i, ijkd, ijmd, j->jkm', c * ws, phi, phi,
                              self.cellmeasure, optimize=True)
            elif isinstance(c, np.ndarray):
                if len(c.shape)==2:
                    M = np.einsum('i, ij, ijkd, ijmd, j->jkm', ws, c, phi, phi, cellmeasure, optimize=True)
                elif len(c.shape)==4:
                    M = np.einsum('i, ijdl, ijkd, ijml, j->jkm', ws, c, phi, phi, cellmeasure, optimize=True)
        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()

        I = np.broadcast_to(cell2dof[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=M.shape)

        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M

    def edge_basis(self, bcs, index=np.s_[:]):
        """
        @brief 计算每条边上的 v \cdot t(即 n \times v) 在重心坐标处的值.
                对于线性元，这个值就是 1/e
        """
        em = self.mesh.entity_measure("edge", index=index)
        shape = bcs.shape[:-1] + (len(em), )
        val = np.broadcast_to(1/em, shape,) #(NQ, NE)
        return val

    def edge_mass_matrix(self, c = 1, index=np.s_[:]):
        """
        @brief (n \times u, n \times v)_{Gamma_{robin}}
               注意 : n_i \times \phi_i = 1/e_i
        @param c 系数, 现在只考虑了 c 是常数的情况
        """
        edge2dof = self.dof.edge_to_dof(index=index)
        em = self.mesh.entity_measure("edge", index=index)
        gdof = self.dof.number_of_global_dofs()

        EM = c*csr_matrix((em.flat, (edge2dof.flat, edge2dof.flat)), shape=(gdof, gdof))
        return EM

    def robin_vector(self, f, isRobinEdge):
        """
        @brief 计算 (f, n\times v)_{\Gamma_{robin}} 其中 n \times v = 1/e
        """
        eint = self.integralalg.edgeintegrator
        bcs, ws = eint.get_quadrature_points_and_weights()
        n = self.mesh.edge_unit_normal()[isRobinEdge]

        point = self.mesh.bc_to_point(bcs, index=isRobinEdge)
        fval = f(point, n)

        e2dof = self.dof.edge_to_dof(index=isRobinEdge)
        gdof = self.dof.number_of_global_dofs()
        F = np.zeros(gdof, dtype=np.float_)
        F[e2dof] = np.einsum("qe, q->e", fval, ws)[:, None] 
        return F

    def curl_matrix(self, c=None, q=None):
        """

        Notes:

        组装 (c*\\nabla \\times u_h, \\nabla \\times u_h) 矩阵 
        """

        mesh = self.mesh
        cellmeasure = mesh.entity_measure('cell')
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = self.curl_basis(bcs)

        if c is None:
            M = np.einsum('i, ijk..., ijm..., j->jkm', ws, phi, phi, cellmeasure, optimize=True)
        else:

            if callable(c):
                if c.coordtype == 'barycentric':
                    c = c(bcs)
                elif c.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bcs)
                    c = c(ps)

            if isinstance(c, (int, float)):
                M = np.einsum('i, ijk, ijm, j->jkm', c * ws, phi, phi, cellmeasure, optimize=True)
            else:
                M = np.einsum('q, qc, qck..., qcm..., c->ckm', ws, c, phi, phi, cellmeasure, optimize=True)

        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()

        I = np.broadcast_to(cell2dof[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=M.shape)

        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M

    def source_vector(self, f):
        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()
        b = self.integralalg.construct_vector_v_v(f, self.basis, cell2dof, gdof=gdof)
        return b

    def face_mass_matrix(self):
        pass

    def set_dirichlet_bc(self, gD, uh, threshold=None, q=None):
        """
        """
        p = 1
        mesh = self.mesh
        edge2dof = self.dof.edge_to_dof()

        qf = self.integralalg.edgeintegrator if q is None else mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]

        t = mesh.edge_unit_tangent(index=index)
        ps = mesh.bc_to_point(bcs, index=index)
        val = gD(ps, t) # (NQ, NBE)
        if len(val.shape)==3:
            val = np.einsum('...ed, ed->...e', val, t)

        measure = self.integralalg.edgemeasure[index]
        gdof = self.number_of_global_dofs()
        idx = edge2dof[index].reshape(-1)
        uh[idx] = np.einsum('q, qe, e->e', ws, val, measure, optimize=True)#.squeeze()
        isDDof = np.zeros(gdof, dtype=np.bool_)
        isDDof[idx] = True
        return isDDof

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof,) + dim

        return np.zeros(shape, dtype=dtype)

    def show_basis(self, fig, index=0, box=None):
        """
        Plot quvier graph for every basis in a fig object
        """
        from .femdof import multi_index_matrix2d

        p = self.p
        mesh = self.mesh

        ldof = self.number_of_local_dofs()

        bcs = multi_index_matrix2d(10) / 10
        ps = mesh.bc_to_point(bcs)
        phi = self.basis(bcs)

        if p == 0:
            m = 1
            n = 3
        elif p == 1:
            m = 4
            n = 2
        elif p == 2:
            m = 5
            n = 3
        for i in range(ldof):
            axes = fig.add_subplot(m, n, i + 1)
            mesh.add_plot(axes, box=box)
            node = ps[:, index, :]
            uv = phi[:, index, i, :]
            axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1],
                        units='xy')

    def curl_error(self, u, uh, celltype=False):

        mesh = self.mesh
        cellmeasure = mesh.entity_measure('cell')
        qf = self.integrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        point = mesh.bc_to_point(bcs)
        uval = u(point)
        uhval = self.curl_value(uh, bcs)
        val = uhval-uval
        error = np.einsum('q, qc, qc, c->c', ws, val, val, cellmeasure, optimize=True)
        if celltype:
            return error
        else:
            return np.sqrt(error.sum())

    def error(self, u, uh, celltype=False):
        mesh = self.mesh
        cellmeasure = mesh.entity_measure('cell')
        qf = self.integrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        point = mesh.bc_to_point(bcs)
        uval = u(point)
        uhval = uh(bcs)
        val = uhval-uval
        error = np.einsum('q, qcl, qcl, c->c', ws, val, val, cellmeasure, optimize=True)
        if celltype:
            return error
        else:
            return np.sqrt(error.sum())

