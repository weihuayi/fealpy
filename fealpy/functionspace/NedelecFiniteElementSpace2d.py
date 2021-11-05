import numpy as np
from numpy.linalg import inv

from scipy.sparse import csr_matrix, coo_matrix
from ..decorator import barycentric
from .Function import Function
from ..quadrature import FEMeshIntegralAlg
from ..decorator import timer

class NDof2d:
    def __init__(self, mesh, spacetype='first'):
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
        self.spacetype = spacetype # 默认的第一类的 Nedelec 元空间
        self.cell2dof = self.cell_to_dof() # 默认的自由度数组

    def boundary_dof(self, threshold=None):
        """
        """
        idx = self.mesh.ds.boundary_edge_index()
        if threshold is not None: #TODO: threshold 可以是一个指标数组
            bc = self.mesh.entity_barycenter('edge', index=idx)
            flag = threshold(bc)
            idx  = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def is_boundary_dof(self, threshold=None):
        """
        """
        idx = self.mesh.ds.boundary_edge_index()
        if threshold is not None: #TODO: threshold 可以是一个指标数组
            bc = self.mesh.entity_barycenter('edge', index=idx)
            flag = threshold(bc)
            idx  = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def edge_to_dof(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edof = self.number_of_local_dofs('edge')
        edge2dof = np.arange(NE*edof).reshape(NE, edof)
        return edge2dof

    def cell_to_dof(self):
        """
        """
        stype = self.spacetype 
        mesh = self.mesh
        
        cell2edge = mesh.ds.cell_to_edge()
        if stype == 'first':
            return cell2edge
        else:
            cell2dof = np.zeros((NC, 6), dtype=np.int_)
            cell2dof[:, 0::2] = 2*cell2edge
            cell2dof[:, 1::2] = cell2dof[:, 0::2] + 1 
            return cell2dof

    def number_of_local_dofs(self, doftype='all'):
        stype = self.spacetype
        if doftype == 'all': # number of all dofs on a cell 
            return 3 if stype == 'first' else 6;
        elif doftype in {'cell', 2}: # number of dofs inside the cell 
            return 0 
        elif doftype in {'face', 'edge', 1}: # number of dofs on a edge 
            return 1 if stype == 'first' else 2;
        elif doftype in {'node', 0}: # number of dofs on a node
            return 0

    def number_of_global_dofs(self):
        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge') 
        gdof = NE*edof
        return gdof 

class NedelecFiniteElementSpace2d:
    def __init__(self, mesh, spacetype='first', q=None, dof=None):
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
        self.spacetype = spacetype 

        if dof is None:
            self.dof = NDof2d(mesh, spacetype)
        else:
            self.dof = dof

        self.integralalg = FEMeshIntegralAlg(self.mesh, q)
        self.integrator = self.integralalg.integrator

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype


    def boundary_dof(self):
        return self.dof.boundary_dof()

    @barycentric
    def face_basis(self, bc, index=None, barycenter=True):
        return self.edge_basis(bc, index, barycenter)


    @barycentric
    def edge_basis(self, bc, index=None, barycenter=True, left=True):
        pass

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
        stype = self.spacetype

        # 每个单元上的全部自由度个数
        ldof = self.number_of_local_dofs(doftype='all') 
        edof = self.number_of_local_dofs(doftype='edge') 

        mesh = self.mesh
        glambda = mesh.grad_lambda() # (NC, 3, 2)
        GD = mesh.geo_dimension()

        shape = bc.shape[:-1] + (ldof, GD)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof, 2)

        cell2edgesign = mesh.ds.cell_to_edge_sign() # (NC, 3)
        
        phi[..., 0::edof, :] =  

        return phi

    @barycentric
    def rot_basis(self, bc, index=np.s_[:], barycenter=True):
        return self.curl_basis(bc, index, barycenter)

    @barycentric
    def curl_basis(self, bc, index=np.s_[:], barycenter=True):
        """

        Parameters
        ----------

        Notes
        -----
        curl [v_0, v_1] = \partial v_1/\partial x - \partial v_0/\partial y

        curl [[ m_k, 0], [0, m_k]] 
        = [-\partial m_k/\partial y,  \partial m_k/\partial x] 

        """
        p = self.p
        ldof = self.number_of_local_dofs('all')
        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        edof = self.smspace.number_of_local_dofs(p=p, doftype='edge') 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        index = index if index is not None else np.s_[:]
        if barycenter:
            ps = mesh.bc_to_point(bc, index=index)
        else:
            ps = bc
        val = self.smspace.grad_basis(ps, p=p+1, index=index) # (NQ, NC, ndof, GD)

        shape = ps.shape[:-1] + (ldof, )
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof)

        c = self.bcoefs[index] # (NC, ldof, ldof) 
        idx = self.smspace.face_index_1(p=p+1)

        x = idx['x']
        y = idx['y']

        phi[:] -= np.einsum('...jm, jmn->...jn', val[..., :cdof, 1], c[:, 0*cdof:1*cdof, :])
        phi[:] += np.einsum('...jm, jmn->...jn', val[..., :cdof, 0], c[:, 1*cdof:2*cdof, :])
        phi[:] -= np.einsum('...jm, jmn->...jn', val[..., cdof+x, 0], c[:, GD*cdof:, :])
        phi[:] -= np.einsum('...jm, jmn->...jn', val[..., cdof+y, 1], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def grad_basis(self, bc):
        """

        Parameters
        ----------

        Notes
        -----
        curl [v_0, v_1] = \partial v_1/\partial x - \partial v_0/\partial y

        curl [[ m_k, 0], [0, m_k]] 
        = [-\partial m_k/\partial y,  \partial m_k/\partial x] 

        """
        p = self.p
        ldof = self.number_of_local_dofs('all')
        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        edof = self.smspace.number_of_local_dofs(p=p, doftype='edge') 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        index = index if index is not None else np.s_[:]
        if barycenter:
            ps = mesh.bc_to_point(bc, index=index)
        else:
            ps = bc
        val = self.smspace.grad_basis(ps, p=p+1, index=index) # (NQ, NC, ndof, GD)

        shape = ps.shape[:-1] + (ldof, GD, GD)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof, GD, GD)

        c = self.bcoefs[index] # (NC, ldof, ldof) 

        idx = self.smspace.face_index_1(p=p+1)
        x = idx['x']
        y = idx['y']

        phi[..., 0, 0] += np.einsum('...jm, jmn->...jn', val[..., :cdof, 0], c[:, 0*cdof:1*cdof, :])
        phi[..., 0, 0] += np.einsum('...jm, jmn->...jn', val[..., cdof+y, 0], c[:, GD*cdof:, :])

        phi[..., 0, 1] += np.einsum('...jm, jmn->...jn', val[..., :cdof, 1], c[:, 0*cdof:1*cdof, :])
        phi[..., 0, 1] += np.einsum('...jm, jmn->...jn', val[..., cdof+y, 1], c[:, GD*cdof:, :])

        phi[..., 1, 0] += np.einsum('...jm, jmn->...jn', val[..., :cdof, 0], c[:, 1*cdof:2*cdof, :])
        phi[..., 1, 0] -= np.einsum('...jm, jmn->...jn', val[..., cdof+x, 0], c[:, GD*cdof:, :])
        
        phi[..., 1, 1] += np.einsum('...jm, jmn->...jn', val[..., :cdof, 1], c[:, 1*cdof:2*cdof, :])
        phi[..., 1, 1] -= np.einsum('...jm, jmn->...jn', val[..., cdof+x, 1], c[:, GD*cdof:, :])

        return phi

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)

    @barycentric
    def value(self, uh, bc, index=np.s_[:], barycenter=True):
        phi = self.basis(bc, index=index, barycenter=barycenter)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof[index]])
        return val

    @barycentric
    def rot_value(self, uh, bc, index=np.s_[:], barycenter=True):
        return self.curl_value(uh, bc, index, barycenter=barycenter)

    @barycentric
    def curl_value(self, uh, bc, index=np.s_[:], barycenter=True):
        cphi = self.curl_basis(bc, index=index, barycenter=barycenter)
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
        return self.interpolation(u)

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
            idof = self.number_of_local_dofs('cell') # dofs inside the cell 
            cell2dof = NE*edof+ np.arange(NC*idof).reshape(NC, idof)

            @barycentric
            def f1(bc): #TODO: check here
                ps = mesh.bc_to_point(bc)
                return np.einsum('ijk, ijm->ijkm', u(ps), self.smspace.basis(ps, p=p-1))

            val = self.integralalg.cell_integral(f1)
            uh[cell2dof[:, 0:idof//2]] = val[:, 0, :] 
            uh[cell2dof[:, idof//2:]] = val[:, 1, :]
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
                M = np.einsum('i, ijkd, ijmd, j->jkm', c*ws, phi, phi,
                        self.cellmeasure, optimize=True)
            else:
                M = np.einsum('i, ijdn, ijkn, ijmd, j->jkm', ws, c, phi, phi, cellmeasure, optimize=True)

        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()

        I = np.broadcast_to(cell2dof[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=M.shape)

        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M 

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
                M = np.einsum('i, ijk, ijm, j->jkm', c*ws, phi, phi, cellmeasure, optimize=True)
            else:
                M = np.einsum('i, ij, ijk, ijm, j->jkm', ws, c, phi, phi, cellmeasure, optimize=True)

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

    def set_dirichlet_bc(self, gD, uh, threshold=None, q=None):
        """
        """
        p = self.p
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
        val = gD(ps, t)
        phi = self.smspace.edge_basis(ps, index=index)

        measure = self.integralalg.edgemeasure[index]
        gdof = self.number_of_global_dofs()
        uh[edge2dof[index]] = np.einsum('i, ij, ijm, j->jm', ws, val, phi,
                measure, optimize=True)
        isDDof = np.zeros(gdof, dtype=np.bool_) 
        isDDof[edge2dof[index]] = True
        return isDDof


    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim

        return np.zeros(shape, dtype=dtype)

    def show_basis(self, fig, index=0, box=None):
        """
        Plot quvier graph for every basis in a fig object
        """
        from .femdof import multi_index_matrix2d

        p = self.p
        mesh = self.mesh

        ldof = self.number_of_local_dofs()

        bcs = multi_index_matrix2d(10)/10
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
            axes = fig.add_subplot(m, n, i+1)
            mesh.add_plot(axes, box=box)
            node = ps[:, index, :]
            uv = phi[:, index, i, :]
            axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1], 
                    units='xy')

