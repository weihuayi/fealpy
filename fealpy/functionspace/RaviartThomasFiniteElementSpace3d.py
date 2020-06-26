import numpy as np
from numpy.linalg import inv

from .Function import Function
from .ScaledMonomialSpace3d import ScaledMonomialSpace3d

from ..decorator import barycentric # 导入默认的坐标类型, 这个空间是重心坐标

class RTDof3d:
    def __init__(self, mesh, p):
        """
        Parameters
        ----------
        mesh : TetrahedronMesh or HalfEdgeMesh3d object
        p : the space order, p>=0

        Notes
        -----

        Reference
        ---------
        """
        self.mesh = mesh
        self.p = p # 默认的空间次数 p >= 0
        self.itype = mesh.itype

        self.cell2dof = self.cell_to_dof() # 默认的自由度数组
        
    def boundary_dof(self, threshold=None):
        idx = self.mesh.ds.boundary_face_index()
        if threshold is not None:
            bc = self.mesh.entity_barycenter('face', index=idx)
            flag = threshold(bc)
            idx  = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        face2dof = self.face_to_dof()
        isBdDof[face2dof[idx]] = True
        return isBdDof

    def face_to_dof(self):
        p = self.p
        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs('face') 
        face2dof = np.arange(NF*fdof).reshape(NF, fdof)
        return face2dof

    def cell_to_dof(self):
        """
        """
        p = self.p 
        mesh = self.mesh
        if p == 0:
            cell2face = mesh.ds.cell_to_face()
            return cell2face
        else:
            ldof = self.number_of_local_dofs(doftype='all')
            cdof = self.number_of_local_dofs(doftype='cell') # 单元内部的自由度个数
            fdof = self.number_of_local_dofs(doftype='face') # 面上的自由度个数
            NC = mesh.number_of_cells()
            NF = mesh.number_of_faces()
            cell2dof = np.zeros((NC, ldof), dtype=self.itype)

            face2dof = self.face_to_dof()
            face2cell = mesh.ds.face_to_cell()

            cell2dof[face2cell[:, [0]], face2cell[:, [2]]*fdof + np.arange(fdof)] = face2dof
            cell2dof[face2cell[:, [1]], face2cell[:, [3]]*fdof + np.arange(fdof)] = face2dof

            cell2dof[:, 4*fdof:] = NF*fdof+ np.arange(NC*cdof).reshape(NC, cdof)
            return cell2dof

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all':
            return (p+1)*(p+2)*(p+4)//2 
        elif doftype in {'cell', 3}:
            return p*(p + 1)*(p + 2)//2 
        elif doftype in {'face', 2}:
            return (p+1)*(p+2)//2
        else:
            return 0 

    def number_of_global_dofs(self):
        p = self.p
        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs(doftype='face') 
        gdof = NF*fdof
        if p > 0:
            cdof = self.number_of_local_dofs(doftype='cell')
            NC = self.mesh.number_of_cells()
            gdof += NC*cdof
        return gdof 

class RaviartThomasFiniteElementSpace3d:
    def __init__(self, mesh, p=0, q=None, dof=None):
        """
        Parameters
        ----------
        mesh : TriangleMesh
        p : the space order, p>=0
        q : the index of quadrature fromula
        dof : the object for degree of freedom

        Note
        ----
        RT_p : [P_{p-1}]^d(T) + [m_1, m_2, m_3]^T \\bar P_{p-1}(T)

        """
        self.p = p
        self.mesh = mesh
        self.smspace = ScaledMonomialSpace3d(mesh, p, q=q)

        if dof is None:
            self.dof = RTDof3d(mesh, p)
        else:
            self.dof = dof

        self.integralalg = self.smspace.integralalg
        self.integrator = self.smspace.integrator

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

        self.bcoefs = self.basis_coefficients()

    def basis_coefficients(self):
        """

        Parameters
        ----------

        Notes
        -----

        """
        p = self.p
        ldof = self.number_of_local_dofs(doftype='all')

        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell') 
        fdof = self.smspace.number_of_local_dofs(p=p, doftype='face')

        mesh = self.mesh
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()

        LM, RM = self.smspace.face_cell_mass_matrix()
        A = np.zeros((NC, ldof, ldof), dtype=self.ftype)

        face = mesh.entity('face')
        face2cell = mesh.ds.face_to_cell()
        n = mesh.face_unit_normal() 

        idx = self.smspace.face_index_1(p=p+1)
        x = idx['x']
        y = idx['y']
        z = idx['z']
        idx2 = np.arange(cdof)[None, None, :]
        idx3 = np.arange(GD*cdof, GD*cdof+fdof)[None, None, :]

        # left cell
        idx0 = face2cell[:, 0][:, None, None]
        idx1 = (face2cell[:, [2]]*fdof + np.arange(fdof))[:, :, None]
        for i in range(GD):
            A[idx0, idx1, i*cdof + idx2] = n[:, i, None, None]*LM[:, :, :cdof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*LM[:, :,  cdof+x] + \
                n[:, 1, None, None]*LM[:, :, cdof+y] + \
                n[:, 2, None, None]*LM[:, :, cdof+z]

        # right cell
        idx0 = face2cell[:, 1][:, None, None]
        idx1 = (face2cell[:, [3]]*fdof + np.arange(fdof))[:, :, None]
        for i in range(GD):
            A[idx0, idx1, i*cdof + idx2] = n[:, i, None, None]*RM[:, :, :cdof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*RM[:, :,  cdof+x] + \
                n[:, 1, None, None]*RM[:, :, cdof+y] + \
                n[:, 2, None, None]*RM[:, :, cdof+z]

        if p > 0:
            M = self.smspace.cell_mass_matrix()
            idx = self.smspace.diff_index_1()
            idof = self.smspace.number_of_local_dofs(p=p-1, doftype='cell') 
            for i, key in enumerate(idx.keys()):
                index = np.arange(4*fdof + i*idof, 4*fdof+ (i+1)*idof)[:, None]
                A[:, index, i*cdof + np.arange(cdof)] = M[:, :idof, :]
                A[:, index, GD*cdof + np.arange(fdof)] = M[:,  idx[key][0], cdof-fdof:]
        return inv(A)

    @barycentric
    def face_basis(self, bc, index=None, barycenter=True):
        """

        Paramerters
        -----------

        Notes
        -----
            给定三角形面中的一组重心坐标或者欧氏坐标点，计算这个三角形面对应的基函数在
            这些点处的函数值。
        """
        p = self.p

        ldof = self.number_of_local_dofs(doftype='all')

        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        fdof = self.smspace.number_of_local_dofs(p=p, doftype='face') 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        face2cell = mesh.ds.face_to_cell()

        index = index if index is not None else np.s_[:]
        if barycenter:
            ps = mesh.bc_to_point(bc, etype='face', index=index)
        else:
            ps = bc
        val = self.smspace.basis(ps, p=p+1, index=face2cell[index, 0]) # (NQ, NF, ndof)

        shape = ps.shape[:-1] + (fdof, GD)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NF, fdof, 2)

        idx0 = face2cell[index, 0][:, None]
        idx2 = face2cell[index[:, None], [2]]*fdof + np.arange(fdof)
        c = self.bcoefs[idx0, :, idx2].swapaxes(-1, -2) # (NF, ldof, edof) 
        idx = self.smspace.face_index_1(p=p+1)

        for i, key in enumerate(idx.keys()):
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., :cdof], c[:, i*cdof:(i+1)*cdof, :])
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., cdof+idx[key]], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def basis(self, bc, index=None, barycenter=True):
        """

        Notes
        -----

        """
        p = self.p

        # 每个单元上的全部自由度个数
        ldof = self.number_of_local_dofs(doftype='all') 

        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        fdof = self.smspace.number_of_local_dofs(p=p, doftype='face') 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        index = index if index is not None else np.s_[:]

        if barycenter:
            ps = mesh.bc_to_point(bc, etype='cell', index=index)
        else:
            ps = bc
        val = self.smspace.basis(ps, p=p+1, index=index) # (NQ, NC, ndof)

        shape = ps.shape[:-1] + (ldof, 3)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof, 3)

        c = self.bcoefs[index] # (NC, ldof, ldof) 
        idx = self.smspace.face_index_1(p=p+1)

        for i, key in enumerate(idx.keys()):
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., :cdof], c[:, i*cdof:(i+1)*cdof, :])
            phi[..., i] += np.einsum('ijm, jmn->ijn', val[..., cdof+idx[key]], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def div_basis(self, bc, index=None, barycenter=True):
        p = self.p

        ldof = self.number_of_local_dofs(doftype='all')

        cdof = self.smspace.number_of_local_dofs(p=p, doftype='cell')
        fdof = self.smspace.number_of_local_dofs(p=p, doftype='face') 

        mesh = self.mesh
        GD = mesh.geo_dimension()
        index = index if index is not None else np.s_[:]
        if barycenter:
            ps = mesh.bc_to_point(bc, index=index)
        else:
            ps = bc
        val = self.smspace.grad_basis(ps, p=p+1, index=index) # (NQ, NC, ndof)

        shape = ps.shape[:-1] + (ldof, )
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof)

        c = self.bcoefs[index] # (NC, ldof, ldof) 
        idx = self.smspace.face_index_1(p=p+1)

        for i, key in enumerate(idx.keys()):
            phi[:] += np.einsum('ijm, jmn->ijn', val[..., :cdof, i], c[:, i*cdof:(i+1)*cdof, :])
            phi[:] += np.einsum('ijm, jmn->ijn', val[..., cdof+idx[key], i], c[:, GD*cdof:, :])
        return phi

    @barycentric
    def grad_basis(self, bc):
        pass

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f


    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in [None, 1]:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)




    def stiff_matrix(self):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        M = self.integralalg.construct_matrix(self.basis, cell2dof0=cell2dof,
                gdof0=gdof)
        return M

    def div_matrix(self):
        p = self.p
        gdof0 = self.number_of_global_dofs()
        cell2dof0 = self.cell_to_dof()
        gdof1 = self.smspace.number_of_global_dofs()
        cell2dof1 = self.smspace.cell_to_dof()
        basis0 = self.div_basis
        basis1 = lambda bc : self.smspace.basis(self.mesh.bc_to_point(bc), p=p)

        D = self.integralalg.construct_matrix(basis0, basis1=basis1, 
                cell2dof0=cell2dof0, gdof0=gdof0,
                cell2dof1=cell2dof1, gdof1=gdof1)
        return D 

    def source_vector(self, f, dim=None):
        cell2dof = self.smspace.cell_to_dof()
        gdof = self.smspace.number_of_global_dofs()
        b = -self.integralalg.construct_vector_s_s(f, self.smspace.basis, cell2dof, 
                gdof=gdof) 
        return b

    def neumann_boundary_vector(self, g, threshold=None, q=None):
        """
        Parameters
        ----------

        Notes
        ----
        """
        p = self.p
        mesh = self.mesh

        fdof = self.smspace.number_of_local_dofs(doftype='face') 
        face2cell = mesh.ds.face_to_cell()
        face2dof = self.dof.face_to_dof() 

        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        fn = mesh.face_unit_normal(index=index)
        phi = self.face_basis(bcs, index=index) 

        ps = mesh.bc_to_point(bcs, etype='face', index=index)
        val = -g(ps)
        measure = self.integralalg.facemeasure[index]

        gdof = self.number_of_global_dofs()
        F = np.zeros(gdof, dtype=self.ftype)
        bb = np.einsum('i, ij, ijmk, jk, j->jm', ws, val, phi, fn, measure, optimize=True)
        np.add.at(F, face2dof[index], bb)
        return F 

    def set_dirichlet_bc(self, uh, g, threshold=None, q=None):
        """

        Parameters
        ----------

        Notes
        -----
        """
        p = self.p
        mesh = self.mesh
        face2dof = self.dof.face_to_dof() 

        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        ps = mesh.bc_to_point(bcs, etype='face', index=index)
        fn = mesh.face_unit_normal(index=index)
        val = -g(ps, fn)
        phi = self.smspace.face_basis(ps, index=index)

        measure = self.integralalg.facemeasure[index]
        gdof = self.number_of_global_dofs()
        uh[face2dof[index]] = np.einsum('i, ij, ijm, j->jm', ws, val, phi,
                measure, optimize=True)
        isDDof = np.zeros(gdof, dtype=np.bool_) 
        isDDof[face2dof[index]] = True
        return isDDof

    def cell_to_dof(self):
        return self.dof.cell_to_dof()

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def show_basis(self, fig, index=0):
        """
        Plot quvier graph for every basis in a fig object
        """
        from .femdof import multi_index_matrix3d

        p = self.p
        mesh = self.mesh

        ldof = self.number_of_local_dofs()

        bcs = multi_index_matrix3d(4)/4
        ps = mesh.bc_to_point(bcs)
        phi = self.basis(bcs)
        if p == 0:
            m = 2
            n = 2
        elif p == 1:
            m = 5
            n = 3
        elif p == 2:
            m = 6
            n = 6
        for i in range(ldof):
            axes = fig.add_subplot(m, n, i+1, projection='3d')
            mesh.add_plot(axes)
            node = ps[:, index, :]
            v = phi[:, index, i, :]
            l = np.max(np.sqrt(np.sum(v**2, axis=-1)))
            v /=l
            axes.quiver(
                    node[:, 0], node[:, 1], node[:, 2], 
                    v[:, 0], v[:, 1], v[:, 2], length=1)
