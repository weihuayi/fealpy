import numpy as np
from numpy.linalg import inv

from .function import Function
from .ScaledMonomialSpace3d import ScaledMonomialSpace3d

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
            cdof = self.number_of_local_dofs('cell')
            fdof = self.number_of_local_dofs('face') 
            NC = mesh.number_of_cells()
            NF = mesh.number_of_faces()
            cell2dof = np.zeros((NC, cdof), dtype=self.itype)

            face2dof = self.face_to_dof()
            face2cell = mesh.ds.face_to_cell()

            cell2dof[face2cell[:, [0]], face2cell[:, [2]]*fdof + np.arange(fdof)] = face2dof
            cell2dof[face2cell[:, [1]], face2cell[:, [3]]*fdof + np.arange(fdof)] = face2dof

            idof = cdof - 4*fdof 
            cell2dof[:, 4*fdof:] = NF*fdof+ np.arange(NC*idof).reshape(NC, idof)
            return cell2dof

    def number_of_local_dofs(self, etype='cell'):
        p = self.p
        if etype == 'cell':
            return (p+1)*(p+2)*(p+4)//2 
        elif etype == 'face':
            return (p+1)*(p+2)//2

    def number_of_global_dofs(self):
        p = self.p
        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs('face') 
        gdof = NF*fdof
        if p > 0:
            cdof = self.number_of_local_dofs('cell')
            idof = cdof - 4*fdof
            NC = self.mesh.number_of_cells()
            gdof += NC*ldof
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
        self.mesh = mesh
        self.p = p
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

        Notes
        -----
        3(p+1)(p+2)(p+3)/6 + (p+1)(p+2)/2 = (p+1)(p+2)(p+4)/2
        """
        p = self.p
        ldof = self.number_of_local_dofs()
        
        cdof = self.smspace.number_of_local_dofs(p=p, etype='cell')
        fdof = self.smspace.number_of_local_dofs(p=p, etype='face') 

        mesh = self.mesh
        NC = mesh.number_of_cells()

        LM, RM = self.smspace.face_cell_mass_matrix()
        A = np.zeros((NC, ldof, ldof), dtype=self.ftype)

        face = mesh.entity('face')
        face2cell = mesh.ds.face_to_cell()
        n = mesh.face_unit_normal() 

        idx2 = np.arange(cdof)[None, None, :]
        idx3 = np.arange(3*cdof, 3*cdof+fdof)[None, None, :]

        idx0 = face2cell[:, [0]][:, None, None]
        idx1 = (face2cell[:, [2]]*fdof + np.arange(fdof))[:, :, None]

        A[idx0, idx1, 0*cdof + idx2] = n[:, 0, None, None]*LM[:, :, :cdof]
        A[idx0, idx1, 1*cdof + idx2] = n[:, 1, None, None]*LM[:, :, :cdof]
        A[idx0, idx1, 2*cdof + idx2] = n[:, 2, None, None]*LM[:, :, :cdof]

        idx = self.smspace.face_index_1(p=p+1)
        x = idx['x']
        y = idx['y']
        z = idx['z']
        A[idx0, idx1, idx3] = n[:, 0, None, None]*LM[:, :,  cdof+x] + \
                n[:, 1, None, None]*LM[:, :, cdof+y] + \
                n[:, 2, None, None]*LM[:, :, cdof+z]

        idx0 = face2cell[:, [1]][:, None, None]
        idx1 = (face2cell[:, [3]]*fdof + np.arange(fdof))[:, :, None]

        A[idx0, idx1, 0*cdof + idx2] = n[:, 0, None, None]*RM[:, :, :cdof]
        A[idx0, idx1, 1*cdof + idx2] = n[:, 1, None, None]*RM[:, :, :cdof]
        A[idx0, idx1, 2*cdof + idx2] = n[:, 2, None, None]*RM[:, :, :cdof]
        A[idx0, idx1, idx3] = n[:, 0, None, None]*RM[:, :,  cdof+x] + \
                n[:, 1, None, None]*RM[:, :, cdof+y] + \
                n[:, 2, None, None]*RM[:, :, cdof+z]

        if p > 0:
            M = self.smspace.cell_mass_matrix()
            idx = self.smspace.diff_index_1()
            x = idx['x']
            y = idx['y']
            z = idx['z']

            idof = p*(p+1)*(p+2)//6 
            idx1 = np.arange(4*fdof+0*idof, 4*fdof+1*idof)[:, None]
            A[:, idx1, 0*cdof + np.arange(cdof)] = M[:, :idof, :]
            A[:, idx1, 3*cdof:] = M[:,  x[0], cdof-fdof:]

            idx1 = np.arange(4*fdof+1*idof, 4*fdof+2*idof)[:, None]
            A[:, idx1, 1*cdof + np.arange(cdof)] = M[:, :idof, :]
            A[:, idx1, 3*cdof:] = M[:,  y[0], cdof-fdof:]

            idx1 = np.arange(4*fdof+2*idof, 4*fdof+3*idof)[:, None]
            A[:, idx1, 2*cdof + np.arange(cdof)] = M[:, :idof, :]
            A[:, idx1, 3*cdof:] = M[:,  z[0], cdof-fdof:]

        return inv(A)

    def basis(self, bc):
        """
        """
        p = self.p
        mesh = self.mesh

        ldof = self.number_of_local_dofs()
        
        cdof = self.smspace.number_of_local_dofs(p=p, etype='cell')
        fdof = self.smspace.number_of_local_dofs(p=p, etype='face') 

        ps = mesh.bc_to_point(bc)
        shape = ps.shape[:-1] + (ldof, 3)
        phi = np.zeros(shape, dtype=self.ftype) # (NQ, NC, ldof, 3)

        c = self.bcoefs # (NC, ldof, ldof) 

        val = self.smspace.basis(ps, p=p+1) # (NQ, NC, ndof)
        idx = self.smspace.face_index_1(p=p+1)
        x = idx['x']
        y = idx['y']
        z = idx['z']

        phi[..., 0] += np.einsum('ijm, jmn->ijn', val[..., :cdof], c[:, 0*cdof:1*cdof, :])
        phi[..., 1] += np.einsum('ijm, jmn->ijn', val[..., :cdof], c[:, 1*cdof:2*cdof, :])
        phi[..., 2] += np.einsum('ijm, jmn->ijn', val[..., :cdof], c[:, 2*cdof:3*cdof, :])
        phi[..., 0] += np.einsum('ijm, jmn->ijn', val[..., cdof+x], c[:, 3*cdof:, :])
        phi[..., 1] += np.einsum('ijm, jmn->ijn', val[..., cdof+y], c[:, 3*cdof:, :])
        phi[..., 2] += np.einsum('ijm, jmn->ijn', val[..., cdof+z], c[:, 3*cdof:, :])
        return phi

    def grad_basis(self, bc):
        pass

    def div_basis(self, bc):
        pass

    def cell_to_dof(self):
        return self.dof.cell_to_dof()

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()


    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    # helper function for understand RT finite element  

    def show_face_frame(self, axes, index):
        pass

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
