import numpy as np
from scipy.sparse import csr_matrix, spdiags, bmat

from ..decorator import barycentric 
from ..quadrature import FEMeshIntegralAlg

from .Function import Function

class CRDof():
    def __init__(self, mesh):
        self.mesh = mesh
        self.cell2dof = self.cell_to_dof()

    def boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        face2dof = self.face_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof[index]] = True
        return isBdDof

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        face2dof = self.face_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof[index]] = True
        return isBdDof

    def face_to_dof(self):
        mesh = self.mesh
        NF = mesh.number_of_faces()
        return np.arange(NF).reshape(-1, 1)

    def cell_to_dof(self, index=np.s_[:]):
        mesh = self.mesh
        cell2dof = mesh.ds.cell_to_face()
        return cell2dof[index]

    def interpolation_points(self):
        mesh = self.mesh
        ipoint = mesh.entity_barycenter('face')
        return ipoint

    def number_of_global_dofs(self):
        gdof = self.mesh.number_of_faces()
        return gdof

    def number_of_local_dofs(self, doftype='cell'):
        mesh = self.mesh        
        TD = mesh.top_dimension()
        if doftype in {'cell'}:
            return TD+1 
        elif doftype in {'face'}:
            return 1
        elif doftype in {'node'}:
            return 0

class CrouzeixRaviartFiniteElementSpace():
    def __init__(self, mesh, q=None):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')

        self.dof = CRDof(mesh)
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = 3 
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

    def number_of_global_dofs(self):
        return self.mesh.number_of_faces()

    def number_of_local_dofs(self):
        return self.TD+1

    def interpolation_points(self):
        return self.dof.interpolation_points() 

    def cell_to_dof(self, index=np.s_[:]):
        return self.dof.cell_to_dof(index=index)

    def face_to_dof(self, index=np.s_[:]):
        NF = self.mesh.number_of_faces()
        return np.arange(NF)[index]

    def boundary_dof(self, threshold=None):
        return  self.dof.boundary_dof(threshold=threshold)

    def is_boundary_dof(self, threshold=None):
        return self.dof.is_boundary_dof(threshold=threshold) 

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    @barycentric
    def basis(self, bc):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`
        Returns
        -------
        phi : numpy.ndarray
            the shape of 'phi' can be `(1, ldof)` or `(NQ, 1, ldof)`

        See Also
        --------

        Notes
        -----

        """
        phi = 1 - self.GD*bc
        return phi[..., None, :] # (..., 1, ldof)

    @barycentric
    def grad_basis(self, bc, index=np.s_[:]):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`

        Returns
        -------
        gphi : numpy.ndarray
            the shape of `gphi` can b `(NC, ldof, GD)' or
            `(NQ, NC, ldof, GD)'

        See also
        --------

        Notes
        -----

        """
        mesh = self.mesh
        gphi = -self.GD*mesh.grad_lambda()[index] # (NC, TD+1, GD)
        if len(bc.shape) == 1:
            return gphi 
        else:
            return gphi[None, ...] # 多个积分点, 增加一个轴

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof[index]])
        return val

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        index = index if index is not None else np.s_[:]
        val = np.einsum(s1, gphi, uh[cell2dof[index]])
        return val

    @barycentric
    def div_value(self, uh, bc, index=np.s_[:]):
        dim = len(uh.shape)
        GD = self.geo_dimension()
        if (dim == 2) & (uh.shape[1] == GD):
            val = self.grad_value(uh, bc, index=index)
            return val.trace(axis1=-2, axis2=-1)
        else:
            raise ValueError("The shape of uh should be (gdof, gdim)!")

    def linear_elasticity_matrix(self, lam, mu):
        """

        Notes
        -----
        """
        
        A = self.stiff_matrix(c=mu)
        B = self.div_matrix(c=lam+mu, format='list')
        GD = self.GD 
        for i in range(GD):
            B[i][i] += A

        A = bmat(B, format='csr')
        return A
        
    def stiff_matrix(self, c=None):
        """
        Notes
        -----
            (\\nabla u, \\nabla v) 

            GD == 2
            [[phi, 0], [0, phi]]

            [d_x phi, d_y phi]

            GD == 3
            [[phi, 0, 0], [0, phi, 0], [0, 0, phi]]

            [d_x phi, d_y phi, d_z phi]
        """

        mesh = self.mesh
        GD = self.GD 
        bc = np.array(GD*[1/GD], dtype=self.ftype) # 梯度为分片常数
        gphi = self.grad_basis(bc)

        gdof = self.number_of_global_dofs()
        c2d = self.cell_to_dof() # (NC, ldof)

        A = np.einsum('cim, cjm, c->cij', gphi, gphi, self.cellmeasure)
        I = np.broadcast_to(c2d[:, :, None], shape=A.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=A.shape)

        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        if c is None:
            return A
        else:
            return c*A

    def div_matrix(self, c=None, format='csr'):
        """

        Notes
        -----
        (div u, div v) 

        GD == 2
        [[phi, 0], [0, phi]]

        [d_x phi, d_y phi]

        GD == 3
        [[phi, 0, 0], [0, phi, 0], [0, 0, phi]]

        [d_x phi, d_y phi, d_z phi]
        """

        mesh = self.mesh
        GD = mesh.geo_dimension()
        bc = np.array(GD*[1/GD], dtype=self.ftype)

        gphi = self.grad_basis(bc)

        gdof = self.number_of_global_dofs()
        c2d = self.cell_to_dof() # (NC, ldof)

        shape = c2d.shape + c2d.shape[-1:]
        I = np.broadcast_to(c2d[:, :, None], shape=shape)
        J = np.broadcast_to(c2d[:, None, :], shape=shape)

        B = []
        for i in range(GD):
            M = []
            for j in range(GD):
                D = np.einsum('ci, cj, c->cij', gphi[..., i], gphi[..., j], self.cellmeasure)
                D = csr_matrix((D.flat, (I.flat, J.flat)), shape=(gdof, gdof))
                if c is None:
                    M += [D]
                else:
                    M += [c*D]
            B += [M]

        if format == 'csr':
            return bmat(B, format='csr') # format = bsr ??
        elif format == 'bsr':
            return bmat(B, format='bsr')
        elif format == 'list':
            return B 

    def source_vector(self, f, dim=None, q=None):
        """

        Notes
        -----
        """
        cellmeasure = self.cellmeasure
        bcs, ws = self.integrator.get_quadrature_points_and_weights()

        if f.coordtype == 'cartesian':
            pp = self.mesh.bc_to_point(bcs)
            fval = f(pp)
        elif f.coordtype == 'barycentric':
            fval = f(bcs)

        gdof = self.number_of_global_dofs()
        shape = gdof if dim is None else (gdof, dim)
        b = np.zeros(shape, dtype=self.ftype)

        if type(fval) in {float, int}:
            if fval == 0.0:
                return b
            else:
                phi = self.basis(bcs)
                bb = np.einsum('m, mik, i->ik...', 
                        ws, phi, self.cellmeasure)
                bb *= fval
        else:
            phi = self.basis(bcs)
            bb = np.einsum('m, mi..., mik, i->ik...',
                    ws, fval, phi, self.cellmeasure)
        cell2dof = self.cell_to_dof() #(NC, ldof)
        if dim is None:
            np.add.at(b, cell2dof, bb)
        else:
            np.add.at(b, (cell2dof, np.s_[:]), bb)

        return b

    def interpolation_matrix(self):
        TD = self.TD
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NF = mesh.number_of_faces()

        face = mesh.entity('face')

        c = np.array([[1/TD]], dtype=self.ftype)
        f2d = self.dof.face_to_dof()

        val = np.broadcast_to(c, shape=(NF, TD))
        I = np.broadcast_to(f2d, shape=(NF, TD))
        J = face
        I = csr_matrix((val.flat, (I.flat, J.flat)), shape=(NF, NN))
        return I

    def interpolation(self, u, dim=None):
        ipoint = self.dof.interpolation_points()
        uI = u(ipoint)
        return self.function(dim=dim, array=uI)

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array, coordtype='barycentric')
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)

    def set_dirichlet_bc(self, gD, uh, threshold=None):
        """
        初始化解 uh  的第一类边界条件。
        """
        ipoints = self.interpolation_points()
        isBdDof = self.is_boundary_dof(threshold=threshold)
        uh[isBdDof] = gD(ipoints[isBdDof])
        return isBdDof
