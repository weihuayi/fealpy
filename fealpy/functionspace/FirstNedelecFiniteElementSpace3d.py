import numpy as np
from numpy.linalg import inv

from scipy.sparse import csr_matrix, coo_matrix
from ..decorator import barycentric
from .Function import Function
from ..quadrature import FEMeshIntegralAlg
from ..decorator import timer

class NDof3d:
    def __init__(self, mesh):
        """
        Parameters
        ----------
        mesh : TetrahedronMesh object

        Notes
        -----

        Reference
        ---------
        """
        self.mesh = mesh
        self.cell2dof = self.cell_to_dof() # 默认的自由度数组

    def boundary_dof(self, threshold=None):
        """
        """
        pass

    def is_boundary_dof(self, threshold=None):
        """
        """
        pass

    def edge_to_dof(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edof = self.number_of_local_dofs('edge')
        edge2dof = np.arange(NE*edof).reshape(NE, edof)
        return edge2dof

    def face_to_dof(self):
        pass

    def cell_to_dof(self):
        pass

    def number_of_local_dofs(self, doftype='all'):
        stype = self.spacetype
        if doftype == 'all': # number of all dofs on a cell 
            return 3
        elif doftype in {'cell', 3}: # number of dofs inside the cell 
            return 0 
        elif doftype in {'face', 2}: # number of dofs on a face 
            return 1
        elif doftype in {'edge', 1}: # number of dofs on a edge
            return 1
        elif doftype in {'node', 0}: # number of dofs on a node
            return 0

    def number_of_global_dofs(self):
        pass



class FirstNedelecFiniteElementSpace3d:
    def __init__(self, mesh, q=None, dof=None):
        """
        Parameters
        ----------
        mesh : TriangleMesh
        q : the index of quadrature fromula
        dof : the object for degree of freedom

        Note
        ----

        """
        self.mesh = mesh

        if dof is None:
            self.dof = FNDof3d(mesh)
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
        (NC, NQ, ldof, 3)
        (NC, ldof, NQ, 3)
        """

        p = self.p

        if p == 1:
            pass
        elif p == 2:
            pass
        elif p == 3:
            pass

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

        """

    @barycentric
    def grad_basis(self, bc, index=np.s_[:]):
        """

        Parameters
        ----------

        Notes
        -----

        """
        pass

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
        val = np.einsum(s1, phi, uh[cell2dof[index]])
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
        return self.interpolation(u)

    def interpolation(self, u):
        pass

    def mass_matrix(self, c=None, q=None):
        pass

    def curl_matrix(self, c=None, q=None):
        """

        Notes:

        组装 (c*\\nabla \\times u_h, \\nabla \\times u_h) 矩阵 
        """
        pass


    def source_vector(self, f):
        pass

    def set_dirichlet_bc(self, gD, uh, threshold=None, q=None):
        """
        """
        pass


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
        from .femdof import multi_index_matrix3d

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

