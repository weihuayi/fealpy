import numpy as np
from ..decorator import barycentric
from .fem_dofs import *

class BernsteinFESpace:
    DOF = { 'C': {
                "IntervalMesh": IntervalMeshCFEDof,
                "TriangleMesh": TriangleMeshCFEDof,
                "TetrahedronMesh": TetrahedronMeshCFEDof,
                "EdgeMesh": EdgeMeshCFEDof,
                }, 
            'D':{
                "IntervalMesh": IntervalMeshDFEDof,
                "TriangleMesh": TriangleMeshDFEDof,
                "TetrahedronMesh": TetrahedronMeshDFEDof,
                "EdgeMesh": EdgeMeshDFEDof, 
                }
        } 

    def __init__(self, 
            mesh, 
            p: int=1, 
            spacetype: str='C', 
            doforder: str='vdims'):
        """
        @brief Initialize the Lagrange finite element space.

        @param mesh The mesh object.
        @param p The order of interpolation polynomial, default is 1.
        @param spacetype The space type, either 'C' or 'D'.
        @param doforder The convention for ordering degrees of freedom in vector space, either 'sdofs' (default) or 'vdims'.

        @note 'sdofs': 标量自由度优先排序，例如 x_0, x_1, ..., y_0, y_1, ..., z_0, z_1, ...
              'vdims': 向量分量优先排序，例如 x_0, y_0, z_0, x_1, y_1, z_1, ...
        """
        self.mesh = mesh
        self.p = p
        assert spacetype in {'C', 'D'} 
        self.spacetype = spacetype
        self.doforder = doforder

        mname = type(mesh).__name__
        self.dof = self.DOF[spacetype][mname](mesh, p)
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        self.cellmeasure = mesh.entity_measure('cell')
        self.itype = mesh.itype
        self.ftype = mesh.ftype

    @barycentric
    def basis(self, bc, index=np.s_[:], p=None):
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
        if p==None:
            p = self.p

        NQ = bc.shape[0]
        TD = bc.shape[1]-1
        multiIndex = self.mesh.multi_index_matrix(p, etype=TD)
        ldof = multiIndex.shape[0]

        B = bc
        B = np.ones((NQ, p+1, TD+1), dtype=np.float_)
        B[:, 1:] = bc[:, None, :]
        B = np.cumprod(B, axis=1)

        P = np.arange(p+1)
        P[0] = 1
        P = np.cumprod(P).reshape(1, -1, 1)
        B = B/P

        # B : (NQ, p+1, TD+1) 
        # B[:, multiIndex, np.arange(TD+1).reshape(1, -1)]: (NQ, ldof, TD+1)
        phi = P[0, -1, 0]*np.prod(B[:, multiIndex, np.arange(TD+1).reshape(1, -1)], 
                axis=-1)
        return phi[..., None, :] 

    @barycentric
    def grad_basis(self, bc, index=np.s_[:], p=None):
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
        if p==None:
            p = self.p

        NQ = bc.shape[0]
        TD = bc.shape[1]-1
        multiIndex = self.multi_index_matrix[TD](p)
        ldof = multiIndex.shape[0]

        B = bc
        B = np.ones((NQ, p+1, TD+1), dtype=self.ftype)
        B[:, 1:] = bc[:, None, :]
        B = np.cumprod(B, axis=1)

        P = np.arange(p+1)
        P[0] = 1
        P = np.cumprod(P).reshape(1, -1, 1)
        B = B/P

        F = np.zeros(B.shape, dtype=np.float_)
        F[:, 1:] = B[:, :-1]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            idx = np.array(idx, dtype=np.int_)
            R[..., i] = np.prod(B[..., multiIndex[:, idx], idx.reshape(1, -1)],
                    axis=-1)*F[..., multiIndex[:, i], [i]]

        Dlambda = self.mesh.grad_lambda()
        gphi = P[0, -1, 0]*np.einsum("qlm, cmd->qcld", R, Dlambda)
        return gphi

    @barycentric
    def value(self, 
            uh: np.ndarray, 
            bc: np.ndarray, 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        """
        @brief Computes the value of a finite element function `uh` at a set of
        barycentric coordinates `bc` for each mesh cell.

        @param uh: numpy.ndarray, the dof coefficients of the basis functions.
        @param bc: numpy.ndarray, the barycentric coordinates with shape (NQ, TD+1).
        @param index: Union[numpy.ndarray, slice], index of the entities (default: np.s_[:]).
        @return numpy.ndarray, the computed function values.

        This function takes the dof coefficients of the finite element function `uh` and a set of barycentric
        coordinates `bc` for each mesh cell. It computes the function values at these coordinates
        and returns the results as a numpy.ndarray.
        """
        gdof = self.number_of_global_dofs()
        phi = self.basis(bc, index=index) # (NQ, NC, ldof)
        cell2dof = self.dof.cell_to_dof(index=index)

        dim = len(uh.shape) - 1
        s0 = 'abdefg'
        if self.doforder == 'sdofs':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (..., gdof)
            # uh[..., cell2dof].shape == (..., NC, ldof)
            # val.shape == (NQ, ..., NC)
            s1 = f"...ci, {s0[:dim]}ci->...{s0[:dim]}c"
            val = np.einsum(s1, phi, uh[..., cell2dof])
        elif self.doforder == 'vdims':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (gdof, ...)
            # uh[cell2dof, ...].shape == (NC, ldof, ...)
            # val.shape == (NQ, NC, ...)
            s1 = f"...ci, ci{s0[:dim]}->...c{s0[:dim]}"
            val = np.einsum(s1, phi, uh[cell2dof, ...])
        else:
            raise ValueError(f"Unsupported doforder: {self.doforder}. Supported types are: 'sdofs' and 'vdims'.")
        return val


    @barycentric
    def grad_value(self, 
            uh: np.ndarray, 
            bc: np.ndarray, 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        """
        @note
        """
        gdof = self.number_of_global_dofs()
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.dof.cell_to_dof(index=index)
        dim = len(uh.shape) - 1
        s0 = 'abdefg'

        if dim == 0: # 如果
            # gphi.shape == (NQ, NC, ldof, GD)
            # uh.shape == (gdof, )
            # uh[cell2dof].shape == (NC, ldof)
            # val.shape == (NQ, NC, GD)
            val = np.einsum('...cim, ci->...cm', gphi, uh[cell2dof[index]])
        elif self.doforder == 'sdofs':
            # gphi.shape == (NQ, NC, ldof, GD)
            # uh.shape == (..., gdof)
            # uh[..., cell2dof].shape == (..., NC, ldof)
            # val.shape == (NQ, ..., GD, NC)
            s1 = '...cim, {}ci->...{}mc'.format(s0[:dim], s0[:dim])
            val = np.einsum(s1, gphi, uh[..., cell2dof])
        elif self.doforder == 'vdims':
            # gphi.shape == (NQ, NC, ldof, GD)
            # uh.shape == (gdof, ...)
            # uh[cell2dof, ...].shape == (NC, ldof, ...)
            # val.shape == (NQ, NC, ..., GD)
            s1 = '...cim, ci{}->...c{}m'.format(s0[:dim], s0[:dim])
            val = np.einsum(s1, gphi, uh[cell2dof[index], ...])
        else:
            raise ValueError(f"Unsupported doforder: {self.doforder}. Supported types are: 'sdofs' and 'vdims'.")

        return val

    def function(self, dim=None, array=None, dtype=np.float64):
        return Function(self, dim=dim, array=array, 
                coordtype='barycentric', dtype=dtype)

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            dim = tuple() 
        if type(dim) is int:
            dim = (dim, )

        if self.doforder == 'sdofs':
            shape = dim + (gdof, )
        elif self.doforder == 'vdims':
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)
