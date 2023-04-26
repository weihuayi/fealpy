import numpy as np
from typing import Optional, Union, Callable
from .Function import Function
from ..decorator import barycentric
from .fem_dofs import *

class LagrangeFESpace():
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

    def __str__(self):
        return "Lagrange finite element space on linear mesh!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell'):
        if self.spacetype == 'C':
            return self.dof.number_of_local_dofs(doftype=doftype)
        elif self.spacetype == 'D':
            return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self, index=np.s_[:]):
        return self.dof.cell_to_dof()[index]

    def face_to_dof(self, index=np.s_[:]):
        return self.dof.face_to_dof()[index] #TODO: index

    def edge_to_dof(self, index=np.s_[:]):
        return self.dof.edge_to_dof() #TODO：index

    def is_boundary_dof(self, threshold=None):
        if self.spacetype == 'C':
            return self.dof.is_boundary_dof(threshold=threshold)
        else:
            raise ValueError('This space is a discontinuous space!')

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    @barycentric
    def basis(self, bc, index=np.s_[:]):
        p = self.p
        phi = self.mesh.shape_function(bc, p=p)
        return phi[..., None, :]

    @barycentric
    def grad_basis(self, bc, index=np.s_[:]):
        """
        @brief 
        @note 注意这里调用的实际上不是形状函数的梯度，而是网格空间基函数的梯度
        """
        return self.mesh.grad_shape_function(bc, p=self.p, index=index)
    
    @barycentric
    def face_basis(self, bc, index=np.s_[:]):
        """
        @brief 计算 face 上的基函数在给定积分点处的函数值
        """
        p = self.p
        phi = self.mesh.shape_function(bc, p=p, etype='face')
        return phi[..., None, :]

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

    def interpolate(self):
        pass


    def boundary_interpolate(self, 
            gD: Union[Callable, int, float, np.ndarray], 
            uh: np.ndarray, 
            threshold: Union[Callable, np.ndarray, None]=None) -> np.ndarray:
        """
        @brief Set the first type (Dirichlet) boundary conditions.

        @param gD: boundary condition function or value (can be a callable, int, float, or numpy.ndarray).
        @param uh: numpy.ndarray, FE function uh .
        @param threshold: optional, threshold for determining boundary degrees of freedom (default: None).

        @return numpy.ndarray, a boolean array indicating the boundary degrees of freedom.

        This function sets the Dirichlet boundary conditions for the FE function `uh`. It supports
        different types for the boundary condition `gD`, such as a function, a scalar, or a numpy array.
        """
        ipoints = self.interpolation_points() # TODO: 直接获取过滤后的插值点
        isDDof = self.is_boundary_dof(threshold=threshold)
        GD = self.geo_dimension()

        if callable(gD): 
            gD = gD(ipoints[isDDof])


        if (len(uh.shape) == 1) or (self.doforder == 'vdims'):
            if len(uh.shape) == 1 and gD.shape[-1] == 1:
                gD = gD[..., 0]
            uh[isDDof] = gD 
        elif self.doforder == 'sdofs':
            if isinstance(gD, (int, float)):
                uh[..., isDDof] = gD 
            elif isinstance(gD, np.ndarray):
                if gD.shape == (GD, ):
                    uh[..., isDDof] = gD[:, None]
                else:
                    uh[..., isDDof] = gD.T
            else:
                raise ValueError("Unsupported type for gD. Must be a callable, int, float, or numpy.ndarray.")

        if len(uh.shape) > 1:
            if self.doforder == 'sdofs':
                shape = (len(uh.shape)-1)*(1, ) + isDDof.shape
            elif self.doforder == 'vdims':
                shape = isDDof.shape + (len(uh.shape)-1)*(1, )
            isDDof = np.broadcast_to(isDDof.reshape(shape), shape=uh.shape) 
        return isDDof

    set_dirichlet_bc = boundary_interpolate 


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

    def show_function(self, axes):
        pass

