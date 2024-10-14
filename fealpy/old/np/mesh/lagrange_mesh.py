from typing import Optional, Union, List,Tuple

import numpy as np
from numpy.typing import NDArray

from fealpy.np.mesh.mesh_base import _S
from fealpy.np.mesh.quadrature import Quadrature

from .. import logger
from . import functional as F
from .mesh_base import HomogeneousMesh, estr2dim

Index = Union[NDArray, int, slice]
_dtype = np.dtype
_S = slice(None)

class LagrangeMesh(HomogeneousMesh):
    # shape function
    def shape_function(self, bc: NDArray, p=None, index: Index=_S):
        raise NotImplementedError

    def grad_shape_function(self, bc: NDArray, p=None, index: Index=_S):
        raise NotImplementedError

    # jacobi matrix
    def jacobi_matrix(self, bc: Union[NDArray, Tuple[NDArray]], p=None, index: Index=_S, return_grad=False):
        """
        Compute the Jacobian matrix from reference element 'u' to the actual element.

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}

        Parameters:
        - bc: Barycentric coordinates, either a tuple or ndarray.
        - p: Order of the polynomial, defaults to self.p if None.
        - etype: Type of the entity, either int or str.
        - index: Index of the entity, default is _S.
        - return_grad: Boolean flag to determine if gradient should be returned, default is False.

        Returns:
        - J: The Jacobian matrix.
        - (Optional) gphi: Gradient of the shape functions if return_grad is True.
        """ 
        p = self.p if p is None else p

        if isinstance(bc, tuple):
            TD = len(bc)
        elif isinstance(bc, np.ndarray):
            TD = bc.shape[-1] - 1
        else:
            raise ValueError(' `bc` should be a tuple or ndarray!')

        entity = self.entity(TD,index)
        gphi = self.grad_shape_function(bc, p=p)
        J = np.einsum(
                'cin, ...cim->...cnm',
                self.node[entity, :], gphi) #(NC,ldof,GD),(NQ,NC,ldof,TD)
        if return_grad is False:
            return J #(NQ,NC,GD,TD)
        else:
            return J, gphi

    # fundamental form
    def first_fundamental_form(self, bc: Union[NDArray, Tuple[NDArray]], 
            index: Index=_S, return_jacobi=False, return_grad=False):
        """
        Compute the first fundamental form of a mesh surface at integration points.

        Parameters:
        - bc: Barycentric coordinates, either a tuple or ndarray.
        - index: Index of the entity, default is _S.
        - return_jacobi: Boolean flag to determine if the Jacobian matrix should be returned, default is False.
        - return_grad: Boolean flag to determine if gradient should be returned, default is False.

        Returns:
        - G: The first fundamental form matrix.
        - (Optional) J: The Jacobian matrix if return_jacobi is True.
        - (Optional) gphi: Gradient of the shape functions if return_grad is True.
        """        
        if isinstance(bc, tuple):
            TD = len(bc)
        elif isinstance(bc, np.ndarray):
            TD = bc.shape[-1] - 1
        else:
            raise ValueError(' `bc` should be a tuple or ndarray!')

        J = self.jacobi_matrix(bc, index=index,
                return_grad=return_grad)
        
        if return_grad:
            J, gphi = J

        shape = J.shape[0:-2] + (TD, TD)
        G = np.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = np.sum(J[..., i]**2, axis=-1)
            for j in range(i+1, TD):
                G[..., i, j] = np.sum(J[..., i]*J[..., j], axis=-1)
                G[..., j, i] = G[..., i, j]
        if (return_jacobi is False) & (return_grad is False):
            return G
        elif (return_jacobi is True) & (return_grad is False): 
            return G, J
        elif (return_jacobi is False) & (return_grad is True): 
            return G, gphi 
        else:
            return G, J, gphi

    def second_fundamental_form(self, bc: Union[NDArray, Tuple[NDArray]], 
            index=np.s_[:], return_jacobi=False, return_grad=False):
        """
        Compute the second fundamental form of a mesh surface at integration points.

        Parameters:
        - bc: Barycentric coordinates, either a tuple or ndarray.
        - index: Index of the entity, default is np.s_[:].
        - return_jacobi: Boolean flag to determine if the Jacobian matrix should be returned, default is False.
        - return_grad: Boolean flag to determine if gradient should be returned, default is False.

        Returns:
        - 
        """
        if isinstance(bc, tuple):
            TD = len(bc)
        elif isinstance(bc, np.ndarray):
            TD = bc.shape[-1] - 1
        else:
            raise ValueError(' `bc` should be a tuple or ndarray!')

    def vtk_cell_type(self, etype='cell'):
        """
        @berif 返回网格单元对应的 vtk 类型。
        """
        raise NotImplementedError

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
        """
        Parameters
        ----------

        @berif 把网格转化为 VTK 的格式
        """
        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        idx = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell[:, idx]]
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)
