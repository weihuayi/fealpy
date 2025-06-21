
from typing import Union, TypeVar, Generic, Callable, Optional

import torch
from torch import Tensor

from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof


_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, Tensor]
Number = Union[int, float]
_S = slice(None)


class LagrangeFESpace(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh: _MT, p: int=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型

        if ctype == 'C':
            self.dof = LinearMeshCFEDof(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self) -> int:
        return self.dof.number_of_global_dofs()

    def interpolation_points(self) -> Tensor:
        return self.dof.interpolation_points()

    def cell_to_dof(self) -> Tensor:
        return self.dof.cell_to_dof()

    def face_to_dof(self) -> Tensor:
        return self.dof.face_to_dof()

    def is_boundary_dof(self, threshold=None) -> Tensor:
        if self.ctype == 'C':
            return self.dof.is_boundary_dof(threshold)
        else:
            raise RuntimeError("boundary dof is not supported by discontinuous spaces.")

    def interpolate(self, source: Union[Callable[..., Tensor], Tensor, Number],
                    uh: Tensor, dim: Optional[int]=None, index: Index=_S) -> Tensor:
        """Interpolate the given Tensor or function `source` to the dofs.
        This is an in-place operation for uh.

        Parameters:
            source (Callable | Tensor | Number): The source to be interpolated.

            uh (Tensor): The output Tensor.

            dim (int | None): The dimension of the dofs in the uh and source.
                This arg will be set to -1 if the `doforder` of space is 'sdofs' when not given,\
                otherwise 0.
            index (Index, optional): The index of the dofs to be interpolated.

        Returns:
            Tensor: The interpolated `uh`.
        """
        if callable(source):
            ipoints = self.interpolation_points() # TODO: 直接获取过滤后的插值点
            source = source(ipoints[index])

        if uh.ndim == 1:
            if isinstance(source, Tensor) and source.shape[-1] == 1:
                source = source.squeeze(-1)
            uh[index] = source
            return uh

        if dim is None:
            dim = -1 if (getattr(self, 'doforder', None) == 'sdofs') else 0

        slicing = [slice(None)] * uh.ndim
        slicing[dim] = index
        uh[slicing] = source

        return uh

    def basis(self, bc: Tensor, index: Index=_S, variable='u'):
        return self.mesh.shape_function(bc, self.p, index=index, variable=variable)

    def grad_basis(self, bc: Tensor, index: Index=_S, variable='u'):
        """
        """
        return self.mesh.grad_shape_function(bc, self.p, index=index, variable=variable)

    def hess_basis(self, bc: Tensor, index: Index=_S, variable='u'):
        """
        """
        return self.mesh.hess_shape_function(bc, self.p, index=index, variable=variable)

    def value(self, uh: Tensor, bc: Tensor,
              dim: Optional[int]=None, index: Index=_S) -> Tensor:
        """Calculate the value of the finite element function.

        Parameters:
            uh (Tensor): Dofs of the function, shaped (..., gdof) for 'sdofs' and\
            'batched', (gdof, ...) for 'vdims'.

            bc (Tensor): Input points in barycentric coordinates, shaped (NQ, NVC).

            dim (int | None, optional):

            index (Index, optional): _description_.

        Raises:
            ValueError: _description_

        Returns:
            Tensor: Function value. Its shape varies according to the `doforder` of space.
            - Returns a Tensor of shape (NQ, ..., NC) if `doforder` is 'sdofs'.
            - Returns a Tensor of shape (NQ, NC, ...) if `doforder` is 'vdims'.
            - Returns a Tensor of shape (..., NQ, NC) if `doforder` is 'batched'.

            Defaults to 'batched' if the space does not have `doforder` attribute.
        """
        phi = self.basis(bc, index=index, variable='x')
        cell2dof = self.dof.cell_to_dof(index)

        dim = len(uh.shape) - 1
        s0 = 'abdefg'
        doforder = 'batched' if not hasattr(self, 'doforder') else self.doforder

        if doforder == 'sdofs':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (..., GD, gdof)
            # uh[..., cell2dof].shape == (..., NC, ldof)
            # val.shape == (NQ, ..., NC)
            s1 = f"...ci, {s0[:dim]}ci->...{s0[:dim]}c"
            val = torch.einsum(s1, phi, uh[..., cell2dof])
        elif doforder == 'vdims':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (gdof, GD)
            # uh[cell2dof, ...].shape == (NC, ldof, ...)
            # val.shape == (NQ, NC, ...)
            s1 = f"c...i, ci{s0[:dim]}->c...{s0[:dim]}"
            val = torch.einsum(s1, phi, uh[cell2dof, ...])
        elif doforder == 'batched':
            # Here 'batched' case is added.
            s1 = f"c...i, {s0[:dim]}ci ->{s0[:dim]}c..."
            val = torch.einsum(s1, phi, uh[..., cell2dof])
        else:
            raise ValueError(f"Unsupported doforder: {self.doforder}. Supported types are: 'sdofs' and 'vdims'.")
        return val

    def grad_value(self, uh: Tensor, bc: Tensor, index: Index=_S):
        pass
