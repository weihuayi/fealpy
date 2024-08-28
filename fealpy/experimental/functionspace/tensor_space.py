
from typing import Tuple, Union, Callable
from math import prod

from ..backend import backend_manager as bm
from ..typing import TensorLike, Size, _S
from .functional import generate_tensor_basis, generate_tensor_grad_basis
from .space import FunctionSpace, _S, Index
from .utils import to_tensor_dof
from fealpy.decorator import barycentric, cartesian


class TensorFunctionSpace(FunctionSpace):
    def __init__(self, scalar_space: FunctionSpace, shape: Tuple[int, ...]) -> None:
        """_summary_

        Parameters:
            scalar_space (FunctionSpace): The scalar space to build tensor space from.\n
            shape (int, ...): Shape of each dof.
                Requires a `-1` be the first or last element to mark the priority
                of the DoF in arrangement.
        """
        self.scalar_space = scalar_space
        self.shape = shape

        if len(shape) < 2:
            raise ValueError('shape must be a tuple of at least two element')

        if shape[0] == -1:
            self.dof_shape = tuple(shape[1:])
            self.dof_priority = False
        elif shape[-1] == -1:
            self.dof_shape = tuple(shape[:-1])
            self.dof_priority = True
        else:
            raise ValueError('`-1` is required as the first or last element')

    @property
    def mesh(self):
        return self.scalar_space.mesh

    @property
    def device(self): return self.scalar_space.device
    @property
    def ftype(self): return self.scalar_space.ftype
    @property
    def itype(self): return self.scalar_space.itype

    @property
    def dof_numel(self) -> int:
        return prod(self.dof_shape)

    @property
    def dof_ndim(self) -> int:
        return len(self.dof_shape)

    def number_of_global_dofs(self) -> int:
        return self.dof_numel * self.scalar_space.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.dof_numel * self.scalar_space.number_of_local_dofs(doftype)

    def basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        phi = self.scalar_space.basis(p, index, **kwargs) # (NC, NQ, ldof)
        return generate_tensor_basis(phi, self.dof_shape, self.dof_priority)

    def grad_basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        gphi = self.scalar_space.grad_basis(p, index, **kwargs)
        return generate_tensor_grad_basis(gphi, self.dof_shape, self.dof_priority)

    def cell_to_dof(self) -> TensorLike:
        """Get the cell to dof mapping.

        Returns:
            Tensor: Cell to dof mapping, shaped (NC, ldof*dof_numel).
        """
        return to_tensor_dof(
            self.scalar_space.cell_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_priority
        )

    def face_to_dof(self) -> TensorLike:
        """Get the face to dof mapping.

        Returns:
            Tensor: Face to dof mapping, shaped (NF, ldof*dof_numel).
        """
        return to_tensor_dof(
            self.scalar_space.face_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_priority
        )

    def interpolation_points(self) -> TensorLike:

        return self.scalar_space.interpolation_points()
    
    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike], ) -> TensorLike:

        if self.dof_priority:
            uI = self.scalar_space.interpolate(u)
            ndim = len(self.shape)
            uI = bm.swapaxes(uI, ndim-1, ndim-2) 
        else:
            uI = self.scalar_space.interpolate(u)   

        return uI.reshape(-1)

    def is_boundary_dof(self, threshold=None, direction=None) -> TensorLike:
        """Return bools indicating boundary dofs.

        Parameters:
        threshold (callable or None, optional): A function used to determine boundary edges based on their 
            positions. This function should return a boolean array indicating which edges are on the boundary.

        direction (callable or TensorLike, optional): A function or a boolean array specifying which directions 
            to apply the boundary condition. If a function, it should return a boolean array indicating the 
            directions (e.g., [True, False] to apply only to the first component). If None, all directions are 
            considered.

        Returns:
            TensorLike: shaped (scalar_gdof * dof_numel,)
        """
        scalar_gdof = self.scalar_space.number_of_global_dofs()
        scalar_is_bd_dof = self.scalar_space.is_boundary_dof(threshold)

        if self.dof_priority:
            is_bd_dof = bm.reshape(scalar_is_bd_dof, (-1,) * self.dof_ndim + (scalar_gdof,))
            is_bd_dof = bm.broadcast_to(is_bd_dof, self.dof_shape + (scalar_gdof,))
        else:
            is_bd_dof = bm.reshape(scalar_is_bd_dof, (scalar_gdof,) + (-1,) * self.dof_ndim)
            is_bd_dof = bm.broadcast_to(is_bd_dof, (scalar_gdof,) + self.dof_shape)

        if direction is not None:
            if callable(direction):
                direction_flags = direction()
            else:
                direction_flags = bm.array(direction)
            if self.dof_priority:
                direction_flags_broadcast = bm.reshape(direction_flags, (-1, 1))
                direction_flags_broadcast = bm.broadcast_to(direction_flags_broadcast, is_bd_dof.shape)
            else:
                direction_flags_broadcast = bm.broadcast_to(direction_flags, is_bd_dof.shape)

            is_bd_dof = is_bd_dof & direction_flags_broadcast

        return is_bd_dof.reshape(-1)
    
    def boundary_interpolate(self,
        gD: Union[Callable, int, float, TensorLike],
        uh: TensorLike,
        threshold: Union[Callable, TensorLike, None]=None) -> TensorLike:

        ipoints = self.interpolation_points()
        scalar_space = self.scalar_space
        isScalarBDof = scalar_space.is_boundary_dof(threshold=threshold)

        if callable(gD):
            gD = gD(ipoints[isScalarBDof])
        
        isTensorBDof = self.is_boundary_dof(threshold=threshold)
        if self.dof_priority:
            uh = bm.set_at(uh, isTensorBDof, gD.T.reshape(-1))
        else:
            uh = bm.set_at(uh, isTensorBDof, gD.reshape(-1))

        return uh

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        phi = self.basis(bc, index=index)
        c2dof = self.cell_to_dof()[index]
        if self.dof_priority:
            val = bm.einsum('cql..., cl... -> cq...', phi, uh[c2dof, ...])
        else:
            val = bm.einsum('cql, ...cl -> ...cq', phi, uh[..., c2dof])
        return val
