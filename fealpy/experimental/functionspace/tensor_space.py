
from typing import Tuple
from math import prod

from ..backend import backend_manager as bm
from ..typing import TensorLike, Size, _S
from .functional import generate_tensor_basis, generate_tensor_grad_basis
from .space import FunctionSpace, _S, Index
from .utils import to_tensor_dof


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
        scalar_gdof = self.scalar_space.number_of_global_dofs()
        scalar_ips = self.scalar_space.interpolation_points()

        # Based on the priority of degrees of freedom, choose different processing methods
        if self.dof_priority:
            # Component priority
            # Reshape the shape of the scalar interpolation points to obtain an array of shape (dof_ndim, scalar_dof, dim), 
                # in order to copy the interpolation points for each degree of freedom component
            ips = bm.reshape(scalar_ips, (-1,)*self.dof_ndim + (scalar_gdof, scalar_ips.shape[-1]))
            # Broadcast the interpolation points to the shape of the vector space, 
                # resulting in an array of shape (dof_shape, scalar_gdof, dim)
            ips = bm.broadcast_to(ips, self.dof_shape + (scalar_gdof, scalar_ips.shape[-1]))

        else:
            # Node priority
            # Reshape the shape of the scalar interpolation points to obtain an array of shape (scalar_dof, dof_ndim, dim), 
                # in order to copy the interpolation points for each degree of freedom component
            ips = bm.reshape(scalar_ips, (scalar_gdof,) + (-1,)*self.dof_ndim + (scalar_ips.shape[-1],))
            # Broadcast the interpolation points to the shape of the vector space,
                # resulting in an array of shape (scalar_gdof, dof_shape, dim)
            ips = bm.broadcast_to(ips, (scalar_gdof,) + self.dof_shape + (scalar_ips.shape[-1],))


        # Reshape the shape to make the degree of freedom order flat (total_dofs, dim)
        ips = bm.reshape(ips, (-1, scalar_ips.shape[-1]))

        return ips


    def is_boundary_dof(self, threshold=None) -> TensorLike:
        """Return bools indicating boundary dofs.

        Returns:
            TensorLike: shaped (scalar_gdof * dof_numel,)
        """
        scalar_gdof = self.scalar_space.number_of_global_dofs()
        scalar_is_bd_dof = self.scalar_space.is_boundary_dof(threshold)

        if self.dof_priority:
            is_bd_dof = bm.reshape(scalar_is_bd_dof, (-1,)*self.dof_ndim + (scalar_gdof,))
            is_bd_dof = bm.broadcast_to(is_bd_dof, self.dof_shape + (scalar_gdof,))

        else:
            is_bd_dof = bm.reshape(scalar_is_bd_dof, (scalar_gdof,) + (-1,)*self.dof_ndim)
            is_bd_dof = bm.broadcast_to(is_bd_dof, (scalar_gdof,) + self.dof_shape)

        return is_bd_dof.reshape(-1)
