
from typing import Tuple

from ..typing import Tensor, Size, _S
from .functional import generate_tensor_basis
from .space import FunctionSpace, _S, Index
from .utils import to_tensor_dof


class TensorFunctionSpace(FunctionSpace):
    def __init__(self, scalar_space: FunctionSpace, shape: Tuple[int, ...], *,
                 dof_last: bool=True) -> None:
        self.scalar_space = scalar_space
        self.shape = Size(shape)
        self.dof_last = dof_last

    @property
    def mesh(self):
        return self.scalar_space.mesh

    @property
    def dof_numel(self) -> int:
        return self.shape.numel()

    @property
    def dof_ndim(self) -> int:
        return len(self.shape)

    def number_of_global_dofs(self) -> int:
        return self.dof_numel * self.scalar_space.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.dof_numel * self.scalar_space.number_of_local_dofs(doftype)

    def basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor:
        phi = self.scalar_space.basis(p, index, **kwargs) # (NC, NQ, ldof)
        return generate_tensor_basis(phi, self.shape, self.dof_last)

    def grad_basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor:
        gphi = self.scalar_space.grad_basis(p, index, **kwargs)
        # TODO: finish this
        pass

    def cell_to_dof(self) -> Tensor:
        """Get the cell to dof mapping.

        Returns:
            Tensor: Cell to dof mapping, shaped (NC, ldof*dof_numel).
        """
        return to_tensor_dof(
            self.scalar_space.cell_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_last
        )

    def face_to_dof(self) -> Tensor:
        """Get the face to dof mapping.

        Returns:
            Tensor: Face to dof mapping, shaped (NF, ldof*dof_numel).
        """
        return to_tensor_dof(
            self.scalar_space.face_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_last
        )
