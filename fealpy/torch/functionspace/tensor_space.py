
from typing import Tuple

from ..typing import Tensor, Size, _S
from .functional import generate_tensor_basis, generate_tensor_grad_basis
from .space import FunctionSpace, _S, Index
from .utils import to_tensor_dof


class TensorFunctionSpace(FunctionSpace):
    def __init__(self, scalar_space: FunctionSpace, shape: Tuple[int, ...], *,
                 dof_priority: bool=True) -> None:
        """_summary_

        Parameters:
            scalar_space (FunctionSpace): The scalar space to build tensor space from.\n
            shape (Tuple[int, ...]): Shape of each dof.\n
            dof_priority (bool, optional): _description_. Defaults to True.
        """
        self.scalar_space = scalar_space
        self.shape = Size(shape)
        self.dof_priority = dof_priority

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
        return generate_tensor_basis(phi, self.shape, self.dof_priority)

    def grad_basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor:
        gphi = self.scalar_space.grad_basis(p, index, **kwargs)
        return generate_tensor_grad_basis(gphi, self.shape, self.dof_priority)

    def cell_to_dof(self) -> Tensor:
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

    def face_to_dof(self) -> Tensor:
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
