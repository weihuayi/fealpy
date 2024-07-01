
import torch

from fealpy.torch.functionspace.space import _S

from .space import FunctionSpace, _S, Index
from . import utils as U

_Size = torch.Size
Tensor = torch.Tensor


class TensorFunctionSpace(FunctionSpace):
    def __init__(self, scalar_space: FunctionSpace, shape: _Size, *,
                 dof_last: bool=True) -> None:
        self.scalar_space = scalar_space
        self.shape = torch.Size(shape)
        self.dof_last = dof_last

    @property
    def ndim_dof(self) -> int:
        return self.shape.numel()

    def number_of_global_dofs(self) -> int:
        return self.ndim_dof * self.scalar_space.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.ndim_dof * self.scalar_space.number_of_local_dofs(doftype)

    def basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor:
        phi = self.scalar_space.basis(p, index, **kwargs) # (NC, NQ, ldof)
        # TODO: finish this

    def grad_basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor:
        gphi = self.scalar_space.grad_basis(p, index, **kwargs)
        # TODO: finish this

    def strain(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor:
        """_summary_

        Parameters:
            p (Tensor): Inputs for the grad_basis.\n
            index (Index, optional): indices of entities.

        Returns:
            Tensor: _description_
        """
        gphi = self.scalar_space.grad_basis(p, index, **kwargs)
        ldof, GD = gphi.shape[-2:]

        if self.dof_last:
            indices = U.flatten_indices((ldof, GD), (1, 0))
        else:
            indices = U.flatten_indices((ldof, GD), (0, 1))

        return torch.cat([U.normal_strain(gphi, indices),
                          U.shear_strain(gphi, indices)], dim=-2)
