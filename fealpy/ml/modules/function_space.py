
from typing import Optional

import torch
from torch import Tensor, dtype, device
from torch.nn import Module, Linear, init, Parameter

from .module import TensorMapping


class FunctionSpaceBase(Module):
    dtype: dtype
    device: device

    def number_of_basis(self) -> int:
        """
        @brief Return the number of basis.
        """
        raise NotImplementedError

    def basis(self, p: Tensor) -> Tensor:
        """
        @brief Return the value of basis, with shape (#samples, #basis, #dims),\
               or (#samples, #basis).
        """
        raise NotImplementedError

    def value(self, um: Tensor, p: Tensor) -> Tensor:
        """
        @brief Return value of a function.
        """
        func = Function(self, um=um)
        return func(p)


class Function(TensorMapping):
    """
    @brief Scaler functions in a linear function space.
    """
    def __init__(self, space: FunctionSpaceBase, gd=1, um: Optional[Tensor]=None) -> None:
        """
        @brief Initialize a scaler function in a linear function space.
        """
        super().__init__()
        dtype = space.dtype
        device = space.device
        M = space.number_of_basis()

        self.space = space
        self._tensor = Parameter(torch.empty((M, gd), dtype=dtype, device=device))
        if um is None:
            init.zeros_(self._tensor)
        else:
            self.set_um_inplace(um)

    @property
    def um(self):
        """
        @brief The `um` Tensor object inside the module, with shape (M, GD).
        """
        return self._tensor

    @property
    def gd(self):
        return self._tensor.shape[-1]

    def numpy(self):
        """
        @brief Return `um` as numpy array, with shape (M, GD), where M is number of\
               basis.
        """
        return self.um.detach().cpu().numpy()

    def set_um_inplace(self, value: Tensor):
        """
        @brief Set values of um inplace. `value` must be in shape of\
               (M, GD), where M is the number of basis.
        """
        if value.ndim >= 3:
            raise RuntimeError("Value must be 1 or 2 dimensional.")
        else:
            if value.ndim == 1:
                value = value[:, None]
            with torch.no_grad():
                self._tensor[:] = value

    def forward(self, p: Tensor):
        return torch.einsum("nf, fd -> nd", self.space.basis(p), self._tensor)
