
from typing import Optional

import torch
from torch import Tensor, dtype, device
from torch.nn import Module, Linear, init

from .module import TensorMapping


class TensorSpace(Module):
    dtype: dtype
    device: device

    def number_of_basis(self) -> int:
        raise NotImplementedError

    def basis_value(self, p: Tensor) -> Tensor:
        raise NotImplementedError

    def value(self, um: Tensor, p: Tensor) -> Tensor:
        func = Function(self, um=um)
        return func(p)


class Function(TensorMapping):
    """
    @brief Scaler functions in a linear function space.
    """
    def __init__(self, space: TensorSpace, um: Optional[Tensor]) -> None:
        """
        @brief Initialize a scaler function in a linear function space.
        """
        super().__init__()
        dtype = space.dtype
        device = space.device
        M = space.number_of_basis()

        self.space = space
        self.uml = Linear(M, 1, bias=False, device=device, dtype=dtype)
        if um is None:
            init.zeros_(self.uml.weight)
        else:
            self.set_um_inplace(um)

    @property
    def dim(self):
        return self.uml.out_features

    @property
    def um(self):
        """
        @brief The `um` Tensor object inside the module, with shape (1, M).
        """
        return self.uml.weight

    def numpy(self):
        """
        @brief Return `um` as numpy array, with shape (M, ), where M is number of\
               basis.
        """
        return self.um.detach().cpu().numpy()[0, :]

    def set_um_inplace(self, value: Tensor):
        """
        @brief Set values of um inplace. `value` must be in shape of (1, M) or\
               (M, ), where M is the number of basis.
        """
        if value.ndim == 1:
            value = value[None, :]
        with torch.no_grad():
            self.uml.weight[:] = value

    def forward(self, p: Tensor): # (N, 1)
        return self.uml(self.space.basis_value(p)) # (N, 1)
