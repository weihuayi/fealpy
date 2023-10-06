
from typing import Optional, Generic, TypeVar

import torch
from torch import Tensor, dtype, device
from torch.nn import init, Parameter
from torch.func import vmap, jacfwd, hessian

from ..nntyping import S as _S
from .module import TensorMapping


class FunctionSpace():
    """The base class for all function spaces in ML module."""
    def __init__(self, in_dim: int=1, out_dim: int=1,
                 dtype: dtype=None, device: device=None, **kwargs) -> None:
        super().__init__()
        assert in_dim >= 1, "Input dimension should be a positive integer."
        assert out_dim >= 1, "Output dimension should be a positive integer."
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dtype = dtype
        self.device = device

    def number_of_basis(self) -> int:
        """
        @brief Return the number of basis.
        """
        raise NotImplementedError

    def basis(self, p: Tensor, *, index=_S) -> Tensor:
        """
        @brief Return the value of basis, with shape (..., #basis).

        @param p: input Tensor. In the shape of (..., #dims).
        @param index: indices of basis.

        @return: Tensor.
        """
        raise NotImplementedError

    def value(self, um: Tensor, p: Tensor) -> Tensor:
        """
        @brief Return value of a function.
        """
        func = Function(self, um=um)
        return func(p)

    def grad_basis(self, p: Tensor, *, index=_S) -> Tensor:
        """
        @brief Return gradient vector of basis, with shape (..., #basis, #dims).

        @param p: input Tensor. In the shape of (..., #dims).
        @param index: indices of basis.

        @return: Tensor.
        """
        func = lambda p: self.basis(p, index=index)
        if p.ndim == 1:
            return jacfwd(func)(p)
        return vmap(jacfwd(func), 0, 0)(p)

    def hessian_basis(self, p: Tensor, *, index=_S) -> Tensor:
        """
        @brief Return hessian matrix of basis, with shape (..., #basis, #dims, #dims).

        @param p: input Tensor. In the shape of (..., #dims).
        @param index: indices of basis.

        @return: Tensor.
        """
        func = lambda p: self.basis(p, index=index)
        if p.ndim == 1:
            return hessian(func)(p)
        return vmap(hessian(func), 0, 0)(p)

    def convect_basis(self, p: Tensor, *, coef: Tensor, index=_S) -> Tensor:
        """
        @brief Return convection item, with shape (..., #basis).

        @param p: input Tensor. In the shape of (..., #dims).
        @param coef: Tensor. The coefficient of the gradient term, or the velocity\
               of the flow field. `coef` must has shape (#dims, ) or (..., #dims).
        @param index: indices of basis.

        @return: Tensor.
        """
        if coef.ndim == 1:
            return torch.einsum('d, ...fd -> ...f', coef, self.grad_basis(p, index=index))
        else:
            return torch.einsum('...d, ...fd -> ...f', coef, self.grad_basis(p, index=index))

    def laplace_basis(self, p: Tensor, *, index=_S) -> Tensor:
        """
        @brief Return value of the Laplacian operator acting on the basis functions,\
               with shape (..., #basis).

        @param p: input Tensor. In the shape of (..., #dims).
        @param index: indices of basis.

        @return: Tensor.
        """
        hes_func = hessian(lambda p: self.basis(p, index=index))
        lap_func = lambda x: torch.sum(torch.diagonal(hes_func(x), 0, -1, -2), dim=-1)
        if p.ndim == 1:
            return lap_func(p)
        return vmap(lap_func, 0, 0)(p)

    def derivative_basis(self, p: Tensor, *idx: int, index=_S) -> Tensor:
        """
        @brief Return specified partial derivatives of basis, with shape (..., #basis).

        @param p: input Tensor. In the shape of (..., #dims).
        @param *idx: int. Index of the independent variable to take partial derivatives.

        @return: Tensor.
        """
        raise NotImplementedError(f"derivative_basis is not supported by {self.__class__.__name__}"
                                  "or it has not been implmented.")


_FS = TypeVar('_FS', bound=FunctionSpace)

class Function(TensorMapping, Generic[_FS]):
    """
    @brief Functions in a linear function space.
    """
    def __init__(self, space: _FS, gd=1, um: Optional[Tensor]=None) -> None:
        """
        @brief Initialize a function in a linear function space.
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
