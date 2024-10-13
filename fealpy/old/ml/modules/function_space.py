
from warnings import warn
from typing import Generic, TypeVar, Optional

import torch
from torch import Tensor, dtype, device
from torch.nn import Module, Parameter
from torch.func import vmap, jacfwd, hessian

from ..nntyping import S as _S
from ..nntyping import TensorOrFunc
from .module import TensorMapping


# NOTE: Inherit from Module to make it able to be saved as '.pth' file.
class FunctionSpace(Module):
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

    def function(self, um: Tensor, *, keepdim=True, requires_grad=False):
        """
        @brief Initialize a function in the space.

        @param um: value Tensor for dofs.
        @param keepdim: bool. The feature axis/dim of output will always be kept\
               if `True`. Defaults to `True`.
        @param requires_grad: bool. If value of dofs requires gradient.\
               Defaults to `False`.

        @return: Function.
        """
        return Function(self, um, keepdim=keepdim, requires_grad=requires_grad)

    def fit(self, sample: Tensor, func: TensorOrFunc, *, weight_decay=0.,
            keepdim=True, requires_grad=False):
        """
        @brief Fit a function in the space. This is done by solving linear\
               equations, using `torch.linalg.solve`.

        @param sample: collocation points Tensor for building linear equations.
        @param func: target function. This callable should receive one single\
               Tensor and return also a single Tensor.
        @param keepdim: bool. See `Function.__init__`.
        @param requires_grad: bool.

        @return: Function.
        """
        from torch.linalg import solve
        from ..solvertools import ridge

        assert sample.ndim == 2
        N = sample.shape[0]
        basis = self.basis(sample)
        NF = basis.shape[-1]
        if N < NF:
            warn(f"Only {N} sample point(s) provided for space with {NF} basis.")
        active_basis_flag = torch.any(torch.abs(basis) > 1e-8, dim=0)
        A = basis[:, active_basis_flag]
        A_ = ridge(A.T@A, lambda_=weight_decay, inplace=False)

        if callable(func):
            source = func(sample)
        else:
            source = func

        # for scaler functions
        if source.ndim == 1 or source.shape[-1] == 1:
            b = source
            um = torch.zeros((NF, ), dtype=sample.dtype, device=sample.device)
            um[active_basis_flag] = solve(A_, A.T@b)

        # for vector functions
        else:
            NS = source.shape[-1]
            b = source
            um = torch.zeros((NF, NS), dtype=sample.dtype, device=sample.device)
            um[active_basis_flag, :] = solve(A_, A.T@b)

        return Function(self, um, keepdim=keepdim, requires_grad=requires_grad)

    def basis(self, p: Tensor, *, index=_S) -> Tensor:
        """
        @brief Return the value of basis, with shape (..., #basis).

        @param p: input Tensor. In the shape of (..., #dims).
        @param index: indices of basis.

        @return: Tensor.
        """
        raise NotImplementedError

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

    def integral_basis(self, quadpts: Tensor, weights: Optional[Tensor]=None, *, index=_S):
        """
        @brief Return integral of basis functions, with shape (#basis, ).

        @param quadpts: quadrature points Tensor. In the shape of (..., #dims).
        @param weights: weights Tensor of quadrature points, with shape (...).

        @return: Tensor.
        """
        val = self.basis(quadpts, index=index)
        if weights is None:
            mean_dim = tuple(range(quadpts.ndim))[:-1]
            return torch.mean(val, dim=mean_dim)
        return torch.einsum('...f, ... -> f', val, weights)


_FS = TypeVar('_FS', bound=FunctionSpace)

class Function(TensorMapping, Generic[_FS]):
    """
    @brief Functions in a linear function space.
    """
    def __init__(self, space: _FS, um: Tensor, *, keepdim=True, requires_grad=False) -> None:
        """
        @brief Initialize a function in a linear function space.

        @param space: FunctionSpace.
        @param um: Tensor. Size of `um` in the first axis must match the number\
               of basis, or a ValueError will be generated.
        @param keepdim: bool. The feature axis/dim of output will always be kept\
               if `True`. Defaults to `True`.
        @param requires_grad: bool. If value of dofs requires gradient.\
               Defaults to `False`.
        """
        super().__init__()
        dtype = space.dtype
        device = space.device
        M = space.number_of_basis()
        if um.shape[0] != M:
            raise ValueError(f"There are {M} basis in the space, but got um with "
                             f"size {um.shape[0]} in the first axis.")

        self.space = space
        self._tensor = Parameter(torch.empty(um.shape, dtype=dtype, device=device),
                                 requires_grad=requires_grad)
        self.set_um_inplace(um)
        self.keepdim=keepdim

    @property
    def um(self):
        """
        @brief The `um` Tensor object inside the module, with shape (M, GD).
        """
        return self._tensor

    @property
    def gd(self):
        return self._tensor.shape[-1]

    def is_scaler(self):
        """
        @brief If this is a scaler function.
        """
        um = self._tensor
        if um.ndim > 1 and um.shape[1] >= 2:
            return False
        else:
            return True

    def numpy(self):
        """
        @brief Return `um` as numpy array, with shape (M, GD), where M is number of\
               basis.
        """
        return self.um.detach().cpu().numpy()

    def set_um_inplace(self, um: Tensor):
        """
        @brief Set values of um inplace. `um` must be in shape of\
               (M, GD) or (M, ), where M is the number of basis.
        """
        if um.ndim >= 3:
            raise RuntimeError("`um` must be 1 or 2 dimensional.")
        else:
            with torch.no_grad():
                self._tensor[:] = um

    def forward(self, p: Tensor) -> Tensor:
        return self.value(p)

    def value(self, p: Tensor) -> Tensor:
        basis = self.space.basis(p)
        um = self._tensor
        if um.ndim == 1:
            ret = torch.einsum("...f, f -> ...", basis, self._tensor)
            return ret.unsqueeze(-1) if self.keepdim else ret
        return torch.einsum("...f, fe -> ...e", basis, self._tensor)

    def gradient(self, p: Tensor) -> Tensor:
        grad = self.space.grad_basis(p)
        um = self._tensor
        if um.ndim == 1:
            return torch.einsum("...fd, f -> ...d", grad, um)
        return torch.einsum("...fd, fe -> ...ed", grad, um)

    def convect(self, p: Tensor, coef: Tensor) -> Tensor:
        convect = self.space.convect_basis(p, coef=coef)
        um = self._tensor
        if um.ndim == 1:
            ret = torch.einsum("...f, f -> ...", convect, um)
            return ret.unsqueeze(-1) if self.keepdim else ret
        return torch.einsum("...f, fe -> ...fe", convect, um)

    def hessian(self, p: Tensor) -> Tensor:
        hessian = self.space.hessian_basis(p)
        um = self._tensor
        if um.ndim == 1:
            return torch.einsum("...fd, f -> ...dd", hessian, um)
        return torch.einsum("...fdd, fe -> ...edd", hessian, um)

    @classmethod
    def zeros(cls, space: _FS, gd: int=0, *, keepdim=True, requires_grad=False):
        """
        @brief Initialize a zero function.

        @param space: FunctionSpace.
        @param gd: int. Output dimension of the function. If `gd == 0`, um will\
               have no extra dims, being with shape (#basis, ). If `gd >= 1`, the\
               shape of um is (#basis, gd).
        @param keepdim: bool. See `Function.__init__`.
        @param requires_grad: bool. See `Function.__init__`.

        @return: Function.
        """
        assert gd >= 0
        M = space.number_of_basis()
        if gd == 0:
            um = torch.zeros((M, ))
        else:
            um = torch.zeros((M, gd))
        return Function(space, um, keepdim=keepdim, requires_grad=requires_grad)
