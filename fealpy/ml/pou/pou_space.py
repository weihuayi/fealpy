
from typing import TypeVar, Generic, Callable, List, Optional

import torch
from torch import Tensor

from ..modules import FunctionSpaceBase
from .pou import PoU

_FS = TypeVar('_FS', bound=FunctionSpaceBase)
SpaceFactory = Callable[[int], _FS]


def assemble(extra_dim: int=0, use_coef=False):
    """
    @brief Assemble data from partitions.

    The function that this decorator acts on must have at least two inputs:\
    the partition index number and the input tensor. The functions that are\
    acted upon by this decorator automatically collect the output of the\
    original function within the partition and assemble it into a total matrix.

    If `use_coef`, an optional parameter `coef` can be accepted by the wrapper\
    and will be passed into the wrapped function, which is like:
    ```
        def func(idx: int, p: Tensor, coef: Tensor):
            ...
    ```
    Tensor full of `1.0` will be used if coef is not given.
    """
    if use_coef:
        def assemble_(func: Callable[..., Tensor]):
            def wrapper(self: "PoUSpace", p: Tensor, coef: Optional[Tensor]=None,
                        *args, **kwargs) -> Tensor:
                N = p.shape[0]
                M = self.number_of_basis()
                GD = p.shape[-1]
                ret = torch.zeros((N, M) + (GD, )*extra_dim,
                                dtype=self.dtype, device=self.device)
                lp = self.pou.global_to_local(p)
                basis_cursor = 0

                if coef is None:
                    coef = torch.ones_like(p)
                if coef.ndim == 1:
                    coef = coef[None, :].broadcast_to(p.shape[0], -1)
                assert coef.shape[0] == p.shape[0]

                for idx, part in enumerate(self.partitions):
                    NF = part.number_of_basis()
                    x = lp[:, idx, :]
                    flag = part.flag(x)
                    ret[flag, basis_cursor:basis_cursor+NF, ...]\
                        += func(self, idx, x[flag, ...], coef[flag, ...], *args, **kwargs)
                    basis_cursor += NF
                return ret
            return wrapper
    else:
        def assemble_(func: Callable[..., Tensor]):
            def wrapper(self: "PoUSpace", p: Tensor, *args, **kwargs) -> Tensor:
                N = p.shape[0]
                M = self.number_of_basis()
                GD = p.shape[-1]
                ret = torch.zeros((N, M) + (GD, )*extra_dim,
                                dtype=self.dtype, device=self.device)
                lp = self.pou.global_to_local(p)
                basis_cursor = 0

                for idx, part in enumerate(self.partitions):
                    NF = part.number_of_basis()
                    x = lp[:, idx, :]
                    flag = part.flag(x)
                    ret[flag, basis_cursor:basis_cursor+NF, ...]\
                        += func(self, idx, x[flag, ...], *args, **kwargs)
                    basis_cursor += NF
                return ret
            return wrapper
    return assemble_


class PoULocalSpace(FunctionSpaceBase, Generic[_FS]):
    def __init__(self, pou_fn, space: _FS) -> None:
        super().__init__()
        self.pou_fn = pou_fn
        self.space = space

    def flag(self, p: Tensor):
        return self.pou_fn.flag(p)

    def number_of_basis(self) -> int:
        return self.space.number_of_basis()

    def basis(self, p: Tensor) -> Tensor:
        return self.space.basis(p) * self.pou_fn(p)

    def grad_basis(self, p: Tensor) -> Tensor:
        space = self.space
        ret = torch.einsum("nd, nf -> nfd", self.pou_fn.gradient(p), space.basis(p))
        ret += self.pou_fn(p)[..., None] * space.grad_basis(p)
        return ret

    def hessian_basis(self, p: Tensor) -> Tensor:
        space = self.space
        ret = torch.einsum("nxy, nf -> nfxy", self.pou_fn.hessian(p), space.basis(p))
        cross = torch.einsum("nx, nfy -> nfxy", self.pou_fn.gradient(p),
                             space.grad_basis(p))
        ret += cross + torch.transpose(cross, -1, -2)
        ret += self.pou_fn(p)[..., None, None] * space.hessian_basis(p)
        return ret

    def laplace_basis(self, p: Tensor, coef: Optional[Tensor]=None) -> Tensor:
        space = self.space
        if coef is None:
            ret = torch.einsum("ndd, nf -> nf", self.pou_fn.hessian(p), space.basis(p))
            ret += 2 * torch.einsum("nd, nfd -> nf", self.pou_fn.gradient(p),
                                    space.grad_basis(p))
        else:
            if coef.ndim == 1:
                    coef = coef[None, :].broadcast_to(p.shape[0], -1)
            assert coef.shape[0] == p.shape[0]
            ret = torch.einsum("ndd, nf, nd -> nf", self.pou_fn.hessian(p), space.basis(p), coef)
            ret += 2 * torch.einsum("nd, nfd, nd -> nf", self.pou_fn.gradient(p),
                                    space.grad_basis(p), coef)
        ret += self.pou_fn(p) * space.laplace_basis(p, coef=coef)
        return ret

    def derivative_basis(self, p: Tensor, *idx: int) -> Tensor:
        N = p.shape[0]
        nf = self.number_of_basis()
        order = len(idx)
        space = self.space
        ret = torch.zeros((N, nf), dtype=self.dtype, device=self.device)

        if order == 0:
            ret[:] = self.basis(p)
        elif order == 1:
            ret += self.pou_fn.derivative(p, idx[0]) * space.basis(p)
            ret += self.pou_fn(p) * space.derivative_basis(p, idx[0])
        elif order == 2:
            ret += self.pou_fn.derivative(p, idx[0], idx[1]) * space.basis(p)
            ret += self.pou_fn.derivative(p, idx[0]) * space.derivative_basis(p, idx[1])
            ret += self.pou_fn.derivative(p, idx[1]) * space.derivative_basis(p, idx[0])
            ret += self.pou_fn(p) * space.derivative_basis(p, idx[0], idx[1])
        elif order == 3:
            ret += self.pou_fn.derivative(p, idx[0], idx[1], idx[2]) * space.basis(p)
            ret += self.pou_fn.derivative(p, idx[0], idx[1]) * space.derivative_basis(p, idx[2])
            ret += self.pou_fn.derivative(p, idx[1], idx[2]) * space.derivative_basis(p, idx[0])
            ret += self.pou_fn.derivative(p, idx[2], idx[0]) * space.derivative_basis(p, idx[1])
            ret += self.pou_fn.derivative(p, idx[0]) * space.derivative_basis(p, idx[2], idx[1])
            ret += self.pou_fn.derivative(p, idx[1]) * space.derivative_basis(p, idx[0], idx[2])
            ret += self.pou_fn.derivative(p, idx[2]) * space.derivative_basis(p, idx[1], idx[0])
            ret += self.pou_fn(p) * space.derivative_basis(p, idx[0], idx[1], idx[2])

        elif order == 4:
            pass
        # TODO: finish this
        else:
            raise NotImplementedError("Derivatives higher than order 4 have bot been implemented.")
        return ret


class PoUSpace(FunctionSpaceBase, Generic[_FS]):
    def __init__(self, space_factory: SpaceFactory[_FS], pou: PoU, pou_fn,
                 print_status=False) -> None:
        super().__init__()

        self.pou = pou
        self.partitions: List[PoULocalSpace[_FS]] = []
        self.in_dim = -1
        self.out_dim = -1

        for i in range(pou.number_of_partitions()):
            part = PoULocalSpace(pou_fn=pou_fn, space=space_factory(i))
            if self.in_dim == -1:
                self.in_dim = part.space.in_dim
                self.out_dim = part.space.out_dim
            else:
                if self.in_dim != part.space.in_dim:
                    raise RuntimeError("Can not group together local spaces with"
                                       "different input dimension.")
                if self.out_dim != part.space.out_dim:
                    raise RuntimeError("Can not group together local spaces with"
                                       "differnet output dimension.")

            self.partitions.append(part)
            self.add_module(f"part_{i}", part)

    def number_of_partitions(self):
        return self.pou.number_of_partitions()

    def number_of_basis(self):
        return sum(p.number_of_basis() for p in self.partitions)

    def partition_basis_slice(self, idx: int):
        """
        @brief Returns the index slice of the local basis function of the\
               specified partition in the global model.

        @param idx: int. The index of the partition.
        """
        assert idx >= 0
        assert idx < self.number_of_partitions()

        start = sum((p.number_of_basis() for p in self.partitions[:idx]), 0)
        stop = start + self.partitions[idx].number_of_basis()
        return slice(start, stop, None)

    @assemble(0)
    def basis(self, idx: int, p: Tensor) -> Tensor:
        return self.partitions[idx].basis(p)

    @assemble(0, use_coef=True)
    def convect_basis(self, idx: int, p: Tensor, coef: Tensor) -> Tensor:
        scale = self.pou.grad_global_to_local(p, index=idx)[:, 0, :] # (samples, GD)
        if coef is not None:
            scale *= coef
        return self.partitions[idx].convect_basis(p, coef=scale)

    @assemble(0, use_coef=True)
    def laplace_basis(self, idx: int, p: Tensor, coef: Tensor) -> Tensor:
        scale = self.pou.grad_global_to_local(p, index=idx)[:, 0, :]**2 # (samples, GD)
        if coef is not None:
            scale *= coef
        return self.partitions[idx].laplace_basis(p, coef=scale)

    @assemble(0)
    def derivative_basis(self, idx: int, p: Tensor, *dim: int) -> Tensor:
        gg2l = self.pou.grad_global_to_local(p, index=idx) # (samples, 1, GD)
        scale = torch.prod(gg2l[idx, dim], dim=-1) # (samples, 1)
        return self.partitions[idx].derivative_basis(p, *dim) * scale

    @assemble(1)
    def grad_basis(self, idx: int, p: Tensor) -> Tensor:
        gg2l = self.pou.grad_global_to_local(p, index=idx) # (samples, 1, GD)
        return self.partitions[idx].grad_basis(p) * gg2l # (samples, basis, GD)
