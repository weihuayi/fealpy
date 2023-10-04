
from typing import TypeVar, Generic, Callable, List, Optional

import torch
from torch import Tensor

from ..nntyping import S as _S
from ..modules.function_space import FunctionSpaceBase
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
            def wrapper(self: "PoUSpace", p: Tensor, *, coef: Optional[Tensor]=None,
                        index=_S, **kwargs) -> Tensor:
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
                        += func(self, idx, x[flag, ...], coef[flag, ...], **kwargs)
                    basis_cursor += NF
                return ret
            return wrapper
    else:
        def assemble_(func: Callable[..., Tensor]):
            def wrapper(self: "PoUSpace", p: Tensor, *, index=_S, **kwargs) -> Tensor:
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
                        += func(self, idx, x[flag, ...], **kwargs)
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

    def basis(self, p: Tensor, *, index=_S) -> Tensor:
        return self.space.basis(p, index=index) * self.pou_fn(p)

    def grad_basis(self, p: Tensor, *, index=_S, trans=None) -> Tensor:
        space = self.space
        ret = torch.einsum("nd, nf -> nfd", self.pou_fn.gradient(p), space.basis(p, index=index))
        ret += self.pou_fn(p)[..., None] * space.grad_basis(p, index=index, trans=trans)
        return ret

    def hessian_basis(self, p: Tensor, *, index=_S, trans=None) -> Tensor:
        space = self.space
        ret = torch.einsum("nxy, nf -> nfxy", self.pou_fn.hessian(p), space.basis(p, index=index))
        cross = torch.einsum("nx, nfy -> nfxy", self.pou_fn.gradient(p),
                             space.grad_basis(p, index=index, trans=trans))
        ret += cross + torch.transpose(cross, -1, -2)
        ret += self.pou_fn(p)[..., None, None] * space.hessian_basis(p, index=index, trans=trans)
        return ret

    def laplace_basis(self, p: Tensor, *, coef: Optional[Tensor]=None,
                      index=_S, trans=None) -> Tensor:
        space = self.space
        if coef is None:
            ret = torch.einsum("ndd, nf -> nf", self.pou_fn.hessian(p), space.basis(p, index=index))
            ret += 2 * torch.einsum("nd, nfd -> nf", self.pou_fn.gradient(p),
                                    space.grad_basis(p, index=index, trans=trans))
        else:
            if coef.ndim == 1:
                coef = coef[None, :].broadcast_to(p.shape[0], -1)
            assert coef.shape[0] == p.shape[0]
            ret = torch.einsum("ndd, nf, nd -> nf", self.pou_fn.hessian(p), space.basis(p, index=index), coef)
            ret += 2 * torch.einsum("nd, nfd, nd -> nf", self.pou_fn.gradient(p),
                                    space.grad_basis(p, index=index, trans=trans), coef)
        ret += self.pou_fn(p) * space.laplace_basis(p, coef=coef, index=index, trans=trans)
        return ret

    def derivative_basis(self, p: Tensor, *idx: int, index=_S, trans=None) -> Tensor:
        N = p.shape[0]
        nf = self.number_of_basis()
        order = len(idx)
        space = self.space
        ret = torch.zeros((N, nf), dtype=self.dtype, device=self.device)
        kwargs = {index: index, trans: trans}

        if order == 0:
            ret[:] = self.basis(p)
        elif order == 1:
            ret += self.pou_fn.derivative(p, idx[0]) * space.basis(p, index=index)
            ret += self.pou_fn(p) * space.derivative_basis(p, idx[0], **kwargs)
        elif order == 2:
            ret += self.pou_fn.derivative(p, idx[0], idx[1]) * space.basis(p, index=index)
            ret += self.pou_fn.derivative(p, idx[0]) * space.derivative_basis(p, idx[1], **kwargs)
            ret += self.pou_fn.derivative(p, idx[1]) * space.derivative_basis(p, idx[0], **kwargs)
            ret += self.pou_fn(p) * space.derivative_basis(p, idx[0], idx[1], **kwargs)
        elif order == 3:
            ret += self.pou_fn.derivative(p, idx[0], idx[1], idx[2]) * space.basis(p, index=index)
            ret += self.pou_fn.derivative(p, idx[0], idx[1]) * space.derivative_basis(p, idx[2], **kwargs)
            ret += self.pou_fn.derivative(p, idx[1], idx[2]) * space.derivative_basis(p, idx[0], **kwargs)
            ret += self.pou_fn.derivative(p, idx[2], idx[0]) * space.derivative_basis(p, idx[1], **kwargs)
            ret += self.pou_fn.derivative(p, idx[0]) * space.derivative_basis(p, idx[2], idx[1], **kwargs)
            ret += self.pou_fn.derivative(p, idx[1]) * space.derivative_basis(p, idx[0], idx[2], **kwargs)
            ret += self.pou_fn.derivative(p, idx[2]) * space.derivative_basis(p, idx[1], idx[0], **kwargs)
            ret += self.pou_fn(p) * space.derivative_basis(p, idx[0], idx[1], idx[2], **kwargs)

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

        if print_status:
            pass

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

    def basis(self, p: Tensor, *, index=_S) -> Tensor:
        N = p.shape[0]
        M = self.number_of_basis()
        ret = torch.zeros((N, M), dtype=self.dtype, device=self.device)
        lp = self.pou.global_to_local(p)
        basis_cursor = 0

        for idx, part in enumerate(self.partitions):
            NF = part.number_of_basis()
            x = lp[:, idx, :]
            flag = part.flag(x)
            ret[flag, basis_cursor:basis_cursor+NF, ...]\
                += part.basis(x[flag, ...], index=index)
            basis_cursor += NF
        return ret

    @assemble(0, use_coef=True)
    def convect_basis(self, idx: int, p: Tensor, *, coef: Tensor, index=_S, trans=None) -> Tensor:
        new_trans = self.pou.grad_global_to_local(index=idx)[0, :, :] # (lGD, gGD)
        if trans is None:
            trans = new_trans
        else:
            trans = new_trans @ trans
        return self.partitions[idx].convect_basis(p, coef=coef, index=index, trans=trans) # (samples, basis)

    @assemble(0, use_coef=True)
    def laplace_basis(self, idx: int, p: Tensor, *, coef: Tensor, index=_S, trans=None) -> Tensor:
        new_trans = self.pou.grad_global_to_local(index=idx)[0, :, :] # (lGD, gGD)
        if trans is None:
            trans = new_trans
        else:
            trans = new_trans @ trans
        return self.partitions[idx].laplace_basis(p, coef=coef, index=index, trans=trans) # (samples, basis)

    # NOTE: Is this unable to be implemented?
    def derivative_basis(self, p: Tensor, *dim: int, index=_S, trans=None) -> Tensor:
        """NOT supported!"""
        raise NotImplementedError(f"derivative basis is not supported by PoUSpace.")

    @assemble(1)
    def grad_basis(self, idx: int, p: Tensor, *, index=_S, trans=None) -> Tensor:
        gg2l = self.pou.grad_global_to_local(index=idx)[0, ...] # (lGD, gGD)
        return torch.einsum('nfl, lg -> nfg',
                            self.partitions[idx].grad_basis(p),
                            gg2l) # (samples, basis, lGD), (lGD, gGD) -> (samples, basis, gGD)
