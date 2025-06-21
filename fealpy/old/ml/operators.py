
from functools import reduce
from typing import (
    List, Union, Tuple, Sequence, Literal, Optional, Generator,
    Generic, TypeVar, overload
)

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .nntyping import TensorFunction, S
from .modules import FunctionSpace, Function, PoUSpace
from . import solvertools as ST


FuncOrTensor = Union[TensorFunction, Tensor, list, tuple]
FuncOrNumber = Union[TensorFunction, Tensor, float, int]
FuncOrTensorLike = Union[FuncOrNumber, FuncOrTensor]

def _to_tensor(sample: Tensor, func_or_tensor: Optional[FuncOrTensorLike]):
    if callable(func_or_tensor):
        func_or_tensor = func_or_tensor(sample)
    if func_or_tensor is None:
        ret = torch.tensor(0.0, dtype=sample.dtype, device=sample.device)
    elif isinstance(func_or_tensor, (float, int, list, tuple)):
        ret = torch.tensor(func_or_tensor, dtype=sample.dtype, device=sample.device)
    else:
        ret = func_or_tensor
    return ret

_FS = TypeVar('_FS', bound=FunctionSpace)


class Form(Generic[_FS]):
    """
    @brief Forms for collecting conditions.

    1. use the `add` method to collect samples, operators and sources.
    2. use the `assembly` method to generate A.T@A and A.T@b.
    """
    def __init__(self, space: _FS, gd: int=1) -> None:
        """
        @brief Build a form to collecting conditions.

        @param space: FunctionSpace.
        @param gd: int. Dimension of the source, used for initializing.
        """
        self.space = space
        self.gd = gd
        self.samples: List[Tensor] = []
        self.operators: List[Tuple[Operator, ...]] = []
        self.sources: List[Tensor] = []

    @overload
    def add(self, sample: Tensor, operator: "Operator",
            source: Optional[FuncOrNumber]=None): ...
    @overload
    def add(self, sample: Tensor, operator: Sequence["Operator"],
            source: Optional[FuncOrNumber]=None): ...
    def add(self, sample: Tensor, operator: Union[Sequence["Operator"], "Operator"],
            source: Optional[FuncOrNumber]=None):
        """
        @brief Add a condition to the form.

        @param sample: collocation points Tensor.
        @param operator: one or sequence of operator(s) applying to the space.
        @param source: function or Tensor of source, optional. Source defaults to\
               zero(s) if not provided.
        """
        self.samples.append(sample)
        # NOTE: samples are allow to have more than 2 dims.
        if isinstance(operator, Operator):
            self.operators.append((operator, ))
        else:
            self.operators.append(tuple(operator))
        b = _to_tensor(sample, source)
        # NOTE: Here we did not check the shape of sample and source, as the number
        # of conditions may not match the number of samples.
        self.sources.append(b)

    def apply_all(self, op_idx=S) -> Generator[Tuple[Tensor, Tensor], None, None]:
        space = self.space
        if isinstance(op_idx, int):
            sub_list = [op_idx, ]
        else:
            sub_list = range(len(self.samples))[op_idx]
        for i in sub_list:
            pts = self.samples[i]
            ops = self.operators[i]
            basis = reduce(torch.add, (op.apply(pts, space) for op in ops))
            src = self.sources[i].broadcast_to(basis.shape[0], self.gd)
            yield basis, src

    @overload
    def assembly(self, *, rescale: Optional[float]=1.0, allow_inplace=True) -> Tuple[csr_matrix, csr_matrix]: ...
    @overload
    def assembly(self, *, rescale: Optional[float]=1.0,
                 return_sparse: Literal[True]=True, allow_inplace=True) -> Tuple[csr_matrix, csr_matrix]: ...
    @overload
    def assembly(self, *, rescale: Optional[float]=1.0,
                 return_sparse: Literal[False]=False, allow_inplace=True) -> Tuple[Tensor, Tensor]: ...
    def assembly(self, *, rescale: Optional[float]=1.0, return_sparse=True, allow_inplace=True):
        """
        @brief Assemble least-square matrix for the linear equations.

        @param rescale: float.
        @param return_sparse: bool, optional. Return in csr_matrix type if `True`.\
               Defaults to `True`.
        @param allow_inplace: bool, optional. Set `False` to avoid in-place operation\
               in assembling. Defaults to `True`.

        @return: Tuple[Tensor, Tensor] or Tuple[csr_matrix, csr_matrix].\
        """
        space = self.space
        NF = space.number_of_basis()
        assert len(self.sources) >= 1
        bshape = (NF, self.gd)

        A = torch.zeros((NF, NF), dtype=space.dtype, device=space.device)
        b = torch.zeros(bshape, dtype=space.dtype, device=space.device)

        for phi, src in self.apply_all():
            if rescale is not None:
                # NOTE: `src` may be from broadcasting, so we clone the `src`
                # to avoid inplace operations. While `phi` is calculated by
                # basis in a space, so it can support inplace operation.
                phi, src = ST.rescale(phi, src.clone(), rescale, inplace=allow_inplace)
            A[:] += phi.T@phi
            b[:] += phi.T@src

        if return_sparse:
            return csr_matrix(A.detach().cpu()), csr_matrix(b.detach().cpu())
        return A, b

    def spsolve(self, *, rescale: Optional[float]=1.0, ridge: Optional[float]=None):
        A_, b_ = self.assembly(rescale=rescale, return_sparse=False)
        if ridge is not None:
            ST.ridge(A_, ridge)
        A_ = csr_matrix(A_)
        b_ = csr_matrix(b_)
        um = spsolve(A_, b_)
        return self.space.function(torch.from_numpy(um))

    def residual(self, um: Tensor, op_index=S):
        ress: List[Tensor] = []
        for basis, src in self.apply_all(op_idx=op_index):
            if um.ndim == 1:
                src.squeeze_(-1)
            ress.append(basis@um - src)
        return torch.cat(ress, dim=0)

    def sample(self, op_idx=S):
        in_feature = self.samples[0].shape[-1]
        data = [s.reshape(-1, in_feature) for s in self.samples[op_idx]]
        return torch.cat(data, dim=0)


### Operators

class Operator():
    """Abstract base class of Operator for functions in space."""
    def __hash__(self) -> int:
        return id(self)

    def __call__(self, func) -> TensorFunction:
        raise NotImplementedError

    def apply(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        """
        @brief Apply to a function space, and return with shape (N, nf, ...).

        @note: Return shape is (N, nf) for `ScalerOperator`, where 'nf' is the\
               number of features.
        """
        raise NotImplementedError

    def integrate(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        """
        @brief Assemble matrix for weak formulation, and return with shape (nf, nf).
        """
        raise NotImplementedError


class ScalerOperator(Operator):
    """Abstract class for scaler operators. Only for typing."""
    def __call__(self, func: Function) -> TensorFunction:
        space = func.space
        um = func.um
        keepdim = func.keepdim
        def new_func(p: Tensor):
            basis = self.apply(p, space)
            if um.ndim == 1:
                ret = torch.einsum("...f, f -> ...", basis, um)
                return ret.unsqueeze(-1) if keepdim else ret
            return torch.einsum("...f, fe -> ...e", basis, um)
        return new_func


class ScalerDiffusion(ScalerOperator):
    """
    @brief -c\\Delta \\phi
    """
    def __init__(self, coef: Optional[FuncOrNumber]=None) -> None:
        """
        @brief Initialize a scaler diffusion operator.

        @param coef: float. Diffusion coefficion, with shape () or (N, ).
        """
        super().__init__()
        self.coef = coef

    def apply(self, p: Tensor, space: FunctionSpace, *, index=S):
        if self.coef is None:
            return -space.laplace_basis(p, index=index)
        coef = _to_tensor(p, self.coef)
        return -space.laplace_basis(p, index=index) * coef

    def integrate(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        gphi = space.grad_basis(p, index=index)
        N = p.shape[0]
        coef = _to_tensor(p, self.coef).broadcast_to((N, ))
        return torch.einsum('ngd, nfd, n -> gf', gphi, gphi, coef)


class ScalerConvection(ScalerOperator):
    """
    @brief \\nabla \\phi \\cdot c
    """
    def __init__(self, coef: FuncOrTensor) -> None:
        """
        @brief Initialize a scaler convection operator.

        @param coef: 1-d or 2-d Tensor, or a function. Velosity of the fluent\
               (Convection coefficient), or the normal direction of boundaries,\
               with shape (GD, ) or (N, GD). See `space.convect_basis`.
        """
        super().__init__()
        self.coef = coef

    def apply(self, p: Tensor, space: FunctionSpace, *, index=S):
        coef = _to_tensor(p, self.coef)
        return space.convect_basis(p, coef=coef, index=index)

    def integrate(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        phi = space.basis(p, index=index)
        gphi = space.grad_basis(p, index=index)
        N, gd = p.shape
        coef = _to_tensor(p, self.coef).broadcast_to(N, gd)
        return torch.einsum('ngd, nf, nd -> gf', gphi, phi, coef)


class ScalerMass(ScalerOperator):
    """
    @brief c \\phi
    """
    def __init__(self, coef: Optional[FuncOrNumber]=None) -> None:
        """
        @brief Initialize a scaler mass operator.

        @param coef: 0-d or 1-d Tensor, or a function. With shape () or (N, ).
        """
        super().__init__()
        self.coef = coef

    def apply(self, p: Tensor, space: FunctionSpace, *, index=S):
        if self.coef is None:
            return space.basis(p, index=index)
        coef = _to_tensor(p, self.coef)
        return space.basis(p, index=index) * coef

    def integrate(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        phi = space.basis(p, index=index)
        N = p.shape[0]
        coef = _to_tensor(p, self.coef).broadcast_to((N, ))
        return torch.einsum('ng, nf, n -> gf', phi, phi, coef)


class Integrator(ScalerOperator):
    def __init__(self) -> None:
        """
        @brief Initialize a integrate operator.
        """
        super().__init__()

    def apply(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        basis = space.basis(p, index=index)
        return torch.mean(basis, dim=0, keepdim=True)

### Continuous Operators

class ContinuousOperator(ScalerOperator):
    """Abstract class for continuity conditions. Only for typing."""
    pass


class Continuous0(ContinuousOperator):
    """
    @brief 0-order continuity condition. Samples should be a Tensor with shape\
           (NVS, NS, GD), where NVS is the number of points in each sub-boundary\
           and NS is the number of sub-boundaries.
    """
    def __init__(self, sub_to_part: Tensor) -> None:
        """
        @brief Initialize a scaler operator to assemble continuous matrix. These\
               continuous conditions are designed for PoUSpace (or DGSpace).

        @param sub_to_part: Tensor. Topology relationship between sub-boundaries\
               and partitions(sub-domains). This tensor should be with shape\
               (#subs, 2) that each row represents a sub-boundary, containing\
               global indices of the two neighbor partitions.

        @example: For example, the partitions\n
        ```
            *----*----*
            |  1 |  3 |
            *----*----*
            |  0 |  2 |
            *----*----*
        ```
        may have a `sub_to_part` tensor like:
        ```
        tensor([[1, 0],
                [3, 2]
                [0, 2]
                [1, 3]], dtype=torch.int32)
        ```
        """
        super().__init__()
        self.sub2part = sub_to_part

    def apply(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        if not isinstance(space, PoUSpace):
            raise TypeError("Continuous0 is designed for PoUSpace, but applied "
                            f"to {space.__class__.__name__}.")
        assert p.ndim == 3 #(NVS, #Subs, #Dims)
        NVS = p.shape[0]
        NS = p.shape[1]
        N_basis = space.number_of_basis()
        sub2part = self.sub2part
        data = torch.zeros((NVS*NS, N_basis), dtype=p.dtype, device=p.device)
        for idx in range(NS):
            sp = p[:, idx, :] #(NVS, #Dims)

            left_idx = int(sub2part[idx, 0].item())
            left_part = space.partitions[left_idx]
            basis_slice_l = space.partition_basis_slice(left_idx)
            left_data = left_part.space.basis(left_part.global_to_local(sp))
            data[idx*NVS:(idx+1)*NVS, basis_slice_l] = left_data

            right_idx = int(sub2part[idx, 1].item())
            right_part = space.partitions[right_idx]
            basis_slice_r = space.partition_basis_slice(right_idx)
            right_data = right_part.space.basis(right_part.global_to_local(sp))
            data[idx*NVS:(idx+1)*NVS, basis_slice_r] = -right_data
        return data


class Continuous1(ContinuousOperator):
    """
    @brief 1-order continuity condition. Samples should be a Tensor with shape\
           (NVS, NS, GD), where NVS is the number of points in each sub-boundary\
           and NS is the number of sub-boundaries.
    """
    def __init__(self, sub_to_part: Tensor, sub_normal: Tensor) -> None:
        """
        @brief Initialize a scaler operator to assemble continuous matrix. These\
               continuous conditions are designed for PoUSpace (or DGSpace).

        @param sub_to_part: Tensor. Topology relationship between sub-boundaries\
               and partitions(sub-domains). This tensor should be with shape\
               (#subs, 2) that each row represents a sub-boundary, containing\
               global indices of the two neighbor partitions.
        @param sub_normal: Tensor. Unit normal direction of sub-boundaries, being\
               with shape (#subs, GD).

        @example: For example, the partitions\n
        ```
            *----*----*
            |  1 |  3 |
            *----*----*
            |  0 |  2 |
            *----*----*
        ```
        may have a `sub_to_part` tensor like:
        ```
        tensor([[1, 0],
                [3, 2]
                [0, 2]
                [1, 3]], dtype=torch.int32)
        ```
        and the `sub_normal` may be:
        ```
        tensor([[0., -1.],
                [0., -1.],
                [1., 0.],
                [1., 0.]], dtype=torch.float64)
        ```
        @note: The order of sub_boundaries in `sub_to_part`, `sub_normal` and\
               samples should match.
        """
        super().__init__()
        self.sub2part = sub_to_part
        self.sub2normal = sub_normal

    def apply(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        if not isinstance(space, PoUSpace):
            raise TypeError("Continuous1 is designed for PoUSpace, but applied "
                            f"to {space.__class__.__name__}.")
        assert p.ndim == 3 #(NVS, #Subs, #Dims)
        NVS = p.shape[0]
        NS = p.shape[1]
        N_basis = space.number_of_basis()
        sub2part = self.sub2part
        data = torch.zeros((NVS*NS, N_basis), dtype=p.dtype, device=p.device)
        for idx in range(NS):
            sp = p[:, idx, :] #(NVS, #Dims)
            n = self.sub2normal[idx, :] #(GD, )

            left_idx = int(sub2part[idx, 0].item())
            left_part = space.partitions[left_idx]
            basis_slice_l = space.partition_basis_slice(left_idx)
            x = left_part.global_to_local(sp)
            left_data = torch.einsum('...fd, d, d -> ...f', left_part.space.grad_basis(x),
                                     1/left_part.radius, n)
            data[idx*NVS:(idx+1)*NVS, basis_slice_l] = left_data

            right_idx = int(sub2part[idx, 1].item())
            right_part = space.partitions[right_idx]
            basis_slice_r = space.partition_basis_slice(right_idx)
            x = right_part.global_to_local(sp)
            right_data = torch.einsum('...fd, d, d -> ...f', right_part.space.grad_basis(x),
                                      1/right_part.radius, n)
            data[idx*NVS:(idx+1)*NVS, basis_slice_r] = -right_data
        return data


### Sources

class Source():
    """Abstract base class of Source."""
    def __init__(self, source: FuncOrNumber) -> None:
        super().__init__()
        self.source = source

    def __hash__(self):
        return id(self)

    def apply(self, p: Tensor) -> Tensor:
        """
        @brief Apply the source function to samples. The result will be broadcasted\
               to (N, #out), where '#out' is the output dimension of the source.\
               For 0-d and 1-d source Tensors, broadcasts to (N, 1).
        """
        ret = _to_tensor(p, self.source)
        if ret.ndim <= 1:
            N = p.shape[0]
            return ret.broadcast_to(N, 1)
        return ret

    __call__ = apply

    def integrate(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        """
        @brief Assemble matrix for weak form, and return with shape (nf, #out).\
               Where '#out' is the output dimension of the source.
        """
        raise NotImplementedError

# NOTE: 'scaler' here means the basis in space are scaler basis. The source
# function may not be scaler.
class ScalerSource(Source):
    def integrate(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        phi = space.basis(p, index=index)
        src = self.apply(p)
        return torch.einsum('ng, n... -> g...', phi, src)
