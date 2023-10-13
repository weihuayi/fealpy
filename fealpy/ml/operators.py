
from functools import reduce
from typing import Union, Tuple, Sequence, List, overload, Literal, Optional
import torch
from torch import Tensor
from scipy.sparse import lil_matrix, csr_matrix
from fealpy.ml.modules import Function, FunctionSpace
from .nntyping import TensorFunction, S
from .modules import FunctionSpace, Function

FuncOrTensor = Union[TensorFunction, Tensor]

def _to_tensor(sample: Tensor, func_or_tensor: Optional[FuncOrTensor], gd=1):
    N = sample.shape[0]
    if func_or_tensor is None:
        ret = torch.tensor(0.0, dtype=sample.dtype, device=sample.device)
    elif callable(func_or_tensor):
        ret = func_or_tensor(sample)
    else:
        ret = func_or_tensor
    if ret.ndim in {0, 1}:
        return ret.broadcast_to(N, gd)
    else:
        return ret


class Form():
    def __init__(self, space: FunctionSpace) -> None:
        self.space = space
        self.samples: List[Tensor] = []
        self.operators: List[Tuple[Operator, ...]] = []
        self.sources: List[Tensor] = []

    @overload
    def add(self, sample: Tensor, operator: "Operator",
            source: Optional[FuncOrTensor]=None): ...
    @overload
    def add(self, sample: Tensor, operator: Sequence["Operator"],
            source: Optional[FuncOrTensor]=None): ...
    def add(self, sample: Tensor, operator: Union[Sequence["Operator"], "Operator"],
            source: Optional[FuncOrTensor]=None):
        """
        @brief Add a condition.

        @param sample: collocation points Tensor.
        @param operator: one or sequence of operator applying to the space.
        @param source: function or Tensor of source.
        """
        assert sample.ndim == 2
        self.samples.append(sample)

        if isinstance(operator, Operator):
            self.operators.append((operator, ))
        else:
            self.operators.append(tuple(operator))

        b = _to_tensor(sample, source)
        assert sample.shape[0] == b.shape[0]
        self.sources.append(b)

    @overload
    def assembly(self) -> Tuple[csr_matrix, csr_matrix]: ...
    @overload
    def assembly(self, return_sparse: Literal[True]) -> Tuple[csr_matrix, csr_matrix]: ...
    @overload
    def assembly(self, return_sparse: Literal[False]) -> Tuple[Tensor, Tensor]: ...
    def assembly(self, return_sparse=True):
        """
        @brief Assemble linear equations for the least-square problem.

        @param return_sparse: bool. Return in csr_matrix type if `True`.

        @return: Tuple[Tensor, Tensor] or Tuple[csr_matrix, csr_matrix].\
        """
        space = self.space
        NF = space.number_of_basis()
        assert len(self.sources) >= 1
        src0 = self.sources[0]

        if src0.ndim == 1:
            bshape: Tuple[int, ...] = (NF, )
        else:
            bshape = (NF, src0.shape[-1])

        if return_sparse:
            dtype = src0.cpu().detach().numpy().dtype
            A = lil_matrix((NF, NF), dtype=dtype)
            b = lil_matrix(bshape, dtype=dtype)
        else:
            A = torch.zeros((NF, NF), dtype=space.dtype, device=space.device)
            b = torch.zeros(bshape, dtype=space.dtype, device=space.device)

        for i in range(len(self.samples)):
            pts = self.samples[i]
            ops = self.operators[i]
            basis = reduce(torch.add, (op.assembly(pts, space) for op in ops))
            src = self.sources[i]
            if return_sparse:
                A[:] += (basis.T@basis).detach().cpu().numpy()
                b[:] += (basis.T@src).detach().cpu().numpy()
            else:
                A[:] += basis.T@basis
                b[:] += basis.T@src

        if return_sparse:
            return A.tocsr(), b.tocsr()
        return A, b


class Operator():
    """Operator for functions in space."""
    def __hash__(self) -> int:
        return id(self)

    def __call__(self, func) -> TensorFunction:
        raise NotImplementedError

    def assembly(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        """
        @brief Assemble matrix for a function space, with shape (N, nf, ...).

        @note: Return shape is (N, nf) for `ScalerOperator`.
        """
        raise NotImplementedError


class ScalerOperator(Operator):
    def __call__(self, func: Function) -> TensorFunction:
        space = func.space
        um = func.um
        keepdim = func.keepdim
        def new_func(p: Tensor):
            basis = self.assembly(p, space)
            if um.ndim == 1:
                ret = torch.einsum("...f, f -> ...", basis, um)
                return ret.unsqueeze(-1) if keepdim else ret
            return torch.einsum("...f, fe -> ...e", basis, um)
        return new_func


class ScalerDiffusion(ScalerOperator):
    """
    @brief -c\\Delta \\phi
    """
    def __init__(self, coef: Optional[FuncOrTensor]=None) -> None:
        """
        @brief Initialize a scaler diffusion operator.

        @param coef: float. Diffusion coefficion, with shape () or (N, ).
        """
        super().__init__()
        self.coef = coef

    def assembly(self, p: Tensor, space: FunctionSpace, *, index=S):
        if self.coef is None:
            return -space.laplace_basis(p, index=index)
        coef = _to_tensor(p, self.coef)
        return -space.laplace_basis(p, index=index) * coef


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

    def assembly(self, p: Tensor, space: FunctionSpace, *, index=S):
        coef = _to_tensor(p, self.coef, gd=p.shape[-1])
        return space.convect_basis(p, coef=coef, index=index)


class ScalerMass(ScalerOperator):
    """
    @brief c \\phi
    """
    def __init__(self, coef: Optional[FuncOrTensor]=None) -> None:
        """
        @brief Initialize a scaler mass operator.

        @param coef: 0-d or 1-d Tensor, or a function. With shape () or (N, ).
        """
        super().__init__()
        self.coef = coef

    def assembly(self, p: Tensor, space: FunctionSpace, *, index=S):
        if self.coef is None:
            return space.basis(p, index=index)
        coef = _to_tensor(p, self.coef)
        return space.basis(p, index=index) * coef


class Integrator(ScalerOperator):
    def __init__(self) -> None:
        """
        @brief Initialize a integrate operator.
        """
        super().__init__()

    def assembly(self, p: Tensor, space: FunctionSpace, *, index=S) -> Tensor:
        basis = space.basis(p, index=index)
        return torch.mean(basis, dim=0, keepdim=True)


class Continuous0(Operator):
    pass


class Continuous1(Operator):
    pass
