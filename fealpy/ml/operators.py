
from typing import Any, Union, Tuple, Sequence, List
import torch
from torch import Tensor

from fealpy.ml.modules import FunctionSpace
from .nntyping import TensorFunction, S
from .modules import FunctionSpace, Function

FuncOrTensor = Union[TensorFunction, Tensor]

def _to_tensor(sample: Tensor, func_or_tensor: Union[FuncOrTensor, None]):
    N = sample.shape[0]
    if func_or_tensor is None:
        return torch.tensor(0.0, dtype=sample.dtype, device=sample.device).broadcast_to(N, 1)
    elif callable(func_or_tensor):
        return func_or_tensor(sample).broadcast_to(N, 1)
    else:
        return func_or_tensor.broadcast_to(N, 1)


class Form():
    def __init__(self, space: FunctionSpace) -> None:
        self.space = space
        self.A_list: List[Tensor] = []
        self.b_list: List[Tensor] = []

    def add(self, sample: Tensor, operators: Sequence["Operator"],
            source: Union[FuncOrTensor, None]=None):
        """
        @brief
        """
        current_idx = len(self.A_list)
        b = _to_tensor(sample, source)
        self.b_list.append(b)
        op = operators[0]
        A = op.assembly(sample, self.space)
        assert b.shape[0] == A.shape[0]
        self.A_list.append(A)
        del A, b
        for op in operators[1:]:
            self.A_list[current_idx] += op.assembly(sample, self.space)

    def assembly(self) -> Tuple[Tensor, Tensor]:
        """
        @brief
        """
        A = torch.cat(self.A_list, dim=0)
        b = torch.cat(self.b_list, dim=0)
        return A, b


class Operator():
    """Operator for functions in space."""
    def __hash__(self) -> int:
        return id(self)

    def __call__(self, func: Function) -> TensorFunction:
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
    @brief -d\\Delta \\phi
    """
    def __init__(self, coef: FuncOrTensor) -> None:
        """
        @brief Initialize a scaler diffusion operator.

        @param coef: float. Diffusion coefficion, with shape () or (N, ).
        """
        super().__init__()
        self.coef = coef

    def assembly(self, p: Tensor, space: FunctionSpace, *, index=S):
        coef = _to_tensor(p, self.coef)
        if coef.ndim == 0:
            return -space.laplace_basis(p, index=index) * coef
        elif coef.ndim == 1:
            return -space.laplace_basis(p, index=index) * coef[:, None]
        else:
            raise ValueError("coef for diffusion should be 0 or 1-dimensional, "
                             f"but got Tensor with shape {coef.shape}.")


class ScalerConvection(ScalerOperator):
    """
    @brief \\nabla \\phi \\cdot n
    """
    def __init__(self, coef: FuncOrTensor) -> None:
        """
        @brief Initialize a scaler convection operator.

        @param coef: 1-d or 2-d Tensor, or a function. Velosity of the fluent\
               (Convection coefficient), or direction of boundaries,\
               with shape (GD, ) or (N, GD). See `space.convect_basis`.
        """
        super().__init__()
        self.coef = coef

    def assembly(self, p: Tensor, space: FunctionSpace, *, index=S):
        coef = _to_tensor(p, self.coef)
        if coef.ndim in {1, 2}:
            return space.convect_basis(p, coef=coef, index=index)
        else:
            raise ValueError("coef for convection should be 1 or 2-dimensional, "
                             f"but got Tensor with shape {coef.shape}.")


class ScalerMass(ScalerOperator):
    def __init__(self, coef: FuncOrTensor) -> None:
        """
        @brief Initialize a scaler mass operator.

        @param coef: 0-d or 1-d Tensor, or a function. With shape () or (N, ).
        """
        super().__init__()
        self.coef = coef

    def assembly(self, p: Tensor, space: FunctionSpace, *, index=S):
        coef = _to_tensor(p, self.coef)
        if coef.ndim == 0:
            return space.basis(p, index=index) * coef
        elif coef.ndim == 1:
            return space.basis(p, index=index) * coef[:, None]
        else:
            raise ValueError("coef for mass should be 0 or 1-dimensional, "
                             f"but got Tensor with shape {coef.shape}.")


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
