
import builtins
from typing import Optional, Union, Callable

from torch import Tensor

from .mesh import HomoMesh

Number = Union[builtins.int, builtins.float]
CoefLike = Union[Number, Tensor, Callable[..., Tensor]]
Index = Union[slice, Tensor, int]


def process_coef_func(
    coef: Optional[CoefLike],
    bcs: Optional[Tensor]=None,
    mesh: Optional[HomoMesh]=None,
    index: Optional[Tensor]=None
):
    r"""Fetch the result Tensor if `coef` is a function."""
    if callable(coef):
        if index is None:
            raise RuntimeError('The index should be provided for coef functions.')
        if getattr(coef, 'coordtype', 'cartesian') == 'cartesian':
            if (mesh is None) or (not isinstance(mesh, HomoMesh)):
                raise RuntimeError('The mesh should be provided for cartesian coef functions.'
                                   'Note that only homogeneous meshes are supported here.')

            ps = mesh.bc_to_point(bcs, index=index)
            coef_val = coef(ps)
        else:
            coef_val = coef(bcs, index=index)
    else:
        coef_val = coef
    return coef_val


def is_scalar(input: Union[int, float, Tensor]) -> bool:
    if isinstance(input, Tensor):
        return input.numel() == 1
    else:
        return isinstance(input, (int, float))


def is_tensor(input: Union[int, float, Tensor]) -> bool:
    if isinstance(input, Tensor):
        return input.numel() >= 2
    return False


def get_coef_subscripts(shape: Tensor, nq: int, nc: int, batched: bool):
    if batched:
        coef_shape = shape[:-1]
        if coef_shape == (nq, nc):
            subs = "qcb"
        elif coef_shape == (nq, ):
            subs = "qb"
        elif coef_shape == (nc, ):
            subs = "cb"
        else:
            raise RuntimeError(f"The shape of the coef should be ({nq}, {nc}, Batch), "
                               f"({nq}, Batch) or ({nc}, Batch), but got {tuple(shape)}.")

    else:
        coef_shape = shape
        if coef_shape == (nq, nc):
            subs = "qc"
        elif coef_shape == (nq, ):
            subs = "q"
        elif coef_shape == (nc, ):
            subs = "c"
        else:
            raise RuntimeError(f"The shape of the coef should be ({nq}, {nc}), "
                               f"({nq}) or ({nc}), but got {tuple(shape)}.")

    return subs
