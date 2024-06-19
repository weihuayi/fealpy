
import builtins
from typing import Optional, Union, Callable

import numpy as np
from numpy.typing import NDArray

from .mesh import HomogeneousMesh, Mesh

Number = Union[builtins.int, builtins.float]
CoefLike = Union[Number, NDArray, Callable[..., NDArray]]
Index = Union[slice, NDArray, int]


def process_coef_func(
    coef: Optional[CoefLike],
    bcs: Optional[NDArray]=None,
    mesh: Optional[HomogeneousMesh]=None,
    etype: Optional[Union[int, str]]=None,
    index: Optional[NDArray]=None
):
    r"""Fetch the result Tensor if `coef` is a function."""
    if callable(coef):
        if index is None:
            raise RuntimeError('The index should be provided for coef functions.')
        if bcs is None:
            raise RuntimeError('The bcs should be provided for coef functions.')
        if etype is None:
            raise RuntimeError('The etype should be provided for coef functions.')
        if getattr(coef, 'coordtype', 'cartesian') == 'cartesian':
            if (mesh is None) or (not isinstance(mesh, HomogeneousMesh)):
                raise RuntimeError('The mesh should be provided for cartesian coef functions.'
                                   'Note that only homogeneous meshes are supported here.')

            ps = mesh.bc_to_point(bcs, etype=etype, index=index)
            coef_val = coef(ps)
        else:
            coef_val = coef(bcs, index=index)
    else:
        coef_val = coef
    return coef_val


def is_scalar(input: Union[int, float, NDArray]) -> bool:
    if isinstance(input, NDArray):
        return input.numel() == 1
    else:
        return isinstance(input, (int, float))


def is_tensor(input: Union[int, float, NDArray]) -> bool:
    if isinstance(input, NDArray):
        return input.numel() >= 2
    return False


def get_coef_subscripts(shape: NDArray, nq: int, nc: int, batched: bool):
    if batched:
        coef_shape = shape[1:]
        if coef_shape == (nq, nc):
            subs = "bqc"
        elif coef_shape == (nq, ):
            subs = "bq"
        elif coef_shape == (nc, ):
            subs = "bc"
        else:
            raise RuntimeError(f"The shape of the coef should be (Batch, {nq}, {nc}), "
                               f"(Batch, {nq}) or (Batch, {nc}), but got {tuple(shape)}.")

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


def process_threshold(threshold: Callable[[NDArray], NDArray],
                      index: NDArray, mesh: Mesh, etype: str):
    if callable(threshold):
        bc = mesh.entity_barycenter(etype, index=index)
        return index[threshold(bc)]
    else:
        raise ValueError("threshold must be callable, "
                         f"but got {type(threshold)}")
