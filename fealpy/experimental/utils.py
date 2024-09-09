
from typing import Optional, Union, Callable


from .backend import backend_manager as bm
from .typing import TensorLike, Index, Number, CoefLike, _S

from .mesh import HomogeneousMesh, Mesh



def process_coef_func(
    coef: Optional[CoefLike],
    bcs: Optional[TensorLike]=None,
    mesh: Optional[HomogeneousMesh]=None,
    etype: Optional[Union[int, str]]=None,
    index: Optional[TensorLike]=None
):
    r"""Fetch the result Tensor if `coef` is a function."""
    if callable(coef):
        if index is None:
            raise RuntimeError('The index should be provided for coef functions.')
        if bcs is None:
            raise RuntimeError('The bcs should be provided for coef functions.')
        if etype is None:
            raise RuntimeError('The etype should be provided for coef functions.')
        if getattr(coef, 'coordtype', 'barycentric') == 'barycentric':
            if (mesh is None) or (not isinstance(mesh, HomogeneousMesh)):
                raise RuntimeError('The mesh should be provided for cartesian coef functions.'
                                   'Note that only homogeneous meshes are supported here.')

            coef_val = coef(bcs, index=index)
        else:
            ps = mesh.bc_to_point(bcs, index=index)
            coef_val = coef(ps)

    else:
        coef_val = coef
    return coef_val


def is_scalar(input: Union[int, float, TensorLike]) -> bool:
    if isinstance(input, TensorLike):
        return bm.size(input) == 1
    else:
        return isinstance(input, (int, float))


def is_tensor(input: Union[int, float, TensorLike]) -> bool:
    if isinstance(input, TensorLike):
        return bm.size(input) >= 2
    return False


def fill_axis(input: TensorLike, ndim: int):
    diff = ndim - input.ndim

    if diff > 0:
        return bm.reshape(input, input.shape + (1, ) * diff)
    elif diff == 0:
        return input
    else:
        raise RuntimeError(f"The dimension of the input should be smaller than {ndim}, "
                           f"but got shape {tuple(input.shape)}.")


def get_coef_subscripts(shape: TensorLike, nq: int, nc: int, batched: bool):
    if batched:
        coef_shape = shape[1:]
        if coef_shape == (nc, nq):
            subs = "bcq"
        elif coef_shape == (nq, ):
            subs = "bq"
        elif coef_shape == (nc, ):
            subs = "bc"
        else:
            raise RuntimeError(f"The shape of the coef should be (Batch, {nc}, {nq}), "
                               f"(Batch, {nq}) or (Batch, {nc}), but got {tuple(shape)}.")
    else:
        coef_shape = shape
        if coef_shape == (nc, nq):
            subs = "cq"
        elif coef_shape == (nq, ):
            subs = "q"
        elif coef_shape == (nc, ):
            subs = "c"
        else:
            raise RuntimeError(f"The shape of the coef should be ({nc}, {nq}), "
                               f"({nq}) or ({nc}), but got {tuple(shape)}.")

    # else:
    #     coef_shape = shape
    #     dim = len(coef_shape)
    #     if coef_shape == (nc, nq):
    #         subs = "cq"
    #     elif coef_shape == (nq, ):
    #         subs = "q"
    #     elif coef_shape == (nc, ):
    #         subs = "c"
    #     elif dim == 3 and coef_shape[:2] == (nc, nq):
    #         subs = "cqd"
    #     elif dim == 3 and coef_shape[0] == nq:
    #         subs = "qd"
    #     elif dim == 3 and coef_shape[0] == nc:
    #         subs = "cd"
    #     else:
    #         raise RuntimeError(f"The shape of the coef should be ({nc}, {nq}), "
    #                            f"({nq}) or ({nc}), but got {tuple(shape)}.")
        
    return subs


def process_threshold(threshold: Callable[[TensorLike], TensorLike],
                      index: TensorLike, mesh: Mesh, etype: str):
    if callable(threshold):
        bc = mesh.entity_barycenter(etype, index=index)
        return index[threshold(bc)]
    else:
        raise ValueError("threshold must be callable, "
                         f"but got {type(threshold)}")
