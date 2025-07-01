from typing import Callable, Dict, Literal, Sequence
from dataclasses import dataclass

from ..backend import TensorLike, backend_manager as bm

BCType = Literal['dirichlet', 'neumann', 'robin']


@dataclass
class BoundaryCondition:
    mask_fn: Callable[[TensorLike], TensorLike]
    kind: BCType
    value_fn: Callable[[TensorLike], TensorLike]


def bc_mask(
        p: TensorLike,
        bcs: Sequence[BoundaryCondition], 
        kind: BCType) -> TensorLike:
    """
    """
    context = bm.context(p)
    context['dtype'] = bm.bool
    mask = bm.zeros_like(p[..., 0], **context)
    for entry in bcs:
        if entry.kind == kind:
            mask = bm.logical_or(mask, entry.mask_fn(p))
    return mask

def bc_value(
        p: TensorLike,
        bcs: Sequence[BoundaryCondition], 
        kind: BCType, 
        value_fn=None
        ) -> TensorLike:
    """
    """
    context = bm.context(p)
    val = bm.zeros_like(p[..., 0], **context)
    for entry in bcs:
        if entry.kind == kind:
            m = entry.mask_fn(p)
            if value_fn is not None:
                v = value_fn(p)
            else:
                v = entry.value_fn(p)
            val = bm.where(m, v, val)
    return val

