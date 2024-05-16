
import builtins
from typing import Optional, Union, Callable

from torch import Tensor

from .mesh import HomoMesh

Number = Union[builtins.int, builtins.float]
CoefLike = Union[Number, Tensor, Callable[..., Tensor]]
Index = Union[slice, Tensor, int]


def process_coef_func(coef: CoefLike, bcs: Optional[Tensor]=None, mesh: Optional[HomoMesh]=None,
                      index: Optional[Tensor]=None):
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


def is_scalar(input: Union[int, float, Tensor]):
    if isinstance(input, Tensor):
        return input.numel() == 1
    else:
        return isinstance(input, (int, float))
