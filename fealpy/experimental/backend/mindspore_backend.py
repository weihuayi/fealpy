from typing import Union, Optional, Tuple, Any
from itertools import combinations_with_replacement
from functools import reduce, partial
from math import factorial
import numpy

try:
    # import torch
    # from torch import vmap, norm, det, cross
    # from torch.func import jacfwd, jacrev

    import mindspore as ms
    from mindspore import jit
    import mindspore.context as context
    import mindspore.ops as ops
    import mindspore.numpy as mnp

except ImportError:
    raise ImportError("Name 'mindspore' cannot be imported. "
                      'Make sure MindSpore is installed before using '
                      'the MindSpore backend in fealpy. '
                      'See https://www.mindspore.cn/ for installation.')

from .base import Backend, ATTRIBUTE_MAPPING, FUNCTION_MAPPING

Tensor = ms.Tensor
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
_device = context.get_context("device_target")


class MindSporeBackend(Backend[Tensor], backend_name='mindspore'):
    DATA_CLASS = ms.Tensor

    @staticmethod
    def set_default_device(device: str) -> None:
        context.set_context(device_target=device)

    @staticmethod
    def get_device() -> str:
        return _device

    @staticmethod
    def to_numpy(tensor_like: Tensor) -> Any:
        return tensor_like.asnumpy()
    
    ### Tensor creation methods ###

    @staticmethod
    def linspace(start, stop, num, *, endpoint=True, retstep=False, dtype=None, **kwargs):
        return mnp.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype, **kwargs)

    @staticmethod
    def eye(n: int, m: Optional[int]=None, /, k: int=0, dtype=None, **kwargs) -> Tensor:
        assert k == 0, "Only k=0 is supported by `eye` in MindSporeBackend."
        return mnp.eye(n, m, k=k, dtype=dtype, **kwargs)
    
    ### Reduction methods ###

    @staticmethod
    def all(a, axis=None, keepdims=False):
        return ops.all(a, axis=axis, keep_dims=keepdims)
    
    @staticmethod
    def any(a, axis=None, keepdims=False):
        return ops.any(a, axis=axis, keep_dims=keepdims)
    
    @staticmethod
    def sum(a, axis=None, dtype=None, keepdims=False, initial=None):
        result = mnp.sum(a, axis=axis, dtype=dtype, keepdim=keepdims, initial=initial)
        return result if (initial is None) else result + initial
    
    @staticmethod
    def prod(a, axis=None, keepdims=False, initial=None):
        result = ops.prod(a, axis=axis, keep_dims=keepdims)
        return result if (initial is None) else result * initial
    
    @staticmethod
    def mean(a, axis=None, dtype=None, keepdims=False):
        return mnp.mean(a, axis=axis, keepdims=keepdims, dtype=dtype)
    
    @staticmethod
    def max(a, axis=None, keepdims=False):
        return mnp.max(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def min(a, axis=None, keepdims=False):
        return mnp.min(a, axis=axis, keepdims=keepdims)
    
    ### Unary methods ###
    # NOTE: all copied

    ### Binary methods ###

    @staticmethod
    def cross(a, b, axis=-1, **kwargs):
        return mnp.cross(a, b, axisa=axis, **kwargs)

    @staticmethod
    def tensordot(a, b, axes):
        return mnp.tensordot(a, b, axes=axes)
    
    ### Other methods ###
    # TODO: unique
    @staticmethod
    def sort(a, axis=0, **kwargs):
        return ops.sort(a, axis=axis, **kwargs)

    @staticmethod
    def nonzero(a):
        return ops.nonzero(a)
    
    @staticmethod
    def cumsum(a, axis=None, dtype=None):
        result = mnp.cumsum(a, axis=axis, dtype=dtype)
        return result

    @staticmethod
    def cumprod(a, axis=None, dtype=None):
        result = mnp.cumprod(a, axis=axis, dtype=dtype)
        return result

    @staticmethod
    def concatenate(arrays, /, axis=0, *, dtype=None):
        if dtype is not None:
            arrays = [a.astype(dtype) for a in arrays]
        result = mnp.concatenate(arrays, axis=axis)
        return result

    @staticmethod
    def stack(arrays, axis=0, *, dtype=None):
        if dtype is not None:
            arrays = [a.astype(dtype=dtype) for a in arrays]
        result = mnp.stack(arrays, axis=axis)
        return result
    
    ### FEALPy functionals ###

    @staticmethod
    def multi_index_matrix(p: int, dim: int, *, dtype=None) -> Tensor:
        dtype = dtype or ms.int_
        rand_op = ops.RandomChoiceWithMask(p + 1)
        indices, _ = rand_op(dim)
        sep = mnp.array(indices).astype(dtype)
        sep = ops.sort(sep, axis=0)
        raw = mnp.zeros((sep.shape[0], dim + 2), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])
    
    @staticmethod
    def edge_length(edge: Tensor, node: Tensor) -> Tensor:
        points = node[edge, :]
        return mnp.norm(points[..., 0, :] - points[..., 1, :], axis=-1)

    staticmethod
    def edge_normal(edge: Tensor, node: Tensor, unit=False, *, out=None) -> Tensor:
        points = node[edge, :]
        if points.shape[-1] != 2:
            raise ValueError("Only 2D meshes are supported.")
        edges = points[..., 1, :] - points[..., 0, :]
        if unit:
            square_sum = ops.ReduceSum(keep_dims=True)(edges ** 2, axis=-1)
            norm_edges = ops.Sqrt()(square_sum)
            edges = edges / norm_edges
        normals = mnp.stack([edges[..., 1], -edges[..., 0]], axis=-1)
        if out is not None:
            out[:] = normals
            return out
        else:
            return normals


attribute_mapping = ATTRIBUTE_MAPPING.copy()
attribute_mapping.update({
    'complex_': 'complex128'
})
MindSporeBackend.attach_attributes(attribute_mapping, ms)
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(array='tensor')
MindSporeBackend.attach_methods(function_mapping, ms)
