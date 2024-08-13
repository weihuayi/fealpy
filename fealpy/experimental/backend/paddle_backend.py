from typing import Optional, Union, Tuple, Any
from functools import reduce
from math import factorial
from itertools import combinations_with_replacement


try:
    import paddle
except ImportError:
    raise ImportError("Name 'paddle' cannot be imported. "
                      'Make sure paddle is installed before using '
                      'the paddle backend in fealpy. '
                      'See https://www.paddlepaddle.org.cn/ for installation.')

from .base import (
    Backend, ATTRIBUTE_MAPPING, FUNCTION_MAPPING
)


Tensor = paddle.Tensor
_device = paddle.device

class PaddleBackend(Backend[Tensor], backend_name='paddle'):
    DATA_CLASS = paddle.Tensor

    linalg = paddle.linalg
    random = paddle.tensor.random

    @staticmethod
    def context(x):
        return {"dtype": x.dtype, "device": x.device}

    @staticmethod
    def set_default_device(device) -> None:
        paddle.device.set_device(device)

    @staticmethod
    def get_device(x, /):
        return x.device 

    @staticmethod
    def to_numpy(tensor_like: Tensor, /) -> Any:
        return tensor_like.numpy()

    @staticmethod
    def from_numpy(x: Any, /) -> Tensor:
        return paddle.to_tensor(x)

    @staticmethod
    def reshape(x, shape):
        if not isinstance(shape, (list, tuple)):
            shape = [shape]
        return x.reshape(shape)
    @staticmethod
    def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=0, **kwargs):
        """
        unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> Tuple[Tensor, Tensor, Tensor]
        """
        b, index, inverse, counts = paddle.unique(a, return_index=True,
                return_inverse=True,
                return_counts=True,
                axis=axis, **kwargs)
        any_return = return_index or return_inverse or return_counts
        if any_return:
            result = (b, )
        else:
            retult = b

        if return_index:
            result += (index, )

        if return_inverse:
            # TODO: 处理 paddle.Tensor.reshape 形状参数不能为整数问题
            inverse = inverse.reshape((-1,))
            result += (inverse, )

        if return_counts:
            result += (counts, )

        return result

    @staticmethod
    def unique_all(x, axis=None, **kwargs):
        """
        unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> Tuple[Tensor, Tensor, Tensor]
        """
        return paddle.unique(x, return_index=True,
                return_inverse=True,
                return_counts=True,
                axis=axis, **kwargs)

    @staticmethod
    def unique_all_(x, axis=None, **kwargs):
        """
        unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> Tuple[Tensor, Tensor, Tensor]
        """
        b, indices0, inverse, counts = paddle.unique(x, return_index=True,
                return_inverse=True,
                return_counts=True,
                axis=axis, **kwargs)
        indices1 = paddle.zeros_like(indices0)
        idx = paddle.arange(inverse.shape[0]-1, -1, -1, dtype=indices0.dtype)
        indices1 = indices1.at[inverse[-1::-1]].set(idx)
        return b, indices0, indices1, inverse, counts

    @staticmethod
    def multi_index_matrix(p: int, dim: int, *, dtype=None) -> Tensor:
        r"""Create a multi-index matrix."""
        dtype = dtype or paddle.int64
        sep = paddle.flip(paddle.to_tensor(
            tuple(combinations_with_replacement(range(p+1), dim)),
            dtype=dtype
        ), axis=0)
        raw = paddle.zeros((sep.shape[0], dim+2), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @staticmethod
    def edge_length(edge: Tensor, node: Tensor, *, out=None) -> Tensor:
        return paddle.linalg.norm(node[edge[:, 0]] - node[edge[:, 1]], axis=-1)

    @staticmethod
    def edge_normal(edge: Tensor, node: Tensor, unit=False, *, out=None) -> Tensor:
        points = node[edge, :]
        if points.shape[-1] != 2:
            raise ValueError("Only 2D meshes are supported.")
        edges = points[..., 1, :] - points[..., 0, :]
        if unit:
            edges /= paddle.linalg.norm(edges, axis=-1, keepdims=True)
        return paddle.stack([edges[..., 1], -edges[..., 0]], axis=-1, name=out)

    @staticmethod
    def edge_tangent(edge: Tensor, node: Tensor, unit=False, *, out=None) -> Tensor:
        edges = node[edge[:, 1], :] - node[edge[:, 0], :]
        if unit:
            l = paddle.linalg.norm(edges, axis=-1, keepdims=True)
            edges /= l
        return paddle.stack([edges[..., 0], edges[..., 1]], axis=-1, name=out)