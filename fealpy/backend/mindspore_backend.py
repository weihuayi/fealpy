from typing import Union, Optional, Tuple, Any
from itertools import combinations_with_replacement
from functools import reduce, partial
from math import factorial
import numpy as np

try:

    import mindspore as ms
    import mindspore.context as context
    import mindspore.ops as ops
    import mindspore.numpy as mnp
    from mindspore.ops import functional as F

except ImportError:
    raise ImportError("Name 'mindspore' cannot be imported. "
                      'Make sure MindSpore is installed before using '
                      'the MindSpore backend in fealpy. '
                      'See https://www.mindspore.cn/ for installation.')

from .base import BackendProxy, ATTRIBUTE_MAPPING, FUNCTION_MAPPING

Tensor = ms.Tensor
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
_device = context.get_context("device_target")


class MindSporeBackend(BackendProxy, backend_name='mindspore'):
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

    @staticmethod
    def from_numpy(tensor_like: Tensor) -> Any:
        return ms.Tensor(tensor_like)

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
        result = mnp.sum(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial)
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
        return mnp.cross(a, b, axis=axis, **kwargs)

    @staticmethod
    def tensordot(a, b, axes):
        return mnp.tensordot(a, b, axes=axes)

    ### Other methods ###
    @staticmethod
    def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=0, **kwargs):

        a = a.asnumpy()
        b, index, inverse, counts = np.unique(a, return_index=True, return_inverse=True,
                return_counts=True,
                axis=axis, **kwargs)
        any_return = return_index or return_inverse or return_counts

        if any_return:
            result = (ms.Tensor(b), )
        else:
            result = ms.Tensor(b)

        if return_index:
            result += (ms.Tensor(index), )

        if return_inverse:
            result += (ms.Tensor(inverse), )

        if return_counts:
            result += (ms.Tensor(counts), )

        return result

    @staticmethod
    def sort(a, axis=0, **kwargs):
        sorted_values, _ = ops.sort(a, axis=axis, **kwargs)
        return sorted_values

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
        sep = mnp.flip(ms.tensor(
            tuple(combinations_with_replacement(range(p+1), dim)),
            dtype=dtype
        ), axis=0)
        raw = mnp.zeros((sep.shape[0], dim+2), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @staticmethod
    def edge_length(edge: Tensor, node: Tensor) -> Tensor:
        points = node[edge, :]
        return mnp.norm(points[..., 0, :] - points[..., 1, :], axis=-1)

    @staticmethod
    def edge_normal(edge: Tensor, node: Tensor, unit=False) -> Tensor:
        points = node[edge, :]
        if points.shape[-1] != 2:
            raise ValueError("Only 2D meshes are supported.")
        edges = points[..., 1, :] - points[..., 0, :]
        if unit:
            square_sum = mnp.sum(mnp.square(edges), axis=-1, keepdims=True)
            norm_edges = mnp.sqrt(square_sum)
            edges = edges / norm_edges
        normals = mnp.stack([edges[..., 1], -edges[..., 0]], axis=-1)
        return normals

    @staticmethod
    def edge_tangent(edge: Tensor, node: Tensor, normalize=False) -> Tensor:
        edge = ms.Tensor(edge.asnumpy()) if isinstance(edge, Tensor) else edge
        node = ms.Tensor(node.asnumpy()) if isinstance(node, Tensor) else node
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        if normalize:
            l = mnp.sqrt(mnp.sum(mnp.square(v), axis=-1, keepdims=True))
            v = v / l
        return v

    @staticmethod
    def tensorprod(*tensors: Tensor) -> Tensor:
        tensors = [ms.Tensor(t.asnumpy()) if isinstance(t, Tensor) else t for t in tensors]
        num = len(tensors)
        NVC = reduce(lambda x, y: x * y.shape[-1], tensors, 1)
        desp1 = 'mnopq'
        desp2 = 'abcde'
        string = ", ".join([f"{desp1[i]}{desp2[i]}" for i in range(num)])
        string += f" -> {desp1[:num]}{desp2[:num]}"

        result = ops.einsum(string, *tensors)
        return result.reshape(-1, int(NVC))

    @classmethod
    def bc_to_points(cls, bcs: Union[Tensor, Tuple[Tensor, ...]], node: Tensor, entity: Tensor) -> Tensor:
        points = node[entity, :]

        if not isinstance(bcs, Tensor):
            bcs = cls.tensorprod(*bcs)
        return ops.einsum('ijk, ...j -> i...k', points, bcs)

    @staticmethod
    def barycenter(entity: Tensor, node: Tensor, loc: Optional[Tensor]=None) -> Tensor:
        return mnp.mean(node[entity, :], dim=1) # TODO: polygon mesh case

    @staticmethod
    def simplex_measure(entity: Tensor, node: Tensor) -> Tensor:
        points = node[entity, :]
        TD = points.shape[-2] - 1
        if TD != points.shape[-1]:
            raise RuntimeError("The geometric dimension of points must be NVC-1"
                            "to form a simplex.")
        edges = points[..., 1:, :] - points[..., :-1, :]
        return ops.det(edges)/(factorial(TD))

    @classmethod
    def _simplex_shape_function_kernel(cls, bc: Tensor, p: int, mi: Optional[Tensor]=None) -> Tensor:
        TD = bc.shape[-1] - 1
        itype = ms.int_
        shape = (1, TD+1)

        if mi is None:
            mi = cls.multi_index_matrix(p, TD, dtype=ms.int_)

        c = mnp.arange(1, p+1, dtype=itype)
        P = 1.0 / mnp.cumprod(c, axis=0)
        t = mnp.arange(0, p, dtype=itype)
        Ap = p * bc.unsqueeze(-2) - t.reshape(-1, 1)
        Ap = mnp.cumprod(Ap, axis=-2)
        Ap = Ap * (P.reshape(-1, 1))
        A = mnp.concatenate([mnp.ones(shape, dtype=bc.dtype), Ap], axis=-2)
        idx = mnp.arange(TD + 1, dtype=itype)
        phi = ops.prod(A[mi, idx], axis=-1)
        return phi

    @classmethod
    def simplex_shape_function(cls, bcs: Tensor, p: int, mi=None) -> Tensor:
        fn = F.vmap(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        )
        return fn(bcs)

    @classmethod
    def simplex_grad_shape_function(cls, bcs: Tensor, p: int, mi=None) -> Tensor:
        fn = F.vmap(F.jacfwd(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        ))
        return fn(bcs)

    @staticmethod
    def tensor_measure(entity: Tensor, node: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError

    @staticmethod
    def interval_grad_lambda(line: Tensor, node: Tensor) -> Tensor:
        points = node[line, :]
        v = points[..., 1, :] - points[..., 0, :] # (NC, GD)
        h2 = mnp.sum(v**2, axis=-1, keepdims=True)
        v = v/(h2)
        return mnp.stack([-v, v], axis=-2)

    @staticmethod
    def triangle_area_3d(tri: Tensor, node: Tensor) -> Tensor:
        points = node[tri, :]
        return mnp.cross(points[..., 1, :] - points[..., 0, :],
                    points[..., 2, :] - points[..., 0, :], axis=-1) / 2.0

    @staticmethod
    def triangle_grad_lambda_2d(tri: Tensor, node: Tensor) -> Tensor:
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :]
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = ops.det(mnp.stack([e0, e1], axis=-2)) # (...)
        e0 = e0.flip([-1])
        e1 = e1.flip([-1])
        e2 = e2.flip([-1])
        result = mnp.stack([e0, e1, e2], axis=-2)
        result[..., 0] * (-1)
        return result / (nv[..., None, None])

    @staticmethod
    def triangle_grad_lambda_3d(tri: Tensor, node: Tensor) -> Tensor:
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :] # (..., 3)
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = mnp.cross(e0, e1, axis=-1) # (..., 3)
        length = mnp.norm(nv, axis=-1, keepdims=True) # (..., 1)
        n = nv / (length)
        return mnp.stack([
            mnp.cross(n, e0, axis=-1),
            mnp.cross(n, e1, axis=-1),
            mnp.cross(n, e2, axis=-1)
        ], axis=-2) / (length.unsqueeze(-2)) # (..., 3, 3)

    @staticmethod
    def quadrangle_grad_lambda_2d(quad: Tensor, node: Tensor) -> Tensor:
        pass

    @classmethod
    def tetrahedron_grad_lambda_3d(cls, tet: Tensor, node: Tensor, localFace: Tensor) -> Tensor:
        NC = tet.shape[0]
        Dlambda = mnp.zeros((NC, 4, 3), dtype=node.dtype)
        volume = cls.simplex_measure(tet, node)
        for i in range(4):
            j, k, m = localFace[i]
            vjk = node[tet[:, k],:] - node[tet[:, j],:]
            vjm = node[tet[:, m],:] - node[tet[:, j],:]
            Dlambda[:, i, :] = mnp.cross(vjm, vjk) / (6*volume.reshape(-1, 1))
        return Dlambda

attribute_mapping = ATTRIBUTE_MAPPING.copy()
attribute_mapping.update({
    'complex_': 'complex128'
})
MindSporeBackend.attach_attributes(attribute_mapping, ms)
MindSporeBackend.attach_attributes(attribute_mapping, mnp)
MindSporeBackend.attach_attributes(attribute_mapping, ops)
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(unique='unique')
MindSporeBackend.attach_methods(function_mapping, ms)
MindSporeBackend.attach_methods(function_mapping, mnp)
MindSporeBackend.attach_methods(function_mapping, ops)
