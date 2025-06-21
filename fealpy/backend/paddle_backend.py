from typing import Optional, Union, Tuple, Any
from functools import reduce, partial
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
    BackendProxy, ATTRIBUTE_MAPPING, FUNCTION_MAPPING
)


Tensor = paddle.Tensor
_device = paddle.device

class PaddleBackend(BackendProxy, backend_name='paddle'):
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

    @staticmethod
    def tensorprod(*tensors: Tensor) -> Tensor:
        num = len(tensors)
        NVC = reduce(lambda x, y: x * y.shape[-1], tensors, 1)
        desp1 = 'mnopq'
        desp2 = 'abcde'
        string = ", ".join([desp1[i] + desp2[i] for i in range(num)])
        string += " -> " + desp1[:num] + desp2[:num]
        return paddle.einsum(string, *tensors).reshape(-1, NVC)

    @classmethod
    def bc_to_points(cls, bcs: Union[Tensor, Tuple[Tensor, ...]], node: Tensor, entity: Tensor) -> Tensor:
        points = node[entity, :]

        if not isinstance(bcs, Tensor):
            bcs = cls.tensorprod(*bcs)
        return paddle.einsum('ijk, ...j -> i...k', points, bcs)

    @staticmethod
    def barycenter(entity: Tensor, node: Tensor, loc: Optional[Tensor] = None) -> Tensor:
        return paddle.mean(node[entity, :], axis=1)  # TODO: polygon mesh case

    @staticmethod
    def simplex_measure(entity: Tensor, node: Tensor) -> Tensor:
        points = node[entity, :]
        TD = points.shape[-2] - 1
        if TD != points.shape[-1]:
            raise RuntimeError("The geometric dimension of points must be NVC-1"
                               "to form a simplex.")
        edges = points[..., 1:, :] - points[..., :-1, :]
        return paddle.linalg.det(edges)/factorial(TD)

    @classmethod
    def _simplex_shape_function_kernel(cls, bc: Tensor, p: int, mi: Optional[Tensor] = None) -> Tensor:
        TD = bc.shape[-1] - 1
        itype = bc.dtype
        device = bc.name
        shape = (1, TD + 1)

        if mi is None:
            mi = cls.multi_index_matrix(p, TD)

        c = paddle.arange(1, p + 1, dtype=itype, name=device)
        P = 1.0 / paddle.cumprod(c, dim=0)
        t = paddle.arange(0, p, dtype=itype, name=device)
        Ap = p * bc.unsqueeze(-2) - t.reshape((-1, 1))
        Ap = paddle.cumprod(Ap, dim=-2).clone()
        Ap = Ap*P.reshape((-1, 1))
        A = paddle.concat([paddle.ones(shape, dtype=bc.dtype, name=device), Ap], axis=-2)
        idx = paddle.arange(TD + 1)
        phi = paddle.prod(A[mi, idx], axis=-1)
        return phi

    @classmethod
    def simplex_shape_function(cls, bcs: Tensor, p: int, mi=None) -> Tensor:
        results = [cls._simplex_shape_function_kernel(bc_slice, p=p, mi=mi) for bc_slice in bcs]

        return paddle.stack(results, axis=0)

    # TODO: paddle 中没有 jacfwd 和 jacrev 方法，使用旧的计算方式
    @classmethod
    def simplex_grad_shape_function(cls, bcs: paddle.Tensor, p: int, mi=None) -> paddle.Tensor:
        TD = bcs.shape[-1] - 1
        if mi is None:
            mi = cls.multi_index_matrix(p, TD)

        ldof = mi.shape[0]  # p 次 Lagrange 形函数的个数

        c = paddle.arange(1, p + 1)
        P = 1.0 / paddle.cumprod(c, dim=0)

        t = paddle.arange(0, p, dtype=bcs.dtype)
        shape = tuple(bcs.shape[:-1]) + (p + 1, TD + 1)
        A = paddle.ones(shape, dtype=bcs.dtype)
        A[..., 1:, :] = p * bcs[..., None, :] - t.reshape((-1, 1))

        FF = paddle.einsum('...jk, m->...kjm', A[..., 1:, :], paddle.ones(p, dtype=bcs.dtype))
        FF[..., range(p), range(p)] = p
        FF = paddle.cumprod(FF, dim=-2)
        F = paddle.zeros(shape, dtype=bcs.dtype)
        temp = paddle.sum(paddle.tril(FF), axis=-1)
        perm = paddle.arange(len(temp.shape))
        perm[-1] = len(perm)-2
        perm[-2] = len(perm)-1
        F[..., 1:, :] = paddle.transpose(temp, perm=perm.tolist())
        F[..., 1:, :] *= P.reshape((-1, 1))

        A = paddle.cumprod(A, dim=-2)
        A[..., 1:, :] *= P.reshape((-1, 1))

        Q = A[..., mi, range(TD + 1)]
        M = F[..., mi, range(TD + 1)]

        shape = tuple(bcs.shape[:-1]) + (ldof, TD + 1)
        R = paddle.zeros(shape, dtype=bcs.dtype)
        for i in range(TD + 1):
            idx = list(range(TD + 1))
            idx.remove(i)
            R[..., i] = M[..., i] * paddle.prod(Q[..., idx], axis=-1)
        return R  # (..., ldof, bc)

    @staticmethod
    def simplex_hess_shape_function(bc: Tensor, p: int, mi=None) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def tensor_measure(entity: Tensor, node: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError

    @staticmethod
    def interval_grad_lambda(line: Tensor, node: Tensor) -> Tensor:
        points = node[line, :]
        v = points[..., 1, :] - points[..., 0, :] # (NC, GD)
        h2 = paddle.sum(v**2, axis=-1, keepdim=True)
        v /= h2
        return paddle.stack([-v, v], axis=-2)

    @staticmethod
    def triangle_area_3d(tri: Tensor, node: Tensor) -> Tensor:
        points = node[tri, :]
        edge1 = points[..., 1, :] - points[..., 0, :]
        edge2 = points[..., 2, :] - points[..., 0, :]
        points = node[tri, :]
        cross_product = paddle.cross(edge1, edge2, axis=-1)
        area = 0.5 * paddle.linalg.norm(cross_product, axis=-1)
        return area

    @staticmethod
    def triangle_grad_lambda_2d(cell: Tensor, node: Tensor) -> Tensor:
        """grad_lambda function for the triangle mesh in 2D.

        Parameters:
            cell (Tensor[NC, 3]): Indices of vertices of triangles.\n
            node (Tensor[NN, 2]): Node coordinates.

        Returns:
            Tensor[..., 3, 2]:
        """
        e0 = node[cell[:, 2]] - node[cell[:, 1]]
        e1 = node[cell[:, 0]] - node[cell[:, 2]]
        e2 = node[cell[:, 1]] - node[cell[:, 0]]
        nv = paddle.linalg.det(paddle.stack([e0, e1], axis=-2)) # (...)
        e0 = paddle.flip(e0, axis=-1)
        e1 = paddle.flip(e1, axis=-1)
        e2 = paddle.flip(e2, axis=-1)
        result = paddle.stack([e0, e1, e2], axis=-2)
        result[..., 0] *= -1
        return result / nv[..., None, None]

    @staticmethod
    def triangle_grad_lambda_3d(cell: Tensor, node: Tensor) -> Tensor:
        """
        Parameters:
            cell (Tensor[NC, 3]): Indices of vertices of triangles.\n
            node (Tensor[NN, 3]): Node coordinates.

        Returns:
            Tensor[..., 3, 3]:
        """
        e0 = node[cell[:, 2]] - node[cell[:, 1]]
        e1 = node[cell[:, 0]] - node[cell[:, 2]]
        e2 = node[cell[:, 1]] - node[cell[:, 0]]
        nv = paddle.cross(e0, e1, axis=-1)  # (..., 3)
        length = paddle.linalg.norm(nv, axis=-1, keepdim=True)  # (..., 1)
        n = nv / length
        return paddle.stack([
            paddle.cross(n, e0, axis=-1),
            paddle.cross(n, e1, axis=-1),
            paddle.cross(n, e2, axis=-1)
        ], axis=-2) / length[..., None]  # (..., 3, 3)

    @staticmethod
    def quadrangle_grad_lambda_2d(quad: Tensor, node: Tensor) -> Tensor:
        pass

    @classmethod
    def tetrahedron_grad_lambda_3d(cls, tet: Tensor, node: Tensor, localFace: Tensor) -> Tensor:
        NC = tet.shape[0]
        Dlambda = paddle.zeros((NC, 4, 3), dtype=node.dtype)
        volume = cls.simplex_measure(tet, node)
        for i in range(4):
            j, k, m = localFace[i]
            vjk = node[tet[:, k],:] - node[tet[:, j],:]
            vjm = node[tet[:, m],:] - node[tet[:, j],:]
            Dlambda[:, i, :] = paddle.cross(vjm, vjk) / (6*volume.reshape((-1, 1)))
        return Dlambda


PaddleBackend.attach_attributes(ATTRIBUTE_MAPPING, paddle)
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(tensor='array')
PaddleBackend.attach_methods(function_mapping, paddle)
