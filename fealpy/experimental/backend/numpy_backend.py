
from typing import Optional, Union, Tuple
from functools import reduce
from math import factorial
from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import det

from .base import (
    Backend, ATTRIBUTE_MAPPING, FUNCTION_MAPPING
)


class NumPyBackend(Backend[NDArray], backend_name='numpy'):
    DATA_CLASS = np.ndarray

    @staticmethod
    def set_default_device(device) -> None:
        raise NotImplementedError("`set_default_device` is not supported by NumPyBackend")

    @staticmethod
    def get_device(tensor_like, /):
        return 'cpu'

    @staticmethod
    def to_numpy(tensor_like: NDArray, /) -> NDArray:
        return tensor_like

    @staticmethod
    def from_numpy(ndarray: NDArray, /) -> NDArray:
        return ndarray

    ### Tensor creation methods ###
    # NOTE: all copied

    ### Reduction methods ###
    # NOTE: all copied

    ### Unary methods ###
    # NOTE: all copied

    ### Binary methods ###

    add_at = staticmethod(np.add.at)

    @staticmethod
    def index_add_(a: NDArray, /, dim, index, src, *, alpha=1):
        assert index.ndim == 1
        indexing = [slice(None)] * a.ndim
        indexing[dim] = index
        np.add.at(a, indexing, alpha*src)
        return a

    ### Other methods ###

    @classmethod
    def nonzero(cls, a, /, as_tuple=True):
        cls.show_unsupported(not as_tuple, 'nonzero', 'as_tuple')
        return np.nonzero(a)

    @staticmethod
    def cat(iterable, axis=0, out=None) -> NDArray:
        return np.concatenate(iterable, axis=axis, out=out)

    ### FEALPy methods ###

    @staticmethod
    def multi_index_matrix(p: int, dim: int, *, dtype=None) -> NDArray:
        sep = np.flip(np.array(
            tuple(combinations_with_replacement(range(p+1), dim)),
            dtype=np.int_
        ), axis=0)
        raw = np.zeros((sep.shape[0], dim+2), dtype=np.int_)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @staticmethod
    def edge_length(edge: NDArray, node: NDArray, *, out=None) -> NDArray:
        assert out == None, "`out` is not supported by edge_length in NumPyBackend"
        points = node[edge, :]
        return np.linalg.norm(points[..., 0, :] - points[..., 1, :], axis=-1)

    @staticmethod
    def edge_normal(edge: NDArray, node: NDArray, normalize=False, *, out=None) -> NDArray:
        points = node[edge, :]
        if points.shape[-1] != 2:
            raise ValueError("Only 2D meshes are supported.")
        edges = points[..., 1, :] - points[..., 0, :]
        if normalize:
            edges /= np.linalg.norm(edges, axis=-1, keepdims=True)
        return np.stack([edges[..., 1], -edges[..., 0]], axis=-1, out=out)

    @staticmethod
    def edge_tangent(edge: NDArray, node: NDArray, normalize=False, *, out=None) -> NDArray:
        v = np.subtract(node[edge[:, 1], :], node[edge[:, 0], :], out=out)
        if normalize:
            l = np.linalg.norm(v, axis=-1, keepdims=True)
            v /= l
        return v

    @staticmethod
    def tensorprod(*tensors: NDArray) -> NDArray:
        num = len(tensors)
        NVC = reduce(lambda x, y: x * y.shape[-1], tensors, 1)
        desp1 = 'mnopq'
        desp2 = 'abcde'
        string = ", ".join([desp1[i]+desp2[i] for i in range(num)])
        string += " -> " + desp1[:num] + desp2[:num]
        return np.einsum(string, *tensors).reshape(-1, NVC)

    @classmethod
    def bc_to_points(cls, bcs: Union[NDArray, Tuple[NDArray, ...]], node: NDArray, entity: NDArray) -> NDArray:
        points = node[entity, :]

        if not isinstance(bcs, np.ndarray):
            bcs = cls.tensorprod(bcs)
        return np.einsum('ijk, ...j -> i...k', points, bcs)

    @staticmethod
    def barycenter(entity: NDArray, node: NDArray, loc: Optional[NDArray]=None) -> NDArray:
        return np.mean(node[entity, :], axis=1)

    @staticmethod
    def simplex_measure(entity: NDArray, node: NDArray) -> NDArray:
        points = node[entity, :]
        TD = points.shape[-2] - 1
        if TD != points.shape[-1]:
            raise RuntimeError("The geometric dimension of points must be NVC-1"
                            "to form a simplex.")
        edges = points[..., 1:, :] - points[..., :-1, :]
        return det(edges)/(factorial(TD))

    @classmethod
    def simplex_shape_function(cls, bc: NDArray, p: int, mi=None) -> NDArray:
        if p == 1:
            return bc
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = cls.multi_index_matrix(p, TD)
        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., mi, idx], axis=-1)
        return phi

    @classmethod
    def simplex_grad_shape_function(cls, bc: NDArray, p: int, mi=None) -> NDArray:
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = cls.multi_index_matrix(p, TD)

        ldof = mi.shape[0] # p 次 Lagrange 形函数的个数

        c = np.arange(1, p+1)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=bc.dtype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., mi, range(TD+1)]
        M = F[..., mi, range(TD+1)]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=bc.dtype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
        return R # (..., ldof, bc)

    @staticmethod
    def simplex_hess_shape_function(bc: NDArray, p: int, mi=None) -> NDArray:
        raise NotImplementedError

    @staticmethod
    def tensor_measure(entity: NDArray, node: NDArray) -> NDArray:
        # TODO
        raise NotImplementedError

    @staticmethod
    def interval_grad_lambda(line: NDArray, node: NDArray) -> NDArray:
        points = node[line, :]
        v = points[..., 1, :] - points[..., 0, :] # (NC, GD)
        h2 = np.sum(v**2, axis=-1, keepdims=True)
        v /= h2
        return np.stack([-v, v], axis=-2)

    @staticmethod
    def triangle_area_3d(tri: NDArray, node: NDArray, out=None) -> NDArray:
        points = node[tri, :]
        edge1 = points[..., 1, :] - points[..., 0, :]
        edge2 = points[..., 2, :] - points[..., 0, :]
        cross_product = np.cross(edge1, edge2, axis=-1)
        area = 0.5 * np.linalg.norm(cross_product, axis=-1)
        if out is not None:
            out[:] = area
            return out
        return area

    @staticmethod
    def triangle_grad_lambda_2d(tri: NDArray, node: NDArray) -> NDArray:
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :]
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = det(np.stack([e0, e1], axis=-2)) # Determinant for 2D case, equivalent to np.linalg.det for 2x2 matrix
        e0 = np.flip(e0, axis=-1)
        e1 = np.flip(e1, axis=-1)
        e2 = np.flip(e2, axis=-1)
        result = np.stack([e0, e1, e2], axis=-2)
        result[..., 0] *= -1
        return result / np.expand_dims(nv, axis=(-1, -2))

    @staticmethod
    def triangle_grad_lambda_3d(tri: NDArray, node: NDArray) -> NDArray:
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :]  # (..., 3)
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = np.cross(e0, e1, axis=-1)  # Normal vector, (..., 3)
        length = np.linalg.norm(nv, axis=-1, keepdims=True)  # Length of normal vector, (..., 1)
        n = nv / length  # Unit normal vector
        # Compute the gradient by crossing the unit normal with each edge
        return np.stack([
            np.cross(n, e0, axis=-1),
            np.cross(n, e1, axis=-1),
            np.cross(n, e2, axis=-1)
        ], axis=-2) / length[..., np.newaxis, np.newaxis]  # Scale by inverse length to normalize

    @staticmethod
    def quadrangle_grad_lambda_2d(quad: NDArray, node: NDArray) -> NDArray:
        pass

    @staticmethod
    def tetrahedron_grad_lambda_3d(tet: NDArray, node: NDArray, local_face: NDArray) -> NDArray:
        pass


NumPyBackend.attach_attributes(ATTRIBUTE_MAPPING, np)
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(tensor='array')
NumPyBackend.attach_methods(function_mapping, np)
