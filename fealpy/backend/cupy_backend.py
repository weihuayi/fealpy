from typing import Optional, Union, Tuple
from functools import reduce
from math import factorial
from itertools import combinations_with_replacement

import numpy as np
import cupy as cp 
import cupyx as cpx
from cupy.typing import NDArray
from cupy.linalg import det

from .base import (
    BackendProxy, ATTRIBUTE_MAPPING, FUNCTION_MAPPING
)


class CuPyBackend(BackendProxy, backend_name='cupy'):
    DATA_CLASS = cp.ndarray

    linalg = cp.linalg
    random = cp.random

    @staticmethod
    def context(x):
        return {"dtype": x.dtype, "device": x.device}

    @staticmethod
    def set_default_device(device) -> None:
        raise NotImplementedError("`set_default_device` is not supported by NumPyBackend")

    @staticmethod
    def get_device(x, /):
        return x.device 

    @staticmethod
    def to_numpy(x: NDArray, /) -> np.ndarray:
        return cp.ndarray.get(x) 

    @staticmethod
    def from_numpy(x: np.ndarray, /) -> NDArray:
        return cp.array(x) 

    @staticmethod
    def scatter(x, indices, val):
        """
        """
        x[indices] = val
        return x

    @staticmethod
    def scatter_add(x, indices, val):
        """
        Apply in place operation on the operand ``x`` for elements specified by
        ``indices``
        """
        cp.add.at(x, indices, val)
        return x

    @staticmethod
    def unique_all_(a, axis=None, **kwargs):
        """
        """
        b, indices0, inverse, counts = cp.unique(a, 
                                                 return_index=True,
                                                 return_inverse=True,
                                                 return_counts=True,
                                                 axis=axis, **kwargs)
        indices1 = cp.zeros_like(indices0)
        indices1[inverse] = range(inverse.shape[0]);
        return b, indices0, indices1, inverse, counts

    ### FEALPy methods ###

    @staticmethod
    def multi_index_matrix(p: int, dim: int, *, dtype=cp.int32) -> NDArray:
        sep = cp.flip(cp.array(
            tuple(combinations_with_replacement(range(p+1), dim)),
            dtype=dtype
        ), axis=0)
        raw = cp.zeros((sep.shape[0], dim+2), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @staticmethod
    def edge_length(edge: NDArray, node: NDArray) -> NDArray:
        """
        """
        points = node[edge, :]
        return cp.linalg.norm(node[edge[:, 1]] - node[edge[:, 0]], axis=-1)

    @staticmethod
    def edge_normal(edge: NDArray, node: NDArray, unit=False, *, out=None) -> NDArray:
        """
        """
        if node.shape[-1] != 2:
            raise ValueError("Only 2D meshes are supported.")
        v = node[edge[:, 1]] - node[edge[:, 0]] 
        if unit:
            v /= cp.linalg.norm(v, axis=-1, keepdims=True)
        return cp.stack([v[..., 1], -v[..., 0]], axis=-1, out=out)

    @staticmethod
    def edge_tangent(edge: NDArray, node: NDArray, unit=False, *, out=None) -> NDArray:
        """
        """
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        if unit:
            v /= cp.linalg.norm(v, axis=-1, keepdims=True)
        return v

    @staticmethod
    def tensorprod(*tensors: NDArray) -> NDArray:
        num = len(tensors)
        NVC = reduce(lambda x, y: x * y.shape[-1], tensors, 1)
        desp1 = 'mnopq'
        desp2 = 'abcde'
        string = ", ".join([desp1[i]+desp2[i] for i in range(num)])
        string += " -> " + desp1[:num] + desp2[:num]
        return cp.einsum(string, *tensors).reshape(-1, NVC)

    @classmethod
    def bc_to_points(cls, 
                     bcs: Union[NDArray, Tuple[NDArray, ...]], 
                     node: NDArray, entity: NDArray) -> NDArray:
        points = node[entity, :]

        if not isinstance(bcs, cp.ndarray):
            bcs = cls.tensorprod(bcs)
        return cp.einsum('ijk, ...j -> i...k', points, bcs)

    @staticmethod
    def barycenter(entity: NDArray, 
                   node: NDArray, loc: Optional[NDArray]=None) -> NDArray:
        return cp.mean(node[entity, :], axis=1)

    @staticmethod
    def simplex_measure(entity: NDArray, node: NDArray) -> NDArray:
        """
        """
        TD = entity.shape[-1] - 1
        points = node[entity, :]
        if TD != node.shape[-1]:
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
        c = cp.arange(1, p+1, dtype=cp.int32)
        P = 1.0/cp.multiply.accumulate(c)
        t = cp.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = cp.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)
        cp.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = cp.arange(TD+1)
        phi = cp.prod(A[..., mi, idx], axis=-1)
        return phi

    @classmethod
    def simplex_grad_shape_function(cls, bc: NDArray, p: int, mi=None) -> NDArray:
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = cls.multi_index_matrix(p, TD)

        ldof = mi.shape[0] # p 次 Lagrange 形函数的个数

        c = cp.arange(1, p+1)
        P = 1.0/cp.multiply.accumulate(c)

        t = cp.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = cp.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)

        FF = cp.einsum('...jk, m->...kjm', A[..., 1:, :], cp.ones(p))
        FF[..., range(p), range(p)] = p
        cp.cumprod(FF, axis=-2, out=FF)
        F = cp.zeros(shape, dtype=bc.dtype)
        F[..., 1:, :] = cp.sum(cp.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        cp.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., mi, range(TD+1)]
        M = F[..., mi, range(TD+1)]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = cp.zeros(shape, dtype=bc.dtype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*cp.prod(Q[..., idx], axis=-1)
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
        """
        """
        points = node[line, :]
        v = points[..., 1, :] - points[..., 0, :]
        v /= cp.sum(v**2, axis=-1, keepdims=True)
        return cp.stack([-v, v], axis=-2)

    @staticmethod
    def triangle_area_3d(tri: NDArray, node: NDArray, out=None) -> NDArray:
        """
        """
        v1 = node[tri[:, 1]] - node[tri[:, 0]]
        v2 = node[tri[:, 2]] - node[tri[:, 0]]
        v = cp.cross(v1, v2, axis=-1)
        area = 0.5 * cp.linalg.norm(v, axis=-1)
        if out is not None:
            out[:] = area
            return out
        return area

    @staticmethod
    def triangle_grad_lambda_2d(tri: NDArray, node: NDArray) -> NDArray:
        """
        """
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :]
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = det(cp.stack([e0, e1], axis=-2)) # Determinant for 2D case, equivalent to cp.linalg.det for 2x2 matrix
        e0 = cp.flip(e0, axis=-1)
        e1 = cp.flip(e1, axis=-1)
        e2 = cp.flip(e2, axis=-1)
        result = cp.stack([e0, e1, e2], axis=-2)
        result[..., 0] *= -1
        return result / cp.expand_dims(nv, axis=(-1, -2))

    @staticmethod
    def triangle_grad_lambda_3d(tri: NDArray, node: NDArray) -> NDArray:
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :]  # (..., 3)
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = cp.cross(e0, e1, axis=-1)  # Normal vector, (..., 3)
        length = cp.linalg.norm(nv, axis=-1, keepdims=True)  # Length of normal vector, (..., 1)
        n = nv / length  # Unit normal vector
        return cp.stack([
            cp.cross(n, e0, axis=-1),
            cp.cross(n, e1, axis=-1),
            cp.cross(n, e2, axis=-1)
        ], axis=-2) / length[..., cp.newaxis]  # Scale by inverse length to normalize

    @staticmethod
    def quadrangle_grad_lambda_2d(quad: NDArray, node: NDArray) -> NDArray:
        pass

    @classmethod
    def tetrahedron_grad_lambda_3d(cls, tet: NDArray, node: NDArray, localFace: NDArray) -> NDArray:
        NC = tet.shape[0]
        Dlambda = cp.zeros((NC, 4, 3), dtype=node.dtype)
        volume = cls.simplex_measure(tet, node)
        for i in range(4):
            j, k, m = localFace[i]
            vjk = node[tet[:, k],:] - node[tet[:, j],:]
            vjm = node[tet[:, m],:] - node[tet[:, j],:]
            Dlambda[:, i, :] = cp.cross(vjm, vjk) / (6*volume.reshape(-1, 1))
        return Dlambda

CuPyBackend.attach_attributes(ATTRIBUTE_MAPPING, cp)
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(tensor='array')
CuPyBackend.attach_methods(function_mapping, cp)
