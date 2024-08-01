
from typing import Union, Optional, Tuple, Any
from itertools import combinations_with_replacement
from functools import reduce, partial
from math import factorial

try:
    import torch
    from torch import vmap, norm, det, cross
    from torch.func import jacfwd, jacrev

except ImportError:
    raise ImportError("Name 'torch' cannot be imported. "
                      'Make sure PyTorch is installed before using '
                      'the PyTorch backend in fealpy. '
                      'See https://pytorch.org/ for installation.')

from .base import Backend, ATTRIBUTE_MAPPING, FUNCTION_MAPPING

Tensor = torch.Tensor
_device = torch.device


class PyTorchBackend(Backend[Tensor], backend_name='pytorch'):
    DATA_CLASS = torch.Tensor
    linalg = torch.linalg
    random = torch.random

    @staticmethod
    def set_default_device(device: Union[str, _device]) -> None:
        torch.set_default_device(device)

    @staticmethod
    def get_device(tensor_like: Tensor, /):
        return tensor_like.device

    @staticmethod
    def to_numpy(tensor_like: Tensor, /) -> Any:
        return tensor_like.detach().cpu().numpy()

    from_numpy = torch.from_numpy

    ### Tensor creation methods ###

    @staticmethod
    def linspace(start, stop, num, *, endpoint=True, retstep=False, dtype=None, **kwargs):
        """
        """
        assert endpoint == True
        vmap_fun = partial(torch.linspace, dtype=dtype, **kwargs)
        for _ in range(start.ndim):
            vmap_fun = vmap(vmap_fun, in_dims=(0, 0, None), out_dims=0)
        return vmap_fun(start, stop, num)

    @staticmethod
    def eye(n: int, m: Optional[int]=None, /, k: int=0, dtype=None, **kwargs) -> Tensor:
        assert k == 0, "Only k=0 is supported by `eye` in PyTorchBackend."
        return torch.eye(n, m, dtype=dtype, **kwargs)

    @staticmethod
    def meshgrid(*xi, copy=None, sparse=None, indexing='xy', **kwargs) -> Tuple[Tensor, ...]:
        return torch.meshgrid(*xi, indexing=indexing)

    ### Reduction methods ###

    @staticmethod
    def all(a, axis=None, keepdims=False):
        return torch.all(a, dim=axis, keepdim=keepdims)

    @staticmethod
    def any(a, axis=None, keepdims=False):
        return torch.any(a, dim=axis, keepdim=keepdims)

    @staticmethod
    def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None):
        result = torch.sum(a, dim=axis, keepdim=keepdims, dtype=dtype, out=out)
        return result if (initial is None) else result + initial

    @staticmethod
    def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None):
        result = torch.prod(a, dim=axis, keepdim=keepdims, dtype=dtype, out=out)
        return result if (initial is None) else result * initial

    @staticmethod
    def mean(a, axis=None, dtype=None, out=None, keepdims=False):
        return torch.mean(a, dim=axis, keepdim=keepdims, dtype=dtype, out=out)

    @staticmethod
    def max(a, axis=None, out=None, keepdims=False):
        if axis is None:
            return torch.max(a, keepdim=keepdims, out=out)
        else:
            return torch.max(a, axis, keepdim=keepdims, out=out)


    @staticmethod
    def min(a, axis=None, out=None, keepdims=False):
        return torch.min(a, dim=axis, keepdim=keepdims, out=out)

    @staticmethod
    def argmax(a, axis=None, out=None, keepdims=False):
        return torch.argmax(a, dim=axis, out=out, keepdim=keepdims)

    @staticmethod
    def argmin(a, axis=None, out=None, keepdims=False):
        return torch.argmin(a, dim=axis, out=out, keepdim=keepdims)

    ### Unary methods ###
    # NOTE: all copied

    ### Binary methods ###

    @staticmethod
    def add_at(a: Tensor, indices: Tensor, src: Tensor, /):
        a.index_add_(indices.ravel(), src.ravel())

    @staticmethod
    def cross(a, b, axis=-1, **kwargs):
        return torch.cross(a, b, dim=axis, **kwargs)

    @staticmethod
    def tensordot(a, b, axes):
        return torch.tensordot(a, b, dims=axes)

    ### Other methods ###

    @staticmethod
    def copy(a, /, **kwargs):
        return torch.clone(a, **kwargs)

    @staticmethod
    def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=0, **kwargs):
        """
        unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> Tuple[Tensor, Tensor, Tensor]
        """
        b, inverse, counts = torch.unique(a, return_inverse=True,
                return_counts=True,
                dim=axis, **kwargs)
        any_return = return_index or return_inverse or return_counts
        if any_return:
            result = (b, )
        else:
            retult = b

        if return_index:
            kwargs = {'dtype': inverse.dtype, 'device': inverse.device}
            indices = torch.zeros(counts.shape, **kwargs)
            indices[inverse.flip(dims=[0])] = torch.arange(a.shape[axis]-1, -1, -1, **kwargs)
            result += (indices, )

        if return_inverse:
            result += (inverse, )

        if return_counts:
            result += (counts, )

        return result

    @staticmethod
    def sort(a, axis=0, **kwargs):
        return torch.sort(a, dim=axis, **kwargs)[0]

    @staticmethod
    def argsort(a, axis=-1, **kwargs):
        return torch.argsort(a, dim=axis, **kwargs)

    @staticmethod
    def nonzero(a, /, as_tuple=True):
        return torch.nonzero(a, as_tuple=as_tuple)

    @staticmethod
    def cumsum(a, axis=None, dtype=None, out=None):
        return torch.cumsum(a, dim=axis, dtype=dtype, out=out)

    @staticmethod
    def cumprod(a, axis=None, dtype=None, out=None):
        return torch.cumprod(a, dim=axis, dtype=dtype, out=out)

    @staticmethod
    def concatenate(arrays, /, axis=0, out=None, *, dtype=None):
        if dtype is not None:
            arrays = [a.to(dtype) for a in arrays]
        return torch.cat(arrays, dim=axis, out=out)

    @staticmethod
    def stack(arrays, axis=0, out=None, *, dtype=None):
        if dtype is not None:
            arrays = [a.to(dtype=dtype) for a in arrays]
        return torch.stack(arrays, dim=axis, out=out)

    @staticmethod
    def flip(a, axis=None):
        if axis is None:
            axis = list(range(a.dim()))
        elif isinstance(axis, int):
            axis = [axis]
        elif isinstance(axis, tuple):
            axis = list(axis)
        return torch.flip(a, dims=axis)

    @staticmethod
    def where(cond, x=None, y=None, out=None):
        if x is None or y is None:
            return torch.as_tensor(cond).nonzero(as_tuple=True)
        else:
            return torch.where(cond, x, y, out=out)

    ### Functional programming
    @staticmethod
    def apply_along_axis(func1d, axis, x, *args, **kwargs):
        """
        Parameters:
            func1d : function (M,) -> (Nj...)
            This function should accept 1-D arrays. It is applied to 1-D slices of `arr` along the specified axis.
            axis : integer
                Axis along which `arr` is sliced.
            arr : ndarray (Ni..., M, Nk...)
            Input array.
            args : any Additional arguments to `func1d`.
            kwargs : any Additional named arguments to `func1d`.
        """
        if axis==0:
            x = torch.transpose(x)
        return vmap(func1d)(x)


    ### FEALPy functionals ###


    @staticmethod
    def multi_index_matrix(p: int, dim: int, *, dtype=None) -> Tensor:
        dtype = dtype or torch.int
        sep = torch.flip(torch.tensor(
            tuple(combinations_with_replacement(range(p+1), dim)),
            dtype=dtype
        ), dims=(0,))
        raw = torch.zeros((sep.shape[0], dim+2), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @staticmethod
    def edge_length(edge: Tensor, node: Tensor, *, out=None) -> Tensor:
        points = node[edge, :]
        return norm(points[..., 0, :] - points[..., 1, :], dim=-1, out=out)

    @staticmethod
    def edge_normal(edge: Tensor, node: Tensor, unit=False, *, out=None) -> Tensor:
        points = node[edge, :]
        if points.shape[-1] != 2:
            raise ValueError("Only 2D meshes are supported.")
        edges = points[..., 1, :] - points[..., 0, :]
        if unit:
            edges = edges.div_(norm(edges, dim=-1, keepdim=True))
        return torch.stack([edges[..., 1], -edges[..., 0]], dim=-1, out=out)

    @staticmethod
    def edge_tangent(edge: Tensor, node: Tensor, unit=False, *, out=None) -> Tensor:
        v = torch.sub(node[edge[:, 1], :], node[edge[:, 0], :], out=out)
        if unit:
            l = torch.norm(v, dim=-1, keepdim=True)
            v.div_(l)
        return v

    @staticmethod
    def tensorprod(*tensors: Tensor) -> Tensor:
        num = len(tensors)
        NVC = reduce(lambda x, y: x * y.shape[-1], tensors, 1)
        desp1 = 'mnopq'
        desp2 = 'abcde'
        string = ", ".join([desp1[i]+desp2[i] for i in range(num)])
        string += " -> " + desp1[:num] + desp2[:num]
        return torch.einsum(string, *tensors).reshape(-1, NVC)

    @classmethod
    def bc_to_points(cls, bcs: Union[Tensor, Tuple[Tensor, ...]], node: Tensor, entity: Tensor) -> Tensor:
        points = node[entity, :]

        if not isinstance(bcs, Tensor):
            bcs = cls.tensorprod(*bcs)
        return torch.einsum('ijk, ...j -> i...k', points, bcs)

    @staticmethod
    def barycenter(entity: Tensor, node: Tensor, loc: Optional[Tensor]=None) -> Tensor:
        return torch.mean(node[entity, :], dim=1) # TODO: polygon mesh case

    @staticmethod
    def simplex_measure(entity: Tensor, node: Tensor) -> Tensor:
        points = node[entity, :]
        TD = points.size(-2) - 1
        if TD != points.size(-1):
            raise RuntimeError("The geometric dimension of points must be NVC-1"
                            "to form a simplex.")
        edges = points[..., 1:, :] - points[..., :-1, :]
        return det(edges).div(factorial(TD))

    @classmethod
    def _simplex_shape_function_kernel(cls, bc: Tensor, p: int, mi: Optional[Tensor]=None) -> Tensor:
        TD = bc.shape[-1] - 1
        itype = torch.int
        device = bc.device
        shape = (1, TD+1)

        if mi is None:
            mi = cls.multi_index_matrix(p, TD, dtype=torch.int)

        c = torch.arange(1, p+1, dtype=itype, device=device)
        P = 1.0 / torch.cumprod(c, dim=0)
        t = torch.arange(0, p, dtype=itype, device=device)
        Ap = p*bc.unsqueeze(-2) - t.reshape(-1, 1)
        Ap = torch.cumprod(Ap, dim=-2).clone()
        Ap = Ap.mul(P.reshape(-1, 1))
        A = torch.cat([torch.ones(shape, dtype=bc.dtype, device=device), Ap], dim=-2)
        idx = torch.arange(TD + 1, dtype=itype, device=device)
        phi = torch.prod(A[mi, idx], dim=-1)
        return phi

    @classmethod
    def simplex_shape_function(cls, bcs: Tensor, p: int, mi=None) -> Tensor:
        fn = vmap(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        )
        return fn(bcs)

    @classmethod
    def simplex_grad_shape_function(cls, bcs: Tensor, p: int, mi=None) -> Tensor:
        fn = vmap(jacfwd(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        ))
        return fn(bcs)

    @classmethod
    def simplex_hess_shape_function(cls, bcs: Tensor, p: int, mi=None) -> Tensor:
        fn = vmap(jacrev(jacfwd(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        )))
        return fn(bcs)

    @staticmethod
    def tensor_measure(entity: Tensor, node: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError

    @staticmethod
    def interval_grad_lambda(line: Tensor, node: Tensor) -> Tensor:
        points = node[line, :]
        v = points[..., 1, :] - points[..., 0, :] # (NC, GD)
        h2 = torch.sum(v**2, dim=-1, keepdim=True)
        v = v.div(h2)
        return torch.stack([-v, v], dim=-2)

    @staticmethod
    def triangle_area_3d(tri: Tensor, node: Tensor, out: Optional[Tensor]=None) -> Tensor:
        points = node[tri, :]
        cross_product = cross(points[..., 1, :] - points[..., 0, :],
                    points[..., 2, :] - points[..., 0, :], dim=-1, out=out) / 2.0
        result = norm(cross_product, dim=-1)
        return result

    @staticmethod
    def triangle_grad_lambda_2d(tri: Tensor, node: Tensor) -> Tensor:
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :]
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = det(torch.stack([e0, e1], dim=-2)) # (...)
        e0 = e0.flip(-1)
        e1 = e1.flip(-1)
        e2 = e2.flip(-1)
        result = torch.stack([e0, e1, e2], dim=-2)
        result[..., 0].mul_(-1)
        return result.div_(nv[..., None, None])

    @staticmethod
    def triangle_grad_lambda_3d(tri: Tensor, node: Tensor) -> Tensor:
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :] # (..., 3)
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = cross(e0, e1, dim=-1) # (..., 3)
        length = norm(nv, dim=-1, keepdim=True) # (..., 1)
        n = nv.div_(length)
        return torch.stack([
            cross(n, e0, dim=-1),
            cross(n, e1, dim=-1),
            cross(n, e2, dim=-1)
        ], dim=-2).div_(length.unsqueeze(-2)) # (..., 3, 3)

    @staticmethod
    def quadrangle_grad_lambda_2d(quad: Tensor, node: Tensor) -> Tensor:
        pass

    @classmethod
    def tetrahedron_grad_lambda_3d(cls, tet: Tensor, node: Tensor, localFace: Tensor) -> Tensor:
        NC = tet.shape[0]
        Dlambda = torch.zeros((NC, 4, 3), dtype=node.dtype)
        volume = cls.simplex_measure(tet, node)
        for i in range(4):
            j, k, m = localFace[i]
            vjk = node[tet[:, k],:] - node[tet[:, j],:]
            vjm = node[tet[:, m],:] - node[tet[:, j],:]
            Dlambda[:, i, :] = cross(vjm, vjk, dim=-1) / (6*volume.reshape(-1, 1))
        return Dlambda


attribute_mapping = ATTRIBUTE_MAPPING.copy()
attribute_mapping.update({
    'bool_': 'bool',
    'int_': 'int',
    'float_': 'float',
    'complex_': 'complex'
})
PyTorchBackend.attach_attributes(attribute_mapping, torch)
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(
        array='tensor', 
        power='pow', 
        transpose='permute',
        repeat='repeat_interleave', 
        index_add_= 'index_add_',
        copy='clone')
PyTorchBackend.attach_methods(function_mapping, torch)

PyTorchBackend.random.rand = torch.rand
PyTorchBackend.random.rand_like = torch.rand_like
PyTorchBackend.random.randint = torch.randint
PyTorchBackend.random.randint_like = torch.randint_like
PyTorchBackend.random.randn = torch.randn
PyTorchBackend.random.randn_like = torch.randn_like
PyTorchBackend.random.randperm = torch.randperm
