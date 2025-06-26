
from typing import Union, Optional, Tuple, Any
from itertools import combinations_with_replacement
from functools import reduce, partial
from math import factorial, prod
from scipy.spatial import KDTree

try:
    import torch
    from torch import vmap, norm, det, cross
    from torch.func import jacfwd, jacrev

except ImportError:
    raise ImportError("Name 'torch' cannot be imported. "
                      'Make sure PyTorch is installed before using '
                      'the PyTorch backend in fealpy. '
                      'See https://pytorch.org/ for installation.')

from .. import logger
from .base import (
    BackendProxy, ModuleProxy,
    ATTRIBUTE_MAPPING, FUNCTION_MAPPING, TRANSFORMS_MAPPING
)

Tensor = torch.Tensor
_device = torch.device

def _dim_to_axis(func):
    def wrapper(*args, axis=None, **kwargs):
        if axis is None:
            return func(*args, **kwargs)
        return func(*args, dim=axis, **kwargs)
    return wrapper

def _dims_to_axes(func):
    def wrapper(*args, axes=None, **kwargs):
        if axes is None:
            return func(*args, **kwargs)
        return func(*args, dims=axes, **kwargs)
    return wrapper

def _size_to_shape(func):
    def wrapper(*args, shape=None, **kwargs):
        if shape is None:
            return func(*args, **kwargs)
        return func(*args, size=shape, **kwargs)
    return wrapper

def _axis_keepdims_dispatch(func, **defaults):
    if len(defaults) > 0:
        def wrapper(*args, **kwargs):
            if 'axis' in kwargs:
                kwargs['dim'] = kwargs.pop('axis')
            if 'keepdims' in kwargs:
                kwargs['keepdim'] = kwargs.pop('keepdims')
            defaults.update(kwargs)
            kwargs = defaults
            return func(*args, **kwargs)
    else:
        def wrapper(*args, **kwargs):
            if 'axis' in kwargs:
                kwargs['dim'] = kwargs.pop('axis')
            if 'keepdims' in kwargs:
                kwargs['keepdim'] = kwargs.pop('keepdims')
            return func(*args, **kwargs)
    return wrapper


class PyTorchBackend(BackendProxy, backend_name='pytorch'):
    DATA_CLASS = torch.Tensor
    linalg = torch.linalg
    random = torch.random

    @staticmethod
    def context(tensor: Tensor, /):
        return {"dtype": tensor.dtype, "device": tensor.device}

    @staticmethod
    def set_default_device(device: Union[str, _device]) -> None:
        torch.set_default_device(device)

    @staticmethod
    def device_type(tensor_like: Tensor, /): return tensor_like.device.type

    @staticmethod
    def device_index(tensor_like: Tensor, /): return tensor_like.device.index

    @staticmethod
    def get_device(tensor_like: Tensor, /): return tensor_like.device

    @staticmethod
    def device_put(tensor_like: Tensor, /, device=None) -> Tensor:
        return tensor_like.to(device=device)

    @staticmethod
    def to_numpy(tensor_like: Tensor, /) -> Any:
        return tensor_like.detach().cpu().numpy()

    from_numpy = staticmethod(torch.from_numpy)

    @staticmethod
    def tolist(tensor: Tensor, /): return tensor.tolist()

    ### Creation Functions ###
    # python array API standard v2023.12
    @staticmethod
    def arange(start, /, stop=None, step=1, *, dtype=None, device=None):
        if stop is None:
            stop = start
            start = 0
        return torch.arange(start, stop, step, dtype=dtype, device=device)

    @staticmethod
    def eye(n: int, m: Optional[int]=None, /, k: int=0, dtype=None, **kwargs) -> Tensor:
        assert k == 0, "Only k=0 is supported by `eye` in PyTorchBackend."
        if m is None:
            m = n
        return torch.eye(n, m, dtype=dtype, **kwargs)

    @staticmethod
    def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
        assert endpoint == True
        if isinstance(start, (int, float)) and isinstance(stop, (int, float)):
            return torch.linspace(start, stop, num, dtype=dtype, device=device)
        else:
            vmap_fun = partial(torch.linspace, dtype=dtype, device=device)
            for _ in range(start.ndim):
                vmap_fun = vmap(vmap_fun, in_dims=(0, 0, None), out_dims=0)
            return vmap_fun(start, stop, num)

    @staticmethod
    def tril(x, /, *, k=0): return torch.tril(x, k)

    @staticmethod
    def triu(x, /, *, k=0): return torch.triu(x, k)

    ### Data Type Functions ###
    # python array API standard v2023.12
    @staticmethod
    def astype(x, dtype, /, *, copy=True, device=None):
        return x.to(dtype=dtype, device=device, copy=copy)

    ### Element-wise Functions ###
    @staticmethod # NOTE: PyTorch's build-in equal is actually `all(equal(x1, x2))`
    def equal(x1, x2, /): return x1 == x2

    ### Indexing Functions ###

    ### Linear Algebra Functions ###
    # python array API standard v2023.12
    @staticmethod
    def matrix_transpose(x, /): return x.transpose(-1, -2)

    tensordot = staticmethod(_dims_to_axes(torch.tensordot))
    vecdot = staticmethod(_dim_to_axis(torch.linalg.vecdot))

    # non-standard
    cross = staticmethod(_dim_to_axis(torch.cross))

    @staticmethod
    def dot(x1, x2, /, *, axis=-1):
        return torch.tensordot(x1, x2, dims=[[axis], [axis]])

    @staticmethod
    def trace(x, /, *, offset: int = 0, axis1=0, axis2=1):
        data = torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)
        return torch.sum(data, dim=-1)

    ### Manipulation Functions ###
    # python array API standard v2023.12
    broadcast_to = staticmethod(_size_to_shape(torch.broadcast_to))
    concat = staticmethod(_dim_to_axis(torch.concat))
    expand_dims = staticmethod(_dim_to_axis(torch.unsqueeze))
    @staticmethod
    def flip(a, axis=None):
        if axis is None:
            axis = list(range(a.dim()))
        elif isinstance(axis, int):
            axis = [axis]
        elif isinstance(axis, tuple):
            axis = list(axis)
        return torch.flip(a, dims=axis)

    permute_dims = staticmethod(_dims_to_axes(torch.permute))
    repeat = staticmethod(_dim_to_axis(torch.repeat_interleave))
    @staticmethod
    def roll(x, /, shift, *, axis=None):
        return torch.roll(x, shifts=shift, dims=axis)

    squeeze = staticmethod(_dim_to_axis(torch.squeeze))
    stack = staticmethod(_dim_to_axis(torch.stack))
    unstack = staticmethod(_dim_to_axis(torch.unbind))

    # non-standard
    @staticmethod
    def concatenate(arrays, /, axis=0, out=None, *, dtype=None):
        if dtype is not None:
            arrays = [a.to(dtype) for a in arrays]
        return torch.cat(arrays, dim=axis, out=out)

    @staticmethod
    def insert(x, obj, values, /, *, axis=None):
        kwargs = {'dtype': x.dtype, 'device': x.device}
        ndim = x.ndim
        if axis is None:
            if ndim != 1:
                x = x.ravel()
            ndim = x.ndim
            axis = ndim - 1
        else: # normalize axis
            if axis < -ndim or axis > ndim:
                raise IndexError(f"index {axis} is out of bounds for axis {axis} "
                                f"with size {ndim}")
            axis = axis if axis >=0 else axis + ndim

        slobj = [slice(None), ] * ndim
        N = x.shape[axis]
        newshape = list(x.shape)

        if isinstance(obj, slice):
            # turn it into a range object
            indices = torch.arange(*obj.indices(N), dtype=torch.int64, device=x.device)
        else:
            # need to copy obj, because indices will be changed in-place
            indices = torch.tensor(obj, dtype=torch.int64, device=x.device)
            if indices.dtype == bool:
                indices = torch.as_tensor(indices, dtype=torch.int64, device=x.device)
            elif indices.ndim > 1:
                raise ValueError(
                    "index array argument obj to insert must be one dimensional "
                    "or scalar")
        if indices.numel() == 1:
            index = indices.item()
            if index < -N or index > N:
                raise IndexError(f"index {obj} is out of bounds for axis {axis} "
                                f"with size {N}")
            if (index < 0):
                index += N

            values = torch.tensor(values, **kwargs)
            if values.ndim < ndim:
                values = values.reshape((1,)*(ndim - values.ndim) + (-1, ))
            if indices.ndim == 0:
                # broadcasting is very different here, since a[:,0,:] = ... behaves
                # very different from a[:,[0],:] = ...! This changes values so that
                # it works likes the second case. (here a[:,0:1,:])
                values = torch.moveaxis(values, 0, axis)
            numnew = values.shape[axis]
            newshape[axis] += numnew
            new = torch.empty(newshape, **kwargs)
            slobj[axis] = slice(None, index)
            new[tuple(slobj)] = x[tuple(slobj)]
            slobj[axis] = slice(index, index+numnew)
            print(values.shape, new.shape, slobj)
            new[tuple(slobj)] = values
            slobj[axis] = slice(index+numnew, None)
            slobj2 = [slice(None)] * ndim
            slobj2[axis] = slice(index, None)
            new[tuple(slobj)] = x[tuple(slobj2)]

            return new

        elif indices.numel() == 0 and not isinstance(obj, Tensor):
            # Can safely cast the empty list to int64
            indices = torch.as_tensor(indices, torch.int64)

        indices[indices < 0] += N

        numnew = len(indices)
        order = indices.argsort(stable=True)   # stable sort
        indices[order] += torch.arange(numnew, dtype=torch.int64, device=x.device)

        newshape[axis] += numnew
        old_mask = torch.ones(newshape[axis], dtype=torch.bool, device=x.device)
        old_mask[indices] = False

        new = torch.empty(newshape, **kwargs)
        slobj2 = [slice(None)]*ndim
        slobj[axis] = indices
        slobj2[axis] = old_mask
        new[tuple(slobj)] = values
        new[tuple(slobj2)] = x

        return new

    @staticmethod
    def split(x, indices_or_sections, /, *, axis=0):
        if isinstance(indices_or_sections, int):
            chunk_size = x.shape[axis] // indices_or_sections
        elif isinstance(indices_or_sections, Tensor):
            if indices_or_sections.ndim != 1:
                raise ValueError("indices_or_sections must be 1-dimensional")
            kwargs = {'dtype': indices_or_sections.dtype, 'device': indices_or_sections.device}
            HEAD = torch.tensor([0], **kwargs)
            TAIL = torch.tensor([x.shape[axis]], **kwargs)
            indices_or_sections = torch.cat([HEAD, indices_or_sections, TAIL])
            chunk_size = (indices_or_sections[1:] - indices_or_sections[:-1]).tolist()
        else:
            raise ValueError("indices_or_sections must be a scalar or 1D Tensor")

        return torch.split(x, chunk_size, dim=axis)

    ### Searching Functions ###
    # python array API standard v2023.12
    argmax = staticmethod(_axis_keepdims_dispatch(torch.argmax))
    argmin = staticmethod(_axis_keepdims_dispatch(torch.argmin))

    @staticmethod
    def nonzero(x, /):
        return torch.nonzero(x, as_tuple=True)

    ### Set Functions ###
    # python array API standard v2023.12
    @staticmethod
    def unique_all(a, axis=None, **kwargs):
        if axis is None:
            a = torch.flatten(a)
            axis = 0
        b, inverse, counts = torch.unique(a, return_inverse=True,
                return_counts=True,
                dim=axis, **kwargs)
        kwargs = {'dtype': inverse.dtype, 'device': inverse.device}
        indices = torch.zeros(counts.shape, **kwargs)
        idx = torch.arange(a.shape[axis]-1, -1, -1, **kwargs)
        indices.scatter_(0, inverse.flip(dims=[0]), idx)
        return b, indices, inverse, counts

    # non-standard
    @staticmethod
    def unique_all_(a, axis=None, **kwargs):
        if axis is None:
            a = torch.flatten(a)
            axis = 0
        b, inverse, counts = torch.unique(a, return_inverse=True,
                return_counts=True,
                dim=axis, **kwargs)
        kwargs = {'dtype': inverse.dtype, 'device': inverse.device}

        indices0 = torch.zeros(counts.shape, **kwargs)
        indices1 = torch.zeros(counts.shape, **kwargs)

        idx = torch.arange(a.shape[axis]-1, -1, -1, **kwargs)
        indices0.scatter_(0, inverse.flip(dims=[0]), idx)

        idx = torch.arange(a.shape[0], **kwargs)
        indices1.scatter_(0, inverse, idx)
        return b, indices0, indices1, inverse, counts

    @staticmethod
    def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=0, **kwargs):
        b, inverse, counts = torch.unique(a, return_inverse=True,
                return_counts=True,
                dim=axis, **kwargs)
        any_return = return_index or return_inverse or return_counts
        if any_return:
            result = (b, )
        else:
            result = b

        if return_index:
            kwargs = {'dtype': inverse.dtype, 'device': inverse.device}
            indices = torch.zeros(counts.shape, **kwargs)
            idx = torch.arange(a.shape[axis]-1, -1, -1, **kwargs)
            indices.scatter_(0, inverse.flip(dims=[0]), idx)
            result += (indices, )

        if return_inverse:
            result += (inverse, )

        if return_counts:
            result += (counts, )

        return result

    ###Sorting Functions ###
    # python array API standard v2023.12
    argsort = staticmethod(_dim_to_axis(torch.argsort))
    @staticmethod
    def sort(x, /, *, axis=-1, descending=False, stable=True):
        return torch.sort(x, dim=axis, descending=descending, stable=stable)[0]

    # non-standard
    @staticmethod
    def lexsort(keys: Tuple[Tensor, ...], /, *, axis: int = -1):
        if keys[0].ndim < 1:
            raise ValueError("keys must be at least 2 dimensional, but got "
                             f"shape {keys.shape}.")
        if len(keys) == 0:
            raise ValueError(f"Must have at least 1 key.")

        idx = keys[0].argsort(dim=axis, stable=True)

        for k in keys[1:]:
            idx = idx.gather(axis, k.gather(axis, idx).argsort(dim=axis, stable=True))

        return idx

    ### Statistical Functions ###
    # python array API standard v2023.12
    @staticmethod
    def max(x, /, *, axis=None, keepdims=False):
        if axis is None:
            return torch.max(x)
        return torch.max(x, axis, keepdim=keepdims)[0]

    mean = staticmethod(_axis_keepdims_dispatch(torch.mean))

    @staticmethod
    def min(x, /, *, axis=None, keepdims=False):
        if axis is None:
            return torch.min(x)
        return torch.min(x, axis, keepdim=keepdims)[0]

    prod = staticmethod(_axis_keepdims_dispatch(torch.prod))
    std = staticmethod(_axis_keepdims_dispatch(torch.std, correction=0.))
    sum = staticmethod(_axis_keepdims_dispatch(torch.sum))
    var = staticmethod(_axis_keepdims_dispatch(torch.var, correction=0.))

    # non-standard
    cumsum = staticmethod(_dim_to_axis(torch.cumsum))
    cumprod = staticmethod(_dim_to_axis(torch.cumprod))

    ### Utility Functions ###
    # python array API standard v2023.12
    all = staticmethod(_axis_keepdims_dispatch(torch.all))
    any = staticmethod(_axis_keepdims_dispatch(torch.any))

    # non-standard
    @staticmethod
    def size(x, /, *, axis=None):
        if axis is None:
            return x.numel()
        else:
            return x.size(axis)

    ### Other Functions ###

    @staticmethod
    def set_at(a: Tensor, indices, src, /):
        a[indices] = src
        return a

    @staticmethod
    def add_at(a: Tensor, indices, src, /):
        logger.warning("When indices are not unique, the behavior is non-deterministic "
                       "for the PyTorch backend "
                       "(one of the values from src will be picked arbitrarily). "
                       "Use index_add instead for deterministic behavior.")
        a[indices] += src
        return a

    @staticmethod
    def index_add(a: Tensor, index, src, /, *, axis: int=0, alpha=1):
        axis = a.ndim + axis if (axis < 0) else axis
        src_flat_shape = a.shape[:axis] + (index.numel(), ) + a.shape[axis+1:]

        if isinstance(src, (int, float, complex)):
            src = torch.full([1,]*len(src_flat_shape), src, dtype=a.dtype, device=a.device)
            src = torch.broadcast_to(src, src_flat_shape)
        else:
            src_shape = a.shape[:axis] + index.shape + a.shape[axis+1:]
            src = torch.broadcast_to(src, src_shape).reshape(src_flat_shape)

        return a.index_add_(axis, index.ravel(), src, alpha=alpha)

    @staticmethod
    def scatter(x: Tensor, index, src, /, *, axis: int=0):
        x.scatter_(dim=axis, index=index, src=src)
        return x

    @staticmethod
    def scatter_add(x: Tensor, index, src, /, *, axis: int=0):
        x.scatter_add_(dim=axis, index=index, src=src)
        return x

    ### Functional programming ###

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

    @staticmethod
    def vmap(func, /, in_axes=0, out_axes=0, **kwargs):
        return torch.vmap(func, in_dims=in_axes, out_dims=out_axes, **kwargs)

    ### Sparse Functions ###

    @staticmethod
    def coo_spmm(indices, values, shape, other):
        if values.ndim == 1:
            mat = torch.sparse_coo_tensor(indices, values, size=shape)
            return PyTorchBackend._spmm(mat, other)
        else:
            raise NotImplementedError("Batch sparse matrix multiplication has "
                                      "not been supported yet.")

    @staticmethod
    def csr_spmm(crow, col, values, shape, other):
        if values.ndim == 1:
            mat = torch.sparse_csr_tensor(crow, col, values, size=shape)
            return PyTorchBackend._spmm(mat, other)
        else:
            raise NotImplementedError("Batch sparse matrix multiplication has "
                                      "not been supported yet.")

    @staticmethod
    def csr_spspmm(crow1, col1, values1, shape1, crow2, col2, values2, shape2):
        mat1 = torch.sparse_csr_tensor(crow1, col1, values1, size=shape1)
        mat2 = torch.sparse_csr_tensor(crow2, col2, values2, size=shape2)
        mat3: Tensor = PyTorchBackend._spmm(mat1, mat2)
        return mat3.crow_indices(), mat3.col_indices(), mat3.values(), mat3.shape

    @staticmethod
    def _spmm(mat, other):
        if other.ndim == 1:
            return torch.sparse.mm(mat, other[:, None])[:, 0]
        else:
            return torch.sparse.mm(mat, other)

    @staticmethod
    def coo_tocsr(indices, values, shape):
        mat = torch.sparse_coo_tensor(indices, values, size=shape)
        mat = mat.to_sparse_csr()
        return mat.crow_indices(), mat.col_indices(), mat.values()

    @staticmethod
    def query_point(x, y, h, box_size, mask_self=True, periodic=[True, True, True]):
        if not isinstance(periodic, list) or len(periodic) != 3 or not all(isinstance(p, bool) for p in periodic):
            raise TypeError("periodic type is：[bool, bool, bool]")
        def map_points(a, b, r, positions):
            x, y = positions[:, 0], positions[:, 1]
            cond_1 = (0 <= x) & (x <= r) & (r < y) & (y < b - r)  # 区域[(0, r), (r, b-r)]
            cond_2 = (a - r <= x) & (x <= a) & (r < y) & (y < b - r)  # 区域[(a-r, a), (r, b-r)]
            cond_3 = (r < x) & (x < a - r) & (0 <= y) & (y <= r)  # 区域[(r, a-r), (0, r)]
            cond_4 = (r < x) & (x < a - r) & (b - r <= y) & (y <= b)  # 区域[(r, a-r), (b-r, b)]
            cond_5 = (0 <= x) & (x <= r) & (0 <= y) & (y <= r)  # 角区域[(0, r), (0, r)]
            cond_6 = (a - r <= x) & (x <= a) & (0 <= y) & (y <= r)  # 角区域[(a-r, a), (0, r)]
            cond_7 = (0 <= x) & (x <= r) & (b - r <= y) & (y <= b)  # 角区域[(0, r), (b-r, b)]
            cond_8 = (a - r <= x) & (x <= a) & (b - r <= y) & (y <= b)  # 角区域[(a-r, a), (b-r, b)]
            cond_9 = ~(cond_1 | cond_2 | cond_3 | cond_4 | cond_5 | cond_6 | cond_7 | cond_8) # 其他区域
            idx_positions = torch.arange(len(positions))
            bool_positions = torch.ones(len(positions), dtype=bool)

            cond_5_1 = torch.concatenate([(x[cond_5] + a).reshape(-1,1), (y[cond_5]).reshape(-1,1)], axis=-1) # 右侧映射
            cond_5_2 = torch.concatenate([(x[cond_5]).reshape(-1,1), (y[cond_5] + b).reshape(-1,1)], axis=-1) # 上方映射 
            cond_5_3 = torch.concatenate([(x[cond_5] + a).reshape(-1,1), (y[cond_5] + b).reshape(-1,1)], axis=-1) # 右上方映射
            _cond_5 = torch.concatenate([cond_5_1, cond_5_2, cond_5_3], axis=0)
            idx_cond_5_1 = torch.nonzero(cond_5).flatten()
            idx_5 = torch.concatenate([idx_cond_5_1, idx_cond_5_1, idx_cond_5_1], axis=0)
            bool_cond_5_1 = torch.zeros(len(cond_5_1), dtype=bool)
            bool_5 = torch.concatenate([bool_cond_5_1, bool_cond_5_1, bool_cond_5_1], axis=0)

            _cond_3 = torch.concatenate([(x[cond_3]).reshape(-1,1), (y[cond_3] + b).reshape(-1,1)], axis=-1) # 上侧映射 
            idx_3 = torch.nonzero(cond_3).flatten()
            bool_3 = torch.zeros(len(_cond_3), dtype=bool)

            cond_6_1 = torch.concatenate([(x[cond_6] - a).reshape(-1,1), (y[cond_6]).reshape(-1,1)], axis=-1) # 左侧映射
            cond_6_2 = torch.concatenate([(x[cond_6]).reshape(-1,1), (y[cond_6] + b).reshape(-1,1)], axis=-1) # 上方映射
            cond_6_3 = torch.concatenate([(x[cond_6] - a).reshape(-1,1), (y[cond_6] + b).reshape(-1,1)], axis=-1) # 左上方映射
            _cond_6 = torch.concatenate([cond_6_1, cond_6_2, cond_6_3], axis=0)
            idx_cond_6_1 = torch.nonzero(cond_6).flatten()
            idx_6 = torch.concatenate([idx_cond_6_1, idx_cond_6_1, idx_cond_6_1], axis=0)
            bool_cond_6_1 = torch.zeros(len(cond_6_1), dtype=bool)
            bool_6 = torch.concatenate([bool_cond_6_1, bool_cond_6_1, bool_cond_6_1], axis=0)

            _cond_1 = torch.concatenate([(x[cond_1] + a).reshape(-1,1), y[cond_1].reshape(-1,1)], axis=-1) # 右侧映射
            idx_1 = torch.nonzero(cond_1).flatten()
            bool_1 = torch.zeros(len(_cond_1), dtype=bool)

            _cond_2 = torch.concatenate([(x[cond_2] - a).reshape(-1,1), y[cond_2].reshape(-1,1)], axis=-1) # 左侧映射
            idx_2 = torch.nonzero(cond_2).flatten()
            bool_2 = torch.zeros(len(_cond_2), dtype=bool)

            cond_7_1 = torch.concatenate([(x[cond_7] + a).reshape(-1,1), (y[cond_7]).reshape(-1,1)], axis=-1) # 右侧映射
            cond_7_2 = torch.concatenate([(x[cond_7]).reshape(-1,1), (y[cond_7] - b).reshape(-1,1)], axis=-1) # 下方映射
            cond_7_3 = torch.concatenate([(x[cond_7] + a).reshape(-1,1), (y[cond_7] - b).reshape(-1,1)], axis=-1) # 右下方映射
            _cond_7 = torch.concatenate([cond_7_1, cond_7_2, cond_7_3], axis=0)
            idx_cond_7_1 = torch.nonzero(cond_7).flatten()
            idx_7 = torch.concatenate([idx_cond_7_1, idx_cond_7_1, idx_cond_7_1], axis=0)
            bool_cond_7_1 = torch.zeros(len(cond_7_1), dtype=bool)
            bool_7 = torch.concatenate([bool_cond_7_1, bool_cond_7_1, bool_cond_7_1], axis=0)

            _cond_4 = torch.concatenate([(x[cond_4]).reshape(-1,1), (y[cond_4] - b).reshape(-1,1)], axis=-1) #下侧映射
            idx_4 = torch.nonzero(cond_4).flatten()
            bool_4 = torch.zeros(len(_cond_4), dtype=bool)

            cond_8_1 = torch.concatenate([(x[cond_8] - a).reshape(-1,1), (y[cond_8]).reshape(-1,1)], axis=-1) # 左侧映射
            cond_8_2 = torch.concatenate([(x[cond_8]).reshape(-1,1), (y[cond_8] - b).reshape(-1,1)], axis=-1) # 下侧映射
            cond_8_3 = torch.concatenate([(x[cond_8] - a).reshape(-1,1), (y[cond_8] - b).reshape(-1,1)], axis=-1) # 左下方映射
            _cond_8 = torch.concatenate([cond_8_1, cond_8_2, cond_8_3], axis=0)
            idx_cond_8_1 = torch.nonzero(cond_8).flatten()
            idx_8 = torch.concatenate([idx_cond_8_1, idx_cond_8_1, idx_cond_8_1], axis=0)
            bool_cond_8_1 = torch.zeros(len(cond_8_1), dtype=bool)
            bool_8 = torch.concatenate([bool_cond_8_1, bool_cond_8_1, bool_cond_8_1], axis=0)

            mapped_positions = torch.concatenate([positions, _cond_5, _cond_3, _cond_6, _cond_1, _cond_2, _cond_7, _cond_4, _cond_8], axis=0)
            mapped_indices = torch.concatenate([idx_positions, idx_5, idx_3, idx_6, idx_1, idx_2, idx_7, idx_4, idx_8], axis=0)
            mapped_bool = torch.concatenate([bool_positions, bool_5, bool_3, bool_6, bool_1, bool_2, bool_7, bool_4, bool_8], axis=0)
            return mapped_positions, mapped_indices, mapped_bool

        if all(periodic):
            map_x, map_idx_x, map_bool_x = map_points(box_size[0], box_size[1], h, x)
            map_y, map_idx_y, map_bool_y= map_points(box_size[0], box_size[1], h, y)
            map_x = map_x.to('cpu')
            map_y = map_y.to('cpu')
            tree = KDTree(map_x)
            neighbors = tree.query_ball_point(map_y, h)
            lengths = torch.tensor([len(sublist) for sublist in neighbors]) 
            a = torch.arange(len(lengths))
            node_self = torch.repeat_interleave(a, lengths)
            neighbors = torch.concatenate([torch.tensor(c) for c in neighbors])
            map_bool = map_bool_x[node_self]
            neighbors = map_idx_x[neighbors]
            neighbors = neighbors[map_bool]
            node_self = node_self[node_self < x.shape[0]]
            if not mask_self:
                mask = node_self == neighbors
                node_self = node_self[~mask]
                neighbors = neighbors[~mask]
            
        elif not any(periodic):
            tree = KDTree(x)
            neighbors = tree.query_ball_point(y, h)
            lengths = torch.tensor([len(sublist) for sublist in neighbors]) 
            a = torch.arange(len(lengths))
            node_self = torch.repeat_interleave(a, lengths)
            neighbors = torch.concatenate([torch.tensor(c) for c in neighbors])
            if not mask_self:
                mask = node_self == neighbors
                node_self = node_self[~mask]
                neighbors = neighbors[~mask]

        else:
            for dim in range(3):
                if not periodic[dim]:
                    raise NotImplementedError(f"Single-side periodic boundary condition for dimension {dim} is not implemented yet.")
                    pass
        return node_self, neighbors

    ### FEALPy functionals ###

    @staticmethod
    def multi_index_matrix(p: int, dim: int, *, dtype=None) -> Tensor:
        """
        TODO:
            1. context?
        """
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
        P = 1.0 / torch.cumprod(c, dim=0, dtype=bc.dtype)
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
        fn = vmap(jacfwd(jacfwd(
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
        shape = tri.shape[:-1] + (3, 2)
        result = torch.zeros(shape, dtype=node.dtype, device=node.device)

        result[..., 0, :] = node[tri[..., 2]] - node[tri[..., 1]]
        result[..., 1, :] = node[tri[..., 0]] - node[tri[..., 2]]
        result[..., 2, :] = node[tri[..., 1]] - node[tri[..., 0]]

        nv = result[..., 0, 0]*result[..., 1, 1] - result[..., 0, 1]*result[..., 1, 0]

        result = result.flip(-1)
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


PyTorchBackend.attach_attributes(ATTRIBUTE_MAPPING, torch)
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(
    array='tensor',
    bitwise_invert='bitwise_not',
    broadcast_arrays='broadcast_tensors',
    copy='clone',
    compile='compile'
)
PyTorchBackend.attach_methods(function_mapping, torch)
PyTorchBackend.attach_methods(TRANSFORMS_MAPPING, torch.func)

PyTorchBackend.random.rand = torch.rand
PyTorchBackend.random.rand_like = torch.rand_like
PyTorchBackend.random.randint = torch.randint
PyTorchBackend.random.randint_like = torch.randint_like
PyTorchBackend.random.randn = torch.randn
PyTorchBackend.random.randn_like = torch.randn_like
PyTorchBackend.random.randperm = torch.randperm


##################################################
### Random Submodule
##################################################

class PyTorchRandom(ModuleProxy):
    def seed(self, seed: int):
        torch.manual_seed(seed)

    def rand(self, *size, dtype=None, device=None):
        return torch.rand(*size, dtype=dtype, device=device)

    def randint(self, low, high=None, size=None, dtype=None, device=None):
        kwargs = {'dtype': dtype, 'device': device}
        if size is None:
            size = (1,)
        elif isinstance(size, int):
            size = (size,)

        if high is None: return torch.randint(low, size, **kwargs)
        return torch.randint(low, high, size, **kwargs)

    def randn(self, *size, dtype=None, device=None):
        return torch.randn(*size, dtype=dtype, device=device)


PyTorchRandom.attach_methods({'seed': 'manual_seed'}, torch)
