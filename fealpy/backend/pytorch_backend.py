
try:
    import torch
except ImportError:
    raise ImportError("Name 'torch' cannot be imported. "
                      'Make sure PyTorch is installed before using '
                      'the PyTorch backend in fealpy. '
                      'See https://pytorch.org/ for installation.')

from .base import (
    Backend, ATTRIBUTE_MAPPING, CREATION_MAPPING, REDUCTION_MAPPING,
    UNARY_MAPPING, BINARY_MAPPING, OTHER_MAPPING
)

Tensor = torch.Tensor


class PyTorchBackend(Backend[Tensor], backend_name='pytorch'):
    DATA_CLASS = torch.Tensor

    ### Tensor creation methods ###

    @staticmethod
    def linspace(start, stop, num, *, endpoint=True, retstep=False, dtype=None, **kwargs):
        return torch.linspace(start, stop, steps=num, dtype=dtype, **kwargs)

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
        return torch.max(a, dim=axis, keepdim=keepdims, out=out)

    @staticmethod
    def min(a, axis=None, out=None, keepdims=False):
        return torch.min(a, dim=axis, keepdim=keepdims, out=out)

    ### Unary methods ###
    # NOTE: all copied

    ### Binary methods ###

    @staticmethod
    def cross(a, b, axis=-1, **kwargs):
        return torch.cross(a, b, dim=axis, **kwargs)

    @staticmethod
    def tensordot(a, b, axes):
        return torch.tensordot(a, b, dims=axes)

    ### Other methods ###
    # TODO: unique

    @staticmethod
    def sort(a, axis=0, **kwargs):
        return torch.sort(a, dim=axis, **kwargs)

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
        return torch.cat(arrays, dim=axis, out=out)

    @staticmethod
    def stack(arrays, axis=0, out=None, *, dtype=None):
        return torch.stack(arrays, dim=axis, out=out)

    ### FEALPy functionals ###


attribute_mapping = ATTRIBUTE_MAPPING.copy()
attribute_mapping.update({
    'bool_': 'bool',
    'int_': 'int',
    'float_': 'float',
    'complex_': 'complex'
})
PyTorchBackend.attach_attributes(attribute_mapping, torch)
creation_mapping = CREATION_MAPPING.copy()
creation_mapping.update(array='tensor')
PyTorchBackend.attach_methods(CREATION_MAPPING, torch)
PyTorchBackend.attach_methods(REDUCTION_MAPPING, torch)
PyTorchBackend.attach_methods(UNARY_MAPPING, torch)
PyTorchBackend.attach_methods(BINARY_MAPPING, torch)
other_mapping = OTHER_MAPPING.copy()
other_mapping.update(transpose='permute')
PyTorchBackend.attach_methods(OTHER_MAPPING, torch)
