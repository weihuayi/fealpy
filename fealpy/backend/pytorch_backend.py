
try:
    import torch
except ImportError:
    raise ImportError("Name 'torch' cannot be imported. "
                      'Make sure PyTorch is installed before using '
                      'the PyTorch backend in fealpy. '
                      'See https://pytorch.org/ for installation.')

from .base import Backend, ATTRIBUTE_MAPPING

Tensor = torch.Tensor


class PyTorchBackend(Backend[Tensor], backend_name='pytorch'):
    DATA_CLASS = torch.Tensor


attribute_mapping = ATTRIBUTE_MAPPING.copy()
attribute_mapping.update({
    'bool_': 'bool',
    'int_': 'int',
    'float_': 'float',
    'complex_': 'complex'
})
PyTorchBackend.attach_attributes(attribute_mapping, torch)
