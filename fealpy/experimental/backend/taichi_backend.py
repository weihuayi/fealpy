from typing import Any, Union, Optional, TypeVar
import numpy as np

try:
    import taichi as ti
    import taichi.math as tm
except ImportError:
    raise ImportError("Name 'taichi' cannot be imported. "
                      'Make sure  Taichi is installed before using '
                      'the Taichi backend in FEALPy. '
                      'See https://taichi-lang.cn/ for installation.')

from .base import Backend, _make_default_mapping

ATTRIBUTE_MAPPING = _make_default_mapping(
    'pi', 'e', 'nan', 'inf', 
    'dtype', 'device',
    'uint8', 'uint16', 'uint32', 'uint64',
    'bool_', 'bool', 
    'int_', 'int8', 'int16', 'int32', 'int64',
    'float_', 'float16', 'float32', 'float64',
    'complex_', 'complex64', 'complex128'
)

Field = ti.Field 
_device = ti._lib.core.Arch 

class TaichiBackend(Backend[Field], backend_name='taichi'):
    DATA_CLASS = Field 

    @staticmethod
    def set_default_device(device: Union[str, _device]) -> None:
        """
        Parameters:
            device : ti.cuda ti.cpu

        TODO:
            1. Set default value
        """
        ti.init(arch=device)

    @staticmethod
    def to_numpy(ti_field: Field, /) -> np.ndarray:
        return ti_field.to_numpy() 

    @staticmethod
    def from_numpy(numpy_array: Array, /) -> Any:
        """
        
        TODO:
            1. add support to `device` agument
        """
        return jax.device_put(numpy_array)

    ### constant ###
    pi = tm.pi
    e = tm.e
    nan = tm.nan 
    inf = tm.inf
    dtype = ti._lib.core.DataType
    device = ti._lib.core.Arch 

    ### 
    
    # Creation functions
    # 'array', 'tensor', 'arange', 'linspace',
    #'empty', 'zeros', 'ones', 'empty_like', 'zeros_like', 'ones_like', 'eye',
    # 'meshgrid',

    # Reduction functions
    # 'all', 'any', 'sum', 'prod', 'mean', 'max', 'min',

    # Unary functions
    #'abs', 'sign', 'sqrt', 'log', 'log10', 'log2', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',

    # Binary functions
    #'add', 'subtract', 'multiply', 'divide', 'power', 'matmul', 'dot', 'cross', 'tensordot',

    # Other functions
    #'reshape', 'broadcast_to', 'einsum', 'unique', 'sort', 'nonzero',
    #'cumsum', 'cumprod', 'cat', 'concatenate', 'stack', 'repeat', 'transpose', 'swapaxes'
