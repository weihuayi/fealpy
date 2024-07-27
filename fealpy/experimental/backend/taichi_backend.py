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

from .base import Backend, ATTRIBUTE_MAPPING, FUNCTION_MAPPING

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

    ### math constant ###

    e = ti.math.e
    pi = ti.math.pi
