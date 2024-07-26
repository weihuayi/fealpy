from typing import Any, Union, Optional, TypeVar
import numpy as np

try:
    import taichi as ti
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
        ti.init(arch=device)
