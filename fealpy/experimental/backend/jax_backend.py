from typing import Optional, Union, Tuple
from functools import reduce
from math import factorial
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import config

    config.update("jax_enable_x64", True)

except ImportError:
    raise ImportError("Name 'jax' cannot be imported. "
                      'Make sure JAX is installed before using '
                      'the JAX backend in FEALPy. '
                      'See https://github.com/google/jax for installation.')

from .base import Backend, ATTRIBUTE_MAPPING, FUNCTION_MAPPING

Array = jax.Array 
_device = jax.Device

class JAXBackend(Backend[Array], backend_name='jax'):
    DATA_CLASS = Array 

    @staticmethod
    def set_default_device(device: Union[str, _device]) -> None:
        jax.default_device = device 

    @staticmethod
    def to_numpy(jax_array: Array, /) -> Any:
        return np.array(jax_array) 

    @staticmethod
    def from_numpy(numpy_array: Array, /) -> Any:
        """
        
        TODO:
            1. add support to `device` agument
        """
        return jax.device_put(numpy_array)

    ### Tensor creation methods ###
    # NOTE: all copied

    ### Reduction methods ###
    # NOTE: all copied

    ### Unary methods ###
    # NOTE: all copied

    ### Binary methods ###
    # NOTE: all copied
