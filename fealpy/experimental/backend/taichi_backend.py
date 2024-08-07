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

from .base import Backend

ATTRIBUTE_MAPPING = _make_default_mapping(
    'pi', 'e', 'nan', 'inf', 'dtype', 'device',
    'bool', 
    'uint8', 'uint16', 'uint32', 'uint64', 
    'int8', 'int16', 'int32', 'int64',
    'float16', 'float32', 'float64',
    'complex64', 'complex128'
)

Field = ti.Field 
Dtype = ti._lib.core.DataType
Device = ti._lib.core.Arch 

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

    ### dtype ###
    bool = ti.uint8
    uint8 = ti.uint8
    uint16 = ti.uint16
    uint32 = ti.uint32
    uint64 = ti.uint64
    int8 = ti.int8
    int16 = ti.int16
    int32 = ti.int32
    int64 = ti.int64
    float16 = ti.float16
    float32 = ti.float32
    float64 = ti.float64
    complex64 = None  # 不支持
    complex128 = None # 不支持

    ### constant ###
    pi = tm.pi
    e = tm.e
    nan = tm.nan 
    inf = tm.inf
    Dtype = ti._lib.core.DataType
    Device = ti._lib.core.Arch 

    ### creation functions ###
    @staticmethod
    def arange(start: Union[int, float], 
               stop: Optional[Union[int, float]] = None, 
               step: Union[int, float] = 1, 
               dtype: Optional[Dtype] =None,
               device: Optional[Device] = None) -> Field:
        """
        Create a Taichi field with evenly spaced values within a given interval.

        Parameters:
            start (int or float): Start of the interval.
            stop (int or float, optional): End of the interval. If not provided, start is treated as 0 and start is used as stop.
            step (int or float, optional): Spacing between values. Default is 1.
            dtype (DataType, optional): Data type of the field. Default is None.

        Returns:
            field: Taichi field with evenly spaced values.
        Signitures:
            def arange(start: Union[int, float], /, stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Array
        """
        if stop is None:
            start, stop = 0, start

        # Automatically determine the dtype if it's not provided
        if dtype is None:
            if isinstance(start, int) and isinstance(step, int):
                dtype = ti.i32  # Use integer type
            else:
                dtype = ti.f32  # Use float type

        d = stop - start
        #num_elements = d//step + d%step 
        num_elements = int((d // step) + (d % step != 0))  # Ensure num_elements is an integer
        field = ti.field(dtype, shape=(num_elements,))

        @ti.kernel
        def fill_arange():
            for i in range(num_elements):
                field[i] = start + i * step

        fill_arange()
        return field

    @staticmethod
    def zeros(shape, dtype=ti.i32):
        """
        Create a Taichi field filled with zeros.

        Parameters:
            shape (tuple): Shape of the field.
            dtype (taichi.DataType): Data type of the field. Default is ti.i32.

        Returns:
            field: Taichi field filled with zeros.
        """
        field = ti.field(dtype, shape=shape)
        field.fill(0)
        return field

    def ones(shape, dtype=ti.i32):
        """
        Create a Taichi field filled with ones.

        Parameters:
            shape (tuple): Shape of the field.
            dtype (taichi.DataType): Data type of the field. Default is ti.i32.

        Returns:
            field: Taichi field filled with ones.
        """
        field = ti.field(dtype, shape=shape)
        field.fill(1)
        return field



    ### element-wise functions ###

    ### indexing functions ###

    ### manipulation functions ###

    ### searching functions ###

    ### set functions ###

    ### sorting functions ###

    ### statistical functions ###

    ### utility functions ###




