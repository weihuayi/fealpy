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

Field = ti.Field
Dtype = ti._lib.core.DataType
Device = ti._lib.core.Arch 

# dtype map from numpy to taichi
dtype_map = {
        np.dtype(np.bool): ti.u8,
        np.dtype(np.float32): ti.f32,
        np.dtype(np.float64): ti.f64,
        np.dtype(np.int32): ti.i32,
        np.dtype(np.int64): ti.i64,
        np.dtype(np.uint32): ti.u32,
        np.dtype(np.uint64): ti.u64,
}

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
    def to_numpy(field: ti.Field, /, *) -> np.ndarray:
        return field.to_numpy() 

    @staticmethod
    def from_numpy(x: Array, /, *) -> Field:
        """
        """
        return ti.from_numpy(x) 

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
    def array(x: Any, 
              dtype: Optional[Dtype] = None, device: Optional[Device] = None) -> Field:
        """
        Create a Taichi field from a given array-like input.

        Parameters:
            input_array (Any): The input data to be converted to a Taichi field. Can 
                be list, tuple, or other array-like object.
            dtype (optional): The data type of the Taichi field. If None, the data 
                type of the input array is used.

        Returns:
            ti.field: A Taichi field containing the input data.
        """
        x = np.asarray(x)
        shape = x.shape
        if dtype is None:
            dtype = dtype_map[x.dtype]
        field = ti.field(dtype=dtype, shape=shape)
        field.from_numpy(x)
        return ti_field

    tensor = array

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
            stop (int or float, optional): End of the interval. If not provided, 
                start is treated as 0 and start is used as stop.
            step (int or float, optional): Spacing between values. Default is 1.
            dtype (DataType, optional): Data type of the field. Default is None.

        Returns:
            field: Taichi field with evenly spaced values.
        """
        if stop is None:
            start, stop = 0, start

        # Automatically determine the dtype if it's not provided
        if dtype is None:
            if isinstance(start, int) and isinstance(step, int):
                dtype = ti.i32  # Use integer type
            else:
                dtype = ti.f64  # Use float type

        d = stop - start
        num_elements = d//step + d%step 
        #num_elements = int((d // step) + (d % step != 0))  # Ensure num_elements is an integer
        field = ti.field(dtype, shape=(num_elements,))

        @ti.kernel
        def fill_arange():
            for i in range(num_elements):
                field[i] = start + i * step

        fill_arange()
        return field

    asarray = array

    @staticmethod
    def empty(shape: Union[int, Tuple[int, ...]],
              dtype: Optional[Dtype] = None,
              device: Optional[Device] = None) -> ti.Field:
        """
        Create a Taichi field with uninitialized data having a specified shape.

        Parameters:
            shape (Union[int, Tuple[int, ...]]): Output array shape.
            dtype (Optional[Dtype]): Output array data type. If None, defaults to
                the default real-valued floating-point data type.
            device (Optional[Device]): Device on which to place the created array.

        Returns:
            out (ti.Field): An array containing uninitialized data.
        """
        # If dtype is not provided, use default floating-point type
        if dtype is None:
            dtype = ti.f64  # Default to single-precision float

        # Ensure shape is a tuple
        if isinstance(shape, int):
            shape = (shape,)

        # Create the field with specified dtype and shape
        field = ti.field(dtype, shape=shape)

        return field

    @staticmethod
    def empty_like(x: Field, 
                   dtype: Optional[Dtype] = None,
                   device: Optional[Device] = None) -> Field:
        """
        Create a Taichi field with uninitialized data having the same shape as an input array x.

        Parameters:
            x (ti.Field): Input array from which to derive the output array shape.
            dtype (Optional[Dtype]): Output array data type. If None, infer from x.
            device (Optional[Device]): Device on which to place the created array.

        Returns:
            out (ti.Field): An array having the same shape as x and containing uninitialized data.
        """
        shape = x.shape
        if dtype is None:
            dtype = x.dtype
        field = ti.field(dtype, shape=shape)
        return field

    @staticmethod
    def eye(n_rows: int, 
            n_cols: Optional[int] = None, 
            k: int = 0, 
            dtype: Optional[Dtype] = None, 
            device: Optional[Device] = None) -> Field:
        """
        Create a two-dimensional Taichi field with ones on the kth diagonal and zeros elsewhere.

        Parameters:
            n_rows (int): Number of rows in the output array.
            n_cols (Optional[int]): Number of columns in the output array. If None, defaults to n_rows.
            k (int): Index of the diagonal. Positive for upper, negative for lower, 0 for main diagonal.
            dtype (Optional[Dtype]): Output array data type. If None, defaults to the default floating-point type.
            device (Optional[Device]): Device on which to place the created array.

        Returns:
            out (Field): An array where all elements are zero except for the kth diagonal, which are ones.
        """
        if dtype is None:
            dtype = ti.f64

        if n_cols is None:
            n_cols = n_rows

        field = ti.field(dtype, shape=(n_rows, n_cols))

        @ti.kernel
        def fill_eye():
            for i, j in ti.ndrange(n_rows, n_cols):
                if j - i == k:
                    field[i, j] = 1
                else:
                    field[i, j] = 0
        fill_eye()
        return field

    @staticmethod
    def from_dlpack(/, *)-> Field:
        raise NotImplementedError

    @staticmethod
    def full(shape: Union[int, Tuple[int, ...]],
             fill_value: Union[bool, int, float, complex],
             dtype: Optional[Dtype] = None,
             device: Optional[Device] = None) -> Field:
        """
        Create a Taichi field with a specified shape and fill it with fill_value.

        Parameters:
            shape (Union[int, Tuple[int, ...]]): Output array shape.
            fill_value (Union[bool, int, float, complex]): Fill value.
            dtype (Optional[Dtype]): Output array data type. If None, infer from fill_value.
            device (Optional[Device]): Device on which to place the created array.

        Returns:
            out (Field): An array where every element is equal to fill_value.
        """
        if dtype is None:
            if isinstance(fill_value, bool):
                dtype = ti.u8  # Boolean type in Taichi
            elif isinstance(fill_value, int):
                dtype = ti.i32  # Default integer type
            elif isinstance(fill_value, float):
                dtype = ti.f64  # Default floating-point type
            elif isinstance(fill_value, complex):
                # Taichi does not have built-in complex number support
                # We will use a workaround by creating a field with two components
                # for the real and imaginary parts if complex is needed.
                raise NotImplementedError("Complex data type is not directly supported in Taichi.")
            else:
                raise TypeError("Unsupported fill_value type.")

        # Ensure shape is a tuple
        if isinstance(shape, int):
            shape = (shape,)

        # Create the field with specified dtype and shape
        field = ti.field(dtype, shape=shape)

        @ti.kernel
        def fill_field():
            for I in ti.grouped(field):
                field[I] = fill_value

        fill_field()
        return field

    @staticmethod
    def full_like(x: Field, 
                  fill_value: Union[bool, int, float, complex], 
                  dtype: Optional[Dtype] = None, 
                  device: Optional[Device] = None) -> ti.Field:
        """
        Create a Taichi field with the same shape as x and fill it with fill_value.

        Parameters:
            x (ti.Field): Input array from which to derive the output array shape.
            fill_value (Union[bool, int, float, complex]): Fill value.
            dtype (Optional[Dtype]): Output array data type. If None, infer from x.
            device (Optional[Device]): Device on which to place the created array.

        Returns:
            out (Field): An array having the same shape as x and filled with fill_value.
        """
        # Infer dtype from x if not provided
        if dtype is None:
            dtype = x.dtype

        # Handle complex numbers
        if isinstance(fill_value, complex):
            raise NotImplementedError("Complex data type is not directly supported in Taichi.")

        # Create the field with the same shape and specified dtype
        field = ti.field(dtype, shape=x.shape)

        @ti.kernel
        def fill_field():
            for I in ti.grouped(field):
                field[I] = fill_value

        fill_field()
        return field

    @staticmethod
    def linspace(start: Union[int, float, complex], 
                 stop: Union[int, float, complex], 
                 num: int, 
                 dtype: Optional[Dtype] = None, 
                 device: Optional[Device] = None, 
                 endpoint: bool = True) -> field:
        """
        Create a Taichi field with evenly spaced numbers over a specified interval.

        Parameters:
            start (Union[int, float, complex]): The start of the interval.
            stop (Union[int, float, complex]): The end of the interval.
            num (int): Number of samples. Must be a nonnegative integer value.
            dtype (Optional[Dtype]): Output array data type. If None, infer based on start and stop.
            device (Optional[Device]): Device on which to place the created array.
            endpoint (bool): Whether to include stop in the interval. Default: True.

        Returns:
            out (field): A one-dimensional array containing evenly spaced values.

        TODO:
            1. tensor input case
        """
        # Determine the data type based on start and stop if dtype is not provided
        if dtype is None:
            if isinstance(start, complex) or isinstance(stop, complex):
                raise NotImplementedError("Complex data type is not directly supported in Taichi.")
            else:
                dtype = ti.f64  # Default floating-point type

        # Handle edge cases
        if num <= 0:
            raise ValueError("num must be a positive integer.")

        # Determine step size
        if endpoint:
            num_intervals = num - 1
        else:
            num_intervals = num

        # Calculate step size for real or complex intervals
        step = (stop - start) / num_intervals if num_intervals > 0 else 0

        # Create the field with the specified number of samples
        field = ti.field(dtype, shape=(num,))

        @ti.kernel
        def fill_linspace():
            for i in range(num):
                field[i] = start + i * step

        fill_linspace()
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




