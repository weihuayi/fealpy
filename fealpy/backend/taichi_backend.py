from typing import Any, Union, Optional, TypeVar, Tuple
import numpy as np

try:
    import taichi as ti
except ImportError:
    raise ImportError(
        "Name 'taichi' cannot be imported. "
        "Make sure  Taichi is installed before using "
        "the Taichi backend in FEALPy. "
        "See https://taichi-lang.cn/ for installation."
    )

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

from fealpy.backend.base import (
    BackendProxy,
    ModuleProxy,
    ATTRIBUTE_MAPPING,
    FUNCTION_MAPPING,
    TRANSFORMS_MAPPING,
)


# 假设 BackendProxy 是你自己定义的基类
class TaichiBackend(BackendProxy, backend_name="taichi"):
    DATA_CLASS = ti.Field
    # Holds the current Taichi arch (e.g., ti.cpu or ti.cuda)
    _device: Union[ti.cpu, ti.cuda, None] = None  # type: ignore

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
    complex128 = None  # 不支持

    @staticmethod
    def context(tensor: ti.Field, /):
        """
        Retrieve context information for a Taichi Field.

        Parameters:
            tensor (ti.Field): A Taichi scalar or vector field.

        Returns:
            dict: Contains 'dtype' (Taichi data type) and 'device' (string indicating CPU or GPU).
        """
        arch = ti.cfg.arch
        device = "cpu" if arch == ti.cpu else "cuda" if arch == ti.cuda else str(arch)
        return {"dtype": tensor.dtype, "device": device}

    @staticmethod
    def set_default_device(device: Union[str, ti.cpu, ti.cuda]) -> None: 
        """
        Configure the default execution device for Taichi.
        This affects where all subsequent Field allocations and kernel runs occur.

        Parameters:
            device (str | ti.cpu | ti.cuda): Target device, either as string 'cpu'/'cuda'
                or the Taichi arch object ti.cpu/ti.cuda.

        Raises:
            ValueError: If the provided device string is unsupported.
        """
        # Accept string aliases and convert to Taichi arch
        if isinstance(device, str):
            if device.lower() == "cpu":
                device = ti.cpu
            elif device.lower() == "cuda":
                device = ti.cuda
            else:
                raise ValueError(f"Unsupported device string: {device}")

        # Initialize Taichi runtime (ignored if already initialized)
        try:
            ti.init(arch=device, default_ip=ti.i32, default_fp=ti.f64)
        except Exception:
            # A subsequent init call may throw; safely ignore
            pass

        # Store the chosen device for future reference
        TaichiBackend._device = device

    @staticmethod
    def device_type(field: ti.Field, /):  # TODO
        arch = ti.cfg.arch
        device = "cpu" if arch == ti.cpu else "cuda" if arch == ti.cuda else str(arch)
        return device

    @staticmethod
    def to_numpy(field: ti.Field, /) -> np.ndarray:
        """
        Convert a Taichi Field into a NumPy ndarray.

        Parameters:
            field (ti.Field): The Taichi field to convert.

        Returns:
            np.ndarray: A NumPy array containing the field data.
        """
        return field.to_numpy()

    @staticmethod
    def from_numpy(ndarray: np.ndarray, /) -> ti.Field:
        """
        Creates a Taichi field from a NumPy ndarray.

        This static method converts a NumPy array into a Taichi field with compatible data types.
        It automatically maps the NumPy data type to the corresponding Taichi data type and initializes
        the field with the values from the input array.

        Args:
            ndarray (np.ndarray): 
                The input NumPy array to be converted. This must be a valid NumPy ndarray.

        Returns:
            ti.Field:
                A Taichi field with the same shape and data as the input NumPy array.
                The data type of the field is automatically determined based on the input array's dtype.

        Raises:
            KeyError: If the NumPy array's data type cannot be mapped to a supported Taichi data type.
            ValueError: If the input is not a valid NumPy ndarray.

        Notes:
            - The mapping from NumPy to Taichi data types is defined in the `dtype_map` dictionary.
            - The resulting Taichi field will be stored on the same device as specified by the current Taichi context.

        Example:
            # Convert a 1D NumPy array to a Taichi field
            numpy_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            taichi_field = MyClass.from_numpy(numpy_array)
        """
        field = ti.field(dtype=dtype_map[ndarray.dtype], shape=ndarray.shape)
        field.from_numpy(ndarray)
        return field

    @staticmethod
    def tolist(field: ti.Field, /) -> list:
        """
        Converts a Taichi field to a nested Python list.

        This static method recursively traverses a Taichi field of any dimension
        and converts its elements into a nested list structure. The conversion
        preserves the original shape and element order of the field.

        Args:
            field (ti.Field): 
                The Taichi field to be converted. Can be a scalar or multi-dimensional field.

        Returns:
            list:
                A nested list containing all elements of the Taichi field.
                For a 0D field, returns a list with a single element.
                For higher dimensions, returns nested lists representing the field's shape.

        Raises:
            AttributeError: If the input is not a valid Taichi field with shape and element access.
            Exception: Propagates any other exceptions that occur during element access.

        Notes:
            - The conversion is performed recursively, which may cause performance issues for very large fields.
            - The returned list structure follows standard Python indexing conventions (e.g., row-major order for 2D fields).

        Example:
            # Convert a 2D Taichi field to a nested list
            field = ti.field(dtype=ti.f32, shape=(2, 3))
            field[0, 0] = 1.0
            field[0, 1] = 2.0
            # ... (set other values)
            nested_list = MyClass.tolist(field)
            # Result: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        """
        if field is None:
            return []
        try:
            shape = field.shape

            def rec(idx, dim):
                if dim == len(shape):
                    return field[tuple(idx)]
                return [rec(idx + [i], dim + 1) for i in range(shape[dim])]

            if len(shape) == 0:
                return [field[None]]
            return rec([], 0)
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    @staticmethod
    def arange(*args, dtype=ti.f64) -> ti.Field: #TODO @ti.kernel
        """
        Generates a Taichi field containing a sequence of evenly spaced values.

        This static method mimics the functionality of NumPy's `arange`, producing
        a Taichi field with values starting from `start`, ending before `stop`,
        and incremented by `step`. The output field's data type can be specified.

        Args:
            *args: 
                - 1 argument: `stop` (int/float)
                - 2 arguments: `start` (int/float), `stop` (int/float)
                - 3 arguments: `start` (int/float), `stop` (int/float), `step` (int/float)

            dtype (ti.dtype, optional): 
                Data type of the generated field (default: ti.f64).

        Returns:
            ti.Field: 
                A 1D Taichi field containing the generated sequence.
                Returns an empty list if no valid elements are in the range.

        Raises:
            ValueError: 
                - If `stop` is not provided when using 1 argument.
                - If `step` is zero.
                - If invalid argument counts are provided.
                - If the range is invalid (e.g., start >= stop with positive step).

        Notes:
            - The generated values are computed as `start + i * step` for `i` in `[0, n)`,
            where `n` is the number of elements determined by the range and step.
            - This method is currently implemented in Python and not yet optimized with Taichi kernels,
             in order to ensure the accuracy of the floating-point number in the output
            (see TODO annotation).

        Examples:
            # Generate field with stop value
            >>> MyClass.arange(5)
            [0.0, 1.0, 2.0, 3.0, 4.0]

            # Generate field with start and stop
            >>> MyClass.arange(2, 7)
            [2.0, 3.0, 4.0, 5.0, 6.0]

            # Generate field with start, stop, and step
            >>> MyClass.arange(1, 10, 2)
            [1.0, 3.0, 5.0, 7.0, 9.0]

            # Generate field with negative step
            >>> MyClass.arange(5, 0, -1)
            [5.0, 4.0, 3.0, 2.0, 1.0]
        """
        if len(args) == 1:
            if args[0] is None:
                raise ValueError("arange() requires stop to be specified.")
            if args[0] <= 0:
                return []
            if args[0] > 0:
                start, stop, step = 0, args[0], 1
        elif len(args) == 2:
            a, b = args
            if a >= b:
                return []
            else:
                start, stop, step = a, b, 1
        elif len(args) == 3:
            a, b, c = args
            if c == 0:
                raise ValueError("step must not be zero")
            else:
                if a >= b and c > 0:
                    return []
                if a <= b and c < 0:
                    return []
                else:
                    start, stop, step = a, b, c
        else:
            raise ValueError(
                "arange expects 1~3 arguments (stop | start, stop | start, stop, step)"
            )

        n = max(1, int(abs((stop - start + (step - (1 if step > 0 else -1))) // step)))

        last = start + n * step
        if step > 0:
            while last < stop:
                n += 1
                last += step
    
        if step < 0:
            while last > stop:
                n += 1
                last += step
            
        field = ti.field(dtype=dtype, shape=n)

        for i in range(n):
            field[i] = start + i * step

        return field

    @staticmethod
    def eye(N, M=None, k :int =0, dtype=ti.f64) -> ti.Field:
        """
        Creates a Taichi field representing a 2D identity matrix with specified dimensions and diagonal offset.

        This static method generates a 2D Taichi field with ones on the specified diagonal and zeros elsewhere,
        similar to NumPy's `eye` function. The main diagonal is defined by default, but can be shifted using the `k` parameter.

        Args:
            N (int): 
                Number of rows in the output matrix. Must be a positive integer.
            M (int, optional): 
                Number of columns in the output matrix. If None (default), defaults to `N`.
            k (int, optional): 
                Index of the diagonal:
                - k=0 (default): main diagonal
                - k>0: upper diagonal
                - k<0: lower diagonal
            dtype (ti.dtype, optional): 
                Data type of the output field (default: ti.f64).

        Returns:
            ti.Field: 
                A 2D Taichi field of shape (N, M) with ones on the specified diagonal and zeros elsewhere.
                Returns an empty list if either `N` or `M` is zero.

        Raises:
            ValueError: 
                - If both `N` and `M` are None.
                - If `N` is None.
                - If `N` or `M` are negative.
                - If `N` or `M` are non-integer floats.
                - If `k` is a non-integer float.
            TypeError: 
                - If `N` or `M` are not integers or integer-convertible floats.

        Notes:
            - The diagonal is determined by the formula `field[i, i + k] = 1` for valid indices `i`.
            - The field is filled using a Taichi kernel for efficient parallel computation.

        Examples:
            # 3x3 identity matrix (default)
            >>> MyClass.eye(3)
            [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]

            # 3x4 matrix with main diagonal
            >>> MyClass.eye(3, 4)
            [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]]

            # 3x3 matrix with upper diagonal (k=1)
            >>> MyClass.eye(3, k=1)
            [[0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]]

            # 3x3 matrix with lower diagonal (k=-1)
            >>> MyClass.eye(3, k=-1)
            [[0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]]
        """
        if N is None and M is None:
            raise ValueError(
                "Both N and M are None. At least one dimension must be specified for eye()."
            )
        if N is None:
            raise ValueError(
                "N is None. The number of rows must be specified for eye()."
            )
        if M is None:
            M = N

        for name, v in [("N", N), ("M", M)]:
            if isinstance(v, float):
                if not v.is_integer():
                    raise TypeError(f"{name} must be an integer, got {v}.")
        N = int(N)
        M = int(M)

        if N == 0 or M == 0:
            return []
        if N < 0 or M < 0:
            raise ValueError(f"N and M must be positive integers, got N={N}, M={M}.")
        field = ti.field(dtype=dtype, shape=(N, M))

        @ti.kernel
        def fill_eye():
            field.fill(0)  
            for i in range(max(0, -k), min(N, M - k)):  
                field[i, i + k] = 1

        fill_eye()
        return field

    @staticmethod
    def zeros(shape: Union[int, ti.Field], dtype=ti.f64) -> ti.Field:
        """
        Creates a Taichi field filled with zeros.

        This static method generates a Taichi field of specified shape and data type,
        initialized with zeros. The shape can be either a single integer (for a 1D field)
        or a tuple of integers (for multi-dimensional fields).

        Args:
            shape (Union[int, tuple]): 
                The shape of the output field. 
                - If an integer, creates a 1D field with that length.
                - If a tuple, each element defines the size of the corresponding dimension.
            dtype (ti.dtype, optional): 
                The data type of the field (default: ti.f64).

        Returns:
            ti.Field: 
                A Taichi field of the specified shape and data type, filled with zeros.
                Returns an empty list if the shape is a single integer 0.

        Raises:
            ValueError: 
                - If a single integer shape is negative.
                - If any dimension in a tuple shape is zero (Taichi does not support zero-sized dimensions).
            TypeError: 
                - If tuple shape elements are not integers or integer-convertible floats.

        Notes:
            - Floats in tuple shapes are converted to integers if they are whole numbers (e.g., 3.0 → 3).
            - Taichi does not support fields with zero-sized dimensions (e.g., (0, 3) is invalid).

        Examples:
            # 1D field with length 5
            >>> MyClass.zeros(5)
            [0.0, 0.0, 0.0, 0.0, 0.0]

            # 2D field with shape (2, 3)
            >>> MyClass.zeros((2, 3))
            [[0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]]

            # 3D field with shape (2, 2, 2)
            >>> MyClass.zeros((2, 2, 2))
            [[[0.0, 0.0],
            [0.0, 0.0]],
            [[0.0, 0.0],
            [0.0, 0.0]]]
        """
        if isinstance(shape, int):
            if shape == 0:
                return []
            if shape < 0:
                raise ValueError(f"Shape must be a non-negative integer, got {shape}.")
        if isinstance(shape, tuple):
            try:
                new_shape = [
                    int(s) if isinstance(s, float) and s.is_integer() else 
                    (s if isinstance(s, int) else _raise_type_error(s))
                    for s in shape
                ]
            except TypeError as e:
                raise e
            def _raise_type_error(s):
                raise TypeError(f"Shape elements must be integers, got {type(s).__name__} {s}.")
            shape = tuple(new_shape)
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
        field = ti.field(dtype=dtype, shape=shape)

        return field

    @staticmethod
    def zeros_like(field: ti.Field) -> ti.Field:
        """
        Creates a Taichi field of zeros with the same shape and data type as the input field.

        This static method generates a new Taichi field filled with zeros, mirroring the 
        dimensions and data type of the provided input field. The output field will be 
        compatible with operations requiring the same shape and data type as the input.

        Args:
            field (ti.Field): 
                The input Taichi field whose shape and data type will be replicated.

        Returns:
            ti.Field: 
                A new Taichi field with the same shape and data type as the input field,
                initialized with zeros.

        Raises:
            ValueError: 
                - If the input field is None.
                - If the input field's shape contains any zero dimensions (Taichi does not support zero-sized fields).
            TypeError: 
                - If the input does not have 'shape' or 'dtype' attributes (i.e., is not a valid Taichi field).

        Notes:
            - The output field is always a new instance, even if the input field is already zero-filled.
            - Taichi does not support fields with zero-sized dimensions (e.g., (0, 3) is invalid).

        Example:
            # Create an input field
            input_field = ti.field(dtype=ti.f32, shape=(2, 3))
            input_field[0, 0] = 1.0
            # ... (set other values)

            # Create zeros_like field
            zeros_field = MyClass.zeros_like(input_field)
            # zeros_field is now [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] 
            # with dtype ti.f32 and shape (2, 3)
        """
        if field is None:
            raise ValueError(
                "Input field is None. Please provide a valid Taichi field."
            )
        if not hasattr(field, "shape") or not hasattr(field, "dtype"):
            raise TypeError(
                "Input is not a valid Taichi field: missing 'shape' or 'dtype' attribute."
            )
        shape = field.shape
        if any(s == 0 for s in shape):
            raise ValueError(
                f"Input field has zero in its shape {shape}, which is not supported by Taichi."
            )
        out = ti.field(dtype=field.dtype, shape=shape)

        return out

    @staticmethod
    def tril(x: Union[ti.Field, list], k: int = 0) -> ti.Field:
        """
        Returns the lower triangular part of a matrix or a 1D array converted to a matrix.

        This method extracts the lower triangular portion of a matrix (2D or higher) or 
        constructs a lower triangular matrix from a 1D array. The lower triangular part 
        includes all elements on or below the specified diagonal.

        Args:
            x (Union[ti.Field, list]): 
                Input data. Can be:
                - A 1D Taichi field or list: Converted to a square matrix where each row is a copy of the array.
                - A 2D Taichi field: Treated as a matrix.
                - A Taichi field with >2 dimensions: The last two dimensions are treated as rows and columns.
            k (int, optional): 
                Diagonal offset. Defaults to 0 (main diagonal).
                - k=0: Main diagonal and below.
                - k>0: Diagonals above the main diagonal.
                - k<0: Diagonals below the main diagonal.

        Returns:
            ti.Field: 
                Lower triangular part of the input. 
                - If input is 1D: Returns an M×M matrix (M = length of input).
                - If input is 2D or higher: Returns a field of the same shape with upper triangular elements zeroed.

        Raises:
            ValueError: 
                - If input field is None.
                - If input field has any zero dimensions.
            TypeError: 
                - If input is not a Taichi field or list.
                - If input is a list with non-numeric elements.

        Notes:
            - For 1D inputs, each row of the output matrix is a copy of the input array.
            - For ND inputs (N>2), the operation is applied to the last two dimensions.
            - The diagonal offset `k` shifts the triangular region up or down.

        Examples:
            # 1D array input
            >>> MyClass.tril([1, 2, 3])
            [[1, 0, 0],
            [1, 2, 0],
            [1, 2, 3]]

            # 2D matrix input
            >>> matrix = ti.field(dtype=ti.f32, shape=(3, 3))
            >>> # Initialize matrix to [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            >>> MyClass.tril(matrix)
            [[1, 0, 0],
            [4, 5, 0],
            [7, 8, 9]]

            # Diagonal offset (k=1)
            >>> MyClass.tril(matrix, k=1)
            [[1, 2, 0],
            [4, 5, 6],
            [7, 8, 9]]
        """
        if isinstance(x, list):
            M = len(x)
            if M == 0:
                return []
            x_field = ti.field(dtype=ti.f64, shape=(M,))
            for i in range(M):
                x_field[i] = x[i]
            out = ti.field(dtype=ti.f64, shape=(M, M))
            @ti.kernel
            def fill_matrix():
                out.fill(0) 
                for i, j in ti.ndrange(M, M):  
                    if j - i <= k:
                        out[i, j] = x_field[j]  
            fill_matrix()
            return out
        if isinstance(x, ti.Field):
            field = x
            if field is None:
                raise ValueError("Input field is None. Please provide a valid Taichi field.")
            shape = field.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
            dtype = field.dtype
            
            if len(shape) == 1:
                M = shape[0]
                out = ti.field(dtype=dtype, shape=(M, M))
                @ti.kernel
                def fill_tril_1d():
                    out.fill(0) 
                    for i, j in ti.ndrange(M, M):  
                        if j - i <= k:
                            out[i, j] = field[j] 
                fill_tril_1d()
                return out

            elif len(shape) > 1:
                out = ti.field(dtype=dtype, shape=shape)
                n_dim = len(shape)

                @ti.kernel
                def fill_tril_nd():
                    for I in ti.grouped(out):
                        i = I[n_dim - 2]
                        j = I[n_dim - 1]
                        if j - i <= k:
                            out[I] = field[I]
                        else:
                            out[I] = 0
                fill_tril_nd()
                return out
        else:
            raise TypeError("Unsupported type for tril: {type(x)}. Expected ti.Field or list.").format(type(x)) 
            
    @staticmethod
    def abs(
        x: Union[int, float, ti.Field]
    ) -> Union[int, float, ti.Field]:
        """
        Computes the absolute value element-wise for Taichi fields or scalar values.

        This static method calculates the absolute value of input data, supporting both
        scalar values (integers, floats, booleans) and Taichi fields. For fields, the
        operation is performed element-wise, returning a new field with the same shape
        and data type.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field. Can be:
                - A scalar (int, float, bool).
                - A Taichi field of numerical values.

        Returns:
            Union[int, float, ti.Field]:
                - If the input is a scalar, returns the absolute value as a scalar.
                - If the input is a Taichi field, returns a new field containing the
                absolute values of the original elements. For a 0D field, returns the
                scalar value directly.

        Raises:
            ValueError: 
                - If the input field is None.
                - If the input field has any zero dimensions (Taichi does not support zero-sized fields).
            TypeError: 
                - If the input is not a scalar or a valid Taichi field.
                - If the input field lacks 'shape' or 'dtype' attributes.

        Notes:
            - The absolute value is computed using Taichi's `ti.abs` for fields, ensuring
            compatibility with Taichi's data types and backends.
            - For boolean inputs, False is converted to 0 and True to 1 before computing
            the absolute value.

        Examples:
            # Scalar usage:
            >>> MyClass.abs(-5)
            5
            >>> MyClass.abs(3.14)
            3.14
            >>> MyClass.abs(True)
            1

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([-1.0, 2.0, -3.0], dtype=np.float32))
            >>> result = MyClass.abs(x)
            # result is now [1.0, 2.0, 3.0]
        """
        if isinstance(x, (int, float, bool)):
            return abs(x)
        if isinstance(x, ti.Field):
            if x is None:
                raise ValueError(
                    "Input field is None. Please provide a valid Taichi field."
                )
            if not hasattr(x, "shape") or not hasattr(x, "dtype"):
                raise TypeError(
                    "Input is not a valid Taichi field: missing 'shape' or 'dtype' attribute."
                )
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(
                    f"Input field has zero in its shape {shape}, which is not supported by Taichi."
                )
            dtype = x.dtype
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_abs():
                for I in ti.grouped(x):
                    out[I] = ti.abs(x[I])

            fill_abs()
            if len(shape) == 0:
                return out[None]
            else:
                return out
        else:
            raise TypeError(
                f"Unsupported type for abs: {type(x)}. Expected int, float, or ti.Field."
            )

    @staticmethod
    def acos(
        x: Union[int, float, ti.Field]
    ) -> Union[float, ti.Field]:
        """
        Computes the inverse cosine (acos) element-wise for scalar values or Taichi fields.

        This static method calculates the inverse cosine of input data, supporting both scalar values
        (integers or floats) and Taichi fields. For scalar inputs, it returns the result as a float.
        For Taichi fields, the operation is performed element-wise, producing a new field with the same shape.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the inverse cosine.
                - A Taichi field: Computes inverse cosine for each element.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the inverse cosine as a float.
                - If input is a Taichi field, returns a new field with the same shape containing element-wise inverse cosines.
                For 0D fields, returns the scalar value directly.

        Raises:
            ValueError: 
                If the input Taichi field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - Integral dtype fields (e.g., int) are automatically converted to Taichi's default floating-point dtype
            since the inverse cosine result is a floating-point value.
            - Uses Taichi's `ti.acos` function for computation, ensuring compatibility with Taichi's backends and data types.
            - Input values should typically be in the range [-1, 1] to produce real results (out-of-range values may return NaN).

        Examples:
            # Scalar usage:
            >>> MyClass.acos(1.0)
            0.0
            >>> MyClass.acos(0.0)
            1.5707963267948966  # π/2

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([1.0, 0.0, -1.0], dtype=np.float32))
            >>> result = MyClass.acos(x)
            # result contains [0.0, π/2, π] (element-wise)
        """
        if isinstance(x, (int, float)):         
            return ti.acos(float(x))
        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in shape {shape}, not supported by Taichi.")
                
            dtype = x.dtype
            if ti.types.is_integral(dtype):
                dtype = ti.get_default_fp()
                
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_acos(
                field: ti.template(), 
                out: ti.template()
            ):
                for I in ti.grouped(field):
                    val = field[I]
                    out[I] = ti.acos(val)

            fill_acos(x, out)
                
            return out[None] if len(shape) == 0 else out
        
    @staticmethod
    def asin(
        x: Union[int, float, ti.Field]
    ) -> Union[float, ti.Field]:
        """
        Computes the inverse sine (asin) element-wise for scalar values or Taichi fields.

        This static method calculates the inverse sine of input data, supporting both scalar values
        (integers or floats) and Taichi fields. For scalar inputs, it returns the result as a float.
        For Taichi fields, the operation is performed element-wise, producing a new field with the same shape.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the inverse sine.
                - A Taichi field: Computes inverse sine for each element.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the inverse sine as a float.
                - If input is a Taichi field, returns a new field with the same shape containing element-wise inverse sines.
                For 0D fields, returns the scalar value directly.

        Raises:
            ValueError: 
                If the input Taichi field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - Integral dtype fields (e.g., int) are automatically converted to Taichi's default floating-point dtype
            since the inverse sine result is a floating-point value.
            - Uses Taichi's `ti.asin` function for computation, ensuring compatibility with Taichi's backends.
            - Input values should typically be in the range [-1, 1] to produce real results (out-of-range values may return NaN).

        Examples:
            # Scalar usage:
            >>> MyClass.asin(0.0)
            0.0
            >>> MyClass.asin(1.0)
            1.5707963267948966  # π/2

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
            >>> result = MyClass.asin(x)
            # result contains [-π/2, 0.0, π/2] (element-wise)
        """
        if isinstance(x, (int, float)):         
            return ti.asin(float(x))
        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in shape {shape}, not supported by Taichi.")
                
            dtype = x.dtype
            if ti.types.is_integral(dtype):
                dtype = ti.get_default_fp()
                
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_asin(
                field: ti.template(), 
                out: ti.template()
            ):

                for I in ti.grouped(field):
                    val = field[I]
                    out[I] = ti.asin(val)

            fill_asin(x, out)
                
            return out[None] if len(shape) == 0 else out
        
    @staticmethod
    def atan(
        x: Union[int, float, ti.Field]
    ) -> Union[float, ti.Field]:
        """
        Computes the inverse tangent (atan) element-wise for scalar values or Taichi fields.

        This static method calculates the inverse tangent of input data, supporting both scalar values
        (integers or floats) and Taichi fields. For scalar inputs, it returns the result as a float.
        For Taichi fields, the operation is performed element-wise, producing a new field with the same shape.
        The inverse tangent is computed using `ti.atan2(x, 1)`, which is equivalent to `atan(x)`.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the inverse tangent.
                - A Taichi field: Computes inverse tangent for each element.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the inverse tangent as a float.
                - If input is a Taichi field, returns a new field with the same shape containing element-wise inverse tangents.
                For 0D fields, returns the scalar value directly.

        Raises:
            ValueError: 
                If the input Taichi field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - Inverse tangent is computed using `ti.atan2(x, 1)`, which is mathematically equivalent to `atan(x)` 
            since `atan2(y, x)` evaluates to `atan(y/x)` for positive `x`.
            - Integral dtype fields (e.g., int) are automatically converted to Taichi's default floating-point dtype 
            because the inverse tangent result is a floating-point value.
            - The operation preserves the shape of the input field for multi-dimensional inputs.

        Examples:
            # Scalar usage:
            >>> MyClass.atan(0.0)
            0.0
            >>> MyClass.atan(1.0)
            0.7853981633974483  # π/4

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
            >>> result = MyClass.atan(x)
            # result contains [-π/4, 0.0, π/4] (element-wise)
        """
        if isinstance(x, (int, float)):         
            return ti.atan2(float(x),1)
        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in shape {shape}, not supported by Taichi.")
                
            dtype = x.dtype
            if ti.types.is_integral(dtype):
                dtype = ti.get_default_fp()
                
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_atan(
                field: ti.template(), 
                out: ti.template(), 
            ):
                for I in ti.grouped(field):
                    val = field[I]
                    out[I] = ti.atan2(val,1)

            fill_atan(x, out)
                
            return out[None] if len(shape) == 0 else out
        
    @staticmethod
    def atan2( 
        y: Union[int, float, ti.Field],
        x: Union[int, float, ti.Field]
    ) -> Union[float, ti.Field]:
        """
        Computes the element-wise arctangent of y/x using the signs of both arguments to determine the quadrant.

        This static method calculates the four-quadrant arctangent of the quotient y/x, supporting both scalar values
        and Taichi fields. The result is an angle in radians between -π and π, representing the direction from the origin
        to the point (x, y).

        Args:
            y (Union[int, float, ti.Field]): 
                The numerator value(s). Can be a scalar or a Taichi field.
            x (Union[int, float, ti.Field]): 
                The denominator value(s). Can be a scalar or a Taichi field.

        Returns:
            Union[float, ti.Field]:
                - If both inputs are scalars, returns a scalar float.
                - If either input is a field, returns a Taichi field of the same shape containing element-wise results.
                For 0D fields, returns the scalar value directly.

        Raises:
            ValueError: 
                - If input fields have different shapes.
                - If any dimension of the input fields is zero (Taichi does not support zero-sized fields).
            ZeroDivisionError: 
                - If both x and y are zero for any element (results in NaN).

        Notes:
            - Scalar inputs are automatically converted to 0D Taichi fields for computation.
            - Integral dtype fields are converted to Taichi's default floating-point dtype (ti.f64) to preserve precision.
            - Uses Taichi's `ti.atan2` for computation, ensuring compatibility with Taichi's backends.
            - Handles special cases:
                - x=0, y>0 → +π/2
                - x=0, y<0 → -π/2
                - x=0, y=0 → NaN (using ti.math.nan)
                - x<0, y=0 → ±π

        Examples:
            # Scalar usage:
            >>> MyClass.atan2(1.0, 1.0)
            0.7853981633974483  # π/4
            >>> MyClass.atan2(1.0, -1.0)
            2.356194490192345   # 3π/4

            # Field usage:
            >>> y = ti.field(dtype=ti.f32, shape=(2,))
            >>> y.from_numpy(np.array([1.0, -1.0], dtype=np.float32))
            >>> x = ti.field(dtype=ti.f32, shape=(2,))
            >>> x.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
            >>> result = MyClass.atan2(y, x)
            # result contains [π/4, -π/4] (element-wise)
        """
        if isinstance(x, (int, float)):
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
        if isinstance(y, (int, float)):
            b = ti.field(dtype=ti.f64, shape=())
            b[None] = float(y)
            y = b
        shape = y.shape

        if shape != x.shape:
            raise ValueError("Input fields must have the same shape.")
        if any(s == 0 for s in shape):
            raise ValueError(
                f"Input field has zero in its shape {shape}, which is not supported by Taichi."
            )
        dtype = y.dtype
        if ti.types.is_integral(dtype):
            dtype = ti.get_default_fp()
        out = ti.field(dtype=dtype, shape=shape)

        @ti.kernel
        def fill_atan2(
            y_field: ti.template(), x_field: ti.template(), out: ti.template()
        ):
            for I in ti.grouped(y_field):
                if x_field[I] == 0 and y_field[I] == 0:
                    out[I] = ti.math.nan
                else:
                    out[I] = ti.atan2(y_field[I], x_field[I])

        fill_atan2(y, x, out)
        if len(shape) == 0:
            return out[None]
        return out
    
    @staticmethod
    def ceil(x: Union[int, float,  ti.Field]) -> Union[int, float, ti.Field]: 
        """
        Computes the ceiling value element-wise for scalar values or Taichi fields.


        This static method calculates the smallest integer greater than or equal to the input value(s).
        For scalar inputs, it returns the result directly. For Taichi fields, the operation is performed
        element-wise, producing a new field with the same shape as the input.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field. Can be:
                - A scalar (int or float): Computes the ceiling of the single value.
                - A Taichi field: Computes the ceiling for each element individually.

        Returns:
            Union[int, float, ti.Field]:
                - If input is a scalar, returns the ceiling value as an int (if input is integral) or float (if input is fractional).
                - If input is a Taichi field, returns a new field with the same shape containing element-wise ceiling values.
                For 0D fields, returns the scalar ceiling value directly.

        Raises:
            ValueError: 
                If the input Taichi field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - Integral dtype fields (e.g., int) are automatically converted to Taichi's default floating-point dtype
            since the ceiling operation on integers typically retains integer values but is represented as a float.
            - Uses Taichi's `ti.ceil` function for computation, ensuring compatibility with Taichi's backends and data types.
            - The ceiling of a non-integer float is the smallest integer greater than the input (e.g., ceil(2.3) = 3.0).

        Examples:
            # Scalar usage:
            >>> MyClass.ceil(2.3)
            3.0
            >>> MyClass.ceil(-1.8)
            -1.0

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([1.2, 2.7, -3.1], dtype=np.float32))
            >>> result = MyClass.ceil(x)
            # result contains [2.0, 3.0, -3.0] (element-wise)
        """
        if isinstance(x, (int, float)):         
            return ti.ceil(float(x))
        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
            dtype = x.dtype
            if ti.types.is_integral(dtype):
                dtype = ti.get_default_fp()
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_ceil(field: ti.template(), out: ti.template()):
                for I in ti.grouped(field):
                    out[I] = ti.ceil(field[I])

            fill_ceil(x, out)
            if len(shape) == 0:
                return out[None]
            return out
        
    @staticmethod
    def clip(
        x: Union[int, float, ti.Field],
        *args, 
        **kwargs  
    ) -> Union[int, float, ti.Field]:
        """
        Clips values of a scalar or Taichi field to a specified range [min_val, max_val].

        This static method restricts input values to lie within a given range. Values below min_val are set to min_val,
        and values above max_val are set to max_val. Both scalar inputs and Taichi fields are supported, with element-wise
        clipping for fields.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field to be clipped. Can be:
                - A scalar (int or float): Clipped to the specified range.
                - A Taichi field: Each element is clipped individually.
            *args: 
                Positional arguments specifying the clipping range:
                - 1 argument: (min_val) → clips values below min_val.
                - 2 arguments: (min_val, max_val) → clips values to [min_val, max_val].
            **kwargs: 
                Keyword arguments specifying the clipping range:
                - min: Minimum value (alternative to positional min_val).
                - max: Maximum value (alternative to positional max_val).

        Returns:
            Union[int, float, ti.Field]:
                - If input is a scalar, returns the clipped scalar value.
                - If input is a Taichi field, returns a new field with the same shape containing element-wise clipped values.
                For 0D fields, returns the clipped scalar value directly.

        Raises:
            TypeError: 
                - If more than 3 positional arguments are provided.
                - If min_val or max_val is specified via both positional and keyword arguments.
                - If unexpected keyword arguments are provided.
                - If input type is not int, float, or ti.Field.
            ValueError: 
                If the input Taichi field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - min_val and max_val can be omitted (e.g., only min_val to clip lower bounds, or only max_val to clip upper bounds).
            - Integral dtype fields (e.g., int) are handled by converting min_val/max_val to integers to preserve type consistency.
            - Uses a Taichi kernel for field operations, ensuring efficient element-wise clipping.

        Examples:
            # Scalar usage:
            >>> MyClass.clip(3.5, 0, 5)  # Clip to [0, 5]
            3.5
            >>> MyClass.clip(-2, min=0)  # Clip values below 0
            0
            >>> MyClass.clip(7, max=5)   # Clip values above 5
            5

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([-1.2, 3.7, 6.1], dtype=np.float32))
            >>> result = MyClass.clip(x, 0, 5)  # Clip to [0, 5]
            # result contains [0.0, 3.7, 5.0] (element-wise)
        """
        min_val = None
        max_val = None

        if len(args) > 2:
            raise TypeError(
                f"clip() takes at most 3 positional arguments ({len(args)} given)"
            )
        elif len(args) >= 1:
            min_val = args[0]
        if len(args) == 2:
            max_val = args[1]

        if "min" in kwargs:
            if min_val is not None:
                raise TypeError("min specified both as positional and keyword argument")
            min_val = kwargs.pop("min")

        if "max" in kwargs:
            if max_val is not None:
                raise TypeError("max specified both as positional and keyword argument")
            max_val = kwargs.pop("max")

        if kwargs:
            raise TypeError(
                f"clip() got unexpected keyword arguments: {list(kwargs.keys())}"
            )

        if isinstance(x, (int, float)):
            x = float(x)
            if min_val is not None:
                min_val = float(min_val)
                x = max(min_val, x)
            if max_val is not None:
                max_val = float(max_val)
                x = min(x, max_val)
            return x

        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(
                    f"Input field has zero in its shape {shape}, which is not supported by Taichi."
                )
            dtype = x.dtype
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_clip(
                field: ti.template(),
                out: ti.template(),
                min_val: ti.f64,
                max_val: ti.f64,
                use_min: ti.i32,
                use_max: ti.i32,
            ):
                for I in ti.grouped(field):
                    v = field[I]
                    if use_min:
                        v = max(min_val, v)
                    if use_max:
                        v = min(v, max_val)
                    out[I] = v

            use_min = int(min_val is not None)
            use_max = int(max_val is not None)

            if dtype in (ti.f32, ti.f64):
                min_converted = float(min_val) if min_val is not None else 0.0
                max_converted = float(max_val) if max_val is not None else 0.0
            elif dtype in (ti.i32, ti.i64):
                min_converted = int(min_val) if min_val is not None else 0
                max_converted = int(max_val) if max_val is not None else 0
            else:
                min_converted = float(min_val) if min_val is not None else 0.0
                max_converted = float(max_val) if max_val is not None else 0.0

            fill_clip(x, out, min_converted, max_converted, use_min, use_max)

            if len(shape) == 0:
                return out[None]
            return out

        raise TypeError(
            f"Unsupported type for clip: {type(x)}. Expected int, float, bool, or ti.Field."
        )

    @staticmethod
    def cos(x: Union[int, float,  ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the cosine element-wise for scalar values or Taichi fields.

        This static method calculates the cosine of input data, supporting both scalar values
        (integers or floats) and Taichi fields. For scalar inputs, it returns the result as a float.
        For Taichi fields, the operation is performed element-wise, producing a new field with the same shape.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the cosine.
                - A Taichi field: Computes cosine for each element individually.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the cosine value as a float.
                - If input is a Taichi field, returns a new field with the same shape containing element-wise cosine values.
                For 0D fields, returns the scalar cosine value directly.

        Raises:
            ValueError: 
                If the input Taichi field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - Integral dtype fields (e.g., int) are automatically converted to Taichi's default floating-point dtype
            since the cosine operation produces floating-point results.
            - Uses Taichi's `ti.cos` function for computation, ensuring compatibility with Taichi's backends and data types.
            - Input values are treated as angles in radians (consistent with standard mathematical conventions).

        Examples:
            # Scalar usage:
            >>> MyClass.cos(0.0)  # Cosine of 0 radians
            1.0
            >>> MyClass.cos(ti.math.pi)  # Cosine of π radians
            -1.0

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([0.0, ti.math.pi/2, ti.math.pi], dtype=np.float32))
            >>> result = MyClass.cos(x)
            # result contains [1.0, 0.0, -1.0] (element-wise)
        """
        if isinstance(x, (int, float)):         
            return ti.cos(float(x))
        if isinstance(x, ti.Field):
            shape =x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
            dtype = x.dtype
            if ti.types.is_integral(dtype):
                dtype = ti.get_default_fp()
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_cos(field: ti.template(), out: ti.template()):
                for I in ti.grouped(field):
                    out[I] = ti.cos(field[I])

            fill_cos(x, out)
            if len(shape) == 0:
                return out[None]
            return out

    @staticmethod
    def cosh(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the hyperbolic cosine element-wise for scalar values or Taichi fields.

        This static method calculates the hyperbolic cosine of input data, supporting both scalar values
        (integers or floats) and Taichi fields. For scalar inputs, it returns the result as a float.
        For Taichi fields, the operation is performed element-wise, producing a new field with the same shape.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the hyperbolic cosine.
                - A Taichi field: Computes hyperbolic cosine for each element individually.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the hyperbolic cosine value as a float.
                - If input is a Taichi field, returns a new field with the same shape containing element-wise hyperbolic cosine values.
                For 0D fields, returns the scalar hyperbolic cosine value directly.

        Raises:
            ValueError: 
                If the input Taichi field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - Integral dtype fields (e.g., int) are automatically converted to Taichi's default floating-point dtype
            since the hyperbolic cosine operation produces floating-point results.
            - The hyperbolic cosine is computed using the formula: cosh(x) = (exp(x) + exp(-x)) / 2.
            - Uses Taichi's `ti.exp` function for exponential calculations, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage:
            >>> MyClass.cosh(0.0)  # Hyperbolic cosine of 0
            1.0
            >>> MyClass.cosh(1.0)  # Hyperbolic cosine of 1
            1.5430806348152437

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([0.0, 1.0, -1.0], dtype=np.float32))
            >>> result = MyClass.cosh(x)
            # result contains [1.0, 1.5430806, 1.5430806] (element-wise)
        """
        if isinstance(x, (int, float)):         
            return (ti.exp(float(x)) + ti.exp(-float(x))) * 0.5
        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
            dtype = x.dtype
            if ti.types.is_integral(dtype):
                dtype = ti.get_default_fp()
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_cosh(field: ti.template(), out: ti.template()):
                for I in ti.grouped(field):
                    out[I] = (ti.exp(field[I]) + ti.exp(-field[I])) * 0.5

            fill_cosh(x, out)
            if len(shape) == 0:
                return out[None]
            return out
        
    @staticmethod
    def floor(x: Union[int, float, ti.Field]) -> Union[int, float, ti.Field]:
        """
        Computes the floor value element-wise for scalar values or Taichi fields.

        This static method calculates the largest integer less than or equal to the input value(s).
        For scalar inputs, it returns the result directly. For Taichi fields, the operation is performed
        element-wise, producing a new field with the same shape as the input.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field. Can be:
                - A scalar (int or float): Computes the floor of the single value.
                - A Taichi field: Computes the floor for each element individually.

        Returns:
            Union[int, float, ti.Field]:
                - If input is a scalar, returns the floor value as an int (if input is integral) or float (if input is fractional).
                - If input is a Taichi field, returns a new field with the same shape containing element-wise floor values.
                For 0D fields, returns the scalar floor value directly.

        Raises:
            ValueError: 
                If the input Taichi field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - Integral dtype fields (e.g., int) are automatically converted to Taichi's default floating-point dtype
            since the floor operation on integers typically retains integer values but is represented as a float.
            - Uses Taichi's `ti.floor` function for computation, ensuring compatibility with Taichi's backends and data types.
            - The floor of a non-integer float is the largest integer less than the input (e.g., floor(2.7) = 2.0).

        Examples:
            # Scalar usage:
            >>> MyClass.floor(2.7)
            2.0
            >>> MyClass.floor(-1.3)
            -2.0

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([1.8, 2.2, -3.7], dtype=np.float32))
            >>> result = MyClass.floor(x)
            # result contains [1.0, 2.0, -4.0] (element-wise)
        """
        if isinstance(x, (int, float)):         
            return ti.floor(float(x))
        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
            dtype = x.dtype
            if ti.types.is_integral(dtype):
                dtype = ti.get_default_fp()
            out = ti.field(dtype=dtype, shape=shape)

        @ti.kernel
        def fill_floor(field: ti.template(), out: ti.template()):
            for I in ti.grouped(field):
                out[I] = ti.floor(field[I])

        fill_floor(x, out)
        if len(shape) == 0:
            return out[None]
        return out

    @staticmethod
    def floor_divide( #TODO 广播没有实现
        x: Union[int, float,ti.Field],
        y: Union[int, float,ti.Field]
    ) -> Union[int, float, ti.Field]:
        """
        Computes the element-wise floor division of two values or fields (x / y, rounded down).

        This static method performs floor division, which divides x by y and rounds the result down to the nearest integer.
        It supports scalar inputs (int/float) and Taichi fields, with special handling for division by zero and shape compatibility.
        Broadcasting is not yet implemented (see TODO).

        Args:
            x (Union[int, float, ti.Field]): 
                Dividend (numerator). Can be a scalar or Taichi field.
            y (Union[int, float, ti.Field]): 
                Divisor (denominator). Can be a scalar or Taichi field.

        Returns:
            Union[int, float, ti.Field]:
                - If inputs are scalars, returns the floor of x/y as a float.
                - If inputs are Taichi fields, returns a new field with the same shape containing element-wise floor division results.
                For 0D fields, returns the scalar result directly.

        Raises:
            ValueError: 
                - If input fields have different shapes (broadcasting not supported).
                - If input fields contain zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - Handles division by zero:
                - 0/0 → NaN
                - Positive x / 0 → +infinity
                - Negative x / 0 → -infinity
            - Integral dtype fields are converted to Taichi's default floating-point dtype to preserve precision in results.
            - TODO: Broadcasting (element-wise operations between fields/scalars of different shapes) is not yet implemented.
            - Uses Taichi's `ti.floor` for rounding down after division.

        Examples:
            # Scalar usage:
            >>> MyClass.floor_divide(7, 3)  # 7/3 = 2.333... → floor → 2.0
            2.0
            >>> MyClass.floor_divide(-7, 3)  # -7/3 = -2.333... → floor → -3.0
            -3.0

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(2,))
            >>> x.from_numpy(np.array([6.5, -5.2], dtype=np.float32))
            >>> y = ti.field(dtype=ti.f32, shape=(2,))
            >>> y.from_numpy(np.array([2.0, 2.0], dtype=np.float32))
            >>> result = MyClass.floor_divide(x, y)
            # result contains [3.0, -3.0] (element-wise)
        """
        if isinstance(x, (int, float)) and isinstance(y, ti.Field) and y.shape == ():
            if y[None]==0:
                if x==0:
                    return ti.math.nan
                if x<0:
                    return -ti.math.inf
                else:
                    return ti.math.inf
            else:
                return ti.floor(x/y[None])
        if isinstance(y, (int, float)) and isinstance(x, ti.Field) and x.shape == ():
            if y==0:
                if x[None]==0:
                    return ti.math.nan
                if x[None]<0:
                    return -ti.math.inf
                else:
                    return ti.math.inf
            else:
                return ti.floor(x[None]/y)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if y==0:
                return 0
            else:
                return ti.floor(x/y)
        if isinstance(x, ti.Field) and isinstance(y, ti.Field):
            shape = x.shape
            if shape != y.shape:
                raise ValueError("Input fields must have the same shape.")
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
            dtype = x.dtype
            if ti.types.is_integral(dtype):
                dtype = ti.get_default_fp()
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_floor_divide(x_field: ti.template(), y_field: ti.template(), out: ti.template()):
                for I in ti.grouped(y_field):
                    if x_field[I] == 0 and y_field[I] == 0:
                        out[I] = ti.math.nan
                    else:
                        out[I] = ti.floor(x_field[I] / y_field[I])
            
            fill_floor_divide(x, y, out)
            if len(shape) == 0:
                return out[None]
            return out
        
    @staticmethod
    def sin(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the element-wise sine of input values or Taichi fields.

        This static method calculates the sine of input data, supporting both scalar values (integers or floats)
        and Taichi fields. For scalar inputs, it returns the result directly. For Taichi fields, the operation
        is performed element-wise, producing a new field with the same shape as the input.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field. Can be:
                - A scalar (int or float): Treated as an angle in radians, with its sine computed directly.
                - A Taichi field: Each element is treated as an angle in radians, with sine computed individually.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the sine value as a float (range: [-1, 1]).
                - If input is a Taichi field, returns a new field with the same shape containing element-wise sine values.
                For 0D fields, returns the scalar sine value directly.

        Raises:
            ValueError: 
                If the input Taichi field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - Integral dtype fields (e.g., int) are automatically converted to Taichi's default floating-point dtype
            because the sine operation produces floating-point results.
            - Uses Taichi's `ti.sin` function for computation, ensuring compatibility with Taichi's backends and data types.
            - Input values are interpreted as angles in radians (consistent with standard mathematical conventions).

        Examples:
            # Scalar usage:
            >>> MyClass.sin(0.0)  # Sine of 0 radians
            0.0
            >>> MyClass.sin(ti.math.pi / 2)  # Sine of π/2 radians
            1.0

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([0.0, ti.math.pi/2, ti.math.pi], dtype=np.float32))
            >>> result = MyClass.sin(x)
            # result contains [0.0, 1.0, 0.0] (element-wise)
        """
        if isinstance(x, (int, float)):         
            return ti.sin(float(x))
        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
            dtype = x.dtype
            if ti.types.is_integral(dtype):
                dtype = ti.get_default_fp()
            out = ti.field(dtype=dtype, shape=shape)


        @ti.kernel
        def fill_sin(field: ti.template(), out: ti.template()):
            for I in ti.grouped(field):
                out[I] = ti.sin(field[I])

        fill_sin(x, out)
        if len(shape) == 0:
            return out[None]
        return out

    @staticmethod
    def sinh(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the hyperbolic sine element-wise for scalar values or Taichi fields.

        This static method calculates the hyperbolic sine of input data, supporting both scalar values
        (integers or floats) and Taichi fields. For scalar inputs, it returns the result as a float.
        For Taichi fields, the operation is performed element-wise, producing a new field with the same shape.
        The hyperbolic sine is computed using the formula: sinh(x) = (exp(x) - exp(-x)) / 2.

        Args:
            x (Union[int, float, ti.Field]): 
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the hyperbolic sine.
                - A Taichi field: Computes hyperbolic sine for each element individually.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the hyperbolic sine value as a float.
                - If input is a Taichi field, returns a new field with the same shape containing element-wise hyperbolic sine values.
                For 0D fields, returns the scalar hyperbolic sine value directly.

        Raises:
            ValueError: 
                If the input Taichi field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - Integral dtype fields (e.g., int) are automatically converted to Taichi's default floating-point dtype
            since the hyperbolic sine operation produces floating-point results.
            - Uses Taichi's `ti.exp` function for exponential calculations, ensuring compatibility with Taichi's backends.
            - The hyperbolic sine is an odd function: sinh(-x) = -sinh(x).

        Examples:
            # Scalar usage:
            >>> MyClass.sinh(0.0)  # Hyperbolic sine of 0
            0.0
            >>> MyClass.sinh(1.0)  # Hyperbolic sine of 1
            1.1752011936438014

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([0.0, 1.0, -1.0], dtype=np.float32))
            >>> result = MyClass.sinh(x)
            # result contains [0.0, 1.1752012, -1.1752012] (element-wise)
        """
        if isinstance(x, (int, float)):         
            return (ti.exp(float(x)) - ti.exp(-float(x))) * 0.5
        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.)")
            dtype = x.dtype
            if ti.types.is_integral(dtype):
                dtype = ti.get_default_fp()
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_sinh(field: ti.template(), out: ti.template()):
                for I in ti.grouped(field):
                    out[I] = (ti.exp(field[I]) - ti.exp(-field[I])) * 0.5

            fill_sinh(x, out)
            if len(shape) == 0:
                return out[None]
            return out
        
    @staticmethod
    def trace(x: ti.Field, k: int = 0) -> Union[float, int]:#TODO
        """
        Computes the trace of a 2D Taichi field (sum of diagonal elements).

        This static method calculates the trace of a 2D matrix, defined as the sum of the elements
        along the main diagonal (or an offset diagonal specified by `k`). The trace is returned as
        a scalar value of the same data type as the input field.

        Args:
            x (ti.Field): 
                Input 2D Taichi field representing a matrix.
            k (int, optional): 
                Diagonal offset. Defaults to 0 (main diagonal).
                - k=0: Main diagonal.
                - k>0: Diagonals above the main diagonal.
                - k<0: Diagonals below the main diagonal.

        Returns:
            Union[float, int]:
                The trace of the matrix as a scalar value. Returns a float for floating-point dtypes
                and an integer for integer dtypes.

        Raises:
            TypeError: 
                If the input is not a Taichi field.
            ValueError: 
                - If the input field is not 2D.
                - If the offset `k` is not an integer.

        Notes:
            - The trace is computed using a Taichi kernel for efficient parallel execution.
            - For non-square matrices, the trace is computed using the minimum dimension (rows or columns)
            adjusted by the offset `k`.

        Examples:
            # 3x3 matrix trace (main diagonal)
            >>> mat = ti.field(dtype=ti.f32, shape=(3, 3))
            >>> # Initialize matrix to [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            >>> MyClass.trace(mat)
            15.0  # 1 + 5 + 9

            # Offset diagonal (k=1)
            >>> MyClass.trace(mat, k=1)
            8.0  # 2 + 6

            # Offset diagonal (k=-1)
            >>> MyClass.trace(mat, k=-1)
            12.0  # 4 + 8
        """
        if not isinstance(x, ti.Field):
            raise TypeError(
                f"Unsupported type for trace: {type(x)}. Expected ti.Field."
            )
        shape = x.shape
        if len(shape) != 2:
            raise ValueError("Input field must be 2D.")
        if isinstance(k, float):
            if not k.is_integer():
                raise ValueError(f"The offset k for trace must be an integer, but currently it is {k}")
            k = int(k)
        dtype = x.dtype
        trace_value = ti.field(dtype=dtype, shape=())

        @ti.kernel
        def compute_trace(field: ti.template(), trace_value: ti.template()):
            trace_value[None] = 0
            for i in range(shape[0]):
                if 0 <= i + k < shape[1]:
                    trace_value[None] += field[i, i + k]

        compute_trace(x, trace_value)
        return trace_value[None]

    @staticmethod
    def unique(x: ti.Field) -> ti.Field: #TODO
        """
        Returns the sorted unique elements of a Taichi field.

        This static method extracts the unique elements from a Taichi field, sorts them in ascending order,
        and returns a new 1D Taichi field containing these unique values. The input field can be of any shape,
        but it will be flattened before processing.

        Args:
            x (ti.Field): 
                Input Taichi field from which unique elements will be extracted.

        Returns:
            ti.Field: 
                A 1D Taichi field containing the sorted unique elements from the input field.

        Raises:
            TypeError: 
                If the input is not a Taichi field.
            ValueError: 
                If the input field has any zero dimensions (Taichi does not support zero-sized fields).

        Notes:
            - The unique elements are sorted using Python's built-in sorting, which may not match the
            order of elements in the original field.
            - The output field will have the same data type as the input field.

        Examples:
            # 1D field with duplicates
            >>> x = ti.field(dtype=ti.i32, shape=(5,))
            >>> x.from_numpy(np.array([3, 1, 2, 2, 3], dtype=np.int32))
            >>> result = MyClass.unique(x)
            # result is [1, 2, 3]

            # 2D field (flattened before processing)
            >>> x = ti.field(dtype=ti.f32, shape=(2, 2))
            >>> x.from_numpy(np.array([[1.5, 2.0], [1.5, 3.0]], dtype=np.float32))
            >>> result = MyClass.unique(x)
            # result is [1.5, 2.0, 3.0]
        """
        if not isinstance(x, ti.Field):
            raise TypeError("Input x must be a Taichi Field.")
        shape = x.shape
        if any(s == 0 for s in shape):
            raise ValueError(
                f"Input field has zero in its shape {shape}, which is not supported by Taichi."
            )
        dtype = x.dtype

        arr = x.to_numpy().flatten()
        unique_values = sorted(set(arr.tolist()))
        n_unique = len(unique_values)
        unique_field = ti.field(dtype=dtype, shape=(n_unique,))
        for i, v in enumerate(unique_values):
            unique_field[i] = v
        return unique_field
