from typing import Any, Union, Optional, TypeVar, Tuple, List
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
            taichi_field = bm.from_numpy(numpy_array)
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
            nested_list = bm.tolist(field)
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
    def arange(*args, dtype=ti.f64) -> ti.Field:  # TODO @ti.kernel
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
            >>> bm.arange(5)
            [0.0, 1.0, 2.0, 3.0, 4.0]

            # Generate field with start and stop
            >>> bm.arange(2, 7)
            [2.0, 3.0, 4.0, 5.0, 6.0]

            # Generate field with start, stop, and step
            >>> bm.arange(1, 10, 2)
            [1.0, 3.0, 5.0, 7.0, 9.0]

            # Generate field with negative step
            >>> bm.arange(5, 0, -1)
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
    def eye(N, M=None, k: int = 0, dtype=ti.f64) -> ti.Field:
        """
        Creates a Taichi field representing an identity-like matrix with 1s on the specified diagonal and 0s elsewhere.

        This static method generates an N×M matrix (or square N×N matrix if M is not specified) where elements on the k-th diagonal are 1, and all other elements are 0. The diagonal is determined by the offset `k`.

        Args:
            N (int):
                Number of rows in the output matrix. Must be a positive integer.
            M (int, optional):
                Number of columns in the output matrix. If None, defaults to N (resulting in a square matrix). Must be a positive integer if specified.
            k (int, optional):
                Diagonal offset. Defaults to 0 (main diagonal).
                - k=0: Main diagonal (elements where row index = column index).
                - k>0: Diagonals above the main diagonal (e.g., k=1 for elements where column index = row index + 1).
                - k<0: Diagonals below the main diagonal (e.g., k=-1 for elements where column index = row index - 1).
            dtype (ti.DataType, optional):
                Data type of the output field. Defaults to `ti.f64`.

        Returns:
            ti.Field:
                An N×M Taichi field with 1s on the k-th diagonal and 0s elsewhere.

        Raises:
            TypeError:
                If N, M, or k are not integers.
            ValueError:
                - If N or M is non-positive (≤ 0).

        Notes:
            - For a square matrix (N = M), this behaves like a standard identity matrix when k=0.
            - The k-th diagonal is defined such that an element at position (i, j) is on the k-th diagonal if `j = i + k`.
            - If the k-th diagonal lies outside the matrix bounds (e.g., k=2 in a 3×3 matrix), the corresponding positions will remain 0.

        Examples:
            # 3x3 identity matrix (main diagonal, k=0)
            >>> bm.eye(3)
            [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]  # dtype ti.f64, shape (3, 3)

            # 4x5 matrix with 1s on diagonal k=1 (above main diagonal)
            >>> bm.eye(4, 5, k=1)
            [[0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]]  # shape (4, 5)

            # 3x3 matrix with 1s on diagonal k=-1 (below main diagonal)
            >>> bm.eye(3, k=-1)
            [[0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]]  # shape (3, 3)
        """
        if M is None:
            M = N

        if (
            not isinstance(N, (int))
            or not isinstance(M, (int))
            or not isinstance(k, (int))
        ):
            raise TypeError(f"{N} and {M} and {k} must be integers.")

        if N == 0 or M == 0:
            return []
        if N < 0 or M < 0:
            raise ValueError(f"N and M must be positive integers, got N={N}, M={M}.")
        field = ti.field(dtype=dtype, shape=(N, M))

        @ti.kernel
        def fill_eye():
            for i in range(max(0, -k), min(N, M - k)):
                field[i, i + k] = 1

        fill_eye()
        return field

    @staticmethod
    def zeros(shape: Union[int, Tuple[int, ...]], dtype=ti.f64) -> ti.Field:
        """
        Creates a Taichi field filled with zeros.

        This static method generates a Taichi field of specified shape and data type,
        initialized with zeros. The shape can be either a single integer (for a 1D field)
        or a tuple of integers (for multi-dimensional fields).

        Args:
            shape (Union[int, Tuple]):
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
            TypeError:
                - If tuple shape elements are not integers or integer-convertible floats.

        Notes:
            - Floats in tuple shapes are converted to integers if they are whole numbers (e.g., 3.0 → 3).
            - Taichi does not support fields with zero-sized dimensions (e.g., (0, 3) is invalid).

        Examples:
            # 1D field with length 5
            >>> bm.zeros(5)
            [0.0, 0.0, 0.0, 0.0, 0.0]

            # 2D field with shape (2, 3)
            >>> bm.zeros((2, 3))
            [[0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]]

            # 3D field with shape (2, 2, 2)
            >>> bm.zeros((2, 2, 2))
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
                    (
                        int(s)
                        if isinstance(s, float) and s.is_integer()
                        else (s if isinstance(s, int) else _raise_type_error(s))
                    )
                    for s in shape
                ]
            except TypeError as e:
                raise e

            def _raise_type_error(s):
                raise TypeError(
                    f"Shape elements must be integers, got {type(s).__name__} {s}."
                )

            shape = tuple(new_shape)
        field = ti.field(dtype=dtype, shape=shape)

        return field

    @staticmethod
    def zeros_like(field: ti.Field) -> ti.Field:
        """
        Creates a Taichi field of zeros with the same shape and data type as the input field.

        This static method constructs a new Taichi field filled with zeros, mirroring the shape and data type of the provided input field. The output field is initialized to zeros implicitly by Taichi's field creation semantics.

        Args:
            field (ti.Field):
                Input Taichi field whose shape and data type will be mirrored.

        Returns:
            ti.Field:
                A new Taichi field with the same shape and data type as `field`, initialized to zeros.

        Raises:
            TypeError:
                If the input is not a Taichi field.

        Notes:
            - The output field is initialized to zeros by default as per Taichi's field creation behavior.
            - This method does not explicitly fill the field with zeros using a kernel, relying instead on Taichi's automatic zero-initialization.
            - For fields with complex or user-defined data types, the zero initialization follows Taichi's default behavior for that type.

        Examples:
            # 2D field example
            >>> x = ti.field(dtype=ti.f32, shape=(3, 4))
            >>> zeros = bm.zeros_like(x)
            # zeros is a ti.f32 field with shape (3, 4), initialized to zeros

            # 3D field example
            >>> y = ti.field(dtype=ti.i64, shape=(2, 2, 2))
            >>> zeros = bm.zeros_like(y)
            # zeros is a ti.i64 field with shape (2, 2, 2), initialized to zeros
        """
        if not isinstance(field, ti.Field):
            raise TypeError("Input must be a Taichi field.")
        shape = field.shape
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
            TypeError:
                - If input is not a Taichi field or list.
                - If input is a list with non-numeric elements.

        Notes:
            - For 1D inputs, each row of the output matrix is a copy of the input array.
            - For ND inputs (N>2), the operation is applied to the last two dimensions.
            - The diagonal offset `k` shifts the triangular region up or down.

        Examples:
            # 1D array input
            >>> bm.tril([1, 2, 3])
            [[1, 0, 0],
            [1, 2, 0],
            [1, 2, 3]]

            # 2D matrix input
            >>> matrix = ti.field(dtype=ti.f32, shape=(3, 3))
            >>> # Initialize matrix to [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            >>> bm.tril(matrix)
            [[1, 0, 0],
            [4, 5, 0],
            [7, 8, 9]]

            # Diagonal offset (k=1)
            >>> bm.tril(matrix, k=1)
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
                for i, j in ti.ndrange(M, M):
                    if j - i <= k:
                        out[i, j] = x_field[j]

            fill_matrix()
            return out
        if isinstance(x, ti.Field):
            field = x
            if field is None:
                raise ValueError(
                    "Input field is None. Please provide a valid Taichi field."
                )
            shape = field.shape
            dtype = field.dtype

            if len(shape) == 1:
                M = shape[0]
                out = ti.field(dtype=dtype, shape=(M, M))

                @ti.kernel
                def fill_tril_1d():
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

                fill_tril_nd()
                return out
        else:
            raise TypeError(
                f"Unsupported type for tril: {type(x)}. Expected ti.Field or list."
            )

    @staticmethod
    def abs(x: Union[int, float, ti.Field]) -> Union[int, float, ti.Field]:
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
            >>> bm.abs(-5)
            5
            >>> bm.abs(3.14)
            3.14
            >>> bm.abs(True)
            1

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([-1.0, 2.0, -3.0], dtype=np.float32))
            >>> result = bm.abs(x)
            # result is now [1.0, 2.0, 3.0]
        """
        if isinstance(x, (int, float, bool)):
            return ti.abs(x)
        if isinstance(x, ti.Field):
            shape = x.shape
            dtype = x.dtype
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_abs():
                for I in ti.grouped(x):
                    out[I] = ti.abs(x[I])

            fill_abs()

            return out
        else:
            raise TypeError(
                f"Unsupported type for abs: {type(x)}. Expected int, float, or ti.Field."
            )

    @staticmethod
    def acos(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the inverse cosine (arccosine) element-wise for scalar values or Taichi fields.

        This static method calculates the arccosine of input data, where the arccosine of a value `x` is the angle whose cosine is `x`. It supports scalar inputs (integers or floats) and Taichi fields, with element-wise computation for fields.

        Args:
            x (Union[int, float, ti.Field]):
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the arccosine.
                - A Taichi field: Computes arccosine for each element individually.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the arccosine as a float (using `ti.acos`).
                - If input is a Taichi field, returns a new `ti.f64` Taichi field with the same shape as `x`, where each element is the arccosine of the corresponding element in `x`.

        Notes:
            - For Taichi field inputs, elements are cast to `ti.f64` before computing the arccosine to match Taichi's mathematical function type requirements.
            - Input values should typically be in the range [-1, 1] to produce real results. Values outside this range may return `NaN` (not a number) due to the mathematical definition of arccosine.
            - Uses Taichi's `ti.acos` function for computation, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage:
            >>> bm.acos(1.0)   # Arccosine of 1.0 (cos(0) = 1)
            0.0
            >>> bm.acos(0.0)   # Arccosine of 0.0 (cos(π/2) = 0)
            1.5707963267948966  # π/2 radians

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([1.0, 0.0, -1.0], dtype=np.float32))
            >>> result = bm.acos(x)  # result is ti.f64 field with shape (3,)
            # result contains [0.0, π/2, π] (element-wise)
        """
        if isinstance(x, (int, float)):
            return ti.acos(x)

        if isinstance(x, ti.Field):
            shape = x.shape
            out = ti.field(dtype=ti.f64, shape=shape)

            @ti.kernel
            def fill_acos(field: ti.template(), out: ti.template()):
                for I in ti.grouped(field):
                    out[I] = ti.acos(ti.cast(field[I], ti.f64))

            fill_acos(x, out)

            return out

    @staticmethod
    def asin(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the inverse sine (arcsine) element-wise for scalar values or Taichi fields.

        This static method calculates the arcsine of input data, where the arcsine of a value `x` is the angle whose sine is `x`. It supports scalar inputs (integers or floats) and Taichi fields, with element-wise computation for fields.

        Args:
            x (Union[int, float, ti.Field]):
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the arcsine.
                - A Taichi field: Computes arcsine for each element individually.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the arcsine as a float (using `ti.asin`).
                - If input is a Taichi field, returns a new `ti.f64` Taichi field with the same shape as `x`, where each element is the arcsine of the corresponding element in `x`.

        Notes:
            - For Taichi field inputs, elements are cast to `ti.f64` before computing the arcsine to match Taichi's mathematical function type requirements.
            - Input values should typically be in the range [-1, 1] to produce real results. Values outside this range may return `NaN` (not a number) due to the mathematical definition of arcsine.
            - Uses Taichi's `ti.asin` function for computation, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage:
            >>> bm.asin(0.0)   # Arcsine of 0.0 (sin(0) = 0)
            0.0
            >>> bm.asin(1.0)   # Arcsine of 1.0 (sin(π/2) = 1)
            1.5707963267948966  # π/2 radians

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
            >>> result = bm.asin(x)  # result is ti.f64 field with shape (3,)
            # result contains [-π/2, 0.0, π/2] (element-wise)
        """
        if isinstance(x, (int, float)):
            return ti.asin(x)
        if isinstance(x, ti.Field):
            shape = x.shape
            out = ti.field(dtype=ti.f64, shape=shape)

            @ti.kernel
            def fill_asin(field: ti.template(), out: ti.template()):
                for I in ti.grouped(field):
                    out[I] = ti.asin(ti.cast(field[I], ti.f64))

            fill_asin(x, out)

            return out

    @staticmethod
    def atan(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the inverse tangent (arctangent) element-wise for scalar values or Taichi fields.

        This static method calculates the arctangent of input data, where the arctangent of a value `x` is the angle whose tangent is `x`.
        For scalar inputs, it uses `ti.atan2(x, 1)` (mathematically equivalent to `atan(x)`).
        For Taichi fields, it performs element-wise computation and returns a new field with results in `ti.f64` precision.

        Args:
            x (Union[int, float, ti.Field]):
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the arctangent.
                - A Taichi field: Computes arctangent for each element individually.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the arctangent as a float (using `ti.atan2(x, 1)`).
                - If input is a Taichi field, returns a new `ti.f64` Taichi field with the same shape as `x`, where each element is the arctangent of the corresponding element in `x`.

        Notes:
            - The arctangent is computed using `ti.atan2(x, 1)`, which is equivalent to `atan(x)` because `atan2(y, x)` evaluates to the angle of the point `(x, y)`, and `(1, x)` gives the angle whose tangent is `x/1 = x`.
            - The result of arctangent ranges from `-π/2` to `π/2` radians, covering all real numbers as input (no restriction to [-1, 1] like arccosine/arcsine).
            - Uses Taichi's `ti.atan2` function for computation, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage:
            >>> bm.atan(0.0)    # Arctangent of 0 (tan(0) = 0)
            0.0
            >>> bm.atan(1.0)    # Arctangent of 1 (tan(π/4) = 1)
            0.7853981633974483  # π/4 radians
            >>> bm.atan(-1.0)   # Arctangent of -1 (tan(-π/4) = -1)
            -0.7853981633974483  # -π/4 radians

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
            >>> result = bm.atan(x)  # result is ti.f64 field with shape (3,)
            # result contains [-π/4, 0.0, π/4] (element-wise)
        """
        if isinstance(x, (int, float)):
            return ti.atan2(x, 1)
        if isinstance(x, ti.Field):
            shape = x.shape
            out = ti.field(dtype=ti.f64, shape=shape)

            @ti.kernel
            def fill_atan(
                field: ti.template(),
                out: ti.template(),
            ):
                for I in ti.grouped(field):
                    out[I] = ti.atan2(field[I], 1)

            fill_atan(x, out)
            return out

    @staticmethod
    def atan2(
        y: Union[int, float, ti.Field], x: Union[int, float, ti.Field]
    ) -> ti.Field:
        """
        Computes the four-quadrant inverse tangent (arctangent) of y/x, using the signs of both inputs to determine the quadrant.

        This static method calculates the arctangent of the quotient y/x, where the result is an angle in radians between -π and π. Unlike `atan`, it uses the signs of both `y` (numerator) and `x` (denominator) to determine the correct quadrant, making it suitable for converting Cartesian coordinates (x, y) to polar angle.

        Args:
            y (Union[int, float, ti.Field]):
                Numerator value(s). Can be a scalar (int/float) or a Taichi field. Scalars are converted to 0D `ti.f64` fields internally.
            x (Union[int, float, ti.Field]):
                Denominator value(s). Can be a scalar (int/float) or a Taichi field. Scalars are converted to 0D `ti.f64` fields internally.

        Returns:
            Union[float, ti.Field]:
                - If both inputs are scalars, returns the angle as a float (in radians).
                - If inputs are Taichi fields, returns a new `ti.f64` Taichi field with the same shape as `y` and `x`, where each element is the four-quadrant arctangent of the corresponding elements in `y` and `x`.

        Raises:
            ValueError:
                If the input Taichi fields `y` and `x` have different shapes.

        Notes:
            - Scalar inputs (int/float) are automatically converted to 0D `ti.f64` Taichi fields to unify processing with field inputs.
            - The output is always of type `ti.f64` to match Taichi's mathematical function type requirements.
            - The result angle θ satisfies `tan(θ) = y/x` and lies in the correct quadrant based on the signs of `x` and `y`:
                - θ ∈ (0, π/2) if x > 0, y > 0 (first quadrant)
                - θ ∈ (π/2, π) if x < 0, y > 0 (second quadrant)
                - θ ∈ (-π, -π/2) if x < 0, y < 0 (third quadrant)
                - θ ∈ (-π/2, 0) if x > 0, y < 0 (fourth quadrant)
            - Uses Taichi's `ti.atan2` function for computation, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage (different quadrants):
            >>> bm.atan2(1, 1)    # First quadrant: π/4
            0.7853981633974483
            >>> bm.atan2(1, -1)   # Second quadrant: 3π/4
            2.356194490192345
            >>> bm.atan2(-1, -1)  # Third quadrant: -3π/4
            -2.356194490192345
            >>> bm.atan2(-1, 1)   # Fourth quadrant: -π/4
            -0.7853981633974483

            # Field usage:
            >>> y = ti.field(dtype=ti.f32, shape=(2,))
            >>> y.from_numpy(np.array([1.0, -1.0]))
            >>> x = ti.field(dtype=ti.f32, shape=(2,))
            >>> x.from_numpy(np.array([1.0, 1.0]))
            >>> result = bm.atan2(y, x)  # ti.f64 field with shape (2,)
            # result contains [π/4, -π/4]
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

        out = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def fill_atan2(
            y_field: ti.template(), x_field: ti.template(), out: ti.template()
        ):
            for I in ti.grouped(y_field):
                if x_field[I] == 0 and y_field[I] == 0:
                    out[I] = ti.math.nan
                else:
                    out[I] = ti.atan2(
                        ti.cast(y_field[I], ti.f64), ti.cast(x_field[I], ti.f64)
                    )

        fill_atan2(y, x, out)
        return out

    @staticmethod
    def ceil(x: Union[int, float, ti.Field]) -> Union[int, float, ti.Field]:
        """
        Computes the ceiling value element-wise for scalar values or Taichi fields.

        This static method calculates the smallest integer greater than or equal to the input value(s).
        For scalar inputs, it uses `ti.ceil` directly. For Taichi fields, it performs element-wise ceiling computation,
        converting elements to `ti.f64` to match Taichi's mathematical function type requirements.

        Args:
            x (Union[int, float, ti.Field]):
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the ceiling.
                - A Taichi field: Computes ceiling for each element individually.

        Returns:
            Union[int, float, ti.Field]:
                - If input is a scalar, returns the ceiling value (result of `ti.ceil(x)`).
                - If input is a Taichi field, returns a new `ti.f64` Taichi field with the same shape as `x`,
                  where each element is the ceiling of the corresponding element in `x`.

        Notes:
            - For Taichi field inputs, elements are cast to `ti.f64` before computing the ceiling to match Taichi's mathematical function type requirements.
            - The ceiling of a value is the smallest integer greater than or equal to that value (e.g., ceil(2.1) = 3.0, ceil(-1.8) = -1.0).
            - Uses Taichi's `ti.ceil` function for computation, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage:
            >>> bm.ceil(2.3)   # Smallest integer ≥ 2.3
            3.0
            >>> bm.ceil(-1.8)  # Smallest integer ≥ -1.8
            -1.0
            >>> bm.ceil(5)     # Ceiling of integer is itself (as float)
            5.0

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([1.2, 2.7, -3.1]))
            >>> result = bm.ceil(x)  # ti.f64 field with shape (3,)
            # result contains [2.0, 3.0, -3.0]

            # 0D field usage:
            >>> x = ti.field(dtype=ti.i32, shape=())
            >>> x[None] = -5
            >>> bm.ceil(x)  # 0D field returns scalar value
            -5.0
        """
        if isinstance(x, (int, float)):
            return ti.ceil(x)
        if isinstance(x, ti.Field):
            shape = x.shape
            out = ti.field(dtype=ti.f64, shape=shape)

            @ti.kernel
            def fill_ceil(field: ti.template(), out: ti.template()):
                for I in ti.grouped(field):
                    out[I] = ti.ceil(ti.cast(field[I], ti.f64))

            fill_ceil(x, out)
            return out

    @staticmethod
    def clip(
        x: Union[int, float, ti.Field], *args, **kwargs
    ) -> Union[int, float, ti.Field]:
        """
        Clips (limits) the values in a scalar or a Taichi field.

        Given an interval, values outside the interval are clipped to the interval
        edges. For example, if an interval of `[0, 1]` is specified, values smaller
        than 0 become 0, and values larger than 1 become 1.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it returns a clipped scalar. For Taichi fields, it returns
        a new field with the clipped values. The clipping boundaries can be specified
        using positional or keyword arguments.

        Args:
            x (Union[int, float, ti.Field]):
                The input value or `ti.Field` to be clipped.
            min (Union[int, float], optional):
                The lower-bound value. Values in `x` less than this are replaced by it.
                Can be provided as the second positional argument or as the keyword
                argument `min`. If omitted, no lower-clipping is performed.
            max (Union[int, float], optional):
                The upper-bound value. Values in `x` greater than this are replaced by it.
                Can be provided as the third positional argument or as the keyword
                argument `max`. If omitted, no upper-clipping is performed.

        Returns:
            Union[int, float, ti.Field]:
                - If `x` is a scalar, returns a clipped `int` or `float`.
                - If `x` is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the element-wise clipped values.

        Raises:
            TypeError: If the arguments are invalid, such as providing too many
                positional arguments, specifying a bound both positionally and by keyword
                (e.g., `clip(x, 5, max=10)`), providing unexpected keyword arguments,
                or if `x` has an unsupported type.

        Notes:
            - This function's signature and behavior are designed to be similar to `numpy.clip`.
            - For `ti.Field` inputs, the operation is not in-place; a new field is
            allocated and returned.

        Example:
            # Scalar usage with keywords
            result = bm.clip(15, min=0, max=10)
            # result is 10

            # Scalar usage with positional arguments
            result2 = bm.clip(-5, 0, 10)
            # result2 is 0

            # Field usage
            f = ti.field(dtype=ti.f64, shape=(4,))
            f.from_numpy(np.array([-10.0, 5.0, 12.0, 8.0]))
            clipped_f = bm.clip(f, min=0.0, max=10.0)
            # clipped_f will contain [0.0, 5.0, 10.0, 8.0]
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
            if min_val is not None:
                min_val = float(min_val)
                x = max(min_val, x)
            if max_val is not None:
                max_val = float(max_val)
                x = min(x, max_val)
            return x

        if isinstance(x, ti.Field):
            shape = x.shape
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

            min_converted = min_val if min_val is not None else 0
            max_converted = max_val if max_val is not None else 0

            fill_clip(x, out, min_converted, max_converted, use_min, use_max)

            return out

        raise TypeError(
            f"Unsupported type for clip: {type(x)}. Expected int, float or ti.Field."
        )

    @staticmethod
    def cos(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the cosine element-wise for scalar values or Taichi fields.

        This static method calculates the cosine of input data, where the cosine of an angle `x` (in radians) is the ratio of the adjacent side to the hypotenuse in a right triangle. It supports scalar inputs (integers or floats) and Taichi fields, with element-wise computation for fields.

        Args:
            x (Union[int, float, ti.Field]):
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the cosine (interpreted as radians).
                - A Taichi field: Computes cosine for each element individually (elements are interpreted as radians).

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the cosine as a float (using `ti.cos`).
                - If input is a Taichi field, returns a new `ti.f64` Taichi field with the same shape as `x`, where each element is the cosine of the corresponding element in `x`. For 0D fields, returns the scalar value directly.

        Notes:
            - For Taichi field inputs, elements are cast to `ti.f64` before computing the cosine to match Taichi's mathematical function type requirements.
            - Input values are interpreted as angles in radians (consistent with standard mathematical conventions).
            - The cosine function ranges between -1 and 1 for all real inputs, with periodicity 2π (i.e., cos(x + 2π) = cos(x)).
            - Uses Taichi's `ti.cos` function for computation, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage:
            >>> bm.cos(0.0)  # Cosine of 0 radians
            1.0
            >>> bm.cos(ti.math.pi)  # Cosine of π radians
            -1.0

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([0.0, ti.math.pi/2, ti.math.pi]))
            >>> result = bm.cos(x)  # ti.f64 field with shape (3,)
            # result contains [1.0, 0.0, -1.0]

            # 0D field usage:
            >>> x = ti.field(dtype=ti.i32, shape=())
            >>> x[None] = 0
            >>> bm.cos(x)  # 0D field returns scalar value
            1.0
        """
        if isinstance(x, (int, float)):
            return ti.cos(x)
        if isinstance(x, ti.Field):
            shape = x.shape
            out = ti.field(dtype=ti.f64, shape=shape)

            @ti.kernel
            def fill_cos(field: ti.template(), out: ti.template()):
                for I in ti.grouped(field):
                    out[I] = ti.cos(ti.cast(field[I], ti.f64))

            fill_cos(x, out)
            return out

    @staticmethod
    def cosh(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the hyperbolic cosine element-wise for scalar values or Taichi fields.

        This static method calculates the hyperbolic cosine of input data using the formula:
        cosh(x) = (exp(x) + exp(-x)) / 2. It supports scalar inputs (integers or floats) and
        Taichi fields, with element-wise computation for fields.

        Args:
            x (Union[int, float, ti.Field]):
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the hyperbolic cosine.
                - A Taichi field: Computes hyperbolic cosine for each element individually.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the hyperbolic cosine as a float.
                - If input is a Taichi field, returns a new `ti.f64` Taichi field with the same shape as `x`,
                where each element is the hyperbolic cosine of the corresponding element in `x`.

        Notes:
            - For Taichi field inputs, elements are cast to `ti.f64` before computation to match Taichi's mathematical function type requirements.
            - The hyperbolic cosine function is always non-negative (cosh(x) ≥ 1 for all real x).
            - It grows exponentially for large positive x and approaches 1 for x close to 0.
            - Uses Taichi's `ti.exp` function for computation, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage:
            >>> bm.cosh(0.0)  # cosh(0) = (e⁰ + e⁻⁰) / 2 = (1 + 1) / 2 = 1
            1.0
            >>> bm.cosh(1.0)  # Approx. 1.543
            1.5430806348152437

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([-1.0, 0.0, 1.0]))
            >>> result = bm.cosh(x)  # ti.f64 field with shape (3,)
            # result contains [1.5430806, 1.0, 1.5430806] (cosh is even: cosh(-x) = cosh(x))
        """
        if isinstance(x, (int, float)):
            return (ti.exp(x) + ti.exp(-x)) * 0.5
        if isinstance(x, ti.Field):
            shape = x.shape
            out = ti.field(dtype=ti.f64, shape=shape)

            @ti.kernel
            def fill_cosh(field: ti.template(), out: ti.template()):
                for I in ti.grouped(field):
                    out[I] = (
                        ti.exp(ti.cast(field[I], ti.f64))
                        + ti.exp(-ti.cast(field[I], ti.f64))
                    ) * 0.5

            fill_cosh(x, out)
            return out

    @staticmethod
    def floor(x: Union[int, float, ti.Field]) -> Union[int, float, ti.Field]:
        """
        Computes the floor value element-wise for scalar values or Taichi fields.

        This static method calculates the largest integer less than or equal to the input value(s).
        For scalar inputs, it uses `ti.floor` directly. For Taichi fields, it performs element-wise
        floor computation, converting elements to `ti.f64` to match Taichi's mathematical function type requirements.

        Args:
            x (Union[int, float, ti.Field]):
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the floor.
                - A Taichi field: Computes floor for each element individually.

        Returns:
            Union[int, float, ti.Field]:
                - If input is a scalar, returns the floor value (e.g., floor(3.7) = 3.0).
                - If input is a Taichi field, returns a new `ti.f64` Taichi field with the same shape as `x`,
                where each element is the floor of the corresponding element in `x`.

        Notes:
            - For Taichi field inputs, elements are cast to `ti.f64` before computing the floor to match Taichi's mathematical function type requirements.
            - The floor of a value is the largest integer less than or equal to that value
            (e.g., floor(3.7) = 3.0, floor(-2.3) = -3.0).
            - Uses Taichi's `ti.floor` function for computation, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage:
            >>> bm.floor(3.7)   # Largest integer ≤ 3.7
            3.0
            >>> bm.floor(-2.3)  # Largest integer ≤ -2.3
            -3.0

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(2,))
            >>> x.from_numpy(np.array([1.5, -1.5]))
            >>> result = bm.floor(x)  # ti.f64 field with shape (2,)
            # result contains [1.0, -2.0]
        """
        if isinstance(x, (int, float)):
            return ti.floor(x)
        if isinstance(x, ti.Field):
            shape = x.shape
            out = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def fill_floor(field: ti.template(), out: ti.template()):
            for I in ti.grouped(field):
                out[I] = ti.floor(ti.cast(field[I], ti.f64))

        fill_floor(x, out)
        return out

    @staticmethod
    def floor_divide(  # TODO 广播没有实现
        x: Union[int, float, ti.Field], y: Union[int, float, ti.Field]
    ) -> Union[int, float, ti.Field]:
        """
        Computes the element-wise floor of the division of x by y.

        This function first performs the division `x / y` and then computes the floor
        of the result. The mathematical operation is `floor(x / y)`. The function
        supports operations between scalars (int, float) and Taichi fields.

        The return type depends on the input types: if both `x` and `y` are integers,
        the result is an integer. Otherwise, the result is a float.

        Args:
            x (Union[int, float, ti.Field]):
                The dividend (numerator). Can be a scalar or a Taichi field.
            y (Union[int, float, ti.Field]):
                The divisor (denominator). Can be a scalar or a Taichi field.

        Returns:
            Union[int, float, ti.Field]:
                - If both inputs are scalars, returns an `int` (if both inputs are `int`)
                or a `float`.
                - If at least one input is a `ti.Field`, returns a new `ti.Field`
                containing the element-wise result.

        Raises:
            ValueError: If `x` and `y` are both `ti.Field`s but have different shapes.

        Notes:
            - TODO: Broadcasting is not yet implemented. When `x` and `y` are both
            `ti.Field`s, they are required to have the exact same shape.
            - Division by zero is handled to mimic NumPy's behavior:
                - `x / 0` with `x > 0` returns `inf`.
                - `x / 0` with `x < 0` returns `-inf`.
                - `0 / 0` returns `nan`.
            - The operation is not in-place; a new field is always allocated for the result
            when the inputs involve fields.

        Example:
            # Scalar usage (integer inputs produce an integer)
            result_int = bm.floor_divide(10, 3)
            # result_int is 3

            # Scalar usage (float input produces a float)
            result_float = bm.floor_divide(-10.0, 3)
            # result_float is -4.0

            # Field usage
            x = ti.field(dtype=ti.f32, shape=(4,))
            y = ti.field(dtype=ti.f32, shape=(4,))
            x.from_numpy(np.array([10., -10., 7., 0.]))
            y.from_numpy(np.array([3., 3., -2., 0.]))

            result_field = bm.floor_divide(x, y)
            # result_field will contain [3.0, -4.0, -4.0, nan]
        """
        if isinstance(x, (int, float)) and isinstance(y, ti.Field) and y.shape == ():
            if y[None] == 0:
                if x == 0:
                    return ti.math.nan
                if x < 0:
                    return -ti.math.inf
                else:
                    return ti.math.inf
            else:
                return (
                    int(ti.floor(x / y[None]))
                    if isinstance(x, int) and isinstance(y[None], int)
                    else ti.floor(x / y[None])
                )
        if isinstance(y, (int, float)) and isinstance(x, ti.Field) and x.shape == ():
            if y == 0:
                if x[None] == 0:
                    return ti.math.nan
                if x[None] < 0:
                    return -ti.math.inf
                else:
                    return ti.math.inf
            else:
                return (
                    int(ti.floor(x[None] / y))
                    if isinstance(x[None], int) and isinstance(y, int)
                    else ti.floor(x[None] / y)
                )
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if y == 0:
                if x == 0:
                    return ti.math.nan
                if x < 0:
                    return -ti.math.inf
                else:
                    return ti.math.inf
            else:
                return (
                    int(ti.floor(x / y))
                    if isinstance(x, int) and isinstance(y, int)
                    else ti.floor(x / y)
                )
        if isinstance(x, ti.Field) and isinstance(y, ti.Field):
            shape = x.shape
            if shape != y.shape:
                raise ValueError("Input fields must have the same shape.")
            out = ti.field(dtype=ti.f64, shape=shape)

            @ti.kernel
            def fill_floor_divide(
                x_field: ti.template(), y_field: ti.template(), out: ti.template()
            ):
                for I in ti.grouped(y_field):
                    if x_field[I] == 0 and y_field[I] == 0:
                        out[I] = ti.math.nan
                    else:
                        if isinstance(x_field[I], int) and isinstance(y_field[I], int):
                            out[I] = int(ti.floor(x_field[I] / y_field[I]))
                        else:
                            out[I] = ti.floor(x_field[I] / y_field[I])

            fill_floor_divide(x, y, out)
            return out

    @staticmethod
    def sin(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the sine element-wise for scalar values or Taichi fields.

        This static method calculates the sine of input data, where the sine of an angle `x` (in radians)
        is the ratio of the opposite side to the hypotenuse in a right triangle. It supports scalar inputs
        (integers or floats) and Taichi fields, with element-wise computation for fields.

        Args:
            x (Union[int, float, ti.Field]):
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the sine (interpreted as radians).
                - A Taichi field: Computes sine for each element individually (elements are interpreted as radians).

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the sine as a float (using `ti.sin`).
                - If input is a Taichi field, returns a new `ti.f64` Taichi field with the same shape as `x`,
                where each element is the sine of the corresponding element in `x`.

        Notes:
            - For Taichi field inputs, elements are cast to `ti.f64` before computing the sine to match Taichi's mathematical function type requirements.
            - Input values are interpreted as angles in radians (consistent with standard mathematical conventions).
            - The sine function ranges between -1 and 1 for all real inputs, with periodicity 2π
            (i.e., sin(x + 2π) = sin(x)).
            - Uses Taichi's `ti.sin` function for computation, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage:
            >>> bm.sin(0.0)  # Sine of 0 radians
            0.0
            >>> bm.sin(ti.math.pi/2)  # Sine of π/2 radians (90 degrees)
            1.0

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([0.0, ti.math.pi/2, ti.math.pi]))
            >>> result = bm.sin(x)  # ti.f64 field with shape (3,)
            # result contains [0.0, 1.0, 0.0]
        """
        if isinstance(x, (int, float)):
            return ti.sin(x)
        if isinstance(x, ti.Field):
            shape = x.shape
            out = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def fill_sin(field: ti.template(), out: ti.template()):
            for I in ti.grouped(field):
                out[I] = ti.sin(ti.cast(field[I], ti.f64))

        fill_sin(x, out)
        return out

    @staticmethod
    def sinh(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the hyperbolic sine element-wise for scalar values or Taichi fields.

        This static method calculates the hyperbolic sine of input data using the formula:
        sinh(x) = (exp(x) - exp(-x)) / 2. It supports scalar inputs (integers or floats) and
        Taichi fields, with element-wise computation for fields.

        Args:
            x (Union[int, float, ti.Field]):
                Input value or field. Can be:
                - A scalar (int or float): Directly computes the hyperbolic sine.
                - A Taichi field: Computes hyperbolic sine for each element individually.

        Returns:
            Union[float, ti.Field]:
                - If input is a scalar, returns the hyperbolic sine as a float.
                - If input is a Taichi field, returns a new `ti.f64` Taichi field with the same shape as `x`,
                where each element is the hyperbolic sine of the corresponding element in `x`.

        Notes:
            - For Taichi field inputs, elements are cast to `ti.f64` before computation to match Taichi's mathematical function type requirements.
            - The hyperbolic sine function is an odd function (sinh(-x) = -sinh(x)).
            - It grows exponentially for large positive x and decays exponentially for large negative x.
            - Uses Taichi's `ti.exp` function for computation, ensuring compatibility with Taichi's backends.

        Examples:
            # Scalar usage:
            >>> bm.sinh(0.0)  # sinh(0) = (e⁰ - e⁻⁰) / 2 = (1 - 1) / 2 = 0
            0.0
            >>> bm.sinh(1.0)  # Approx. 1.175
            1.1752011936438014

            # Field usage:
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([-1.0, 0.0, 1.0]))
            >>> result = bm.sinh(x)  # ti.f64 field with shape (3,)
            # result contains [-1.1752012, 0.0, 1.1752012]
        """
        if isinstance(x, (int, float)):
            return (ti.exp(x) - ti.exp(-x)) * 0.5
        if isinstance(x, ti.Field):
            shape = x.shape
            out = ti.field(dtype=ti.f64, shape=shape)

            @ti.kernel
            def fill_sinh(field: ti.template(), out: ti.template()):
                for I in ti.grouped(field):
                    out[I] = (ti.exp(field[I]) - ti.exp(-field[I])) * 0.5

            fill_sinh(x, out)
            return out

    @staticmethod
    def trace(x: ti.Field, k: int = 0) -> Union[float, int]:  # TODO
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
            >>> bm.trace(mat)
            15.0  # 1 + 5 + 9

            # Offset diagonal (k=1)
            >>> bm.trace(mat, k=1)
            8.0  # 2 + 6

            # Offset diagonal (k=-1)
            >>> bm.trace(mat, k=-1)
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
                raise ValueError(
                    f"The offset k for trace must be an integer, but currently it is {k}"
                )
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
    def triu(x: Union[ti.Field, list], k: int = 0) -> ti.Field:
        """
        Returns the upper triangular part of a matrix or a 1D array converted to a matrix.

        This method extracts the upper triangular portion of a matrix (2D or higher) or constructs an upper triangular matrix from a 1D array. The upper triangular part includes all elements on or above the specified diagonal, with other elements set to 0.

        Args:
            x (Union[ti.Field, list]):
                Input data. Can be:
                - A 1D Taichi field or list: Converted to a square matrix where the upper triangular part is filled with elements from the input.
                - A 2D Taichi field: Treated as a matrix, with the upper triangular part retained.
                - A Taichi field with >2 dimensions: The last two dimensions are treated as rows and columns, with the upper triangular part of these dimensions retained.
            k (int, optional):
                Diagonal offset. Defaults to 0 (main diagonal).
                - k=0: Main diagonal and above.
                - k>0: Diagonals above the main diagonal (e.g., k=1 skips the main diagonal).
                - k<0: Diagonals below the main diagonal (treated as part of the upper triangle).

        Returns:
            ti.Field:
                Upper triangular part of the input.
                - If input is a list or 1D field: Returns an MxM matrix (M = length of input) with upper triangular elements filled from the input.
                - If input is 2D or higher: Returns a field of the same shape with lower triangular elements zeroed.

        Raises:
            ValueError:
                - If the input Taichi field is None.
                - If the input field has any zero dimensions (Taichi does not support zero-sized fields).
            TypeError:
                - If the input type is not ti.Field or list.

        Notes:
            - For 1D inputs (list or field), the output is an MxM matrix where elements satisfying `i - j ≤ k` (i: row index, j: column index) are filled with the input's j-th element; others are 0.
            - For multi-dimensional fields (>2D), only the last two dimensions are processed (treated as rows and columns).
            - The upper triangular condition is determined by `i - j ≤ k`, where `i` and `j` are indices of the last two dimensions.

        Examples:
            # List input (1D → 3x3 matrix)
            >>> bm.triu([1, 2, 3], k=0)  # Main diagonal and above
            [[1, 2, 3],
            [0, 2, 3],
            [0, 0, 3]]

            # 1D Taichi field (shape (3,) → 3x3 matrix)
            >>> x = ti.field(dtype=ti.f32, shape=(3,))
            >>> x.from_numpy(np.array([1, 2, 3]))
            >>> bm.triu(x, k=-1)  # Includes main diagonal and one below
            [[1, 2, 3],
            [1, 2, 3],
            [0, 2, 3]]

            # 2D Taichi field (shape (3,3))
            >>> mat = ti.field(dtype=ti.f32, shape=(3,3))
            >>> mat.from_numpy(np.array([[1,2,3], [4,5,6], [7,8,9]]))
            >>> bm.triu(mat, k=1)  # Above main diagonal (excludes main diagonal)
            [[0, 2, 3],
            [0, 0, 6],
            [0, 0, 0]]
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
                for i, j in ti.ndrange(M, M):
                    if i - j <= k:
                        out[i, j] = x_field[j]

            fill_matrix()
            return out
        if isinstance(x, ti.Field):
            field = x
            if field is None:
                raise ValueError(
                    "Input field is None. Please provide a valid Taichi field."
                )
            shape = field.shape
            if any(s == 0 for s in shape):
                raise ValueError(
                    f"Input field has zero in its shape {shape}, which is not supported by Taichi."
                )
            dtype = field.dtype

            if len(shape) == 1:
                M = shape[0]
                out = ti.field(dtype=dtype, shape=(M, M))

                @ti.kernel
                def fill_tril_1d():
                    for i, j in ti.ndrange(M, M):
                        if i - j <= k:
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
                        if i - j <= k:
                            out[I] = field[I]

                fill_tril_nd()
                return out
        else:
            raise TypeError(
                f"Unsupported type for tril: {type(x)}. Expected ti.Field or list."
            )

    @staticmethod
    def flatten(field: ti.Field) -> ti.Field:
        """
        Flattens a Taichi field into a 1D array in row-major (C-style) order.

        This static method converts a multi-dimensional Taichi field into a one-dimensional field
        by concatenating all elements in row-major order (last dimension changes fastest). The
        flattened array preserves the data type of the original field.

        Args:
            field (ti.Field):
                Input Taichi field to be flattened. Can be of any dimension (1D, 2D, 3D, etc.).

        Returns:
            ti.Field:
                A new 1D Taichi field containing all elements from the input field in row-major order.

        Raises:
            TypeError:
                If the input is not a Taichi field.

        Notes:
            - Uses Taichi's parallel computation to efficiently flatten the array.
            - Row-major order means that the last dimension is traversed first, followed by the second-to-last, etc.
            For example, a 2D array [[1, 2], [3, 4]] flattens to [1, 2, 3, 4].
            - The flattened array's length is the product of all dimensions of the original field.
            - This method computes strides manually to determine the linear index for each multi-dimensional index.

        Examples:
            # 2D field flattening
            >>> x = ti.field(dtype=ti.i32, shape=(2, 3))
            >>> # Initialize x to [[1, 2, 3], [4, 5, 6]]
            >>> flat = bm.flatten(x)
            # flat is [1, 2, 3, 4, 5, 6]

            # 3D field flattening
            >>> x = ti.field(dtype=ti.f32, shape=(2, 2, 2))
            >>> # Initialize x to [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
            >>> flat = bm.flatten(x)
            # flat is [1, 2, 3, 4, 5, 6, 7, 8]
        """
        ndim = len(field.shape)

        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * field.shape[i + 1]

        total_size = 1
        for s in field.shape:
            total_size *= s

        flat = ti.field(field.dtype, shape=total_size)

        @ti.kernel
        def flt():
            for multi_idx in ti.grouped(field):
                flat_idx = 0
                for i in ti.static(range(ndim)):
                    flat_idx += multi_idx[i] * strides[i]
                flat[flat_idx] = field[multi_idx]

        flt()
        return flat

    @staticmethod
    def unique(
        a: ti.Field,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
        axis: Optional[int] = None,
    ) -> Union[ti.Field, Tuple[ti.Field, ...]]:
        """
        Finds the unique elements of a Taichi field.

        This function is similar to `numpy.unique`. It returns the sorted unique elements of an
        input field. It can optionally also return the indices of the first occurrences of the
        unique values, the inverse indices to reconstruct the input field, and the number of
        times each unique value appears in the input field. The function can operate on a
        flattened version of the field or along a specified axis.

        Args:
            a (ti.Field): The input Taichi field.
            return_index (bool, optional): If True, also return the indices of `a` that
                result in the unique array. Defaults to False.
            return_inverse (bool, optional): If True, also return the indices of the unique
                array that can be used to reconstruct `a`. Defaults to False.
            return_counts (bool, optional): If True, also return the number of times each
                unique value appears in `a`. Defaults to False.
            axis (Optional[int], optional): The axis along which to find unique elements.
                If None, the field is flattened before the operation. Defaults to None.

        Returns:
            Union[ti.Field, Tuple[ti.Field, ...]]:
                - If all `return_*` flags are False, returns a `ti.Field` containing the
                sorted unique elements.
                - If any `return_*` flag is True, returns a tuple of `ti.Field`s:
                `(unique_elements, [unique_indices], [inverse_indices], [unique_counts])`,
                where the optional fields are included based on the flags.

        Raises:
            ValueError: If `axis` is specified for a 1D input field and is not 0 or None.

        Example:
            # 1D Field
            x = ti.field(dtype=ti.i32, shape=(8,))
            x.from_numpy(np.array([1, 3, 2, 3, 1, 4, 2, 1]))
            unique_vals = TaichiBackend.unique(x)
            # unique_vals will contain [1, 2, 3, 4]

            # 2D Field with axis=0
            y = ti.field(dtype=ti.i32, shape=(3, 2))
            y.from_numpy(np.array([[1, 2], [1, 2], [3, 4]]))
            unique_rows = TaichiBackend.unique(y, axis=0)
            # unique_rows will be a 2x2 field containing [[1, 2], [3, 4]]
        """
        ndim = len(a.shape)
        if ndim == 1:
            if axis is None or axis == 0:
                return TaichiBackend._unique_1d(
                    a, a, return_index, return_inverse, return_counts
                )
            raise ValueError("axis is not supported for 1D input.")
        if axis is None:
            a_org = a
            a = TaichiBackend.flatten(a)
            return TaichiBackend._unique_1d(
                a_org, a, return_index, return_inverse, return_counts
            )
        return TaichiBackend._unique_generic(
            a, return_index, return_inverse, return_counts, axis
        )

    @staticmethod
    def _unique_1d(
        a_org: ti.Field,
        a: ti.Field,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
    ) -> Union[ti.Field, Tuple[ti.Field, ...]]:
        """
        Internal helper to find unique elements in a 1D or flattened field.

        This private function implements the core logic for finding unique elements in a 1D
        Taichi field. It operates in several stages:
        1. A Taichi kernel (`find_unique`) iterates through the input field to discover
        unique elements, their first occurrence index, and their counts. It also builds
        a preliminary inverse map.
        2. A second kernel (`sort_unique`) sorts the discovered unique elements. It does
        this by sorting an index map rather than moving the elements directly.
        3. A third kernel (`copy_sorted_results`) constructs the final output fields based on
        the sorted order. It remaps the inverse indices to align with the sorted unique
        elements.
        4. If the original input was a multi-dimensional field that was flattened, the
        inverse index field is reshaped back to the original shape.

        Args:
            a_org (ti.Field): The original input field (before any flattening). This is used
                to correctly shape the `return_inverse` output.
            a (ti.Field): The 1D `ti.Field` to be processed for unique elements.
            return_index (bool): Flag to control returning first-occurrence indices.
            return_inverse (bool): Flag to control returning the inverse map.
            return_counts (bool): Flag to control returning element counts.

        Returns:
            Union[ti.Field, Tuple[ti.Field, ...]]:
                A single `ti.Field` of unique elements, or a tuple containing the unique
                elements and other requested arrays as `ti.Field`s.

        Notes:
            This is a private method and not intended for direct user invocation. The logic
            to remap the inverse indices after sorting can be computationally intensive, as
            it involves a search within the `copy_sorted_results` kernel.
        """
        n = a.shape[0]
        max_unique = n

        unique_elements = ti.field(a.dtype, shape=max_unique)
        inverse = ti.field(ti.i32, shape=n)
        counts = ti.field(ti.i32, shape=max_unique)
        indices = ti.field(ti.i32, shape=max_unique)
        unique_count = ti.field(ti.i32, shape=())
        unique_count[None] = 0

        @ti.kernel
        def find_unique():
            for i in range(n):
                element = a[i]
                is_new = 1
                current_unique = unique_count[None]

                for j in range(current_unique):
                    if unique_elements[j] == element:
                        is_new = 0
                        inverse[i] = j
                        ti.atomic_add(counts[j], 1)
                        break

                if is_new:
                    new_idx = ti.atomic_add(unique_count[None], 1)
                    if new_idx < max_unique:
                        unique_elements[new_idx] = element
                        indices[new_idx] = i
                        inverse[i] = new_idx
                        counts[new_idx] = 1

        find_unique()
        n_unique = unique_count[None]
        sorted_indices = ti.field(ti.i32, shape=n_unique)

        @ti.kernel
        def sort_unique():
            for i in range(n_unique):
                sorted_indices[i] = i

            for i in range(n_unique):
                for j in range(n_unique - 1 - i):
                    if (
                        unique_elements[sorted_indices[j]]
                        > unique_elements[sorted_indices[j + 1]]
                    ):
                        sorted_indices[j], sorted_indices[j + 1] = (
                            sorted_indices[j + 1],
                            sorted_indices[j],
                        )

        sort_unique()

        final_unique = ti.field(a.dtype, shape=n_unique)
        final_indices = ti.field(ti.i32, shape=n_unique)
        final_counts = ti.field(ti.i32, shape=n_unique)
        final_inverse_1d = ti.field(ti.i32, shape=n)

        @ti.kernel
        def copy_sorted_results():
            for i in range(n_unique):
                idx = sorted_indices[i]
                final_unique[i] = unique_elements[idx]
                final_indices[i] = indices[idx]
                final_counts[i] = counts[idx]

            if return_inverse:
                for i in range(n):
                    old_idx = inverse[i]
                    for j in range(n_unique):
                        if sorted_indices[j] == old_idx:
                            final_inverse_1d[i] = j
                            break

        copy_sorted_results()

        result = (final_unique,)

        if return_index:
            result += (final_indices,)

        if return_inverse:
            final_inverse = ti.field(ti.i32, shape=a_org.shape)

            @ti.kernel
            def reshape_inverse():
                for multi_idx in ti.grouped(final_inverse):
                    flat_idx = 0
                    stride = 1
                    for i in ti.static(range(len(a_org.shape) - 1, -1, -1)):
                        flat_idx += multi_idx[i] * stride
                        stride *= a_org.shape[i]
                    final_inverse[multi_idx] = final_inverse_1d[flat_idx]

            reshape_inverse()
            result += (final_inverse,)

        if return_counts:
            result += (final_counts,)

        return result[0] if len(result) == 1 else result

    @staticmethod
    def _unique_generic(
        a: ti.Field,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
        axis: int = 0,
    ) -> Union[ti.Field, Tuple[ti.Field, ...]]:
        """
        Internal helper to find unique slices along a specified axis in an N-D field.

        This private function handles the generic case of finding unique "slices" along a
        given axis in a multi-dimensional field. The process involves:
        1. Transposing the input field `a` so that the specified `axis` becomes the first
        axis (axis 0). This simplifies processing.
        2. Identifying unique slices along the new axis 0.
        3. Sorting these unique slices lexicographically using a bubble sort algorithm adapted
        for Taichi kernels.
        4. If `return_inverse` or `return_counts` is requested, a separate kernel is run to
        re-iterate over the transposed data and compute these values based on the final
        sorted unique slices.
        5. Transposing the final unique slices back so the unique dimension is on the
        original `axis`.
        6. Assembling and returning the final results.

        Args:
            a (ti.Field): The N-dimensional input Taichi field.
            return_index (bool): Flag to control returning first-occurrence indices.
            return_inverse (bool): Flag to control returning the inverse map.
            return_counts (bool): Flag to control returning element counts.
            axis (int): The axis along which to find unique slices.

        Returns:
            Union[ti.Field, Tuple[ti.Field, ...]]:
                A `ti.Field` containing the unique slices, or a tuple containing the unique
                slices and other requested arrays. The shape of the unique slices field will
                match the input shape, except for the dimension of `axis`, which will be the
                number of unique slices found.

        Raises:
            ValueError: If the specified `axis` is out of bounds for the input field `a`.

        Notes:
            This is a private method. The use of transposition is a key strategy to simplify
            the problem. The calculation of inverse and counts is a separate, potentially
            expensive step that requires a full pass over the data after the unique slices have
            been found and sorted.
        """
        shape = a.shape
        ndim = len(shape)
        if axis < 0 or axis >= ndim:
            raise ValueError(
                f"axis {axis} out of bounds for array with {ndim} dimensions"
            )

        axes_order: List[int] = [axis] + [i for i in range(ndim) if i != axis]
        transposed_shape = tuple(shape[i] for i in axes_order)
        transposed_a = ti.field(a.dtype, shape=transposed_shape)

        @ti.kernel
        def transpose_input():
            for indices_vec in ti.grouped(a):
                transposed_indices = ti.Vector([0] * ndim)
                for i in ti.static(range(ndim)):
                    transposed_indices[i] = indices_vec[axes_order[i]]
                transposed_a[transposed_indices] = a[indices_vec]

        transpose_input()

        max_unique = transposed_shape[0]
        unique_elements = ti.field(a.dtype, shape=(max_unique, *transposed_shape[1:]))
        inverse = ti.field(ti.i32, shape=transposed_shape[0])
        counts = ti.field(ti.i32, shape=max_unique)
        indices = ti.field(ti.i32, shape=max_unique)
        unique_count = ti.field(ti.i32, shape=())
        unique_count[None] = 0

        @ti.kernel
        def find_unique_elements():
            for i in range(transposed_shape[0]):
                is_new = 1
                for j in range(unique_count[None]):
                    same = 1
                    for sub_idx in ti.grouped(ti.ndrange(*transposed_shape[1:])):
                        if transposed_a[i, sub_idx] != unique_elements[j, sub_idx]:
                            same = 0
                            break
                    if same:
                        is_new = 0
                        break

                if is_new == 1:
                    new_idx = ti.atomic_add(unique_count[None], 1)
                    if new_idx < max_unique:
                        for sub_idx in ti.grouped(ti.ndrange(*transposed_shape[1:])):
                            unique_elements[new_idx, sub_idx] = transposed_a[i, sub_idx]
                        indices[new_idx] = i

        find_unique_elements()
        n_unique = unique_count[None]

        @ti.func
        def is_greater(idx1: int, idx2: int) -> bool:
            result = False
            is_decided = False
            for sub_idx in ti.grouped(ti.ndrange(*transposed_shape[1:])):
                if not is_decided:
                    val1 = unique_elements[idx1, sub_idx]
                    val2 = unique_elements[idx2, sub_idx]
                    if val1 > val2:
                        result = True
                        is_decided = True
                    elif val1 < val2:
                        result = False
                        is_decided = True
            return result

        @ti.func
        def swap_elements(i: int, j: int):
            for sub_idx in ti.grouped(ti.ndrange(*transposed_shape[1:])):
                tmp = unique_elements[i, sub_idx]
                unique_elements[i, sub_idx] = unique_elements[j, sub_idx]
                unique_elements[j, sub_idx] = tmp

            if ti.static(return_index):
                tmp_idx = indices[i]
                indices[i] = indices[j]
                indices[j] = tmp_idx

        @ti.kernel
        def sort_unique():
            for i in range(n_unique - 1):
                for j in range(n_unique - 1 - i):
                    if is_greater(j, j + 1):
                        swap_elements(j, j + 1)

        if n_unique > 1:
            sort_unique()

        if return_inverse or return_counts:

            @ti.kernel
            def update_inverse_and_counts():
                for i in range(n_unique):
                    counts[i] = 0

                for i in range(transposed_shape[0]):
                    for j in range(n_unique):
                        same = 1
                        for sub_idx in ti.grouped(ti.ndrange(*transposed_shape[1:])):
                            if transposed_a[i, sub_idx] != unique_elements[j, sub_idx]:
                                same = 0
                                break
                        if same:
                            inverse[i] = j
                            ti.atomic_add(counts[j], 1)
                            break

            update_inverse_and_counts()

        final_shape = list(a.shape)
        final_shape[axis] = n_unique
        final_unique = ti.field(a.dtype, shape=tuple(final_shape))

        lm = len(a.shape)
        inverse_axes_order = [axes_order.index(i) for i in range(lm)]

        @ti.kernel
        def transpose_final():
            for full_idx in ti.grouped(final_unique):
                mid_idx = ti.Vector([0] * lm, dt=ti.i32)
                for i in ti.static(range(lm)):
                    mapped_dim = inverse_axes_order[i]
                    mid_idx[mapped_dim] = full_idx[i]
                final_unique[full_idx] = unique_elements[mid_idx]

        transpose_final()

        final_indices = None
        if return_index:
            final_indices = ti.field(ti.i32, shape=n_unique)

            @ti.kernel
            def copy_indices():
                for i in range(n_unique):
                    final_indices[i] = indices[i]

            copy_indices()

        final_counts = None
        if return_counts:
            final_counts = ti.field(ti.i32, shape=n_unique)

            @ti.kernel
            def copy_counts():
                for i in range(n_unique):
                    final_counts[i] = counts[i]

            copy_counts()

        final_inverse = None
        if return_inverse:

            final_inverse = ti.field(ti.i32, shape=transposed_shape[0])
            final_inverse.copy_from(inverse)

        result = (final_unique,)
        if final_indices is not None:
            result += (final_indices,)
        if final_inverse is not None:
            result += (final_inverse,)
        if final_counts is not None:
            result += (final_counts,)

        return result[0] if len(result) == 1 else result

    @staticmethod
    def ones(
        shape: Union[int, Tuple[int, ...]], dtype: Optional[Dtype] = ti.f64
    ) -> ti.Field:
        """
        Creates a Taichi field filled with ones.

        This function generates a Taichi field of a specified shape and data type, where all elements are initialized to 1.
        The function supports both scalar shapes (int) and tuple shapes (Tuple[int, ...]).

        Args:
            shape (Union[int, Tuple[int, ...]]):
                The shape of the output field. Can be:
                    - An `int` for a 1D field.
                    - A `Tuple[int, ...]` for multi-dimensional fields.
            dtype (Optional[ti.DataType]):
                The data type of the field elements. Defaults to `ti.f64`.

        Returns:
            ti.Field:
                A Taichi field of the specified shape and data type, filled with ones.

        Raises:
            ValueError: If the shape is not an int or a tuple of ints.
            ValueError: If any dimension of the shape is zero.

        Notes:
            The function uses `ti.field` to create the field and `ti.cast` to ensure the fill value is of the correct data type.

        Example:
            # 1D field usage:
            field_1d = bm.ones(5)

            # 2D field usage:
            field_2d = bm.ones((3, 4), dtype=bm.i32)
        """
        if not isinstance(shape, (int, tuple)) or (
            isinstance(shape, tuple) and not all(isinstance(dim, int) for dim in shape)
        ):
            raise ValueError("Shape must be an int or a Tuple[int, ...].")

        x = ti.field(shape=shape, dtype=dtype)
        fill_value = ti.cast(1, dtype)

        @ti.kernel
        def fill_like():
            x.fill(fill_value)

        fill_like()
        return x

    @staticmethod
    def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[bool, int, float], dtype: Optional[Dtype] = None) -> ti.Field:  # type: ignore
        """
        Creates a Taichi field filled with a specified value.

        This function generates a Taichi field of a specified shape and data type, where all elements are initialized to the provided `fill_value`.
        The function supports both scalar shapes (int) and tuple shapes (Tuple[int, ...]).

        Args:
            shape (Union[int, Tuple[int, ...]]):
                The shape of the output field. Can be:
                    - An `int` for a 1D field.
                    - A `Tuple[int, ...]` for multi-dimensional fields.
            fill_value (Union[bool, int, float]):
                The value to fill the field with. Can be a boolean, integer, or float.
            dtype (Optional[ti.DataType]):
                The data type of the field elements. If not provided, the data type is inferred from the `fill_value`. Defaults to `None`.

        Returns:
            ti.Field:
                A Taichi field of the specified shape and data type, filled with the specified `fill_value`.

        Raises:
            ValueError: If the shape is not an int or a tuple of ints.
            ValueError: If any dimension of the shape is zero.
            TypeError: If the `fill_value` type is not supported.

        Notes:
            The function uses `ti.field` to create the field and `ti.cast` to ensure the fill value is of the correct data type.
            If `dtype` is not specified, it is inferred based on the type of `fill_value`:
                - `ti.u1` for boolean values.
                - `ti.i32` for integer values.
                - `ti.f64` for float values.

        Example:
            # 1D field usage:
            field_1d = bm.full(5, 2.5)

            # 2D field usage:
            field_2d = bm.full((3, 4), True)

            # 3D field usage:
            field_3d = bm.full((2, 3, 4), 3)

        """
        if not isinstance(shape, (int, tuple)) or (
            isinstance(shape, tuple) and not all(isinstance(dim, int) for dim in shape)
        ):
            raise ValueError("Shape must be an int or a Tuple[int, ...].")

        if dtype is None:
            if isinstance(fill_value, bool):
                dtype = ti.u1  # Boolean type in Taichi
            elif isinstance(fill_value, int):
                dtype = ti.i32  # Default integer type
            elif isinstance(fill_value, float):
                dtype = ti.f64  # Default floating-point type
            else:
                raise TypeError("Unsupported fill_value type.")
        x = ti.field(dtype=dtype, shape=shape)

        @ti.kernel
        def fill_like():
            x.fill(fill_value)

        fill_like()
        return x

    @staticmethod
    def ones_like(field: ti.Field) -> ti.Field:
        """
        Creates a Taichi field with the same shape as the given field, filled with ones.

        This function generates a Taichi field that has the same shape and data type as the provided `field`, and all elements are initialized to 1.
        If the `field` is of boolean type, the fill value will be 1 (True).

        Args:
            field (ti.Field):
                The input field whose shape and data type will be copied.

        Returns:
            ti.Field:
                A Taichi field with the same shape and data type as the input field, filled with ones.

        Raises:
            ValueError: If the shape of the input field is zero.

        Notes:
            The function uses `ti.field` to create the field and `ti.cast` to ensure the fill value is of the correct data type.
            If the `field` is of boolean type (`ti.u1`), the fill value will be 1 (True).

        Example:
            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
            new_field = bm.ones_like(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([1, 2, 3], dtype=np.int32))
            new_field_int = bm.ones_like(field_int)

            # Field usage with boolean dtype:
            field_bool = ti.field(dtype=ti.u1, shape=(3,))
            field_bool.from_numpy(np.array([1, 0, 1], dtype=np.uint8))
            new_field_bool = bm.ones_like(field_bool)
        """
        x = ti.field(shape=field.shape, dtype=field.dtype)
        fill_value = ti.cast(1, field.dtype)

        @ti.kernel
        def fill_like():
            x.fill(fill_value)

        fill_like()
        return x

    @staticmethod
    def full_like(field: ti.Field, fill_value: Union[bool, int, float], dtype: Optional[Dtype] = None) -> ti.Field:  # type: ignore
        """
        Creates a Taichi field with the same shape as the given field, filled with a specified value.

        This function generates a Taichi field that has the same shape as the provided `field`, and all elements are initialized to the provided `fill_value`.
        The data type of the new field can be specified, or it will be inferred from the `fill_value`.

        Args:
            field (ti.Field):
                The input field whose shape will be copied.
            fill_value (Union[bool, int, float]):
                The value to fill the new field with. Can be a boolean, integer, or float.
            dtype (Optional[ti.DataType]):
                The data type of the field elements. If not provided, the data type is inferred from the `fill_value`. Defaults to `None`.

        Returns:
            ti.Field:
                A Taichi field with the same shape as the input field and the specified `fill_value`.

        Raises:
            ValueError: If the shape of the input field is zero.
            TypeError: If the `fill_value` type is not supported.

        Notes:
            The function uses `ti.field` to create the field and `ti.cast` to ensure the fill value is of the correct data type.
            If `dtype` is not specified, it is inferred based on the type of `fill_value`:
                - `ti.u1` for boolean values.
                - `ti.i32` for integer values.
                - `ti.f64` for float values.

        Example:
            # Field usage:
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))

            new_field = bm.full_like(field, 5.0)

        """
        if dtype is None:
            if isinstance(fill_value, bool):
                dtype = ti.u1  # Boolean type in Taichi
            elif isinstance(fill_value, int):
                dtype = ti.i32  # Default integer type
            elif isinstance(fill_value, float):
                dtype = ti.f64  # Default floating-point type
            else:
                raise TypeError("Unsupported fill_value type.")

        x = ti.field(dtype=dtype, shape=field.shape)

        @ti.kernel
        def fill_like():
            x.fill(fill_value)

        fill_like()
        return x

    @staticmethod
    def acosh(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the inverse hyperbolic cosine (acosh) of the input.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed value.
        For Taichi fields, it computes the acosh value for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value(s). Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed acosh values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.

        Notes:
            The inverse hyperbolic cosine is computed as:
                acosh(x) = log(x + sqrt(x * x - 1))
            The input values should be in the range [1, +∞) for real results.

        Example:
            # Scalar usage:
            result_scalar = bm.acosh(1.5)

            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([1.5, 2.0, 3.0], dtype=np.float64))
            result_field = bm.acosh(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([2, 3, 4], dtype=np.int32))
            result_field_int = bm.acosh(field_int)
        """
        if isinstance(x, (int, float)):
            return ti.log(x + ti.sqrt(x * x - 1.0))

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def compute_acosh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                u = ti.cast(field[I], ti.f64)
                result[I] = ti.log(u + ti.sqrt(u * u - 1.0))

        compute_acosh(x, result)

        return result

    @staticmethod
    def asinh(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the inverse hyperbolic sine (asinh) of the input.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed value.
        For Taichi fields, it computes the asinh value for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value(s). Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed asinh values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.

        Notes:
            The inverse hyperbolic sine is computed as:
                asinh(x) = log(x + sqrt(x * x + 1))
            The input values can be any real number.

        Example:
            # Scalar usage:
            result_scalar = bm.asinh(0.5)

            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([0.5, 1.0, 2.0], dtype=np.float64))
            result_field = bm.asinh(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([1, 2, 3], dtype=np.int32))
            result_field_int = bm.asinh(field_int)
        """
        if isinstance(x, (int, float)):
            return ti.log(x + ti.sqrt(x * x + 1.0))

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def compute_asinh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                u = ti.cast(field[I], ti.f64)
                result[I] = ti.log(u + ti.sqrt(u * u + 1.0))

        compute_asinh(x, result)

        return result

    @staticmethod
    def add(
        x: Union[int, float, ti.Field], y: Union[int, float, ti.Field]
    ) -> Union[int, float, ti.Field]:
        """
        Adds two values or fields element-wise.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed sum.
        For Taichi fields, it computes the element-wise sum of the two fields and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The first input value or field. Can be a scalar `float` or `int`, or a `ti.Field`.
            y (Union[ti.Field, float, int]):
                The second input value or field. Must be of the same type as `x` (either both scalars or both Taichi fields).

        Returns:
            Union[float, int, ti.Field]:
                - If both inputs are scalars, returns the sum as a `float` or `int`.
                - If both inputs are Taichi fields, returns a new `ti.Field` of the same shape
                and data type, containing the element-wise sum.

        Raises:
            TypeError: If either input is not a Taichi field when both are not scalars.
            TypeError: If the types of `x` and `y` are mismatched (one is a scalar and the other is a field).
            ValueError: If the input fields do not have the same shape.

        Notes:
            The function uses an element-wise kernel to compute the sum of the fields.
            The behavior of the function is similar to element-wise addition in NumPy.

        Example:
            # Scalar usage:
            result_scalar = bm.add(0.5, 1.5)

            # Field usage with the same dtype:
            field_1 = ti.field(dtype=ti.f64, shape=(3,))
            field_1.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))

            field_2 = ti.field(dtype=ti.f64, shape=(3,))
            field_2.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))

            result_field = bm.add(field_1, field_2)

        """
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return x + y
        if not isinstance(x, ti.Field) or not isinstance(y, ti.Field):
            raise TypeError("Both inputs must be ti.Field or scalar")

        if x.shape != y.shape:
            raise ValueError("Input fields must have the same shape")

        @ti.kernel
        def add_field(x: ti.template(), y: ti.template(), z: ti.template()):

            for I in ti.grouped(x):
                z[I] = x[I] + y[I]

        z = ti.field(dtype=x.dtype, shape=x.shape)
        add_field(x, y, z)
        return z

    @staticmethod
    def atanh(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the inverse hyperbolic tangent (atanh) of the input.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed value.
        For Taichi fields, it computes the atanh value for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value(s). Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed atanh values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.

        Notes:
            The inverse hyperbolic tangent is computed as:
                atanh(x) = 0.5 * log((1 + x) / (1 - x))
            The input values should be in the range (-1, 1) for real results.

        Example:
            # Scalar usage:
            result = bm.atanh(0.5)

            # Field usage:
            x = bm.field(dtype=ti.f64, shape=(4,))
            x.from_numpy(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64))
            result_field = bm.atanh(x)
        """
        if isinstance(x, (int, float)):

            if x == 1.0:
                return ti.math.inf
            else:
                return ti.log((1.0 + x) / (1.0 - x)) / 2.0

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_atanh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                u = ti.cast(field[I], ti.f64)
                if u == 1.0:
                    result[I] = ti.math.inf
                else:
                    result[I] = ti.log((1.0 + u) / (1.0 - u)) / 2.0

        compute_atanh(x, result)

        return result

    @staticmethod
    def equal(
        x: Union[int, float, ti.Field], y: Union[int, float, ti.Field]
    ) -> Union[bool, ti.Field]:  # type: ignore
        """
        Compares two values or fields element-wise for equality.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the result of the equality comparison.
        For Taichi fields, it computes the element-wise equality comparison and returns a new field
        containing the boolean results.

        Args:
            x (Union[ti.Field, float, int]):
                The first input value or field. Can be a scalar `float` or `int`, or a `ti.Field`.
            y (Union[ti.Field, float, int]):
                The second input value or field. Must be of the same type as `x` (either both scalars or both Taichi fields).

        Returns:
            Union[bool, ti.Field]:
                - If both inputs are scalars, returns the result of the equality comparison as a `bool`.
                - If both inputs are Taichi fields, returns a new `ti.Field` of the same shape and boolean data type (`ti.u1`), containing the element-wise equality results.

        Raises:
            TypeError: If either input is not a Taichi field when both are not scalars.
            TypeError: If the types of `x` and `y` are mismatched (one is a scalar and the other is a field).
            ValueError: If the input fields do not have the same shape.

        Notes:
            The function uses an element-wise kernel to perform the equality comparison.
            The result field for Taichi fields is of boolean type (`ti.u1`), where each element is `1` (True) if the corresponding elements in `x` and `y` are equal, and `0` (False) otherwise.
            Broadcasting is not implemented, so both fields must have the same shape.

        Example:
            # Scalar usage:
            result_scalar = bm.equal(0.5, 0.5)

            # Scalar usage with inequality:
            result_scalar_ineq = bm.equal(0.5, 1.5)

            # Field usage with the same shape:
            field_1 = ti.field(dtype=ti.f64, shape=(3,))
            field_1.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))

            field_2 = ti.field(dtype=ti.f64, shape=(3,))
            field_2.from_numpy(np.array([1.0, 2.0, 4.0], dtype=np.float64))

            result_field = bm.equal(field_1, field_2)
        """
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return x == y
        if not isinstance(x, ti.Field) or not isinstance(y, ti.Field):
            raise TypeError("Both inputs must be ti.Field or scalar")
        if x.shape != y.shape:  # TODO 未实现广播操作
            raise ValueError("Input fields must have the same shape")

        @ti.kernel
        def equal_field(x: ti.template(), y: ti.template(), z: ti.template()):

            for I in ti.grouped(x):
                z[I] = x[I] == y[I]

        z = ti.field(dtype=ti.u1, shape=x.shape)
        equal_field(x, y, z)
        return z

    @staticmethod
    def exp(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the exponential (e^x) of the input.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed exponential value.
        For Taichi fields, it computes the exponential value for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value or field. Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed exponential values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.

        Notes:
            The exponential function is computed as:  e^x
            The function uses `ti.exp` to compute the exponential of each element.

        Example:
            # Scalar usage:
            result_scalar = bm.exp(0.5)

            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([0.5, 1.0, 2.0], dtype=np.float64))
            result_field = bm.exp(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([1, 2, 3], dtype=np.int32))
            result_field_int = bm.exp(field_int)
        """
        if isinstance(x, (int, float)):
            return ti.exp(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def compute_exp(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.exp(ti.cast(field[I], ti.f64))

        compute_exp(x, result)

        return result

    @staticmethod
    def expm1(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the exponential of the input minus one (exp(x) - 1).

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed value of `exp(x) - 1`.
        For Taichi fields, it computes the `exp(x) - 1` value for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value or field. Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed `exp(x) - 1` values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.

        Notes:
            The function uses a more accurate approximation for small values of `x` to avoid loss of precision.
            For `|x| < 1e-5`, the approximation used is:
                expm1(x) ≈ x * (1 + x * (0.5 + x * (1 / 3)))
            For larger values of `x`, it directly computes `exp(x) - 1`.

        Example:
            # Scalar usage:
            result_scalar = bm.expm1(0.5)

            # Scalar usage with small value:
            result_scalar_small = bm.expm1(1e-6)

            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([0.5, 1e-6, 2.0], dtype=np.float64))
            result_field = bm.expm1(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([1, 2, 3], dtype=np.int32))
            result_field_int = bm.expm1(field_int)
        """
        if isinstance(x, (int, float)):
            if ti.abs(x) < 1e-5:
                return x * (1 + x * (0.5 + x * (1 / 3)))
            else:
                return ti.exp(x) - 1

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def compute_expm1(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                u = ti.cast(field[I], ti.f64)
                if ti.abs(u) < 1e-5:

                    result[I] = u * (1 + u * (0.5 + u * (1 / 3)))
                else:
                    result[I] = ti.exp(u) - 1

        compute_expm1(x, result)

        return result

    @staticmethod
    def log(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the natural logarithm (logarithm base e) of the input.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed natural logarithm value.
        For Taichi fields, it computes the natural logarithm for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value or field. Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed natural logarithm values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.
            ValueError: If the input values are not in the range (0, +∞) for real results.

        Notes:
            The natural logarithm is computed as:
                log(x)
            The input values should be in the range (0, +∞) for real results.
            The function uses `ti.log` to compute the natural logarithm of each element.

        Example:
            # Scalar usage:
            result_scalar = bm.log(2.718)

            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
            result_field = bm.log(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([1, 2, 3], dtype=np.int32))
            result_field_int = bm.log(field_int)
        """
        if isinstance(x, (int, float)):
            return ti.log(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def compute_log(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.log(ti.cast(field[I], ti.f64))

        compute_log(x, result)

        return result

    @staticmethod
    def log1p(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the natural logarithm (logarithm base e) of 1 + x.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed natural logarithm value.
        For Taichi fields, it computes the natural logarithm for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value or field. Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed natural logarithm values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.

        Notes:
            The natural logarithm of 1 + x is computed as:
                log(1 + x)
            The input values should be in the range (-1, +∞) for real results.
            For small values of x, a Taylor series expansion is used to improve precision.
            The function uses `ti.log` to compute the natural logarithm of each element.

        Example:
            # Scalar usage:
            result_scalar = bm.log1p(0.5)

            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([0.1, 0.2, 0.3], dtype=np.float64))
            result_field = bm.log1p(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([1, 2, 3], dtype=np.int32))
            result_field_int = bm.log1p(field_int)
        """
        if isinstance(x, (int, float)):
            if ti.abs(x) > 1e-4:
                return ti.log(1.0 + x)
            else:
                return x * (1 + x * (-0.5 + x * (1 / 3)))

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def compute_log1p(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                u = ti.cast(field[I], ti.f64)
                if ti.abs(u) > 1e-4:
                    result[I] = ti.log(1.0 + u)
                else:
                    result[I] = u * (1 + u * (-0.5 + u * (1 / 3)))

        compute_log1p(x, result)

        return result

    @staticmethod
    def sqrt(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the square root of the input.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed square root value.
        For Taichi fields, it computes the square root for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value or field. Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed square root values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.
            ValueError: If the input values are negative, as the square root of a negative number is not defined in the real number system.

        Notes:
            The square root is computed as:
                sqrt(x)
            The input values should be non-negative for real results.
            The function uses `ti.sqrt` to compute the square root of each element.

        Example:
            # Scalar usage:
            result_scalar = bm.sqrt(4.0)

            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([1.0, 4.0, 9.0], dtype=np.float64))
            result_field = bm.sqrt(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([1, 4, 9], dtype=np.int32))
            result_field_int = bm.sqrt(field_int)
        """
        if isinstance(x, (int, float)):
            return ti.sqrt(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def compute_sqrt(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.sqrt(ti.cast(field[I], ti.f64))

        compute_sqrt(x, result)

        return result

    @staticmethod
    def sign(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the sign of the input.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed sign value.
        For Taichi fields, it computes the sign for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value or field. Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed sign values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.

        Notes:
            The sign function returns:
                - 1.0 if x is positive.
                - -1.0 if x is negative.
                - 0.0 if x is zero.
            The function uses `ti.math.sign` to compute the sign of each element.

        Example:
            # Scalar usage:
            result_scalar = bm.sign(4.0)  # Returns 1.0
            result_scalar = bm.sign(-2.0) # Returns -1.0
            result_scalar = bm.sign(0.0)  # Returns 0.0

            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([1.0, -4.0, 0.0], dtype=np.float64))
            result_field = bm.sign(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([1, -4, 0], dtype=np.int32))
            result_field_int = bm.sign(field_int)
        """
        if isinstance(x, (int, float)):

            @ti.kernel
            def compute_sign_scalar(x: ti.template()) -> float:
                return ti.math.sign(x)

            return compute_sign_scalar(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_sign(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.math.sign(field[I])

        compute_sign(x, result)

        return result

    @staticmethod
    def tan(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the tangent of the input.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed tangent value.
        For Taichi fields, it computes the tangent for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value or field. Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed tangent values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.

        Notes:
            The tangent function is computed as:
                tan(x)
            The input values should be in radians.
            The tangent function is periodic with a period of π and has vertical asymptotes at x = (2k + 1)π/2 for integer k.
            The function uses `ti.tan` to compute the tangent of each element.

        Example:
            # Scalar usage:
            result_scalar = bm.tan(0.5)  # Returns the tangent of 0.5 radians

            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([0.1, 0.5, 1.0], dtype=np.float64))
            result_field = bm.tan(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([0, 1, 2], dtype=np.int32))
            result_field_int = bm.tan(field_int)
        """
        if isinstance(x, (int, float)):
            return ti.tan(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def compute_tan(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.tan(ti.cast(field[I], ti.f64))

        compute_tan(x, result)

        return result

    @staticmethod
    def tanh(x: Union[int, float, ti.Field]) -> Union[float, ti.Field]:
        """
        Computes the hyperbolic tangent of the input.

        This function supports both scalar values (float or int) and Taichi fields.
        For scalar inputs, it directly returns the computed hyperbolic tangent value.
        For Taichi fields, it computes the hyperbolic tangent for each element and returns a new field
        containing the results.

        Args:
            x (Union[ti.Field, float, int]):
                The input value or field. Can be:
                    - A scalar `float` or `int`.
                    - A `ti.Field` containing numerical values.

        Returns:
            Union[ti.Field, float]:
                - If the input is a scalar, returns a `float`.
                - If the input is a `ti.Field`, returns a new `ti.Field` of the same shape
                and data type, containing the computed hyperbolic tangent values element-wise.

        Raises:
            TypeError: If the input is neither a scalar (float or int) nor a `ti.Field`.

        Notes:
            The hyperbolic tangent is computed as:
                tanh(x)
            The function uses `ti.tanh` to compute the hyperbolic tangent of each element.
            The hyperbolic tangent function is defined for all real numbers and has a range of (-1, 1).

        Example:
            # Scalar usage:
            result_scalar = bm.tanh(0.5)  # Returns the hyperbolic tangent of 0.5

            # Field usage with default dtype (f64):
            field = ti.field(dtype=ti.f64, shape=(3,))
            field.from_numpy(np.array([0.1, 0.5, 1.0], dtype=np.float64))
            result_field = bm.tanh(field)

            # Field usage with default dtype (i32):
            field_int = ti.field(dtype=ti.i32, shape=(3,))
            field_int.from_numpy(np.array([0, 1, 2], dtype=np.int32))
            result_field_int = bm.tanh(field_int)
        """
        if isinstance(x, (int, float)):
            return ti.tanh(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=ti.f64, shape=shape)

        @ti.kernel
        def compute_tanh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.tanh(ti.cast(field[I], ti.f64))

        compute_tanh(x, result)

        return result

    @staticmethod
    def cross(field_1: ti.Field, field_2: ti.Field) -> ti.Field:
        """
        Computes the cross product of two input fields.

        This function supports 1D vectors of length 2 (2D vectors) and length 3 (3D vectors).
        For 2D vectors, it computes the scalar cross product (which is the determinant of the 2x2 matrix formed by the vectors).
        For 3D vectors, it computes the vector cross product and returns a new field containing the result.

        Args:
            field_1 (ti.Field):
                The first input field. Must be a 1D vector of length 2 or 3.
            field_2 (ti.Field):
                The second input field. Must be a 1D vector of length 2 or 3.

        Returns:
            ti.Field:
                - If the input fields are 2D vectors, returns a scalar `ti.Field` containing the result.
                - If the input fields are 3D vectors, returns a new `ti.Field` of the same shape and data type,
                containing the computed cross product vector element-wise.

        Raises:
            TypeError: If both inputs are not `ti.Field`.
            ValueError: If the input fields do not have the same shape or are not 1D vectors of length 2 or 3.

        Notes:
            The cross product for 2D vectors is defined as:
                cross_product = field_1[0] * field_2[1] - field_1[1] * field_2[0]
            The cross product for 3D vectors is defined as:
                cross_product[0] = field_1[1] * field_2[2] - field_2[1] * field_1[2]
                cross_product[1] = field_2[0] * field_1[2] - field_1[0] * field_2[2]
                cross_product[2] = field_1[0] * field_2[1] - field_2[0] * field_1[1]

        Example:
            # 2D vector usage:
            field_1_2d = ti.field(dtype=ti.f64, shape=(2,))
            field_1_2d.from_numpy(np.array([1.0, 2.0], dtype=np.float64))
            field_2_2d = ti.field(dtype=ti.f64, shape=(2,))
            field_2_2d.from_numpy(np.array([3.0, 4.0], dtype=np.float64))
            result_2d = bm.cross(field_1_2d, field_2_2d)

            # 3D vector usage:
            field_1_3d = ti.field(dtype=ti.f64, shape=(3,))
            field_1_3d.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
            field_2_3d = ti.field(dtype=ti.f64, shape=(3,))
            field_2_3d.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float64))
            result_3d = bm.cross(field_1_3d, field_2_3d)
        """
        if not isinstance(field_1, ti.Field) or not isinstance(field_2, ti.Field):
            raise TypeError("Both inputs must be ti.Field")
        if field_1.shape != field_2.shape:
            raise ValueError("Input fields must have the same shape")
        shape = field_1.shape
        if len(shape) != 1 or shape[0] not in (2, 3):
            raise ValueError("Input fields must be 1D vectors of length 2 or 3")
        dim = shape[0]
        result_shape = (1,) if dim == 2 else shape
        result = ti.field(dtype=field_1.dtype, shape=result_shape)

        @ti.kernel
        def compute_cross_2d(
            field_1: ti.template(), field_2: ti.template(), result: ti.template()
        ):
            result[0] = field_1[0] * field_2[1] - field_1[1] * field_2[0]

        @ti.kernel
        def compute_cross_3d(
            field_1: ti.template(), field_2: ti.template(), result: ti.template()
        ):
            result[0] = field_1[1] * field_2[2] - field_2[1] * field_1[2]
            result[1] = field_2[0] * field_1[2] - field_1[0] * field_2[2]
            result[2] = field_1[0] * field_2[1] - field_2[0] * field_1[1]

        if ti.static(dim == 2):
            compute_cross_2d(field_1, field_2, result)
        else:
            compute_cross_3d(field_1, field_2, result)

        return result
