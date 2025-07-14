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
    def set_default_device(device: Union[str, ti.cpu, ti.cuda]) -> None:  # type: ignore
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
            ti.init(arch=device, default_fp=ti.f64, default_ip=ti.i32)
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
        field = ti.field(dtype=dtype_map[ndarray.dtype], shape=ndarray.shape)
        field.from_numpy(ndarray)
        return field

    @staticmethod
    def tolist(field: ti.Field, /) -> list:
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
    def arange(*args, dtype=ti.i32):
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

        n = max(1, abs((stop - start + (step - (1 if step > 0 else -1))) // step))

        field = ti.field(dtype=dtype, shape=(n,))

        @ti.kernel
        def fill():
            for i in range(n):
                field[i] = start + i * step

        fill()
        return field

    @staticmethod
    def eye(N, M=None, k=0, dtype=ti.f32):
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
        if not isinstance(N, int) or not isinstance(M, int):
            raise TypeError(f"N and M must be integers, got N={N}, M={M}.")
        if N == 0 or M == 0:
            return []
        if N < 0 or M < 0:
            raise ValueError(f"N and M must be positive integers, got N={N}, M={M}.")
        field = ti.field(dtype=dtype, shape=(N, M))

        @ti.kernel
        def fill_eye():
            for i, j in ti.ndrange(N, M):
                if j - i == k:
                    field[i, j] = 1
                else:
                    field[i, j] = 0

        fill_eye()
        return field

    @staticmethod
    def zeros(shape, dtype=ti.f32):
        # 支持 int 或 tuple 作为 shape
        if isinstance(shape, int):
            if shape == 0:
                return []
            if shape < 0:
                raise ValueError(f"Shape must be a non-negative integer, got {shape}.")
            if shape > 0:
                shape = (shape,)
        if any(s == 0 for s in shape):
            raise ValueError(
                f"Input field has zero in its shape {shape}, which is not supported by Taichi."
            )
        field = ti.field(dtype=dtype, shape=shape)

        @ti.kernel
        def fill_zeros():
            for I in ti.grouped(field):
                field[I] = 0

        fill_zeros()
        return field

    @staticmethod
    def zeros_like(field: ti.Field):
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

        @ti.kernel
        def fill_zeros():
            for I in ti.grouped(out):
                out[I] = 0

        fill_zeros()
        return out

    @staticmethod
    def tril(field: ti.Field, k: int = 0):
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
        if len(shape) == 0:
            raise ValueError(
                "Input field is a scalar (0D), tril is not defined for scalars."
            )
        dtype = field.dtype

        if len(shape) == 1:
            M = shape[0]
            out = ti.field(dtype=dtype, shape=(M, M))

            @ti.kernel
            def fill_tril_1d():
                for i, j in ti.ndrange(M, M):
                    if j - i <= k:
                        out[i, j] = field[j]
                    else:
                        out[i, j] = 0

            fill_tril_1d()
            return out

        elif len(shape) == 2:
            N, M = shape
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_tril_2d():
                for i, j in ti.ndrange(N, M):
                    if j - i <= k:
                        out[i, j] = field[i, j]
                    else:
                        out[i, j] = 0

            fill_tril_2d()
            return out

        else:
            raise ValueError(
                f"Input field with shape {shape} is not supported. Only 1D and 2D fields are supported."
            )

    @staticmethod
    def abs(field: ti.Field):
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
        dtype = field.dtype
        out = ti.field(dtype=dtype, shape=shape)

        @ti.kernel
        def fill_abs():
            for I in ti.grouped(field):
                out[I] = ti.abs(field[I])

        fill_abs()
        return out

    @staticmethod
    def acos(field: ti.Field):
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
        dtype = field.dtype
        out = ti.field(dtype=dtype, shape=shape)

        @ti.kernel
        def fill_acos():
            for I in ti.grouped(field):
                out[I] = ti.acos(field[I])

        fill_acos()
        return out

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
        if shape == 0 or shape == (0,):
            raise ValueError("Shape dimensions must be greater than 0.")
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
        if shape == 0 or shape == (0,):
            raise ValueError("Shape dimensions must be greater than 0.")
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
    def acosh(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):
            return ti.log(x + ti.sqrt(x * x - 1.0))

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_acosh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.log(field[I] + ti.sqrt(field[I] * field[I] - 1.0))

        compute_acosh(x, result)

        return result

    @staticmethod
    def asinh(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):
            return ti.log(x + ti.sqrt(x * x + 1.0))

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_asinh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.log(field[I] + ti.sqrt(field[I] * field[I] + 1.0))

        compute_asinh(x, result)

        return result

    @staticmethod
    def add(
        x: Union[ti.Field, float, int], y: Union[ti.Field, float, int]
    ) -> Union[float, int, ti.Field]:
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
        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
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
    def atanh(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):

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
                result[I] = ti.log((1.0 + field[I]) / (1.0 - field[I])) / 2.0

        compute_atanh(x, result)

        return result

    @staticmethod
    def equal(
        x: Union[ti.Field, float, int], y: Union[ti.Field, float, int]
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
        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            return x == y

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
    def exp(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):
            return ti.exp(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_exp(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.exp(field[I])

        compute_exp(x, result)

        return result

    @staticmethod
    def expm1(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):
            if ti.abs(x) < 1e-5:
                return x * (1 + x * (0.5 + x * (1 / 3)))
            else:
                return ti.exp(x) - 1

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_expm1(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                if ti.abs(field[I]) < 1e-5:
                    result[I] = field[I] * (1 + field[I] * (0.5 + field[I] * (1 / 3)))
                else:
                    result[I] = ti.exp(field[I]) - 1

        compute_expm1(x, result)

        return result

    @staticmethod
    def log(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):
            return ti.log(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_log(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.log(field[I])

        compute_log(x, result)

        return result

    @staticmethod
    def log1p(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):
            if ti.abs(x) > 1e-4:
                return ti.log(1.0 + x)
            else:
                return x * (1 + x * (-0.5 + x * (1 / 3)))

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_log1p(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                if ti.abs(field[I]) > 1e-4:
                    result[I] = ti.log(1.0 + field[I])
                else:
                    result[I] = field[I] * (1 + field[I] * (-0.5 + field[I] * (1 / 3)))

        compute_log1p(x, result)

        return result

    @staticmethod
    def sqrt(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):
            return ti.sqrt(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_sqrt(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.sqrt(field[I])

        compute_sqrt(x, result)

        return result

    @staticmethod
    def sign(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):

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
    def tan(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):
            return ti.tan(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_tan(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.tan(field[I])

        compute_tan(x, result)

        return result

    @staticmethod
    def tanh(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
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
        if isinstance(x, (float, int)):
            return ti.tanh(x)

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        shape = x.shape

        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_tanh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.tanh(field[I])

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
