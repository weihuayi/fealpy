from typing import Any, Union, Optional, TypeVar, Tuple
import numpy as np

try:
    import taichi as ti
    import taichi.math as tm
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
            ti.init(arch=device)
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

        if not isinstance(shape, (int, tuple)) or (
            isinstance(shape, tuple) and not all(isinstance(dim, int) for dim in shape)
        ):
            raise ValueError("Shape must be an int or a Tuple[int, ...].")
        if shape == 0 or shape == (0,):
            raise ValueError("Shape dimensions must be greater than 0.")
        x = ti.field(shape=shape, dtype=dtype)
        fill_value = ti.cast(1, dtype)
        x.fill(fill_value)
        return x

    @staticmethod
    def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[bool, int, float], dtype: Optional[Dtype] = None) -> ti.Field:  # type: ignore
        if not isinstance(shape, (int, tuple)) or (
            isinstance(shape, tuple) and not all(isinstance(dim, int) for dim in shape)
        ):
            raise ValueError("Shape must be an int or a Tuple[int, ...].")
        if shape == 0 or shape == (0,):
            raise ValueError("Shape dimensions must be greater than 0.")
        if dtype is None:
            if isinstance(fill_value, bool):
                dtype = ti.u8  # Boolean type in Taichi
            elif isinstance(fill_value, int):
                dtype = ti.i32  # Default integer type
            elif isinstance(fill_value, float):
                dtype = ti.f64  # Default floating-point type
            else:
                raise TypeError("Unsupported fill_value type.")
        x = ti.field(dtype=dtype, shape=shape)
        x.fill(fill_value)
        return x

    @staticmethod
    def ones_like(field: ti.Field) -> ti.Field:

        x = ti.field(shape=field.shape, dtype=field.dtype)
        fill_value = ti.cast(1, field.dtype)
        x.fill(fill_value)
        return x

    @staticmethod
    def full_like(field: ti.Field, fill_value: Union[bool, int, float], dtype: Optional[Dtype] = None) -> ti.Field:  # type: ignore

        if dtype is None:
            if isinstance(fill_value, bool):
                dtype = ti.u8  # Boolean type in Taichi
            elif isinstance(fill_value, int):
                dtype = ti.i32  # Default integer type
            elif isinstance(fill_value, float):
                dtype = ti.f64  # Default floating-point type
            else:
                raise TypeError("Unsupported fill_value type.")

        x = ti.field(dtype=dtype, shape=field.shape)
        x.fill(fill_value)
        return x

    @staticmethod
    def acosh(x: Union[ti.Field, float]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, float):
            if x < 1.0:
                raise ValueError(
                    "Input value is out of the domain for acosh (must be >= 1.0)"
                )
            return ti.log(x + ti.sqrt(x * x - 1.0))

        # 如果输入是 ti.Field
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a float")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        # 创建一个标志字段来标记错误
        error_flag = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def compute_acosh(
            field: ti.template(), result: ti.template(), error_flag: ti.template()
        ):
            error_flag[None] = 0  # 初始化错误标志为 0
            for I in ti.grouped(field):
                if field[I] < 1.0:
                    error_flag[None] = 1  # 设置错误标志为 1
                else:
                    result[I] = ti.log(field[I] + ti.sqrt(field[I] * field[I] - 1.0))

        compute_acosh(x, result, error_flag)

        if error_flag[None] == 1:
            raise ValueError(
                "Input value is out of the domain for acosh (must be >= 1.0)"
            )

        if len(shape) == 1 and shape[0] == 1:
            # 如果结果是一个单值的 ti.Field，返回其单值
            return result[None]

        return result

    @staticmethod
    def asinh(x: Union[ti.Field, float]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, float):
            return ti.log(x + ti.sqrt(x * x + 1.0))

        # 如果输入是 ti.Field
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a float")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_asinh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.log(field[I] + ti.sqrt(field[I] * field[I] + 1.0))

        compute_asinh(x, result)

        if len(shape) == 1 and shape[0] == 1:
            # 如果结果是一个单值的 ti.Field，返回其单值
            return result[None]

        return result

    @staticmethod
    def add(x: ti.Field, y: ti.Field) -> ti.Field:
        if not isinstance(x, ti.Field) or not isinstance(y, ti.Field):
            raise TypeError("Both inputs must be ti.Field")

        if x.shape != y.shape:
            raise ValueError("Input fields must have the same shape")

        @ti.kernel
        def add_field(x: ti.template(), y: ti.template(), z: ti.template()):

            for I in ti.grouped(x):
                z[I] = x[I] + y[I]  # taichi math库里面有atomic_add函数

        z = ti.field(dtype=x.dtype, shape=x.shape)
        add_field(x, y, z)
        return z
