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
            ti.init(arch=device,default_ip=ti.i32, default_fp=ti.f64)
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
    def arange(*args, dtype=ti.f64):
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
        if (step > 0 and last < stop - 1e-8) or (step < 0 and last > stop + 1e-8):
            n += 1
            
        if isinstance(n, float):
            if not n.is_integer():
                raise ValueError(f"arange 生成的长度 n 必须为整数，当前为 {n}")
            n = int(n)

        field = ti.field(dtype=dtype, shape=(n,))

        @ti.kernel
        def fill():
            for i in range(n):
                field[i] = start + i * step

        fill()
        return field

    @staticmethod
    def eye(N, M=None, k=0, dtype=ti.f64):
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

        for name, v in [('N', N), ('M', M)]:
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
            for i, j in ti.ndrange(N, M):
                if j - i == k:
                    field[i, j] = 1
                else:
                    field[i, j] = 0

        fill_eye()
        return field

    @staticmethod
    def zeros(shape, dtype=ti.f64):
        if isinstance(shape, float):
            if not shape.is_integer():
                raise TypeError(f"Shape must be an integer or tuple of integers, got float {shape}.")
            shape = int(shape)
        if isinstance(shape, int):
            if shape == 0:
                return []
            if shape < 0:
                raise ValueError(f"Shape must be a non-negative integer, got {shape}.")
            if shape > 0:
                shape = (shape,)
        if isinstance(shape, tuple):
            new_shape = []
            for s in shape:
                if isinstance(s, float):
                    if not s.is_integer():
                        raise TypeError(f"Shape elements must be integers, got float {s}.")
                    s = int(s)
                if not isinstance(s, int):
                    raise TypeError(f"Shape elements must be integers, got {type(s)}.")
                new_shape.append(s)
            shape = tuple(new_shape)
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
    def tril(field: ti.Field, k: int = 0) -> ti.Field:
        if field is None:
            raise ValueError("Input field is None. Please provide a valid Taichi field.")
        if not hasattr(field, "shape") or not hasattr(field, "dtype"):
            raise TypeError("Input is not a valid Taichi field: missing 'shape' or 'dtype' attribute.")
        shape = field.shape
        if any(s == 0 for s in shape):
            raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
        if len(shape) == 0:
            raise ValueError("Input field is a scalar (0D), tril is not defined for scalars.")
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
            raise ValueError("Input field is a scalar (0D), tril is not defined for scalars.")
        

    @staticmethod
    def abs(
        x: Union[int, float, bool, ti.Field]
    ) -> Union[int, float, bool, ti.Field]:
        if isinstance(x, (int, float, bool)):
            return abs(x)
        if isinstance(x, ti.Field):
            if x is None:
                raise ValueError("Input field is None. Please provide a valid Taichi field.")        
            if not hasattr(x, "shape") or not hasattr(x, "dtype"):
                raise TypeError("Input is not a valid Taichi field: missing 'shape' or 'dtype' attribute.")
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
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
                f"Unsupported type for abs: {type(x)}. Expected int, float, bool, or ti.Field."
            )

    @staticmethod
    def acos(
        x: Union[int, float, bool, ti.Field]
    ) -> Union[float, ti.Field]:
        if isinstance(x, (int, float, bool)):
            if x < -1 or x > 1:
                raise ValueError(f"Input value {x} is out of domain for acos (must be in [-1, 1])")
            return float(ti.acos(x))
        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
            dtype = x.dtype
            out = ti.field(dtype=dtype, shape=shape)
            error_flag = ti.field(dtype=ti.i32, shape=())


            @ti.kernel
            def fill_acos(field: ti.template, out: ti.template, error_flag: ti.template):
                error_flag[None] = 0
                for I in ti.grouped(field):
                    if field[I] < -1 or field[I] > 1:
                        error_flag[None] = 1
                    else:
                        out[I] = ti.acos(field[I])

            fill_acos(x, out, error_flag)
            if error_flag[None] == 1:
                raise ValueError("Some elements are out of domain for acos (must be in [-1, 1])")
            if len(shape) == 0:
                return out[None]
            return out
        raise TypeError(
            f"Unsupported type for acos: {type(x)}. Expected int, float, bool, or ti.Field."
        )

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
    def acosh(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):
            return ti.log(x + ti.sqrt(x * x - 1.0))

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_acosh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.log(field[I] + ti.sqrt(field[I] * field[I] - 1.0))

        compute_acosh(x, result)

        if len(shape) == 1 and shape[0] == 1:
            # 如果结果是一个单值的 ti.Field，返回其单值
            return result[None]

        return result

    @staticmethod
    def asinh(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):
            return ti.log(x + ti.sqrt(x * x + 1.0))

        # 如果输入是 ti.Field
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

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
            result = MyClass.atanh(0.5)

            # Field usage:
            x = ti.field(dtype=ti.f32, shape=(4,))
            x.from_numpy(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
            result_field = MyClass.atanh(x)
        """
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):
            if x == 1.0:
                return ti.math.inf
            else:
                return ti.log((1.0 + x) / (1.0 - x)) / 2.0

        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_atanh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.log((1.0 + field[I]) / (1.0 - field[I])) / 2.0

        compute_atanh(x, result)

        return result


    @staticmethod
    def equal(x: ti.Field, y: ti.Field) -> ti.Field:
        if not isinstance(x, ti.Field) or not isinstance(y, ti.Field):
            raise TypeError("Both inputs must be ti.Field")

        if x.shape != y.shape:
            raise ValueError("Input fields must have the same shape")

        @ti.kernel
        def equal_field(x: ti.template(), y: ti.template(), z: ti.template()):

            for I in ti.grouped(x):
                if x[I] == y[I]:
                    z[I] = True
                else:
                    z[I] = False

        z = ti.field(dtype=ti.u1, shape=x.shape)
        equal_field(x, y, z)
        return z


    @staticmethod
    def exp(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):
            return ti.exp(x)

        # 如果输入是 ti.Field
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_exp(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.exp(field[I])

        compute_exp(x, result)

        return result
    

    @staticmethod
    def expm1(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):
            if ti.abs(x) < 1e-5:
                return x + (x * x) / 2 + (x * x * x) / 6
            else:
                return ti.exp(x) - 1
        # 如果输入是 ti.Field
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_expm1(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                if ti.abs(field[I]) < 1e-5:
                    result[I] = (
                        field[I]
                        + (field[I] * field[I]) / 2
                        + (field[I] * field[I] * field[I]) / 6
                    )
                else:
                    result[I] = ti.exp(field[I]) - 1

        compute_expm1(x, result)

        return result


    @staticmethod
    def log(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):
            return ti.log(x)

        # 如果输入是 ti.Field
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_log(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.log(field[I])

        compute_log(x, result)


        return result


    @staticmethod
    def log1p(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):
            if ti.abs(x) > 1e-4:
                return ti.log(1.0 + x)
            else:
                return x - (x * x) / 2 + (x * x * x) / 3

        # 如果输入是 ti.Field
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_log1p(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                if ti.abs(field[I]) > 1e-4:
                    result[I] = ti.log(1.0 + field[I])
                else:
                    result[I] = (
                        field[I]
                        - (field[I] * field[I]) / 2
                        + (field[I] * field[I] * field[I]) / 3
                    )

        compute_log1p(x, result)

        return result


    @staticmethod
    def sqrt(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):
            return ti.sqrt(x)

        # 如果输入是 ti.Field
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_sqrt(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.sqrt(field[I])

        compute_sqrt(x, result)

        return result


    @staticmethod
    def sign(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):

            @ti.kernel
            def compute_sign_scalar(x: ti.template()) -> float:
                return tm.sign(x)

            return compute_sign_scalar(x)
       
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_sign(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = tm.sign(field[I])

        compute_sign(x, result)

        return result

    @staticmethod
    def tan(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):
            return ti.tan(x)

        # 如果输入是 ti.Field
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_tan(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.tan(field[I])

        compute_tan(x, result)

        return result
    
    @staticmethod
    def tanh(x: Union[ti.Field, float, int]) -> Union[ti.Field, float]:
        # 检查输入是否是单值（标量）
        if isinstance(x, (float, int)):
            return ti.tanh(x)

        # 如果输入是 ti.Field
        if not isinstance(x, ti.Field):
            raise TypeError("Input must be a ti.Field or a scalar")

        # 获取矩阵的形状
        shape = x.shape

        # 创建一个新的 ti.Field 来存储结果
        result = ti.field(dtype=x.dtype, shape=shape)

        @ti.kernel
        def compute_tanh(field: ti.template(), result: ti.template()):
            for I in ti.grouped(field):
                result[I] = ti.tanh(field[I])

        compute_tanh(x, result)

        return result

    @staticmethod
    def cross(field_1: ti.Field, field_2: ti.Field) -> ti.Field:

        if not isinstance(field_1, ti.Field) or not isinstance(field_2, ti.Field):
            raise TypeError("Both inputs must be ti.Field")
        if field_1.shape != field_2.shape:
            raise ValueError("Input fields must have the same shape")
        dim = field_1.shape[0]
        if len(field_1.shape) != 1 or dim not in (2, 3):
            raise ValueError("Input fields must be 1D vectors of length 2 or 3")
        if dim == 2:
            result_shape = (1,)
        else:
            result_shape = field_1.shape
        result = ti.field(dtype=field_1.dtype, shape=result_shape)

        @ti.kernel
        def compute_cross(field_1: ti.template(), field_2: ti.template()):

            if ti.static(dim == 2):

                vec1 = ti.Vector([field_1[0], field_1[1]])
                vec2 = ti.Vector([field_2[0], field_2[1]])
                result[0] = tm.cross(vec1, vec2)

            else:

                vec1 = ti.Vector([field_1[0], field_1[1], field_1[2]])
                vec2 = ti.Vector([field_2[0], field_2[1], field_2[2]])
                cross_result = tm.cross(vec1, vec2)

                for i in ti.static(range(3)):
                    result[i] = cross_result[i]

        compute_cross(field_1, field_2)
        return result
    
