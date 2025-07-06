from typing import Any, Union, Optional, TypeVar, Tuple
import numpy as np
try:
    import taichi as ti
    import taichi.math as tm
except ImportError:
    raise ImportError("Name 'taichi' cannot be imported. "
                      'Make sure  Taichi is installed before using '
                      'the Taichi backend in FEALPy. '
                      'See https://taichi-lang.cn/ for installation.')

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

from fealpy.backend.base import BackendProxy, ModuleProxy, ATTRIBUTE_MAPPING, FUNCTION_MAPPING, TRANSFORMS_MAPPING

# 假设 BackendProxy 是你自己定义的基类
class TaichiBackend(BackendProxy, backend_name='taichi'):
    DATA_CLASS = ti.Field
    # Holds the current Taichi arch (e.g., ti.cpu or ti.cuda)
    _device: Union[ti.cpu, ti.cuda, None] = None # type: ignore
    
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
        device = 'cpu' if arch == ti.cpu else 'cuda' if arch == ti.cuda else str(arch)
        return {"dtype": tensor.dtype, "device": device}

    @staticmethod
    def set_default_device(device: Union[str, ti.cpu, ti.cuda]) -> None: # type: ignore
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
            if device.lower() == 'cpu':
                device = ti.cpu
            elif device.lower() == 'cuda':
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
    def device_type(field: ti.Field, /):  # type: ignore
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
            raise ValueError("arange expects 1~3 arguments (stop | start, stop | start, stop, step)")

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
            raise ValueError("Both N and M are None. At least one dimension must be specified for eye().")
        if N is None:
            raise ValueError("N is None. The number of rows must be specified for eye().")
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
            raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
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
            raise ValueError("Input field is None. Please provide a valid Taichi field.")
        if not hasattr(field, "shape") or not hasattr(field, "dtype"):
            raise TypeError("Input is not a valid Taichi field: missing 'shape' or 'dtype' attribute.")
        shape = field.shape
        if any(s == 0 for s in shape):
            raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
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
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
            
        shape = x.shape
        if any(s == 0 for s in shape):
            raise ValueError(f"Input field has zero in shape {shape}, not supported by Taichi.")
            
        dtype = x.dtype
        if ti.types.is_integral(dtype):
            dtype = ti.get_default_fp()
            
        out = ti.field(dtype=dtype, shape=shape)
        error_flag = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def fill_acos(
            field: ti.template(), 
            out: ti.template(), 
            error_flag: ti.template()
        ):
            error_flag[None] = 0
            for I in ti.grouped(field):
                val = field[I]
                if val < -1 or val > 1:
                    error_flag[None] = 1
                else:
                    out[I] = ti.acos(val)

        fill_acos(x, out, error_flag)
            
        if error_flag[None] == 1:
            raise ValueError("Some elements are out of domain for acos (must be in [-1, 1])")
            
        return out[None] if len(shape) == 0 else out
        
    @staticmethod
    def asin(
        x: Union[int, float, bool, ti.Field]
    ) -> Union[float, ti.Field]:

        if isinstance(x, (int, float, bool)):         
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
            
        shape = x.shape
        if any(s == 0 for s in shape):
            raise ValueError(f"Input field has zero in shape {shape}, not supported by Taichi.")
            
        dtype = x.dtype
        if ti.types.is_integral(dtype):
            dtype = ti.get_default_fp()
            
        out = ti.field(dtype=dtype, shape=shape)
        error_flag = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def fill_asin(
            field: ti.template(), 
            out: ti.template(), 
            error_flag: ti.template()
        ):
            error_flag[None] = 0
            for I in ti.grouped(field):
                val = field[I]
                if val < -1 or val > 1:
                    error_flag[None] = 1
                else:
                    out[I] = ti.asin(val)

        fill_asin(x, out, error_flag)
            
        if error_flag[None] == 1:
            raise ValueError("Some elements are out of domain for asin (must be in [-1, 1])")
            
        return out[None] if len(shape) == 0 else out
        
        
    @staticmethod
    def atan(
        x: Union[int, float, bool, ti.Field]
    ) -> Union[float, ti.Field]:

        if isinstance(x, (int, float, bool)):
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
            
        shape = x.shape
        if any(s == 0 for s in shape):
            raise ValueError(f"Input field has zero in shape {shape}, not supported by Taichi.")
            
        dtype = x.dtype
        if ti.types.is_integral(dtype):
            dtype = ti.get_default_fp()
            
        out = ti.field(dtype=dtype, shape=shape)
        error_flag = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def fill_atan(
            field: ti.template(), 
            out: ti.template(), 
        ):
            error_flag[None] = 0
            for I in ti.grouped(field):
                val = field[I]
                out[I] = ti.atan2(val,1)

        fill_atan(x, out)
               
        return out[None] if len(shape) == 0 else out
        
    @staticmethod
    def atan2(
        y: Union[int, float, bool, ti.Field],
        x: Union[int, float, bool, ti.Field]
    ) -> Union[float, ti.Field]:
        if isinstance(x, (int, float, bool)):
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
        if isinstance(y, (int, float, bool)):
            b = ti.field(dtype=ti.f64, shape=())
            b[None] = float(y)
            y = b
        shape = y.shape

        if shape != x.shape:
            raise ValueError("Input fields must have the same shape.")
        if any(s == 0 for s in shape):
            raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
        dtype = y.dtype
        if ti.types.is_integral(dtype):
            dtype = ti.get_default_fp()
        out = ti.field(dtype=dtype, shape=shape)

        @ti.kernel
        def fill_atan2(y_field: ti.template(), x_field: ti.template(), out: ti.template()):
            for I in ti.grouped(y_field):
                out[I] = ti.atan2(y_field[I], x_field[I])

        fill_atan2(y, x, out)
        if len(shape) == 0:
            return out[None]
        return out
        
    @staticmethod
    def ceil(x: Union[int, float, bool, ti.Field]) -> Union[int, float, ti.Field]:
        if isinstance(x, (int, float, bool)):
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
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
        x: Union[int, float, bool, ti.Field],
        a_min: Optional[Union[int, float, bool]] = None,
        a_max: Optional[Union[int, float, bool]] = None
    ) -> Union[int, float, ti.Field]:
        if isinstance(x, (int, float, bool)):
            x = float(x)
            if a_min is not None:
                a_min = float(a_min)
                x = max(a_min, x)
            if a_max is not None:
                a_max = float(a_max)
                x = min(x, a_max)
            return x

        if isinstance(x, ti.Field):
            shape = x.shape
            if any(s == 0 for s in shape):
                raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
            dtype = x.dtype
            out = ti.field(dtype=dtype, shape=shape)

            @ti.kernel
            def fill_clip(field: ti.template(), out: ti.template(), a_min: ti.f64, a_max: ti.f64, use_min: ti.i32, use_max: ti.i32):
                for I in ti.grouped(field):
                    v = field[I]
                    if use_min:
                        v = max(a_min, v)
                    if use_max:
                        v = min(a_max, v)
                    out[I] = v

            use_min = int(a_min is not None)
            use_max = int(a_max is not None)

            if dtype in (ti.f32, ti.f64):
                a_min_val = float(a_min) if a_min is not None else 0.0
                a_max_val = float(a_max) if a_max is not None else 0.0
            elif dtype in (ti.i32, ti.i64, ti.u8, ti.u16, ti.u32, ti.u64):
                a_min_val = int(a_min) if a_min is not None else 0
                a_max_val = int(a_max) if a_max is not None else 0
            else:
                a_min_val = float(a_min) if a_min is not None else 0.0
                a_max_val = float(a_max) if a_max is not None else 0.0

            fill_clip(
                x, out,
                a_min_val,
                a_max_val,
                use_min, use_max
            )
            if len(shape) == 0:
                return out[None]
            return out

        raise TypeError(
            f"Unsupported type for clip: {type(x)}. Expected int, float, bool, or ti.Field."
        )
        
    @staticmethod
    def cos(x: Union[int, float, bool, ti.Field]) -> Union[float, ti.Field]:
        if isinstance(x, (int, float, bool)):
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
        shape = x.shape
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
    def cosh(x: Union[int, float, bool, ti.Field]) -> Union[float, ti.Field]:
        if isinstance(x, (int, float, bool)):
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
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
    def floor(x: Union[int, float, bool, ti.Field]) -> Union[int, float, ti.Field]:
        if isinstance(x, (int, float, bool)):
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
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
    def floor_divide(
        x: Union[int, float, bool, ti.Field],
        y: Union[int, float, bool, ti.Field]
    ) -> Union[int, float, ti.Field]:
        if isinstance(x, (int, float, bool)):
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
        if isinstance(y, (int, float, bool)):
            b = ti.field(dtype=ti.f64, shape=())
            b[None] = float(y)
            y = b
        shape = x.shape
        if shape != y.shape:
            raise ValueError("Input fields must have the same shape.")
        if any(s == 0 for s in shape):
            raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
        dtype = x.dtype
        if ti.types.is_integral(dtype):
            dtype = ti.get_default_fp()
        out = ti.field(dtype=dtype, shape=shape)
        has_zero = ti.field(ti.i32, shape=())

        @ti.kernel
        def fill_floor_divide(x_field: ti.template(), y_field: ti.template(), out: ti.template()):
            for I in ti.grouped(y_field):
                out[I] = ti.floor(x_field[I] / y_field[I])
        
        @ti.kernel
        def check_zeros(y_field: ti.template()):
            for I in ti.grouped(y_field):
                if y_field[I] == 0:
                    has_zero[None] = 1
                    
        has_zero[None] = 0
        check_zeros(y) 
        if has_zero[None] == 1:
            raise ZeroDivisionError("Field contains zero values in divisor")     

        else:
            fill_floor_divide(x, y, out)
            if len(shape) == 0:
                return out[None]
            return out
        
    @staticmethod
    def sin(x: Union[int, float, bool, ti.Field]) -> Union[float, ti.Field]:
        if isinstance(x, (int, float, bool)):
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
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
    def sinh(x: Union[int, float, bool, ti.Field]) -> Union[float, ti.Field]:
        if isinstance(x, (int, float, bool)):
            a = ti.field(dtype=ti.f64, shape=())
            a[None] = float(x)
            x = a
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
    def trace(x: ti.Field) -> Union[float, int]:
        if not isinstance(x, ti.Field):
            raise TypeError(f"Unsupported type for trace: {type(x)}. Expected ti.Field.")
        shape = x.shape
        if len(shape) != 2:
            raise ValueError("Input field must be 2D.")
        if shape[0] != shape[1]:
            raise ValueError("Input field must be square (same number of rows and columns).")
        dtype = x.dtype
        trace_value = ti.field(dtype=dtype, shape=())

        @ti.kernel
        def compute_trace(field: ti.template(), trace_value: ti.template()):
            trace_value[None] = 0
            for i in range(shape[0]):
                trace_value[None] += field[i, i]

        compute_trace(x, trace_value)
        return trace_value[None]
    
    @staticmethod
    def insert(x: ti.Field, values: Union[int, float, bool, ti.Vector], indices: Tuple[int, ...]) -> ti.Field:#TO DO
        if not isinstance(x, ti.Field):
            raise TypeError("Input x must be a Taichi Field.")
        if not isinstance(values, (int, float, bool, ti.Vector)):
            raise TypeError("Values must be an int, float, bool, or ti.Vector.")
        if not isinstance(indices, tuple) or not all(isinstance(i, int) for i in indices):
            raise TypeError("Indices must be a tuple of integers.")

        if len(indices) != len(x.shape):
            raise ValueError(f"Indices length {len(indices)} does not match field shape {x.shape}.")

        # 转为 Taichi 向量（如果是多维索引）
        if len(indices) == 0:
            @ti.kernel
            def insert_value(field: ti.template(), value: ti.template()):
                field[None] = value
            insert_value(x, values)
        else:
            @ti.kernel
            def insert_value(field: ti.template(), value: ti.template(), idx: ti.template()):
                field[idx] = value
            insert_value(x, values, indices)
            
    @staticmethod
    def unique(x: ti.Field) -> ti.Field:
        if not isinstance(x, ti.Field):
            raise TypeError("Input x must be a Taichi Field.")
        shape = x.shape
        if any(s == 0 for s in shape):
            raise ValueError(f"Input field has zero in its shape {shape}, which is not supported by Taichi.")
        dtype = x.dtype

        arr = x.to_numpy().flatten()
        unique_values = sorted(set(arr.tolist()))
        n_unique = len(unique_values)
        unique_field = ti.field(dtype=dtype, shape=(n_unique,))
        for i, v in enumerate(unique_values):
            unique_field[i] = v
        return unique_field


    @staticmethod
    def ones(
        shape: Union[int, Tuple[int, ...]], dtype: Optional[Dtype] = float64
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


if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    a=np.ndarray(dtype=np.float32, shape=(3,4))
    for i in range(3):
        for j in range(4):
            a[i, j] = 1
    print(np.insert(a, [1,2], 0, axis=0))