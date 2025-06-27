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

# from fealpy.backend.base import BackendProxy

from .base import (
    BackendProxy, ModuleProxy,
    ATTRIBUTE_MAPPING, FUNCTION_MAPPING, TRANSFORMS_MAPPING
)

# 假设 BackendProxy 是你自己定义的基类
class TaichiBackend(BackendProxy, backend_name='taichi'):
    DATA_CLASS = ti.Field
    # Holds the current Taichi arch (e.g., ti.cpu or ti.cuda)
    _device: Union[ti.cpu, ti.cuda, None] = None
    
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
    def ones(shape: Union[int, Tuple[int, ...]], ) -> ti.Field:
        x = ti.field(shape=shape,dtype=ti.i32)
        x.fill(1)
        return x
    
    @staticmethod
    def full(shape: Union[int, Tuple[int, ...]], element: Union[bool, int, float], dtype: Optional[Dtype] = None) -> ti.Field:  # type: ignore
        if dtype is None:
            if isinstance(element, bool):
                dtype = ti.u8  # Boolean type in Taichi
            elif isinstance(element, int):
                dtype = ti.i32  # Default integer type
            elif isinstance(element, float):
                dtype = ti.f64  # Default floating-point type
            else:
                raise TypeError("Unsupported fill_value type.")

        x = ti.field(dtype=dtype, shape=shape)
        x.fill(element)
        return x
    
    @staticmethod
    def ones_like(field: ti.Field) -> ti.Field:
        if field.shape == (0,):
            return None
        x = ti.field(dtype=ti.i32, shape=field.shape)
        x.fill(1)
        return x
    
    @staticmethod
    def full_like(field: ti.Field, element: Union[bool, int, float], dtype: Optional[Dtype] = None) -> ti.Field:  # type: ignore

        if dtype is None:
            if isinstance(element, bool):
                dtype = ti.u8  # Boolean type in Taichi
            elif isinstance(element, int):
                dtype = ti.i32  # Default integer type
            elif isinstance(element, float):
                dtype = ti.f64  # Default floating-point type
            else:
                raise TypeError("Unsupported fill_value type.")

        x = ti.field(dtype=dtype, shape=field.shape)
        x.fill(element)
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
