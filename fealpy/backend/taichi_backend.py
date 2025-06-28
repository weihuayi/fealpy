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
    def to_list(field: ti.Field, /) -> list:
        if field is None:
            return []
        try:
            shape = field.shape
            if len(shape) == 0:
                return field[None]
            elif len(shape) == 1:
                return [field[i] for i in range(shape[0])]
            elif len(shape) == 2:
                return [[field[i, j] for j in range(shape[1])] for i in range(shape[0])]
            else:
                raise ValueError("Currently only 0D, 1D and 2D fields are supported.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    @staticmethod
    def arange(*args, dtype=ti.i32):
        # 解析参数
        if len(args) == 1:
            start, stop, step = 0, args[0], 1
        elif len(args) == 2:
            start, stop = args
            step = 1
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise ValueError("arange expects 1~3 arguments (stop | start, stop | start, stop, step)")

        # 计算元素个数
        if step == 0:
            raise ValueError("step must not be zero")
        n = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
        if n == 0:
            return None

        field = ti.field(dtype=dtype, shape=(n,))
        
        @ti.kernel
        def fill():
            for i in range(n):
                field[i] = start + i * step

        fill()
        return field
    
    @staticmethod
    def eye(N, M=None, k=0, dtype=ti.f32):
        if M is None:
            M = N
        if N <= 0 or M <= 0:
            return None
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
            shape = (shape,)
        if any(s == 0 for s in shape):
            return None
        field = ti.field(dtype=dtype, shape=shape)

        # @ti.kernel
        # def fill_zeros():
        #     for I in ti.grouped(field):
        #         field[I] = 0

        # fill_zeros()
        return field
    
    @staticmethod
    def zeros_like(field: ti.Field):
        """
        创建一个和输入 field 形状、dtype 相同的全零 field。
        """
        if field is None:
            return None
        shape = field.shape
        if any(s == 0 for s in shape):
            return None
        dtype = field.dtype
        out = ti.field(dtype=dtype, shape=shape)

        @ti.kernel
        def fill_zeros():
            for I in ti.grouped(out):
                out[I] = 0

        fill_zeros()
        return out

    @staticmethod
    def tril(field_or_shape, k: int = 0, dtype=ti.f32):
        # 如果是 int 或 tuple，先创建全1 field
        if isinstance(field_or_shape, int):
            shape = (field_or_shape, field_or_shape)
            field = ti.field(dtype=dtype, shape=shape)
            @ti.kernel
            def fill_ones():
                for I in ti.grouped(field):
                    field[I] = 1
            fill_ones()
        elif isinstance(field_or_shape, tuple):
            shape = field_or_shape
            field = ti.field(dtype=dtype, shape=shape)
            @ti.kernel
            def fill_ones():
                for I in ti.grouped(field):
                    field[I] = 1
            fill_ones()
        else:
            field = field_or_shape
            shape = field.shape
            dtype = field.dtype

        if field is None or len(shape) != 2 or any(s == 0 for s in shape):
            return None

        out = ti.field(dtype=dtype, shape=shape)
        N, M = shape

        @ti.kernel
        def fill_tril():
            for i, j in ti.ndrange(N, M):
                if j - i <= k:
                    out[i, j] = field[i, j]
                else:
                    out[i, j] = 0

        fill_tril()
        return out

    @staticmethod
    def abs(field: ti.Field):
        """
        返回每个元素的绝对值，结果为新的 field.
        """
        shape = field.shape
        if any(s == 0 for s in shape):
            return None
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
        """
        返回每个元素的反余弦值，结果为新的 field.
        """
        shape = field.shape
        if any(s == 0 for s in shape):
            return None
        dtype = field.dtype
        out = ti.field(dtype=dtype, shape=shape)

        @ti.kernel
        def fill_acos():
            for I in ti.grouped(field):
                out[I] = ti.acos(field[I])

        fill_acos()
        return out


if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    a=TaichiBackend.zeros((3,4))
    print(a)