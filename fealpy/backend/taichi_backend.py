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
