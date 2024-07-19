
from abc import ABCMeta
from typing import Optional, Dict, Any, Tuple, Type, Generic, TypeVar


Size = tuple[int, ...]

class TensorLike(metaclass=ABCMeta):
    @property
    def shape(self) -> Tuple[int, ...]: raise NotImplementedError
    @property
    def dtype(self) -> Any: raise NotImplementedError
    @property
    def device(self) -> Any: raise NotImplementedError
    @property
    def ndim(self) -> int: raise NotImplementedError
    @property
    def size(self) -> int: raise NotImplementedError


_DT = TypeVar("_DT")
ATTRIBUTE_MAPPING = {
    'pi': 'pi',
    'e': 'e',
    'nan': 'nan',
    'inf': 'inf',
    'dtype': 'dtype',
    'device': 'device',
    'bool_': 'bool_',
    'int_': 'int_',
    'int8': 'int8',
    'int16': 'int16',
    'int32': 'int32',
    'int64': 'int64',
    'float_': 'float_',
    'float16': 'float16',
    'float32': 'float32',
    'float64': 'float64',
    'complex_': 'complex_',
    'complex64': 'complex64',
    'complex128': 'complex128'
}

class Backend(Generic[_DT]):
    """
    Base class for all backends.
    """
    DATA_CLASS: Optional[Type[_DT]] = None
    _available_backends: Dict[str, Type["Backend"]] = {}

    def __init_subclass__(cls, backend_name: str, **kwargs):
        super().__init_subclass__(**kwargs)

        if backend_name != "":
            cls._available_backends[backend_name.lower()] = cls
            cls.backend_name = backend_name
            TensorLike.register(cls.DATA_CLASS)
        else:
            raise ValueError("Backend name cannot be empty.")

    @classmethod
    def attach_attributes(cls, mapping: Dict[str, str], source: type):
        for target_key, source_key in mapping.items():
            if (source_key is None) or (source_key == ''):
                continue
            if hasattr(source, source_key):
                setattr(cls, target_key, getattr(source, source_key))
