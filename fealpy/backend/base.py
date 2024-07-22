
from abc import ABCMeta
from typing import Union, Optional, Dict, Any, Tuple, Type, Generic, TypeVar

from .. import logger


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


def _make_default_mapping(*names: str):
    return {k: k for k in names}

Number = Union[int, float, TensorLike]
_DT = TypeVar("_DT")
ATTRIBUTE_MAPPING = _make_default_mapping(
    'pi', 'e', 'nan', 'inf', 'dtype', 'device',
    'bool_', 'uint8', 'int_', 'int8', 'int16', 'int32', 'int64',
    'float_', 'float16', 'float32', 'float64',
    'complex_', 'complex64', 'complex128'
)
CREATION_MAPPING = _make_default_mapping(
    # Creation functions
    'array', 'tensor', 'arange', 'linspace',
    'empty', 'zeros', 'ones', 'empty_like', 'zeros_like', 'ones_like'
)
REDUCTION_MAPPING = _make_default_mapping(
    # Reduction functions
    'all', 'any', 'sum', 'prod', 'mean', 'max', 'min'
)
UNARY_MAPPING = _make_default_mapping(
    # Unary functions
    'abs', 'sign', 'sqrt', 'log', 'log10', 'log2', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
    'reshape', 'ravel', 'flatten', 'broadcast_to', 'einsum'
)
BINARY_MAPPING = _make_default_mapping(
    # Binary functions
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot', 'cross', 'tensordot'
)
OTHER_MAPPING = _make_default_mapping(
    # Other functions
    'reshape', 'broadcast_to', 'einsum', 'unique', 'sort', 'nonzero',
    'cumsum', 'cumprod', 'cat', 'concatenate', 'stack', 'transpose'
)


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
    def attach_attributes(cls, mapping: Dict[str, str], source: Any):
        for target_key, source_key in mapping.items():
            if (source_key is None) or (source_key == ''):
                continue
            if hasattr(source, source_key):
                setattr(cls, target_key, getattr(source, source_key))

    @classmethod
    def attach_methods(cls, mapping: Dict[str, str], source: Any):
        for target_key, source_key in mapping.items():
            if (source_key is None) or (source_key == ''):
                continue
            if hasattr(cls, target_key):
                # Methods will not be copied from source if implemented manually.
                logger.debug(f"{target_key} is already defined. "
                             f"Skip the copy from {source.__name__}.")
                continue
            if hasattr(source, source_key):
                setattr(cls, target_key, staticmethod(getattr(source, source_key)))
            else:
                logger.debug(f"{source_key} is not found in {source.__name__}. "
                             f"Method {target_key} remains unimplemented.")

    @classmethod
    def show_unsupported(cls, signal: bool, function_name: str, arg_name: str) -> None:
        if signal:
            logger.warning(f"{cls.__name__} does not support the "
                           f"'{arg_name}' argument in the function {function_name}. "
                           f"The argument will be ignored.")

    @classmethod
    def is_tensor(cls, obj: Any, /) -> bool:
        return isinstance(obj, cls.DATA_CLASS)
