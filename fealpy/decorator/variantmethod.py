from typing import (
    Tuple, Dict,
    Any, Type, Callable,
    TypeVar, overload, ParamSpec, Generic, Concatenate,
    Optional,
)
from functools import partial

__all__ = "variantmethod", "VariantMeta"

Self = TypeVar('Self', bound='VariantMeta')
_T = TypeVar("_T")
_P = ParamSpec("_P")
_R_co = TypeVar("_R_co", covariant=True)


class Variantmethod(Generic[_T, _P, _R_co]):
    __slots__ = ('virtual_table', 'default_key', 'fselect')
    virtual_table : Dict[Any, Callable]

    def __init__(self, func: Optional[Callable] = None, key: Optional[str] = None):
        if func is None:
            self.virtual_table = {}
        else:
            self.virtual_table = {key: func}
        self.default_key = key
        self.fselect = None

    @property
    def __func__(self) -> Callable[Concatenate[_T, _P], _R_co]:
        return self.virtual_table[self.default_key]

    @property
    def __name__(self) -> str:
        return self.__func__.__name__

    def __get__(self, obj: _T, objtype: Type[_T]) -> Callable[_P, _R_co]:
        if obj is None:
            return self

        if self.fselect is None:
            func = self.virtual_table[self.default_key]
        else:
            key = self.fselect.__get__(obj, objtype)()
            if key in self.virtual_table:
                func = self.virtual_table[key]
            else:
                func = self.virtual_table[self.default_key]

        return func.__get__(obj, objtype)

    def __set__(self, obj: _T, val: Any):
        raise RuntimeError

    def register(self, key: Any, /):
        def decorator(func: Callable):
            self.virtual_table[key] = func
            return self
        return decorator

    def selector(self, fselect: Callable[[_T], Any], /):
        self.fselect = fselect
        return self

    def update(self, other: "Variantmethod", /) -> None:
        if len(self.virtual_table) == 0:
            self.default_key = other.default_key

        self.virtual_table.update(other.virtual_table)

        if other.fselect is not None:
            self.fselect = other.fselect


@overload
def variantmethod(func: Callable[Concatenate[_T, _P], _R_co]) -> Variantmethod[_T, _P, _R_co]: ...
@overload
def variantmethod(key: Any) -> Callable[[Callable[Concatenate[_T, _P], _R_co]], Variantmethod[_T, _P, _R_co]]: ...
def variantmethod(arg):
    if isinstance(arg, Callable):
        return Variantmethod(arg)
    else:
        return partial(Variantmethod, key=arg)


def update_dispatch(dispatch_map: Dict[str, Variantmethod], dispobj: Variantmethod):
    name = dispobj.__name__

    if name not in dispatch_map:
        dispatch_map[name] = Variantmethod()

    dispatch_map[name].update(dispobj)


class VariantMeta(type):
    def __init__(self, name: str, bases: Tuple[type, ...], dict: Dict[str, Any], /, **kwds: Any):
        dispatch_map: Dict[str, Variantmethod] = {}

        for base in bases:
            for attr in dir(base):
                val = getattr(base, attr)
                if isinstance(val, Variantmethod):
                    update_dispatch(dispatch_map, val)

        for val in dict.values():
            if isinstance(val, Variantmethod):
                update_dispatch(dispatch_map, val)

        for attr, disp in dispatch_map.items():
            setattr(self, attr, disp)

        return type.__init__(self, name, bases, dict, **kwds)
