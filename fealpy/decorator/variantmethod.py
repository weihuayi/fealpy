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
    __slots__ = ('virtual_table', 'key_table', 'default_key')
    virtual_table : Dict[Any, Callable]
    key_table : Dict[_T, Any]

    def __init__(self, func: Optional[Callable] = None, key: Any = None):
        if func is None:
            self.virtual_table = {}
        else:
            self.virtual_table = {key: func}
        self.key_table = {}
        self.default_key = key

    @property
    def __func__(self) -> Callable[Concatenate[_T, _P], _R_co]:
        assert len(self.virtual_table) >= 1, "variants can not be empty"
        return self.virtual_table[self.default_key]

    @property
    def __name__(self) -> str:
        return self.__func__.__name__

    # def __get__(self, obj: _T, objtype: Type[_T]) -> Callable[_P, _R_co]:
    @overload
    def __get__(self, obj: None, objtype: Type[_T]) -> "Variantmethod[_T, _P, _R_co]": ...
    @overload
    def __get__(self, obj: _T, objtype: Type[_T]) -> "VariantHandler[_T, _P, _R_co]": ...
    def __get__(self, obj, objtype):
        if obj is None:
            return self

        return VariantHandler(self, obj, objtype)

    def __set__(self, obj: _T, val: Any):
        raise RuntimeError("Variantmethod has no setter.")
    
    def __len__(self) -> int:
        return len(self.virtual_table)

    def __getitem__(self, key: Any) -> Callable[Concatenate[_T, _P], _R_co]:
        if key in self.virtual_table:
            return self.virtual_table[key]
        else:
            return self.__func__

    def register(self, key: Any, /):
        def decorator(func: Callable) -> Variantmethod[_T, _P, _R_co]:
            self.virtual_table[key] = func
            return self
        return decorator

    def get_key(self, obj: _T):
        if obj in self.key_table:
            return self.key_table[obj]
        else:
            return self.default_key

    def set_key(self, obj:_T, val: Any):
        self.key_table[obj] = val

    def update(self, other: "Variantmethod", /) -> None:
        if len(self.virtual_table) == 0:
            self.default_key = other.default_key

        self.virtual_table.update(other.virtual_table)
        self.key_table.update(other.key_table)


class VariantHandler(Generic[_T, _P, _R_co]):
    def __init__(self, vm: Variantmethod[_T, _P, _R_co], obj: _T, objtype: Type[_T]):
        self.vm = vm
        self.instance = obj
        self.owner = objtype

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs):
        key = self.vm.get_key(self.instance)
        func = self.vm[key]
        return func.__get__(self.instance, self.owner)(*args, **kwargs)

    def __getitem__(self, val: Any) -> Callable[_P, _R_co]:
        func = self.vm[val]
        return func.__get__(self.instance, self.owner)

    def set(self, val: Any):
        self.vm.set_key(self.instance, val)


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
