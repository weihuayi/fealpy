from typing import Any, Dict, Tuple, TypeVar, Callable, Optional

_F = TypeVar('_F', bound=Callable[..., Any])


class OperatorMeta(type):
    """
    Metaclass that automatically registers methods decorated with @register decorator.

    When a new class inheriting from OpteratorBase is defined,
    methods marked with the @assemblymethod decorator are automatically registered
    under specified API namespaces.

    Attributes:
        _api_registry (Dict[str, Dict[str, Any]]): 
            A dictionary mapping API namespaces to their registered methods.

    Parameters:
        name (str):
            The name of the class being created.
        bases (Tuple[type, ...]):
            Base classes of the new class.
        mdict (Dict[str, Any]):
            Class namespace dictionary.

    Returns:
        type: The newly created class.

    Raises:
        ValueError: If a method with the same call_name is already registered in the given api_name.
    """
    def __init__(
        self, 
        name: str, 
        bases: Tuple[type, ...], 
        mdict: Dict[str, Any], /,
        **kwds: Any
    ):
        super().__init__(name, bases, mdict, **kwds)
        self._api_registry: Dict[str, Dict[str, Any]] = {}

        for meth_name, meth in mdict.items():
            if self.is_valid_callable(meth):
                api_name = getattr(meth, '__api_name__')
                call_name = getattr(meth, '__call_name__')

                if api_name not in self._api_registry:
                    self._api_registry[api_name] = {}

                api_map = self._api_registry[api_name]

                if call_name in api_map:
                    raise ValueError(
                        f"Method '{call_name}' already registered under API '{api_name}'."
                    )

                api_map[call_name] = meth

    @staticmethod
    def is_valid_callable(meth: Any) -> bool:
        """
        Determine if a method is valid for registration.

        A valid method must:
            - be callable.
            - have '__call_name__' and '__api_name__' attributes.

        Parameters:
            meth (Any): Method to be checked.

        Returns:
            bool: True if valid, False otherwise.
        """
        return (
            callable(meth) and
            hasattr(meth, '__call_name__') and
            hasattr(meth, '__api_name__')
        )


class OpteratorBase(metaclass=OperatorMeta):
    """
    Base class for computational operators.

    Inherit from this class to enable automatic registration of methods
    using the @assemblymethod decorator.
    """

    def __init__(self, method: Optional[str]=None):
        """
        Initialize the operator base.
        
        Parameters:
            method (str, optional): Default method to be called when the operator is called. 
                                Defaults to None.
        """
        self.method = method

    def get_registered_method(self, call_name: str, api_name: Optional[str]=None) -> Callable:
        """
        Retrieve a registered method by call name and API namespace.

        Parameters:
            call_name (str): Call name of the method.
            api_name (str, optional): API namespace of the method. Defaults to None.

        Returns:
            Callable: The registered method.

        Raises:
            KeyError: If no method is registered under the given call_name and api_name.
        """
        # method_name = self._api_registry[api_name][call_name]
        # return getattr(self, method_name)
        method_obj = self._api_registry[api_name][call_name]
        return getattr(self, method_obj.__name__)
    
    def __call__(self):
        """
        Call the operator's default method if specified.
        
        Returns:
            Result of the called method.
            
        Raises:
            ValueError: If the 'method' attribute is not specified.
        """
        if not hasattr(self, 'method') or self.method is None:
            raise ValueError("Method not specified.")
        
        meth = self.get_registered_method(self.method, None)
        
        return meth()


def assemblymethod(call_name: str, api_name: Optional[str]=None) -> Callable[[_F], _F]:
    """
    Decorator to mark methods for automatic registration by the OperatorMeta metaclass.

    Parameters:
        call_name (str): Unique identifier for the method.
        api_name (str, optional): API namespace under which the method is registered. Defaults to None.

    Example:
        @assemblymethod(call_name='solve', api_name='solver')
        def solve_method(self):
            pass
    """
    def decorator(meth: _F) -> _F:
        meth.__call_name__ = call_name
        meth.__api_name__ = api_name
        return meth
    return decorator