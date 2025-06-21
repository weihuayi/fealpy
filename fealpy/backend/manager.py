
from typing import Dict, Optional
import importlib
import threading

from ..import logger
from .base import BackendProxy


class BackendManager():
    # _instance = None

    # def __new__(cls, *, default_backend: str):
    #     if cls._instance is None:
    #         cls._instance = super().__new__(cls)
    #     return cls._instance

    def __init__(self, *, default_backend: Optional[str]=None):
        self._backends: Dict[str, BackendProxy] = {}
        self._THREAD_LOCAL = threading.local()
        self._default_backend_name = default_backend

    def set_backend(self, name: str) -> None:
        """Set the current backend."""
        if name not in self._backends:
            self.load_backend(name)
        self._THREAD_LOCAL.__dict__['backend'] = self._backends[name]

    def load_backend(self, name: str) -> None:
        """Load a backend by name."""
        if name not in BackendProxy._available_backends:
            try:
                importlib.import_module(f"fealpy.backend.{name}_backend")
            except ImportError:
                raise RuntimeError(f"Backend '{name}' is not found.")

        if name in BackendProxy._available_backends:
            if name in self._backends:
                logger.info(f"Backend '{name}' has already been loaded.")
                return
            # NOTE: initialize a backend proxy instance when loading.
            # Backend proxy instances are singletons as there is no need to load twice.
            backend = BackendProxy._available_backends[name]()
            self._backends[name] = backend
        else:
            raise RuntimeError(f"Failed to load backend '{name}'.")

    def get_current_backend(self, logger_msg=None) -> BackendProxy:
        """Get the current backend."""
        if 'backend' not in self._THREAD_LOCAL.__dict__:
            if self._default_backend_name is None:
                raise RuntimeError(
                    f"Backend properties were accessed ({logger_msg}) "
                    "before a backend was specified, "
                    "and no default backend was set in the backend manager."
                )
            self.set_backend(self._default_backend_name)
            logger.info(f"Backend auto-setting triggered by {logger_msg}."
                        "To reduce unnecessary backend loading, "
                        "get backend properties and methods after executing set_backend()")
        return self._THREAD_LOCAL.__dict__['backend']

    def __getattr__(self, item):
        """Redirct attribute access to the current backend."""
        return getattr(self.get_current_backend("GET_ATTR: " + item), item)

    def __setattr__(self, key, value):
        """Redirct attribute access to the current backend."""
        if key in {'_backends', '_THREAD_LOCAL', '_default_backend_name'}:
            super().__setattr__(key, value)
        else:
            setattr(self.get_current_backend("SET_ATTR: " + key), key, value)
