
from typing import Dict
import importlib
import threading

from .base import Backend


class BackendManager():
    # _instance = None

    # def __new__(cls, *, default_backend: str):
    #     if cls._instance is None:
    #         cls._instance = super().__new__(cls)
    #     return cls._instance

    def __init__(self, *, default_backend: str):
        self._backends: Dict[str, Backend] = {}
        self._THREAD_LOCAL = threading.local()
        self.set_backend(default_backend)

    def set_backend(self, name: str) -> None:
        """Set the current backend."""
        if name not in self._backends:
            self.load_backend(name)
        self._THREAD_LOCAL.__dict__['backend'] = self._backends[name]

    def load_backend(self, name: str) -> None:
        """Load a backend by name."""
        if name not in Backend._available_backends:
            try:
                importlib.import_module(f"fealpy.experimental.backend.{name}_backend")
            except ImportError:
                raise RuntimeError(f"Backend '{name}' is not found.")

        if name in Backend._available_backends:
            backend = Backend._available_backends[name]()
            self._backends[name] = backend
        else:
            raise RuntimeError(f"Failed to load backend '{name}'.")

    def get_current_backend(self) -> Backend:
        """Get the current backend."""
        return self._THREAD_LOCAL.__dict__['backend']

    def __getattr__(self, item):
        """Redirct attribute access to the current backend."""
        return getattr(self.get_current_backend(), item)

    def __setattr__(self, key, value):
        """Redirct attribute access to the current backend."""
        if key in {'_backends', '_THREAD_LOCAL'}:
            super().__setattr__(key, value)
        else:
            setattr(self.get_current_backend(), key, value)
