from typing import Any, Optional, Tuple, List, Dict, Literal
import importlib
import threading

from fealpy import logger
from .geometry_kernel_adapter_base import GeometryKernelAdapterBase


class GeometryKernelManager():
    def __init__(self, *, default_adapter: Optional[str]=None):
        self._adapters: Dict[str, GeometryKernelAdapterBase] = {}
        self._THREAD_LOCAL = threading.local()
        self._default_adapter_name = default_adapter

    def load_adapter(self, name: str) -> None:
        if name not in GeometryKernelAdapterBase._available_adapters:
            try:
                importlib.import_module(f"fealpy.geometry.{name}_adapter")
            except ImportError:
                raise RuntimeError(f"Kernel '{name}' is not found.")

        if name in GeometryKernelAdapterBase._available_adapters:
            if name in self._adapters:
                logger.info(f"Kernel '{name}' has already been loaded.")
                return
            adapter = GeometryKernelAdapterBase._available_adapters[name]()
            adapter.initialize()
            self._adapters[name] = adapter
        else:
            raise RuntimeError(f"Failed to load adapter '{name}'.")

    def set_adapter(self, name: str) -> None:
        if name not in self._adapters:
            self.load_adapter(name)
        self._THREAD_LOCAL.__dict__['adapter'] = self._adapters[name]

    def get_current_adapter(self, logger_msg=None) -> GeometryKernelAdapterBase:
        if 'adapter' not in self._THREAD_LOCAL.__dict__:
            if self._default_adapter_name is None:
                raise RuntimeError(
                    f"Adapter properties were accessed ({logger_msg}) "
                    "before a adapter was specified, "
                    "and no default adapter was set in the adapter manager."
                )
            self.set_adapter(self._default_adapter_name)
            logger.info(f"Kernel auto-setting triggered by {logger_msg}."
                        "To reduce unnecessary adapter loading, "
                        "get adapter properties and methods after executing set_adapter()")
        return self._THREAD_LOCAL.__dict__['adapter']

    def __getattr__(self, item):
        return getattr(self.get_current_adapter("GET_ATTR: " + item), item)

    def __setattr__(self, key, value):
        if key in {'_adapters', '_THREAD_LOCAL', '_default_adapter_name'}:
            super().__setattr__(key, value)
        else:
            setattr(self.get_current_adapter("SET_ATTR: " + key), key, value)