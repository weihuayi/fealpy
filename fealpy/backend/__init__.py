"""
FEALPy Backends
===============

This module provides a backend manager for FEALPy.

"""
from .manager import BackendManager
from .base import TensorLike, Size, Number

backend_manager = BackendManager(default_backend='numpy')
bm = backend_manager
