"""
FEALPy Backends
===============

This module provides a backend manager for FEALPy.

"""
from .base import *
from .manager import BackendManager

backend_manager = BackendManager(default_backend='numpy')
bm = backend_manager
Tensor = TensorLike
