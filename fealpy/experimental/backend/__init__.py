
from .manager import BackendManager
from .base import TensorLike

backend_manager = BackendManager(default_backend='numpy')
