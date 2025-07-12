# from .domain import Domain
# from .sizing_function import huniform
from .geometry_kernel_manager import GeometryKernelManager
from .implicit_surface import SphereSurface


geometry_kernel_manager = GeometryKernelManager(default_adapter='occ')