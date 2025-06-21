# from .domain import Domain
# from .sizing_function import huniform
from .geometry_kernel_manager import GeometryKernelManager


geometry_kernel_manager = GeometryKernelManager(default_adapter='occ')