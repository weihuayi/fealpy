from typing import Protocol, Sequence, TypeVar
from ...backend import TensorLike


class WavePDEDataProtocol(Protocol):
    """Protocol interface for wave PDE data components.
    
    Defines the recommended protocol interface for wave partial differential equation solvers.

    This protocol suggests four main categories of methods that implementing classes may provide:
        1. Domain specification methods (geometry ,computational domain, duration)
        2. Equation terms methods (exact solution, grdient, init_solution, init_solution_t and source terms)
        4. Boundary condition methods (Dirichlet, Neumann, Robin types)

    Notes:  
        This protocol serves as a development guideline - implementing classes are encouraged to:
        - Provide implementations for the declared methods
        - Maintain consistent method signatures and return types
        - Implement methods relevant to their use case
    """
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def duration(self) -> Sequence[float]: ...
    def speed(self) -> float: ...
    def init_solution(self, p: TensorLike) -> TensorLike: ...
    def init_solution_t(self, p: TensorLike) -> TensorLike: ...
    def solution(self, p: TensorLike, t: float) -> TensorLike: ...
    def gradient(self, p: TensorLike, t: float) -> TensorLike: ...
    def source(self, p: TensorLike, t: float) -> TensorLike: ...
    def dirichlet(self, p: TensorLike, t:float) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

WavePDEDataT = TypeVar('WavePDEDataT', bound=WavePDEDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # example name: (file_name, class_name)
    "sincos": ("sin_cos_data_2d", "SinCosData2D"),
    "sinmix": ("sin_mix_data_1d", "SinMixData1D")

}
