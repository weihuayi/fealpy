from typing import Protocol, Sequence, TypeVar
from ...backend import TensorLike

class PoissonPDEDataProtocol(Protocol):
    """Protocol interface for poisson PDE data components.
    
    Defines the recommended protocol interface for poisson partial differential equation solvers.

    This protocol suggests four main categories of methods that implementing classes may provide:
        1. Domain specification methods (geometry,computational domain)
        2. Equation terms methods (exact solution, grdient and source terms)
        3. Boundary condition methods (Dirichlet, Neumann, Robin types)

    Notes:  
        This protocol serves as a development guideline - implementing classes are encouraged to:
        - Provide implementations for the declared methods
        - Maintain consistent method signatures and return types
        - Implement methods relevant to their use case
    """
    def get_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def init_mesh(self): ...
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...
    def neumann(self, p: TensorLike) -> TensorLike: ...
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike: ...
    def robin(self, p: TensorLike) -> TensorLike: ...
    def is_robin_boundary(self, p: TensorLike) -> TensorLike: ...

PoissonPDEDataT = TypeVar('PoissonPDEDataT', bound=PoissonPDEDataProtocol)
"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # example name: (file_name, class_name)
    "coscos": ("cos_cos_data_2d", "CosCosData2D"),
    "coscoscos": ("cos_cos_cos_data_3d", "CosCosCosData3D"),
    "sin": ("sin_data_1d", "SinData1D"),
    "sinsin": ("sin_sin_data_2d", "SinSinData2D"),
    "sinsinsin": ("sin_sin_sin_data_3d", "SinSinSinData3D"),
    "interior":("interior_layer_data_2d", "InteriorLayerData2d"),
    "corner":("corner_singularity_data_2d", "CornerSingularityData2d")
    "CosCos_Cos_Dir_2D": ("coscos_cos_dir_2d", "CosCos_Cos_Dir_2D"),
    "CosCosCos_Cos_Dir_3D": ("coscoscos_cos_dir_3d", "CosCosCos_Cos_Dir_3D"),
    "Sin_Sin_Dir_1D": ("sin_sin_dir_1d", "Sin_Sin_Dir_1D"),
    "SinSin_Sin_Dir_2D": ("sinsin_sin_dir_2d", "SinSin_Sin_Dir_2D"),
    "SinSinSin_Sin_Dir_3D": ("sinsinsin_sin_dir_3d", "SinSinSin_Sin_Dir_3D"),
}
