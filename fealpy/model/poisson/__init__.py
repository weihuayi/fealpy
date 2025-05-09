
from typing import Protocol, Sequence, TypeVar

from ...backend import TensorLike
from ...mesh.mesh_base import Mesh
from ..utils import example_import_util


class PoissonPDEDataProtocol(Protocol):
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def init_mesh(self) -> Mesh: ...
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def flux(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

PoissonPDEDataT = TypeVar('PoissonPDEDataT', bound=PoissonPDEDataProtocol)


DATA_TABLE = {
    # example name: (file_name, class_name)
    "coscos": ("cos_cos_data_2d", "CosCosData2D"),
    "sin": ("sin_data_1d", "SinData1D"),
    "sinsin": ("sin_sin_data_2d", "SinSinData2D"),

}


# def get_example(key: str) -> PoissonPDEDataProtocol:
#     return example_import_util("poisson", key, DATA_TABLE)()

def get_example(key:str = "" , flag = False) -> PoissonPDEDataProtocol:
    """
    
    """
    if flag==True:
        print("Available examples name: (file_name, class_name)")
        for key in DATA_TABLE.keys():
            print(f"- {key}:{DATA_TABLE[key]}")
        print("You can obtain the examples you need in this way:\n   pde = get_example('The example name you need')")
    
    if key not in DATA_TABLE:
        raise ValueError(f"example {key} not found, please check the name you input.")
    
    return example_import_util("poisson", key, DATA_TABLE)()
