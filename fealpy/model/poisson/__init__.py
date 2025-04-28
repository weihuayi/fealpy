
from typing import Protocol, Sequence, TypeVar

from ...backend import TensorLike
from ...mesh.mesh_base import Mesh
from ..utils import example_import_util


class PoissonPDEDataProtocol(Protocol):
    description : str
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def init_mesh(self) -> Mesh: ...
    def solution(self, p: TensorLike) -> TensorLike: ...
    def gradient(self, p: TensorLike) -> TensorLike: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

PoissonPDEDataT = TypeVar('PoissonPDEDataT', bound=PoissonPDEDataProtocol)


DATA_TABLE = {
    # example name: (file_name, class_name)
    "coscos": ("coscosdata", "CosCosData"),

}


def get_example(key: str) -> PoissonPDEDataProtocol:
    return example_import_util("poisson", key, DATA_TABLE)
