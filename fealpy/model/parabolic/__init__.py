from typing import Protocol, Sequence, TypeVar

from ...backend import TensorLike
from ...mesh.mesh_base import Mesh
from ..utils import example_import_util


class ParabolicPDEDataProtocol(Protocol):
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def duration(self) -> Sequence[float]: ...
    def init_solution(self, p: TensorLike) -> TensorLike: ...
    def solution(self, p: TensorLike, t: float) -> TensorLike: ...
    def gradient(self, p: TensorLike, t: float) -> TensorLike: ...
    def source(self, p: TensorLike, t: float) -> TensorLike: ...
    def dirichlet(self, p: TensorLike, t:float) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

ParabolicPDEDataT = TypeVar('ParabolicPDEDataT', bound=ParabolicPDEDataProtocol)


DATA_TABLE = {
    # example name: (file_name, class_name)
    # "coscos": ("coscosdata", "CosCosData"),
    "sinexp": ("sin_exp_data_1d", "SinExpData1D"),
    "sinsinexp": ("sin_sin_exp_data_2d", "SinSinExpData2D"),

}


def get_example(key: str) -> ParabolicPDEDataProtocol:
    return example_import_util("parabolic", key, DATA_TABLE)
