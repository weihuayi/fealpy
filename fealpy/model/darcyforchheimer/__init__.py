from typing import Protocol, Sequence, TypeVar
from ...backend import TensorLike


class DarcyforchheimerDataProtocol(Protocol):
    """Protocol interface for Darcy–Forchheimer PDE data components."""
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def pressure(self, p: TensorLike) -> TensorLike: ...
    def velocity(self, p: TensorLike) -> TensorLike: ...
    def grad_pressure(self, p: TensorLike) -> TensorLike: ...
    def f(self, p: TensorLike) -> TensorLike: ...  
    def g(self, p: TensorLike) -> TensorLike: ... 
    def neumann(self, p: TensorLike, n: TensorLike) -> TensorLike: ...

DFDataT = TypeVar('DarcyforchheimerDataT', bound=DarcyforchheimerDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""

DATA_TABLE = {
    "sin": ("sin_data_1d", "SinData1D"),
    "post1":("postdata1","PostData1"),
    "post2":("postdata2","PostData2"),
    "post3":("postdata1","PostData3"),
    "post4":("postdata4","PostData4"),
    "post5":("postdata5","PostData5"),
    "exp":("exp_data_2d","ExpData2D"),
    "coscos":("cos_cos_data_2d","CosCosData2D"),
    "arctan":("arctan_data_2d","ArctanData2D"),

}
