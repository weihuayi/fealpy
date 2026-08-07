from typing import Protocol, Sequence, TypeVar, overload,Optional

from fealpy.backend import TensorLike

class HyperelasticPDEDataProtocol(Protocol):
    '''A protocol for hyperelastic PDE data classes.'''
    
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def init_mesh(self, p: TensorLike) -> None: ...
    
    def body_force(self, p: TensorLike) -> TensorLike: ...
    def dirichlet(self, p: TensorLike) -> Optional[TensorLike]: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> Optional[TensorLike]: ...

HyperelasticPDEDataT = TypeVar('HyperelasticPDEDataT', bound=HyperelasticPDEDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # Add hyperelastic PDE models here (file_name, class_name)
    1: ("yeoh_uniaxial_model", "YeohUniaxialModel"),
}