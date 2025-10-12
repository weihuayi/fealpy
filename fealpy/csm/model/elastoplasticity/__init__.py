from typing import Protocol, Sequence, TypeVar, overload,Optional

from fealpy.backend import TensorLike

class ElastoplasticityPDEDataProtocol(Protocol):
    '''
    A protocol for elastoplasticity PDE data classes.
    This protocol defines the expected methods and properties
    that any elastoplasticity PDE data class should implement.
    The protocol is used to ensure that the classes conform to a specific interface,
    allowing for type checking and better code organization.
    The protocol includes methods for initializing the mesh,
    computing the source term, and retrieving geometric dimensions and domain information.
    The protocol is generic and can be used with different elastoplasticity PDE data classes
    that may have different implementations but share the same interface.
    The protocol is designed to be flexible and extensible,
    allowing for future enhancements and modifications.
    Attributes:
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        yield_strength (float): Yield strength of the material.
        f (float or callable): Distributed load applied to the domain.
        l (float): Characteristic length of the domain.
    '''
    
    def init_mesh(self, p: TensorLike) -> None: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def dirichlet(self, p: TensorLike) -> Optional[TensorLike]: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> Optional[TensorLike]: ...

ElastoplasticityPDEDataT = TypeVar('ElastoplasticityPDEDataT', bound=ElastoplasticityPDEDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # Add elastoplasticity PDE models here (file_name, class_name)
    1: ("elastoplasticity_data_2d", "ElastoplasticityData2D"),
    2: ("elastoplasticity_data_3d", "ElastoplasticityData3D"),
}