from typing import Protocol, Sequence, TypeVar, overload,Optional

from fealpy.backend import TensorLike

class BeamPDEDataProtocol(Protocol):
    '''
    A protocol for beam PDE data classes.
    This protocol defines the expected methods and properties
    that any beam PDE data class should implement.
    The protocol is used to ensure that the classes conform to a specific interface,
    allowing for type checking and better code organization.
    The protocol includes methods for initializing the mesh,
    computing the source term, and retrieving geometric dimensions and domain information.
    The protocol is generic and can be used with different beam PDE data classes
    that may have different implementations but share the same interface.
    The protocol is designed to be flexible and extensible,
    allowing for future enhancements and modifications.
    Attributes:
        E (float): Young's modulus.
        I (float): Moment of inertia.
        A (float): Cross-sectional area.
        f (float): Distributed load.
        L (float): Beam length.
        n (int): Number of mesh elements.
    Methods:
        init_mesh(p: TensorLike) -> None:
            Initialize the mesh for the beam domain.
        source(p: TensorLike) -> TensorLike:
            Compute the source term for the beam PDE.
        geo_dimension() -> int:
            Return the geometric dimension of the domain.
        domain() -> Sequence[float]:
            Return the computational domain [xmin, xmax].
    The protocol is designed to be implemented by specific beam PDE data classes,
    '''
    
    def init_mesh(self, p: TensorLike) -> None: ...
    def source(self, p: TensorLike) -> TensorLike: ...
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def dirichlet(self, p: TensorLike) -> Optional[TensorLike]: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> Optional[TensorLike]: ...

BeamPDEDataT = TypeVar('BeamPDEDataT', bound=BeamPDEDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # Add beam PDE models here (file_name, class_name)
    1: ("euler_bernoulli_beam_data_2d", "EulerBernoulliBeamData2D"),
    2: ("timoshenko_beam_data_3d", "TimoshenkoBeamData3D"),
}