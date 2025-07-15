from typing import Protocol, Sequence, TypeVar, Optional
from ...backend import TensorLike

class IonFlowPDEDataProtocol(Protocol):
    """
    Protocol interface for fully coupled ion-flow field PDE models.
    
    This protocol specifies the structure for PDE models involving
    electrostatic potential φ and ion density ρ in a coupled system.

    Key components:
    1. Domain and geometry description
    2. Exact solutions and boundary data
    3. Source terms (right-hand sides)
    4. Coupling structure of the equations

    Notes:
        - This is a protocol interface and not a concrete implementation.
        - All TensorLike arguments are coordinate arrays of shape (d, N).
    """

    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    
    # Exact solutions
    def potential(self, p: TensorLike) -> TensorLike: ...
    def potential_gradient(self, p: TensorLike) -> TensorLike: ...
    def potential_dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_potential_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

    def density(self, p: TensorLike) -> TensorLike: ...
    def density_gradient(self, p: TensorLike) -> TensorLike: ...
    def density_dirichlet(self, p: TensorLike) -> TensorLike: ...
    def is_density_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

    # RHS source terms
    def source_potential(self, p: TensorLike) -> TensorLike: ...
    def source_density(self, p: TensorLike) -> TensorLike: ...

IonFlowPDEDataT = TypeVar('IonFlowPDEDataT', bound=IonFlowPDEDataProtocol)

"""
DATA_TABLE is a registry for ion-flow test cases.

To register a new model, add its file and class name here.
The solver will dynamically load it using this registry.
"""
DATA_TABLE = {
    "spherical_shell": ("spherical_shell_case", "SphericalShellIonFlowData"),
    # "parallel_plate": ("parallel_plate_case", "ParallelPlateIonFlowData"),
}

