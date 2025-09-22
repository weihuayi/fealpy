from typing import Protocol, Sequence, TypeVar
from ...backend import TensorLike

class StokesPDEDataProtocol(Protocol):
    """Protocol interface for incompressible Stokes equation PDE data components.

    The Stokes system:
        -μ Δu + ∇p = f   in Ω
          ∇·u = 0        in Ω

    This protocol defines interfaces for:
        1. Domain and geometry
        2. Physical parameters (viscosity)
        3. Velocity and pressure fields (exact or initial)
        4. Source terms (forcing function)
        5. Boundary conditions

    """
    # --- Domain specification ---
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...

    # --- Physical parameters ---
    def viscosity(self) -> float: ...

    # --- Source term ---
    def source(self, p: TensorLike) -> TensorLike: ...

    # --- Optional: exact solution (for verification) ---
    def velocity(self, p: TensorLike) -> TensorLike: ...
    def pressure(self, p: TensorLike) -> TensorLike: ...
    def grad_velocity(self, p: TensorLike) -> TensorLike: ...
    def div_velocity(self, p: TensorLike) -> TensorLike: ...

    # --- Dirichlet boundary conditions ---
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike: ...

    # --- Optional: Neumann-type boundary support ---
    def neumann_stress(self, p: TensorLike) -> TensorLike: ...
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike: ...

StokesPDEDataT = TypeVar('StokesPDEDataT', bound=StokesPDEDataProtocol)

DATA_TABLE = {
    # example name: (file_name, class_name)
    1: ("exp0001", "Exp0001"),
    2: ("exp0002", "Exp0002"),
}

