from typing import Protocol, Sequence, TypeVar
from ...typing import TensorLike

class LinearElasticityPDEDataProtocol(Protocol):
    """
    Protocol interface for linear elasticity PDE data components.

    This protocol is designed to define a standardized interface for linear
    elasticity problems in structural mechanics. It supports material properties,
    exact physical fields, body forces, and both essential and natural boundary
    conditions.

    Method categories:
        1. Geometry and domain specification
        2. Material properties (Lame parameters)
        3. Physical fields (displacement, strain, stress)
        4. External loads (body force)
        5. Boundary conditions (essential and natural types)

    Tensor conventions:
        - p: coordinate tensor, shape (..., dim)
        - displacement(p): (..., dim)
        - strain(p): (..., dim, dim)
        - stress(p): (..., dim, dim)
        - source(p): (..., dim)
        - displacement_bc(p): (..., dim)
        - traction_bc(p): (..., dim)
        - is_..._boundary(p): (...,) boolean mask
    """

    # --- Domain and dimension ---
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...

    def init_mesh(self): ...

    # --- Material properties ---
    def lam(self, p: TensorLike) -> TensorLike: ...
    def mu(self, p: TensorLike) -> TensorLike: ...
    def rho(self, p: TensorLike) -> TensorLike: ...

    # --- Exact physical fields ---
    def displacement(self, p: TensorLike) -> TensorLike: ...
    def strain(self, p: TensorLike) -> TensorLike: ...
    def stress(self, p: TensorLike) -> TensorLike: ...

    # --- Body force ---
    def body_force(self, p: TensorLike) -> TensorLike: ...

    # --- Essential boundary condition (Dirichlet) ---
    def displacement_bc(self, p: TensorLike) -> TensorLike: ...
    def is_displacement_boundary(self, p: TensorLike) -> TensorLike: ...

    # --- Natural boundary condition (Neumann) ---
    def traction_bc(self, p: TensorLike) -> TensorLike: ...
    def is_traction_boundary(self, p: TensorLike) -> TensorLike: ...

LinearElasticityPDEDataT = TypeVar("LinearElasticityPDEDataT", bound=LinearElasticityPDEDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # example name: (file_name, class_name)
    "boxpoly3d": ("box_poly_data_3d", "BoxPolyData3d"),
    "boxpoly2d": ("box_poly_data_2d", "BoxPolyData2d"),
    "boxsinsin2d": ("box_sinsin_data_2d", "BoxSinSinData2d"),
    "boxtri2d": ("box_tri_data_2d", "BoxTriData2d"),
    "boxmixed2d": ("box_mixed_data_2d", "BoxMixedData2d"),
}
