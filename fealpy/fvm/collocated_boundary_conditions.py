"""Explicit boundary data graph consumed by collocated SIMPLE and PISO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TypeAlias

from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor
from fealpy.typing import TensorLike

from .collocated_pressure_system import (
    CollocatedPressureSystemControls,
    PressureClosureKind,
)
from .dirichlet_bc import DirichletBC
from .face_gradient import reconstruct_face_gradient
from .fvm_geometry import FVMGeometry
from .gradient_reconstruct import (
    GradientReconstruct,
    ResolvedGradientBoundary,
)
from .simple_controls import SimpleDiscretizationControls
from .solver_controls import PisoSolverControls
from .traction_bc import TractionBC


BoundaryValueFunction: TypeAlias = Callable[[TensorLike], TensorLike]


class NormalizedVelocityBoundary(Protocol):
    """Normalized velocity boundary data on one fixed mesh."""

    dirichlet_faces: TensorLike
    dirichlet_value: BoundaryValueFunction
    neumann_faces: TensorLike
    neumann_value: BoundaryValueFunction
    natural_faces: TensorLike


class NormalizedPressureBoundary(Protocol):
    """Normalized physical-pressure boundary data on one fixed mesh."""

    dirichlet_faces: TensorLike
    dirichlet_value: BoundaryValueFunction


class NormalizedMomentumBoundary(Protocol):
    """Normalized steady-momentum traction data on one fixed mesh."""

    traction_faces: TensorLike
    traction_value: BoundaryValueFunction


class NormalizedCollocatedBoundary(Protocol):
    """Static fixed-mesh boundary data required by SIMPLE/PISO resolvers."""

    geometry: FVMGeometry
    velocity: NormalizedVelocityBoundary
    pressure: NormalizedPressureBoundary
    momentum: NormalizedMomentumBoundary


def vector_face_average(
    geometry: FVMGeometry,
    faces: TensorLike,
    value: BoundaryValueFunction,
    quadrature_order: int,
) -> TensorLike:
    if faces.shape[0] == 0:
        return bm.zeros(
            (0, geometry.GD),
            dtype=geometry.cell_center.dtype,
            device=bm.get_device(geometry.cell_center),
        )

    def _integrand(
        points: TensorLike,
        _face_slice: slice,
    ) -> TensorLike:
        return bm.array(
            value(points),
            dtype=points.dtype,
            device=bm.get_device(points),
        )

    integral = geometry.face_integral(_integrand, q=quadrature_order)
    return integral[faces] / geometry.face_measure[faces, None]


@dataclass(frozen=True)
class ResolvedVelocityBoundary:
    dirichlet_face_values: TensorLike
    neumann_faces: TensorLike
    neumann_sn_grad: TensorLike
    natural_faces: TensorLike
    dirichlet_operator: DirichletBC
    gradient: GradientReconstruct

    def correct_face_gradient(
        self,
        cell_gradient: TensorLike,
        cell_velocity: TensorLike,
        *,
        interpolation_method: str,
    ) -> TensorLike:
        return reconstruct_face_gradient(
            self.gradient.geometry,
            cell_gradient,
            cell_velocity,
            interpolation_method=interpolation_method,
            boundary=self.gradient.boundary,
        )


@dataclass(frozen=True)
class ResolvedPressureStateBoundary:
    gradient: GradientReconstruct


@dataclass(frozen=True)
class ResolvedMomentumBoundary:
    velocity: ResolvedVelocityBoundary
    traction: TractionBC

    def source(self, reference: TensorLike) -> TensorLike:
        return self.traction.source(reference)

    def diffusion_source(
        self,
        diffusion_coef: float,
        reference: TensorLike,
    ) -> TensorLike:
        geometry = self.velocity.gradient.geometry
        faces = self.velocity.neumann_faces
        sn_grad = bm.array(
            self.velocity.neumann_sn_grad,
            dtype=reference.dtype,
            device=bm.get_device(reference),
        )
        weight = diffusion_coef * geometry.mag_S_f[faces]
        source = bm.zeros(
            (geometry.NC, geometry.GD),
            dtype=reference.dtype,
            device=bm.get_device(reference),
        )
        return bm.index_add(
            source,
            geometry.owner[faces],
            weight[:, None] * sn_grad,
            axis=0,
        )

    def apply_diffusion(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
        diffusion_coef: float,
    ) -> tuple[CSRTensor, TensorLike]:
        operator = self.velocity.dirichlet_operator
        matrix = operator.apply_diffusion_matrix(
            matrix,
            coef=diffusion_coef,
            components=1,
        )
        rhs = operator.apply_diffusion_rhs(
            rhs,
            coef=diffusion_coef,
            components=self.velocity.gradient.geometry.GD,
        )
        return matrix, rhs

    def natural_convection_diagonal(
        self,
        convection_face_velocity: TensorLike,
    ) -> TensorLike:
        geometry = self.velocity.gradient.geometry
        faces = self.velocity.natural_faces
        flux = bm.einsum(
            "ij,ij->i",
            convection_face_velocity[faces],
            geometry.S_f[faces],
        )
        return bm.index_add(
            bm.zeros(
                geometry.NC,
                dtype=convection_face_velocity.dtype,
                device=bm.get_device(convection_face_velocity),
            ),
            geometry.owner[faces],
            flux,
            axis=0,
        )

    def convection_source(
        self,
        convection_face_velocity: TensorLike,
        cell_velocity: TensorLike,
        reconstructed_face_velocity: TensorLike,
    ) -> TensorLike:
        return self.traction.convection_source(
            convection_face_velocity,
            cell_velocity,
            reconstructed_face_velocity,
        )

    def apply_pressure_force(
        self,
        cell_force: TensorLike,
        pressure: TensorLike,
        pressure_gradient: TensorLike,
    ) -> TensorLike:
        return self.traction.apply_pressure_force(
            cell_force,
            pressure,
            pressure_gradient,
        )

    def apply_cross_diffusion_flux(self, face_flux: TensorLike) -> TensorLike:
        return self.traction.apply_cross_diffusion_flux(face_flux)


@dataclass(frozen=True)
class ResolvedPressureSystemBoundary:
    dirichlet_operator: DirichletBC
    gradient: GradientReconstruct
    closure: PressureClosureKind


@dataclass(frozen=True)
class ResolvedRhieChowBoundary:
    pressure_state: ResolvedPressureStateBoundary
    zero_gradient_difference_faces: TensorLike

    def apply_pressure_partial(
        self,
        pressure: TensorLike,
        partial: TensorLike,
        distance: TensorLike,
    ) -> TensorLike:
        gradient = self.pressure_state.gradient
        geometry = gradient.geometry
        faces = gradient.boundary.dirichlet_faces
        owner = geometry.owner[faces]
        boundary_partial = (
            gradient.boundary.dirichlet_values - pressure[owner]
        ) / distance[faces]
        return bm.set_at(partial, faces, boundary_partial)

    def apply_gradient_difference(
        self,
        gradient_difference: TensorLike,
    ) -> TensorLike:
        faces = self.zero_gradient_difference_faces
        return bm.set_at(
            gradient_difference,
            faces,
            bm.zeros_like(gradient_difference[faces]),
        )


@dataclass(frozen=True)
class ResolvedPhysicalBoundaryConditions:
    geometry: FVMGeometry
    velocity: ResolvedVelocityBoundary
    momentum: ResolvedMomentumBoundary
    pressure_state: ResolvedPressureStateBoundary


@dataclass(frozen=True)
class ResolvedSimpleBoundaryConditions:
    physical: ResolvedPhysicalBoundaryConditions
    pressure_correction: ResolvedPressureSystemBoundary
    rhie_chow: ResolvedRhieChowBoundary


@dataclass(frozen=True)
class ResolvedPisoBoundaryConditions:
    physical: ResolvedPhysicalBoundaryConditions
    pressure_corrector: ResolvedPressureSystemBoundary
    rhie_chow: ResolvedRhieChowBoundary


def resolve_velocity_boundary(
    boundary: NormalizedCollocatedBoundary,
    *,
    gradient_method: str,
    gradient_layer_weights: tuple[float, float],
    gradient_boundary_weight: float,
    diffusion_method: str,
    diffusion_nonorthogonal_eps: float,
    use_face_average: bool,
    face_quadrature_order: int,
) -> ResolvedVelocityBoundary:
    geometry = boundary.geometry
    velocity_boundary = boundary.velocity
    dirichlet_faces = velocity_boundary.dirichlet_faces
    dirichlet_value = velocity_boundary.dirichlet_value
    dirichlet_center_values = bm.array(
        dirichlet_value(geometry.face_center[dirichlet_faces]),
        dtype=geometry.cell_center.dtype,
        device=bm.get_device(geometry.cell_center),
    )
    dirichlet_face_values = (
        vector_face_average(
            geometry,
            dirichlet_faces,
            dirichlet_value,
            face_quadrature_order,
        )
        if use_face_average
        else dirichlet_center_values
    )

    neumann_faces = velocity_boundary.neumann_faces
    neumann_sn_grad = bm.array(
        velocity_boundary.neumann_value(
            geometry.face_center[neumann_faces]
        ),
        dtype=geometry.cell_center.dtype,
        device=bm.get_device(geometry.cell_center),
    )

    natural_faces = velocity_boundary.natural_faces
    gradient_neumann_faces = bm.concatenate(
        (neumann_faces, natural_faces)
    )
    gradient_neumann_values = bm.concatenate(
        (
            neumann_sn_grad,
            bm.zeros(
                (natural_faces.shape[0], geometry.GD),
                dtype=geometry.cell_center.dtype,
                device=bm.get_device(geometry.cell_center),
            ),
        ),
        axis=0,
    )
    order = bm.argsort(gradient_neumann_faces)
    gradient_neumann_faces = gradient_neumann_faces[order]
    gradient_neumann_values = gradient_neumann_values[order]
    gradient_boundary = ResolvedGradientBoundary(
        dirichlet_faces=dirichlet_faces,
        dirichlet_values=dirichlet_center_values,
        neumann_faces=gradient_neumann_faces,
        neumann_sn_grad=gradient_neumann_values,
    )
    operator = DirichletBC(
        geometry,
        dirichlet_faces,
        dirichlet_center_values,
        diffusion_method=diffusion_method,
        nonorthogonal_eps=diffusion_nonorthogonal_eps,
    )
    return ResolvedVelocityBoundary(
        dirichlet_face_values=dirichlet_face_values,
        neumann_faces=neumann_faces,
        neumann_sn_grad=neumann_sn_grad,
        natural_faces=natural_faces,
        dirichlet_operator=operator,
        gradient=GradientReconstruct(
            geometry,
            gradient_boundary,
            method=gradient_method,
            layer_weights=gradient_layer_weights,
            boundary_weight=gradient_boundary_weight,
        ),
    )


def resolve_pressure_state_boundary(
    boundary: NormalizedCollocatedBoundary,
    *,
    gradient_method: str,
    gradient_layer_weights: tuple[float, float],
    gradient_boundary_weight: float,
) -> ResolvedPressureStateBoundary:
    geometry = boundary.geometry
    pressure_boundary = boundary.pressure
    faces = pressure_boundary.dirichlet_faces
    values = bm.array(
        pressure_boundary.dirichlet_value(geometry.face_center[faces]),
        dtype=geometry.cell_center.dtype,
        device=bm.get_device(geometry.cell_center),
    )
    gradient_boundary = ResolvedGradientBoundary(
        dirichlet_faces=faces,
        dirichlet_values=values,
        neumann_faces=geometry.boundary_faces[:0],
        neumann_sn_grad=bm.zeros(
            0,
            dtype=geometry.cell_center.dtype,
            device=bm.get_device(geometry.cell_center),
        ),
    )
    return ResolvedPressureStateBoundary(
        gradient=GradientReconstruct(
            geometry,
            gradient_boundary,
            method=gradient_method,
            layer_weights=gradient_layer_weights,
            boundary_weight=gradient_boundary_weight,
        ),
    )


def resolve_traction_boundary(
    boundary: NormalizedCollocatedBoundary,
    *,
    use_face_average: bool,
    face_quadrature_order: int,
) -> TractionBC:
    geometry = boundary.geometry
    momentum_boundary = boundary.momentum
    faces = momentum_boundary.traction_faces
    value = momentum_boundary.traction_value
    center_values = bm.array(
        value(geometry.face_center[faces]),
        dtype=geometry.cell_center.dtype,
        device=bm.get_device(geometry.cell_center),
    )
    values = (
        vector_face_average(
            geometry,
            faces,
            value,
            face_quadrature_order,
        )
        if use_face_average
        else center_values
    )
    source = bm.zeros(
        (geometry.NC, geometry.GD),
        dtype=geometry.cell_center.dtype,
        device=bm.get_device(geometry.cell_center),
    )
    source = bm.index_add(
        source,
        geometry.owner[faces],
        geometry.mag_S_f[faces, None] * values,
        axis=0,
    )
    return TractionBC(
        geometry=geometry,
        faces=faces,
        face_average_values=values,
        cell_source=source,
    )


def resolve_physical_boundary_conditions(
    boundary: NormalizedCollocatedBoundary,
    *,
    velocity_gradient_method: str,
    pressure_gradient_method: str,
    gradient_layer_weights: tuple[float, float],
    gradient_boundary_weight: float,
    diffusion_method: str,
    diffusion_nonorthogonal_eps: float,
    use_velocity_face_average: bool,
    use_traction_face_average: bool,
    face_quadrature_order: int,
) -> ResolvedPhysicalBoundaryConditions:
    geometry = boundary.geometry
    velocity = resolve_velocity_boundary(
        boundary,
        gradient_method=velocity_gradient_method,
        gradient_layer_weights=gradient_layer_weights,
        gradient_boundary_weight=gradient_boundary_weight,
        diffusion_method=diffusion_method,
        diffusion_nonorthogonal_eps=diffusion_nonorthogonal_eps,
        use_face_average=use_velocity_face_average,
        face_quadrature_order=face_quadrature_order,
    )
    pressure_state = resolve_pressure_state_boundary(
        boundary,
        gradient_method=pressure_gradient_method,
        gradient_layer_weights=gradient_layer_weights,
        gradient_boundary_weight=gradient_boundary_weight,
    )
    traction = resolve_traction_boundary(
        boundary,
        use_face_average=use_traction_face_average,
        face_quadrature_order=face_quadrature_order,
    )
    momentum = ResolvedMomentumBoundary(
        velocity=velocity,
        traction=traction,
    )
    return ResolvedPhysicalBoundaryConditions(
        geometry=geometry,
        velocity=velocity,
        momentum=momentum,
        pressure_state=pressure_state,
    )


def build_pressure_boundary_operators(
    geometry: FVMGeometry,
    faces: TensorLike,
    values: TensorLike,
    *,
    pressure_gradient_method: str,
    gradient_layer_weights: tuple[float, float],
    gradient_boundary_weight: float,
    diffusion_method: str,
    diffusion_nonorthogonal_eps: float,
) -> tuple[DirichletBC, GradientReconstruct]:
    gradient_boundary = ResolvedGradientBoundary(
        dirichlet_faces=faces,
        dirichlet_values=values,
        neumann_faces=geometry.boundary_faces[:0],
        neumann_sn_grad=bm.zeros(
            0,
            dtype=geometry.cell_center.dtype,
            device=bm.get_device(geometry.cell_center),
        ),
    )
    operator = DirichletBC(
        geometry,
        faces,
        values,
        diffusion_method=diffusion_method,
        nonorthogonal_eps=diffusion_nonorthogonal_eps,
    )
    gradient = GradientReconstruct(
        geometry,
        gradient_boundary,
        method=pressure_gradient_method,
        layer_weights=gradient_layer_weights,
        boundary_weight=gradient_boundary_weight,
    )
    return operator, gradient


def resolve_pressure_system_boundary(
    geometry: FVMGeometry,
    faces: TensorLike,
    values: TensorLike,
    closure: PressureClosureKind,
    *,
    pressure_gradient_method: str,
    gradient_layer_weights: tuple[float, float],
    gradient_boundary_weight: float,
    diffusion_method: str,
    diffusion_nonorthogonal_eps: float,
) -> ResolvedPressureSystemBoundary:
    operator, gradient = build_pressure_boundary_operators(
        geometry,
        faces,
        values,
        pressure_gradient_method=pressure_gradient_method,
        gradient_layer_weights=gradient_layer_weights,
        gradient_boundary_weight=gradient_boundary_weight,
        diffusion_method=diffusion_method,
        diffusion_nonorthogonal_eps=diffusion_nonorthogonal_eps,
    )
    return ResolvedPressureSystemBoundary(
        dirichlet_operator=operator,
        gradient=gradient,
        closure=closure,
    )


def resolve_collocated_simple_boundary_conditions(
    boundary: NormalizedCollocatedBoundary,
    discretization_controls: SimpleDiscretizationControls,
    pressure_system_controls: CollocatedPressureSystemControls,
) -> ResolvedSimpleBoundaryConditions:
    physical = resolve_physical_boundary_conditions(
        boundary,
        velocity_gradient_method=(
            discretization_controls.velocity_gradient_method
        ),
        pressure_gradient_method=(
            discretization_controls.pressure_gradient_method
        ),
        gradient_layer_weights=(
            discretization_controls.gradient_layer_weights
        ),
        gradient_boundary_weight=(
            discretization_controls.gradient_boundary_weight
        ),
        diffusion_method=discretization_controls.diffusion_method,
        diffusion_nonorthogonal_eps=(
            discretization_controls.diffusion_nonorthogonal_eps
        ),
        use_velocity_face_average=(
            discretization_controls.face_flux_correction_scheme != "none"
        ),
        use_traction_face_average=True,
        face_quadrature_order=(
            discretization_controls.face_flux_quadrature_order
        ),
    )
    geometry = physical.geometry
    fixed_faces = bm.sort(
        bm.concatenate(
            (
                physical.pressure_state.gradient.boundary.dirichlet_faces,
                physical.momentum.traction.faces,
            )
        )
    )
    fixed_values = bm.zeros(
        fixed_faces.shape[0],
        dtype=geometry.cell_center.dtype,
        device=bm.get_device(geometry.cell_center),
    )
    closure = (
        PressureClosureKind.DIRICHLET
        if fixed_faces.shape[0] > 0
        else pressure_system_controls.pure_neumann_closure
    )
    pressure_correction = resolve_pressure_system_boundary(
        geometry,
        fixed_faces,
        fixed_values,
        closure,
        pressure_gradient_method=(
            discretization_controls.pressure_gradient_method
        ),
        gradient_layer_weights=(
            discretization_controls.gradient_layer_weights
        ),
        gradient_boundary_weight=(
            discretization_controls.gradient_boundary_weight
        ),
        diffusion_method=discretization_controls.diffusion_method,
        diffusion_nonorthogonal_eps=(
            discretization_controls.diffusion_nonorthogonal_eps
        ),
    )
    return ResolvedSimpleBoundaryConditions(
        physical=physical,
        pressure_correction=pressure_correction,
        rhie_chow=ResolvedRhieChowBoundary(
            pressure_state=physical.pressure_state,
            zero_gradient_difference_faces=physical.momentum.traction.faces,
        ),
    )


def resolve_collocated_piso_boundary_conditions(
    boundary: NormalizedCollocatedBoundary,
    controls: PisoSolverControls,
    pressure_system_controls: CollocatedPressureSystemControls,
) -> ResolvedPisoBoundaryConditions:
    if boundary.momentum.traction_faces.shape[0] > 0:
        raise ValueError(
            "momentum traction is not supported by the PISO boundary resolver."
        )
    physical = resolve_physical_boundary_conditions(
        boundary,
        velocity_gradient_method=controls.velocity_gradient_method,
        pressure_gradient_method=controls.pressure_gradient_method,
        gradient_layer_weights=controls.gradient_layer_weights,
        gradient_boundary_weight=controls.gradient_boundary_weight,
        diffusion_method=controls.diffusion_method,
        diffusion_nonorthogonal_eps=(
            controls.diffusion_nonorthogonal_eps
        ),
        use_velocity_face_average=False,
        use_traction_face_average=False,
        face_quadrature_order=1,
    )
    geometry = physical.geometry
    pressure_gradient_boundary = physical.pressure_state.gradient.boundary
    faces = pressure_gradient_boundary.dirichlet_faces
    values = pressure_gradient_boundary.dirichlet_values
    closure = (
        PressureClosureKind.DIRICHLET
        if faces.shape[0] > 0
        else pressure_system_controls.pure_neumann_closure
    )
    return ResolvedPisoBoundaryConditions(
        physical=physical,
        pressure_corrector=resolve_pressure_system_boundary(
            geometry,
            faces,
            values,
            closure,
            pressure_gradient_method=controls.pressure_gradient_method,
            gradient_layer_weights=controls.gradient_layer_weights,
            gradient_boundary_weight=controls.gradient_boundary_weight,
            diffusion_method=controls.diffusion_method,
            diffusion_nonorthogonal_eps=(
                controls.diffusion_nonorthogonal_eps
            ),
        ),
        rhie_chow=ResolvedRhieChowBoundary(
            pressure_state=physical.pressure_state,
            zero_gradient_difference_faces=geometry.boundary_faces[:0],
        ),
    )


__all__ = [
    "ResolvedPisoBoundaryConditions",
    "ResolvedSimpleBoundaryConditions",
    "resolve_collocated_piso_boundary_conditions",
    "resolve_collocated_simple_boundary_conditions",
]
