"""External boundary descriptions and collocated-algorithm resolvers.

External PDE data describe velocity Dirichlet,
velocity Neumann, velocity natural outlet, pressure Dirichlet/reference
constraints, and steady-momentum traction.  Engineering patch names are first
normalized to that PDE representation, then resolved into algorithm-specific
operators before entering SIMPLE or PISO.  Solver code therefore sees neither
inlet/outlet/wall labels nor raw PDE boundary kinds.

When pressure has no Dirichlet/reference patch, the pressure equation is closed
by the solver's homogeneous Neumann plus gauge route.  That implicit pressure
closure is different from an explicit ``pressure/neumann(value)`` engineering
condition, which is intentionally rejected for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .collocated_boundary_conditions import (
    resolve_collocated_piso_boundary_conditions,
    resolve_collocated_simple_boundary_conditions,
)
from .fvm_geometry import (
    FVMGeometry,
    boundary_face_flag,
    select_boundary_faces,
)


def _boundary_value(value, points, variable: str):
    """Evaluate one boundary value under the ``(..., GD)`` contract."""
    is_callable = callable(value)
    raw = value(points) if is_callable else value
    result = bm.array(
        raw,
        dtype=points.dtype,
        device=bm.get_device(points),
    )
    point_shape = points.shape[:-1]
    dimension = points.shape[-1]

    if variable in {"velocity", "momentum"}:
        if is_callable:
            if result.shape != points.shape:
                raise ValueError(
                    f"{variable} boundary callable must return shape (..., GD)."
                )
            return result
        if result.shape == points.shape:
            return result
        if result.shape in {(dimension,), ()}:
            return bm.broadcast_to(result, points.shape)
        raise ValueError(
            f"{variable} boundary data must be scalar, have shape (GD,), "
            "or have shape (..., GD)."
        )

    if is_callable:
        if result.shape != point_shape:
            raise ValueError(
                "pressure boundary callable must return shape (...)."
            )
        return result
    if result.shape == point_shape:
        return result
    if result.shape == ():
        return bm.broadcast_to(result, point_shape)
    raise ValueError(
        "pressure boundary data must be scalar or have shape (...)."
    )


@dataclass(frozen=True)
class BoundaryPatch:
    """Named boundary-face patch selected by physical face centers."""

    name: str
    selector: Callable


@dataclass(frozen=True)
class BoundaryCondition:
    """Engineering boundary condition assigned to one variable and patch.

    Current supported combinations are ``velocity/dirichlet``,
    ``velocity/neumann``, ``velocity/natural``, ``pressure/dirichlet``,
    ``pressure/reference``, and ``momentum/traction``.  Explicit
    ``pressure/neumann(value)`` is rejected because pressure-correction solvers
    need a separate flux-consistency design for that case.
    """

    variable: str
    patch: str
    kind: str
    value: object = None


@dataclass(frozen=True)
class PDEVelocityBoundary:
    """Normalized velocity boundary data on one fixed mesh."""

    dirichlet_faces: TensorLike
    dirichlet_value: Callable[[TensorLike], TensorLike]
    neumann_faces: TensorLike
    neumann_value: Callable[[TensorLike], TensorLike]
    natural_faces: TensorLike


@dataclass(frozen=True)
class PDEPressureBoundary:
    """Normalized physical-pressure boundary data on one fixed mesh."""

    dirichlet_faces: TensorLike
    dirichlet_value: Callable[[TensorLike], TensorLike]


@dataclass(frozen=True)
class PDEMomentumBoundary:
    """Normalized steady-momentum traction data on one fixed mesh."""

    traction_faces: TensorLike
    traction_value: Callable[[TensorLike], TensorLike]


class PDEBoundaryConditions:
    """Strict PDE boundary data accepted by the boundary resolvers.

    This object has no engineering patch semantics.  It exposes typed
    ``velocity``, ``pressure``, and ``momentum`` data objects containing the
    face tensors and value callables from which the SIMPLE and PISO resolvers
    construct algorithm-specific boundary operators:

    - velocity Dirichlet value;
    - velocity Neumann normal derivative;
    - velocity natural outlet faces;
    - pressure Dirichlet value, including engineering pressure reference.
    - steady-momentum traction value and faces.

    If pressure Dirichlet data are absent, the pressure equation uses the
    solver-side homogeneous Neumann plus gauge route.  That route is not
    represented as a ``pressure/neumann(value)`` boundary condition here.
    """

    def __init__(
        self,
        mesh,
        *,
        dirichlet_velocity=None,
        dirichlet_velocity_selector=None,
        neumann_velocity=None,
        neumann_velocity_selector=None,
        natural_velocity_selector=None,
        dirichlet_pressure=None,
        dirichlet_pressure_selector=None,
        momentum_traction=None,
        momentum_traction_selector=None,
        geometry=None,
    ) -> None:
        self.mesh = mesh
        self.geometry = FVMGeometry(mesh) if geometry is None else geometry
        dirichlet_velocity_faces = self._resolve_faces(
            dirichlet_velocity_selector,
            active=dirichlet_velocity is not None,
            name="velocity Dirichlet",
        )
        neumann_velocity_faces = self._resolve_faces(
            neumann_velocity_selector,
            active=neumann_velocity is not None,
            name="velocity Neumann",
        )
        natural_velocity_faces = self._resolve_faces(
            natural_velocity_selector,
            active=natural_velocity_selector is not None,
            name="velocity natural",
        )
        dirichlet_pressure_faces = self._resolve_faces(
            dirichlet_pressure_selector,
            active=dirichlet_pressure is not None,
            name="pressure Dirichlet",
        )
        momentum_traction_faces = self._resolve_faces(
            momentum_traction_selector,
            active=momentum_traction is not None,
            name="momentum traction",
        )
        self._validate_disjoint(
            dirichlet_velocity_faces,
            neumann_velocity_faces,
            "velocity Dirichlet and Neumann faces must be disjoint.",
        )
        self._validate_disjoint(
            dirichlet_velocity_faces,
            natural_velocity_faces,
            "velocity Dirichlet and natural faces must be disjoint.",
        )
        self._validate_disjoint(
            neumann_velocity_faces,
            natural_velocity_faces,
            "velocity Neumann and natural faces must be disjoint.",
        )
        self._validate_disjoint(
            dirichlet_pressure_faces,
            momentum_traction_faces,
            (
                "pressure Dirichlet/reference and momentum traction "
                "faces must be disjoint."
            ),
        )
        self.velocity = PDEVelocityBoundary(
            dirichlet_faces=dirichlet_velocity_faces,
            dirichlet_value=self._canonical_value(
                dirichlet_velocity,
                variable="velocity",
            ),
            neumann_faces=neumann_velocity_faces,
            neumann_value=self._canonical_value(
                neumann_velocity,
                variable="velocity",
            ),
            natural_faces=natural_velocity_faces,
        )
        self.pressure = PDEPressureBoundary(
            dirichlet_faces=dirichlet_pressure_faces,
            dirichlet_value=self._canonical_value(
                dirichlet_pressure,
                variable="pressure",
            ),
        )
        self.momentum = PDEMomentumBoundary(
            traction_faces=momentum_traction_faces,
            traction_value=self._canonical_value(
                momentum_traction,
                variable="momentum",
            ),
        )

    def _resolve_faces(self, selector, *, active: bool, name: str):
        if not active:
            if selector is not None:
                raise ValueError(
                    f"{name} selector requires corresponding boundary data."
                )
            return self.geometry.boundary_faces[:0]
        if selector is None:
            return self.geometry.boundary_faces
        return select_boundary_faces(self.geometry, selector)

    @staticmethod
    def _validate_disjoint(first_faces, second_faces, message: str) -> None:
        faces = bm.sort(
            bm.concatenate((first_faces, second_faces))
        )
        if faces.shape[0] < 2:
            return
        if bool(bm.to_numpy(bm.any(faces[1:] == faces[:-1]))):
            raise ValueError(message)

    @staticmethod
    def _canonical_value(value, *, variable: str):
        """Return one callable following the Cartesian boundary-value contract."""

        if value is None:
            def canonical(points):
                if variable in {"velocity", "momentum"}:
                    return bm.zeros_like(points)
                return bm.zeros(
                    points.shape[:-1],
                    dtype=points.dtype,
                    device=bm.get_device(points),
                )

            return canonical

        def canonical(points):
            return _boundary_value(value, points, variable)

        return canonical

class EngineeringBoundaryConditions:
    """Map engineering patches to strict PDE boundary-condition data.

    The adapter owns no SIMPLE/PISO algebra.  It converts named patches and
    engineering boundary data into value callables and explicit face tensors.  The
    algorithm-specific resolver then constructs the boundary operators consumed
    by the solver.
    """

    _SUPPORTED_VARIABLES = {"velocity", "pressure", "momentum"}
    _SUPPORTED_KINDS = {
        "dirichlet",
        "neumann",
        "natural",
        "reference",
        "traction",
    }

    def __init__(
        self,
        mesh,
        patches: Iterable[BoundaryPatch],
        conditions: Iterable[BoundaryCondition],
    ) -> None:
        self.mesh = mesh
        self.geometry = FVMGeometry(mesh)
        self.patches = tuple(patches)
        self.conditions = tuple(conditions)
        patch_by_name = {}
        for patch in self.patches:
            if patch.name in patch_by_name:
                raise ValueError(f"duplicate boundary patch: {patch.name!r}.")
            if not callable(patch.selector):
                raise ValueError(f"selector for patch {patch.name!r} must be callable.")
            patch_by_name[patch.name] = patch
        self._patch_by_name = patch_by_name
        self._validate_conditions()

    def _validate_conditions(self) -> None:
        seen = set()
        for condition in self.conditions:
            if condition.variable not in self._SUPPORTED_VARIABLES:
                raise ValueError(f"unsupported boundary variable: {condition.variable!r}.")
            if condition.kind not in self._SUPPORTED_KINDS:
                raise ValueError(f"unsupported boundary kind: {condition.kind!r}.")
            if condition.patch not in self._patch_by_name:
                raise ValueError(f"unknown boundary patch: {condition.patch!r}.")
            # Combination rules keep the engineering layer aligned with the
            # strict PDE boundary protocol accepted by the resolvers.
            if condition.kind == "reference" and condition.variable != "pressure":
                raise ValueError("reference boundary is only supported for pressure.")
            if condition.kind == "natural" and condition.variable != "velocity":
                raise ValueError("natural boundary is only supported for velocity.")
            if condition.kind == "traction" and condition.variable != "momentum":
                raise ValueError("traction boundary is only supported for momentum.")
            if condition.variable == "momentum" and condition.kind != "traction":
                raise ValueError("momentum only supports traction boundary data.")
            if condition.variable == "pressure" and condition.kind == "neumann":
                raise ValueError(
                    "explicit pressure/neumann(value) is not implemented for "
                    "pressure-correction solvers."
                )
            key = (condition.variable, condition.patch)
            if key in seen:
                raise ValueError(
                    f"duplicate condition for {condition.variable!r} on "
                    f"{condition.patch!r}."
                )
            seen.add(key)
            if condition.kind in (
                "dirichlet",
                "neumann",
                "reference",
                "traction",
            ) and condition.value is None:
                raise ValueError(
                    f"{condition.kind.capitalize()} condition on "
                    f"{condition.patch!r} needs value data."
                )
        traction_patches = {
            condition.patch
            for condition in self._conditions("momentum", "traction")
        }
        pressure_value_patches = {
            condition.patch
            for kind in ("dirichlet", "reference")
            for condition in self._conditions("pressure", kind)
        }
        conflicts = traction_patches & pressure_value_patches
        if conflicts:
            patches = ", ".join(sorted(repr(patch) for patch in conflicts))
            raise ValueError(
                "momentum traction conflicts with pressure Dirichlet/reference "
                f"on patch {patches}."
            )

    def _conditions(self, variable: str, kind: str | None = None):
        """Return conditions matching a variable and optionally a kind."""
        return tuple(
            condition
            for condition in self.conditions
            if condition.variable == variable
            and (kind is None or condition.kind == kind)
        )

    def to_pde_boundary(self):
        """Return strict PDE boundary data for solver kernels."""
        return PDEBoundaryConditions(
            self.mesh,
            geometry=self.geometry,
            dirichlet_velocity=(
                self.dirichlet_value("velocity")
                if self.has_dirichlet("velocity")
                else None
            ),
            dirichlet_velocity_selector=(
                self.dirichlet_selector("velocity")
                if self.has_dirichlet("velocity")
                else None
            ),
            neumann_velocity=(
                self.neumann_value("velocity")
                if self._conditions("velocity", "neumann")
                else None
            ),
            neumann_velocity_selector=(
                self.neumann_selector("velocity")
                if self._conditions("velocity", "neumann")
                else None
            ),
            natural_velocity_selector=(
                self.natural_selector("velocity")
                if self._conditions("velocity", "natural")
                else None
            ),
            dirichlet_pressure=(
                self.dirichlet_value("pressure")
                if self.has_dirichlet("pressure")
                else None
            ),
            dirichlet_pressure_selector=(
                self.dirichlet_selector("pressure")
                if self.has_dirichlet("pressure")
                else None
            ),
            momentum_traction=(
                self.traction_value("momentum")
                if self._conditions("momentum", "traction")
                else None
            ),
            momentum_traction_selector=(
                self.traction_selector("momentum")
                if self._conditions("momentum", "traction")
                else None
            ),
        )

    def has_dirichlet(self, variable: str) -> bool:
        """Return whether ``variable`` has at least one Dirichlet patch.

        Pressure reference patches are treated as pressure Dirichlet patches
        before the data reach the solver.
        """
        if variable == "pressure":
            return bool(
                self._conditions("pressure", "dirichlet")
                or self._conditions("pressure", "reference")
            )
        return bool(self._conditions(variable, "dirichlet"))

    def dirichlet_selector(self, variable: str):
        """Return a face-center selector for all Dirichlet patches of variable."""
        kinds = ("dirichlet", "reference") if variable == "pressure" else ("dirichlet",)
        return self.condition_selector(variable, kinds)

    def natural_selector(self, variable: str):
        """Return a face-center selector for all natural patches of variable."""
        return self.condition_selector(variable, ("natural",))

    def neumann_selector(self, variable: str):
        """Return a face-center selector for all Neumann patches of variable."""
        return self.condition_selector(variable, ("neumann",))

    def traction_selector(self, variable: str):
        """Return a face-center selector for all traction patches."""
        return self.condition_selector(variable, ("traction",))

    def condition_selector(self, variable: str, kinds):
        """Return a face-center selector for all patches of selected kinds."""
        conditions = tuple(
            condition
            for kind in kinds
            for condition in self._conditions(variable, kind)
        )

        def selector(points):
            flag = bm.zeros(
                points.shape[:-1],
                dtype=bm.bool,
                device=bm.get_device(points),
            )
            for condition in conditions:
                flag = flag | self._patch_flag(condition.patch, points)
            return flag

        return selector

    def dirichlet_value(self, variable: str):
        """Return a value callable assembled from Dirichlet patch data."""
        kinds = ("dirichlet", "reference") if variable == "pressure" else ("dirichlet",)
        return self.combined_value(variable, kinds)

    def neumann_value(self, variable: str):
        """Return a value callable assembled from Neumann patch data."""
        return self.combined_value(variable, ("neumann",))

    def traction_value(self, variable: str):
        """Return a value callable assembled from traction patch data."""
        return self.combined_value(variable, ("traction",))

    def combined_value(self, variable: str, kinds):
        """Return a value callable assembled from patch data for selected kinds."""
        conditions = tuple(
            condition
            for kind in kinds
            for condition in self._conditions(variable, kind)
        )
        if not conditions:
            raise ValueError(f"{variable!r} has no {kinds!r} boundary conditions.")

        def value(points):
            result = self._empty_value(points, variable)
            for condition in conditions:
                flag = self._patch_flag(condition.patch, points)
                if bool(bm.to_numpy(bm.any(flag))):
                    condition_value = self._condition_value(condition, points, variable)
                    result = bm.set_at(result, flag, condition_value[flag])
            return result

        return value

    def patch_face_mask(self, patch_name: str):
        """Return a boundary-face mask for one named patch."""
        points = self.geometry.face_center[self.geometry.boundary_faces]
        return self._patch_flag(patch_name, points)

    def patch_face_index(self, patch_name: str):
        """Return global face indices belonging to one named patch."""
        return self.geometry.boundary_faces[
            self.patch_face_mask(patch_name)
        ]

    def _patch_flag(self, patch_name: str, points):
        patch = self._patch_by_name[patch_name]
        return boundary_face_flag(points, patch.selector)

    def _empty_value(self, points, variable: str):
        if variable in {"velocity", "momentum"}:
            return bm.zeros(
                points.shape,
                dtype=points.dtype,
                device=bm.get_device(points),
            )
        return bm.zeros(
            points.shape[:-1],
            dtype=points.dtype,
            device=bm.get_device(points),
        )

    def _condition_value(self, condition: BoundaryCondition, points, variable: str):
        return _boundary_value(condition.value, points, variable)


def normalize_pde_boundary_conditions(mesh, boundary_conditions):
    if isinstance(boundary_conditions, EngineeringBoundaryConditions):
        boundary_conditions = boundary_conditions.to_pde_boundary()
    if not isinstance(boundary_conditions, PDEBoundaryConditions):
        raise TypeError(
            "boundary_conditions must be EngineeringBoundaryConditions "
            "or PDEBoundaryConditions."
        )
    if boundary_conditions.mesh is not mesh:
        raise ValueError(
            "mesh-bound boundary conditions require their original mesh."
        )
    return boundary_conditions


def resolve_simple_boundary_conditions(
    mesh,
    boundary_conditions,
    discretization_controls,
    pressure_system_controls,
):
    boundary = normalize_pde_boundary_conditions(mesh, boundary_conditions)
    return resolve_collocated_simple_boundary_conditions(
        boundary,
        discretization_controls,
        pressure_system_controls,
    )


def resolve_piso_boundary_conditions(
    mesh,
    boundary_conditions,
    controls,
    pressure_system_controls,
):
    boundary = normalize_pde_boundary_conditions(mesh, boundary_conditions)
    if boundary.momentum.traction_faces.shape[0] > 0:
        raise ValueError(
            "momentum traction is not supported by the PISO boundary resolver."
        )
    return resolve_collocated_piso_boundary_conditions(
        boundary,
        controls,
        pressure_system_controls,
    )
