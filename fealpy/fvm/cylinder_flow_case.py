"""Cylinder-flow case adapter and Gmsh mesh generator for FVM benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.decorator import cartesian, variantmethod
from fealpy.mesh import TriangleMesh

from .fvm_geometry import FVMGeometry


@dataclass(frozen=True)
class CylinderMeshQuality:
    """Basic mesh-quality diagnostics used by the cylinder benchmark setup."""

    number_of_cells: int
    number_of_faces: int
    number_of_boundary_faces: int
    min_cell_angle: float
    max_cell_angle: float
    mean_cell_angle: float
    max_nonorthogonal_angle: float
    p95_nonorthogonal_angle: float
    mean_nonorthogonal_angle: float
    unclassified_boundary_faces: int
    multi_classified_boundary_faces: int
    boundary_face_counts: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "number_of_cells": self.number_of_cells,
            "number_of_faces": self.number_of_faces,
            "number_of_boundary_faces": self.number_of_boundary_faces,
            "min_cell_angle": self.min_cell_angle,
            "max_cell_angle": self.max_cell_angle,
            "mean_cell_angle": self.mean_cell_angle,
            "max_nonorthogonal_angle": self.max_nonorthogonal_angle,
            "p95_nonorthogonal_angle": self.p95_nonorthogonal_angle,
            "mean_nonorthogonal_angle": self.mean_nonorthogonal_angle,
            "unclassified_boundary_faces": self.unclassified_boundary_faces,
            "multi_classified_boundary_faces": self.multi_classified_boundary_faces,
            "boundary_face_counts": self.boundary_face_counts,
        }


class CylinderFlowCase:
    """DFG-style 2D cylinder-flow benchmark case for collocated FVM solvers.

    The mesh generator uses exact circular arcs in Gmsh only to define the
    geometry and sizing fields.  It always exports first-order triangular
    elements to FEALPy, so all FVM faces remain straight line segments.
    """

    default_mesh_type = "improved_tri"
    supports_geometric_refine = False

    def __init__(
        self,
        re: float = 20.0,
        *,
        box: Sequence[float] = (0.0, 2.2, 0.0, 0.41),
        center: Sequence[float] = (0.2, 0.2),
        radius: float = 0.05,
        mean_velocity: float = 0.2,
        rho: float = 1.0,
        mu: float | None = None,
        mesh_size: float = 0.04,
        cylinder_mesh_size: float = 0.006,
        wake_mesh_size: float = 0.02,
        cylinder_refine_radius: float | None = None,
        wake_length: float | None = None,
        wake_half_width: float | None = None,
        outlet_velocity_policy: str = "profile",
        eps: float = 1.0e-12,
    ) -> None:
        if len(box) != 4:
            raise ValueError("box must be (xmin, xmax, ymin, ymax).")
        if len(center) != 2:
            raise ValueError("center must be (cx, cy).")
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        if re <= 0.0:
            raise ValueError("re must be positive.")
        if rho <= 0.0:
            raise ValueError("rho must be positive.")
        if mesh_size <= 0.0:
            raise ValueError("mesh_size must be positive.")
        if cylinder_mesh_size <= 0.0:
            raise ValueError("cylinder_mesh_size must be positive.")
        if wake_mesh_size <= 0.0:
            raise ValueError("wake_mesh_size must be positive.")
        if outlet_velocity_policy not in {"profile", "zero"}:
            raise ValueError("outlet_velocity_policy must be 'profile' or 'zero'.")

        self.box = tuple(float(value) for value in box)
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius)
        self.mean_velocity = float(mean_velocity)
        self.rho = float(rho)
        self.re = float(re)
        self.mesh_size = float(mesh_size)
        self.cylinder_mesh_size = float(cylinder_mesh_size)
        self.wake_mesh_size = float(wake_mesh_size)
        refine_radius = 3.0 * self.radius if cylinder_refine_radius is None else cylinder_refine_radius
        self.cylinder_refine_radius = float(refine_radius)
        self.wake_length = self.box[1] - (self.center[0] + self.radius) if wake_length is None else float(wake_length)
        self.wake_half_width = 1.5 * self.radius if wake_half_width is None else float(wake_half_width)
        self.outlet_velocity_policy = outlet_velocity_policy
        self.eps = float(eps)

        diameter = 2.0 * self.radius
        if mu is None:
            self.mu = self.rho * abs(self.mean_velocity) * diameter / self.re
        else:
            if mu <= 0.0:
                raise ValueError("mu must be positive.")
            self.mu = float(mu)
            if abs(self.mean_velocity) * diameter > 0.0:
                self.re = self.rho * abs(self.mean_velocity) * diameter / self.mu
        self.nu = self.mu / self.rho

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    @variantmethod("improved_tri")
    def init_mesh(
        self,
        *,
        mesh_size: float | None = None,
        cylinder_mesh_size: float | None = None,
        wake_mesh_size: float | None = None,
    ) -> TriangleMesh:
        """Generate an improved first-order triangular cylinder-flow mesh."""
        return self._init_improved_tri_mesh(
            mesh_size=self.mesh_size if mesh_size is None else float(mesh_size),
            cylinder_mesh_size=(
                self.cylinder_mesh_size
                if cylinder_mesh_size is None
                else float(cylinder_mesh_size)
            ),
            wake_mesh_size=(
                self.wake_mesh_size
                if wake_mesh_size is None
                else float(wake_mesh_size)
            ),
        )

    def _init_improved_tri_mesh(
        self,
        *,
        mesh_size: float,
        cylinder_mesh_size: float,
        wake_mesh_size: float,
    ) -> TriangleMesh:
        import gmsh

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.Smoothing", 10)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cylinder_mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

            gmsh.model.add("fealpy_fvm_cylinder_flow")
            surface, boundary_curves = self._build_gmsh_geometry(
                gmsh, mesh_size, cylinder_mesh_size
            )
            self._add_gmsh_physical_groups(gmsh, surface, boundary_curves)
            self._add_gmsh_size_fields(
                gmsh,
                boundary_curves["cylinder"],
                mesh_size,
                cylinder_mesh_size,
                wake_mesh_size,
            )
            gmsh.model.mesh.generate(2)
            node, cell = self._extract_gmsh_triangles(gmsh)
        finally:
            gmsh.finalize()

        return TriangleMesh(
            bm.array(node, dtype=bm.float64),
            bm.array(cell, dtype=bm.int32),
        )

    def _build_gmsh_geometry(self, gmsh, mesh_size, cylinder_mesh_size):
        xmin, xmax, ymin, ymax = self.box
        cx, cy = self.center
        r = self.radius
        geo = gmsh.model.geo

        p0 = geo.addPoint(xmin, ymin, 0.0, mesh_size)
        p1 = geo.addPoint(xmax, ymin, 0.0, mesh_size)
        p2 = geo.addPoint(xmax, ymax, 0.0, mesh_size)
        p3 = geo.addPoint(xmin, ymax, 0.0, mesh_size)
        l_bottom = geo.addLine(p0, p1)
        l_outlet = geo.addLine(p1, p2)
        l_top = geo.addLine(p2, p3)
        l_inlet = geo.addLine(p3, p0)

        pc = geo.addPoint(cx, cy, 0.0, cylinder_mesh_size)
        pe = geo.addPoint(cx + r, cy, 0.0, cylinder_mesh_size)
        pn = geo.addPoint(cx, cy + r, 0.0, cylinder_mesh_size)
        pw = geo.addPoint(cx - r, cy, 0.0, cylinder_mesh_size)
        ps = geo.addPoint(cx, cy - r, 0.0, cylinder_mesh_size)
        c1 = geo.addCircleArc(pe, pc, pn)
        c2 = geo.addCircleArc(pn, pc, pw)
        c3 = geo.addCircleArc(pw, pc, ps)
        c4 = geo.addCircleArc(ps, pc, pe)

        outer_loop = geo.addCurveLoop([l_bottom, l_outlet, l_top, l_inlet])
        cylinder_loop = geo.addCurveLoop([c1, c2, c3, c4])
        surface = geo.addPlaneSurface([outer_loop, cylinder_loop])
        geo.synchronize()

        return surface, {
            "inlet": [l_inlet],
            "outlet": [l_outlet],
            "walls": [l_bottom, l_top],
            "cylinder": [c1, c2, c3, c4],
        }

    @staticmethod
    def _add_gmsh_physical_groups(gmsh, surface, boundary_curves):
        gmsh.model.addPhysicalGroup(1, boundary_curves["inlet"], tag=1)
        gmsh.model.setPhysicalName(1, 1, "inlet")
        gmsh.model.addPhysicalGroup(1, boundary_curves["outlet"], tag=2)
        gmsh.model.setPhysicalName(1, 2, "outlet")
        gmsh.model.addPhysicalGroup(1, boundary_curves["walls"], tag=3)
        gmsh.model.setPhysicalName(1, 3, "walls")
        gmsh.model.addPhysicalGroup(1, boundary_curves["cylinder"], tag=4)
        gmsh.model.setPhysicalName(1, 4, "cylinder")
        gmsh.model.addPhysicalGroup(2, [surface], tag=5)
        gmsh.model.setPhysicalName(2, 5, "fluid")

    def _add_gmsh_size_fields(
        self,
        gmsh,
        cylinder_curves,
        mesh_size,
        cylinder_mesh_size,
        wake_mesh_size,
    ) -> None:
        field = gmsh.model.mesh.field
        distance = field.add("Distance")
        field.setNumbers(distance, "CurvesList", cylinder_curves)
        field.setNumber(distance, "Sampling", 100)

        cylinder_refinement = field.add("Threshold")
        field.setNumber(cylinder_refinement, "InField", distance)
        field.setNumber(cylinder_refinement, "SizeMin", cylinder_mesh_size)
        field.setNumber(cylinder_refinement, "SizeMax", mesh_size)
        field.setNumber(cylinder_refinement, "DistMin", 0.25 * self.radius)
        field.setNumber(cylinder_refinement, "DistMax", self.cylinder_refine_radius)

        cx, cy = self.center
        wake = field.add("Box")
        field.setNumber(wake, "VIn", wake_mesh_size)
        field.setNumber(wake, "VOut", mesh_size)
        field.setNumber(wake, "XMin", cx + self.radius)
        field.setNumber(wake, "XMax", min(self.box[1], cx + self.wake_length))
        field.setNumber(wake, "YMin", max(self.box[2], cy - self.wake_half_width))
        field.setNumber(wake, "YMax", min(self.box[3], cy + self.wake_half_width))

        minimum = field.add("Min")
        field.setNumbers(minimum, "FieldsList", [cylinder_refinement, wake])
        field.setAsBackgroundMesh(minimum)

    @staticmethod
    def _extract_gmsh_triangles(gmsh):
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = np.asarray(node_coords, dtype=float).reshape(-1, 3)[:, :2]
        tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)
        triangle_nodes = None
        for elem_type, nodes in zip(elem_types, elem_node_tags):
            if int(elem_type) == 2:
                triangle_nodes = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
                break
        if triangle_nodes is None:
            raise RuntimeError("Gmsh did not generate first-order triangle elements.")

        cell = np.array(
            [[tag_to_index[int(tag)] for tag in tri] for tri in triangle_nodes],
            dtype=np.int32,
        )
        return node, cell

    @cartesian
    def source(self, p: TensorLike, t: float | None = None) -> TensorLike:
        return bm.zeros(
            p.shape,
            dtype=p.dtype,
            device=bm.get_device(p),
        )

    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        y = p[..., 1]
        height = self.box[3] - self.box[2]
        y0 = y - self.box[2]
        profile = 6.0 * self.mean_velocity * y0 * (height - y0) / height**2
        zeros = bm.zeros_like(profile)
        return bm.stack([profile, zeros], axis=-1)

    @cartesian
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike:
        result = bm.zeros(
            p.shape,
            dtype=p.dtype,
            device=bm.get_device(p),
        )
        profile = self.inlet_velocity(p)
        inlet = self.is_inlet_boundary(p)
        result = bm.set_at(result, inlet, profile[inlet])
        if self.outlet_velocity_policy == "profile":
            outlet = self.is_outlet_boundary(p)
            result = bm.set_at(result, outlet, profile[outlet])
        return result

    @cartesian
    @cartesian
    def velocity_0(self, p: TensorLike, t: float = 0.0) -> TensorLike:
        return bm.zeros(
            p.shape,
            dtype=p.dtype,
            device=bm.get_device(p),
        )

    @cartesian
    def pressure_0(self, p: TensorLike, t: float = 0.0) -> TensorLike:
        return bm.zeros(
            p.shape[:-1],
            dtype=p.dtype,
            device=bm.get_device(p),
        )

    @cartesian
    def velocity(self, p: TensorLike, t: float | None = None) -> TensorLike:
        return bm.zeros(
            p.shape,
            dtype=p.dtype,
            device=bm.get_device(p),
        )

    @cartesian
    def pressure(self, p: TensorLike, t: float | None = None) -> TensorLike:
        return bm.zeros(
            p.shape[:-1],
            dtype=p.dtype,
            device=bm.get_device(p),
        )

    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        return bm.abs(p[..., 0] - self.box[0]) <= self.eps

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        return bm.abs(p[..., 0] - self.box[1]) <= self.eps

    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        y = p[..., 1]
        return (bm.abs(y - self.box[2]) <= self.eps) | (
            bm.abs(y - self.box[3]) <= self.eps
        )

    @cartesian
    def is_cylinder_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        cx, cy = self.center
        distance = bm.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        tolerance = max(2.0 * self.cylinder_mesh_size, self.eps)
        return bm.abs(distance - self.radius) <= tolerance

    @cartesian
    def engineering_boundary_conditions(self, mesh, pde=None):
        """Return patch-wise engineering boundary conditions for cylinder flow."""
        from .engineering_boundary_conditions import (
            BoundaryCondition,
            BoundaryPatch,
            EngineeringBoundaryConditions,
        )

        def zero_velocity(p):
            return bm.zeros(
                p.shape,
                dtype=p.dtype,
                device=bm.get_device(p),
            )

        patches = [
            BoundaryPatch("inlet", self.is_inlet_boundary),
            BoundaryPatch("outlet", self.is_outlet_boundary),
            BoundaryPatch("walls", self.is_wall_boundary),
            BoundaryPatch("cylinder", self.is_cylinder_boundary),
        ]
        conditions = [
            BoundaryCondition(
                "velocity",
                "inlet",
                "dirichlet",
                self.inlet_velocity,
            ),
            BoundaryCondition("velocity", "walls", "dirichlet", zero_velocity),
            BoundaryCondition("velocity", "cylinder", "dirichlet", zero_velocity),
            BoundaryCondition("velocity", "outlet", "natural", None),
            BoundaryCondition("pressure", "outlet", "dirichlet", 0.0),
        ]
        return EngineeringBoundaryConditions(mesh, patches, conditions)

    def mesh_quality(self, mesh: TriangleMesh) -> dict:
        """Return mesh-quality metrics relevant to FVM cylinder-flow tests."""
        node = np.asarray(bm.to_numpy(mesh.entity("node")))
        cell = np.asarray(bm.to_numpy(mesh.entity("cell")), dtype=np.int64)
        angles = self._triangle_angles(node[cell])
        min_angle = angles.min(axis=1)
        max_angle = angles.max(axis=1)
        nonorthogonal = self._nonorthogonal_angles(mesh)
        geometry = FVMGeometry(mesh)
        boundary_counts, unclassified, multi_classified = self._boundary_counts(
            geometry
        )
        quality = CylinderMeshQuality(
            number_of_cells=geometry.NC,
            number_of_faces=geometry.NF,
            number_of_boundary_faces=int(bm.sum(geometry.is_boundary)),
            min_cell_angle=float(min_angle.min()),
            max_cell_angle=float(max_angle.max()),
            mean_cell_angle=float(angles.mean()),
            max_nonorthogonal_angle=float(nonorthogonal.max()),
            p95_nonorthogonal_angle=float(np.percentile(nonorthogonal, 95)),
            mean_nonorthogonal_angle=float(nonorthogonal.mean()),
            unclassified_boundary_faces=int(unclassified),
            multi_classified_boundary_faces=int(multi_classified),
            boundary_face_counts=boundary_counts,
        )
        return quality.as_dict()

    @staticmethod
    def _triangle_angles(points):
        a = np.linalg.norm(points[:, 1] - points[:, 2], axis=1)
        b = np.linalg.norm(points[:, 2] - points[:, 0], axis=1)
        c = np.linalg.norm(points[:, 0] - points[:, 1], axis=1)
        angles = []
        for edge_a, edge_b, opposite in ((b, c, a), (c, a, b), (a, b, c)):
            cosine = (edge_a**2 + edge_b**2 - opposite**2) / (
                2.0 * edge_a * edge_b
            )
            angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        return np.stack(angles, axis=1)

    @staticmethod
    def _nonorthogonal_angles(mesh):
        geometry = FVMGeometry(mesh)
        sf = np.asarray(bm.to_numpy(geometry.S_f))
        d = np.asarray(bm.to_numpy(geometry.d_f))
        face_to_cell = np.asarray(bm.to_numpy(geometry.face_to_cell), dtype=np.int64)
        interior = face_to_cell[:, 0] != face_to_cell[:, 1]
        sf = sf[interior]
        d = d[interior]
        cosine = np.einsum("ij,ij->i", sf, d) / (
            np.linalg.norm(sf, axis=1) * np.linalg.norm(d, axis=1)
        )
        return np.degrees(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))

    def _boundary_counts(self, geometry):
        boundary_faces = bm.nonzero(geometry.is_boundary)[0]
        face_centers = geometry.face_center[boundary_faces]
        flags = {
            "inlet": np.asarray(bm.to_numpy(self.is_inlet_boundary(face_centers))),
            "outlet": np.asarray(bm.to_numpy(self.is_outlet_boundary(face_centers))),
            "walls": np.asarray(bm.to_numpy(self.is_wall_boundary(face_centers))),
            "cylinder": np.asarray(bm.to_numpy(self.is_cylinder_boundary(face_centers))),
        }
        stacked = np.stack([flag.astype(bool) for flag in flags.values()], axis=0)
        total = stacked.sum(axis=0)
        return (
            {name: int(flag.sum()) for name, flag in flags.items()},
            int((total == 0).sum()),
            int((total > 1).sum()),
        )
