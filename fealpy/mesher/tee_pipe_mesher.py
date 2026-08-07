from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .gmsh_fsi_pipe_mesher import BaseGmshFSIPipeMesher, build_dimtag_maps


DimTag = Tuple[int, int]


class TeePipeMesher(BaseGmshFSIPipeMesher):
    """FSI tee-pipe mesher based on gmsh OCC geometry and FEALPy mesh output."""

    @staticmethod
    def default_parameters():
        return {
            "D_main": 32.0,
            "D_branch": 32.0,
            "intersect_angle": 90.0,
            "R_fillet": 12.25,
            "L_in_ratio": 5.0,
            "L_out_ratio": 10.0,
            "wall_thickness": 5.0,
            "mesh_size_global": None,
            "mesh_size_junction": None,
            "mesh_size_interface": None,
        }

    @classmethod
    def validate_parameters(cls, params):
        defaults = cls.default_parameters()
        unknown_keys = sorted(set(params) - set(defaults))
        if unknown_keys:
            raise ValueError(f"unknown parameter keys: {unknown_keys}")

        raw = {**defaults, **params}
        d_main = float(raw["D_main"])
        d_branch = float(raw["D_branch"])
        intersect_angle = float(raw["intersect_angle"])
        r_fillet = float(raw["R_fillet"])
        l_in_ratio = float(raw["L_in_ratio"])
        l_out_ratio = float(raw["L_out_ratio"])
        wall_thickness = float(raw["wall_thickness"])

        if d_main <= 0.0:
            raise ValueError("D_main must be > 0")
        if d_branch <= 0.0:
            raise ValueError("D_branch must be > 0")
        if wall_thickness <= 0.0:
            raise ValueError("wall_thickness must be > 0")
        if l_in_ratio <= 0.0:
            raise ValueError("L_in_ratio must be > 0")
        if l_out_ratio <= 0.0:
            raise ValueError("L_out_ratio must be > 0")
        if not 45.0 <= intersect_angle <= 100.0:
            raise ValueError("intersect_angle must be between 45 and 100 degrees")
        if r_fillet <= 0.0:
            raise ValueError("R_fillet must be > 0")

        if d_main > d_branch:
            raise NotImplementedError(
                "unsupported now: D_main > D_branch. "
                "Only equal-diameter tee is supported."
            )
        if d_main < d_branch:
            raise NotImplementedError(
                "unsupported now: D_main < D_branch. "
                "Only equal-diameter tee is supported."
            )

        inner_radius = 0.5 * d_branch
        if r_fillet >= inner_radius:
            raise ValueError("R_fillet must be smaller than inner radius")
        outer_radius = inner_radius + wall_thickness
        main_inlet_length = l_in_ratio * d_main
        branch1_length = l_out_ratio * d_branch
        branch2_length = l_out_ratio * d_branch

        default_sizes = {
            "mesh_size_global": 0.1875 * d_branch,
            "mesh_size_junction": 0.125 * d_branch,
            "mesh_size_interface": 0.09375 * d_branch,
        }
        mesh_size_global = (
            default_sizes["mesh_size_global"]
            if raw["mesh_size_global"] is None
            else float(raw["mesh_size_global"])
        )
        mesh_size_junction = (
            default_sizes["mesh_size_junction"]
            if raw["mesh_size_junction"] is None
            else float(raw["mesh_size_junction"])
        )
        mesh_size_interface = (
            default_sizes["mesh_size_interface"]
            if raw["mesh_size_interface"] is None
            else float(raw["mesh_size_interface"])
        )
        if mesh_size_global <= 0.0 or mesh_size_junction <= 0.0 or mesh_size_interface <= 0.0:
            raise ValueError(
                "mesh_size_global/mesh_size_junction/mesh_size_interface must be > 0"
            )

        return {
            "D_main": d_main,
            "D_branch": d_branch,
            "intersect_angle": intersect_angle,
            "R_fillet": r_fillet,
            "L_in_ratio": l_in_ratio,
            "L_out_ratio": l_out_ratio,
            "wall_thickness": wall_thickness,
            "main_inner_diameter": d_main,
            "branch_inner_diameter": d_branch,
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "main_inlet_length": main_inlet_length,
            "branch1_length": branch1_length,
            "branch2_length": branch2_length,
            "branch1_angle_deg": intersect_angle,
            "mesh_size_global": mesh_size_global,
            "mesh_size_junction": mesh_size_junction,
            "mesh_size_interface": mesh_size_interface,
        }

    def __init__(self, params=None, gmsh_module=None):
        external = self.default_parameters() if params is None else dict(params)
        internal = self.validate_parameters(external)
        super().__init__(internal, gmsh_module=gmsh_module)
        self.external_params = external
        self.internal_params = internal

    def model_name(self) -> str:
        return "fealpy_tee_pipe_mesher"

    def _branch1_direction(self, params):
        angle_rad = math.radians(float(params["branch1_angle_deg"]))
        return (-math.cos(angle_rad), math.sin(angle_rad), 0.0)

    def _port_centers(self, params):
        b1 = self._branch1_direction(params)
        b2 = (-b1[0], -b1[1], -b1[2])
        return (
            (-float(params["main_inlet_length"]), 0.0, 0.0),
            (
                b1[0] * float(params["branch1_length"]),
                b1[1] * float(params["branch1_length"]),
                0.0,
            ),
            (
                b2[0] * float(params["branch2_length"]),
                b2[1] * float(params["branch2_length"]),
                0.0,
            ),
        )

    def _single_dimtag(self, dimtags, dim, context):
        matches = [dimtag for dimtag in dimtags if dimtag[0] == dim]
        if len(matches) != 1:
            raise RuntimeError(f"{context}: expected exactly one {dim}D entity, got {matches}")
        return matches[0]

    def _pick_largest_volume(self, gmsh_module, dimtags, context):
        volumes = [dimtag for dimtag in dimtags if dimtag[0] == 3]
        if not volumes:
            raise RuntimeError(f"{context}: expected at least one 3D entity")
        return max(volumes, key=lambda v: gmsh_module.model.occ.getMass(*v))

    def _collect_junction_fillet_curve_tags(self, gmsh_module, volume_dimtag):
        surface_dimtags = gmsh_module.model.getBoundary(
            [volume_dimtag], oriented=False, recursive=False
        )
        candidate_curve_tags = set()
        for surface in surface_dimtags:
            if surface[0] != 2:
                continue
            curve_dimtags = gmsh_module.model.getBoundary(
                [surface], oriented=False, recursive=False
            )
            for curve in curve_dimtags:
                if curve[0] != 1:
                    continue
                up_surfaces, _ = gmsh_module.model.getAdjacencies(1, curve[1])
                if len(up_surfaces) < 2:
                    continue
                surface_types = [gmsh_module.model.getType(2, int(tag)) for tag in up_surfaces]
                if any(surface_type == "Plane" for surface_type in surface_types):
                    continue
                candidate_curve_tags.add(int(curve[1]))
        tags = sorted(candidate_curve_tags)
        if not tags:
            raise RuntimeError("failed to identify junction curves for fillet")
        return tags

    def _build_sharp_tee_volume(self, gmsh_module, radius, params):
        occ = gmsh_module.model.occ
        main_inlet, branch1_outlet, branch2_outlet = self._port_centers(params)

        cylinders = [
            (
                3,
                occ.addCylinder(
                    main_inlet[0],
                    main_inlet[1],
                    main_inlet[2],
                    -main_inlet[0],
                    -main_inlet[1],
                    -main_inlet[2],
                    float(radius),
                ),
            ),
            (
                3,
                occ.addCylinder(
                    0.0,
                    0.0,
                    0.0,
                    branch1_outlet[0],
                    branch1_outlet[1],
                    branch1_outlet[2],
                    float(radius),
                ),
            ),
            (
                3,
                occ.addCylinder(
                    0.0,
                    0.0,
                    0.0,
                    branch2_outlet[0],
                    branch2_outlet[1],
                    branch2_outlet[2],
                    float(radius),
                ),
            ),
        ]
        fused, _ = occ.fuse([cylinders[0]], cylinders[1:])
        return self._single_dimtag(fused, 3, "tee fuse")

    def _build_rounded_tee_volume(self, gmsh_module, radius, fillet_radius, params):
        occ = gmsh_module.model.occ
        sharp_volume = self._build_sharp_tee_volume(gmsh_module, radius, params)
        occ.synchronize()
        curve_tags = self._collect_junction_fillet_curve_tags(gmsh_module, sharp_volume)
        filleted = occ.fillet(
            [sharp_volume[1]],
            curve_tags,
            [float(fillet_radius)],
            removeVolume=True,
        )
        return self._single_dimtag(filleted, 3, "tee fillet")

    def build_fsi_volumes(self, gmsh_module, params):
        occ = gmsh_module.model.occ
        try:
            fluid = self._build_rounded_tee_volume(
                gmsh_module, params["inner_radius"], params["R_fillet"], params
            )
            outer = self._build_rounded_tee_volume(
                gmsh_module, params["outer_radius"], params["R_fillet"], params
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to fillet tee junction with gmsh OCC; "
                "try adjusting intersect_angle or R_fillet"
            ) from exc

        solid_candidates = occ.cut(
            [outer],
            [fluid],
            removeObject=True,
            removeTool=False,
        )[0]
        solid = self._pick_largest_volume(gmsh_module, solid_candidates, "tee solid cut")
        fragment_result = occ.fragment([fluid], [solid])
        fluid_out = self._pick_largest_volume(gmsh_module, fragment_result[1][0], "tee fluid fragment")
        solid_out = self._pick_largest_volume(gmsh_module, fragment_result[1][1], "tee solid fragment")

        keep_tags = {fluid_out[1], solid_out[1]}
        all_fragment_volumes = [
            dimtag
            for group in fragment_result[1]
            for dimtag in group
            if dimtag[0] == 3
        ]
        extra_volumes = [dimtag for dimtag in all_fragment_volumes if dimtag[1] not in keep_tags]
        if extra_volumes:
            occ.remove(extra_volumes, recursive=True)

        return {"fluid": fluid_out, "solid": solid_out}

    def _surface_center(self, gmsh_module, surface_dimtag):
        return gmsh_module.model.occ.getCenterOfMass(*surface_dimtag)

    def _surface_matches_point(self, gmsh_module, surface_dimtag, point, tol):
        center = self._surface_center(gmsh_module, surface_dimtag)
        bbox = gmsh_module.model.occ.getBoundingBox(*surface_dimtag)
        bbox_center = (
            0.5 * (bbox[0] + bbox[3]),
            0.5 * (bbox[1] + bbox[4]),
            0.5 * (bbox[2] + bbox[5]),
        )
        dist_center = math.sqrt(
            (center[0] - point[0]) ** 2
            + (center[1] - point[1]) ** 2
            + (center[2] - point[2]) ** 2
        )
        dist_bbox = math.sqrt(
            (bbox_center[0] - point[0]) ** 2
            + (bbox_center[1] - point[1]) ** 2
            + (bbox_center[2] - point[2]) ** 2
        )
        return min(dist_center, dist_bbox) <= tol

    def _surface_boundary_count(self, gmsh_module, surface_dimtag):
        return len(gmsh_module.model.getBoundary([surface_dimtag], oriented=False, recursive=False))

    def _pick_smallest_surface(self, gmsh_module, candidates):
        if not candidates:
            return []
        best = min(candidates, key=lambda s: gmsh_module.model.occ.getMass(*s))
        return [best]

    def classify_boundaries(self, gmsh_module, volumes, params):
        main_inlet, branch1_outlet, branch2_outlet = self._port_centers(params)
        tol = max(
            float(params["main_inlet_length"]),
            float(params["branch1_length"]),
            float(params["branch2_length"]),
        ) * 1e-5

        fluid_surfaces = gmsh_module.model.getBoundary([volumes["fluid"]], oriented=False, recursive=False)
        solid_surfaces = gmsh_module.model.getBoundary([volumes["solid"]], oriented=False, recursive=False)
        solid_surface_tags = {s[1] for s in solid_surfaces}
        shared_surfaces = [s for s in fluid_surfaces if s[1] in solid_surface_tags]
        interface_surfaces = [s for s in shared_surfaces if self._surface_boundary_count(gmsh_module, s) > 1]
        interface_tags = {s[1] for s in interface_surfaces}

        main_candidates: List[DimTag] = []
        branch1_candidates: List[DimTag] = []
        branch2_candidates: List[DimTag] = []
        for surface in fluid_surfaces:
            if surface[1] in interface_tags:
                continue
            if self._surface_boundary_count(gmsh_module, surface) != 1:
                continue
            if self._surface_matches_point(gmsh_module, surface, main_inlet, tol):
                main_candidates.append(surface)
            elif self._surface_matches_point(gmsh_module, surface, branch1_outlet, tol):
                branch1_candidates.append(surface)
            elif self._surface_matches_point(gmsh_module, surface, branch2_outlet, tol):
                branch2_candidates.append(surface)

        main_inlet_surface = self._pick_smallest_surface(gmsh_module, main_candidates)
        branch1_outlet_surface = self._pick_smallest_surface(gmsh_module, branch1_candidates)
        branch2_outlet_surface = self._pick_smallest_surface(gmsh_module, branch2_candidates)

        solid_main_candidates: List[DimTag] = []
        solid_branch1_candidates: List[DimTag] = []
        solid_branch2_candidates: List[DimTag] = []
        for surface in solid_surfaces:
            if surface[1] in interface_tags:
                continue
            if self._surface_boundary_count(gmsh_module, surface) < 2:
                continue
            if self._surface_matches_point(gmsh_module, surface, main_inlet, tol):
                solid_main_candidates.append(surface)
            elif self._surface_matches_point(gmsh_module, surface, branch1_outlet, tol):
                solid_branch1_candidates.append(surface)
            elif self._surface_matches_point(gmsh_module, surface, branch2_outlet, tol):
                solid_branch2_candidates.append(surface)

        solid_main_inlet_end = self._pick_smallest_surface(gmsh_module, solid_main_candidates)
        solid_branch1_end = self._pick_smallest_surface(gmsh_module, solid_branch1_candidates)
        solid_branch2_end = self._pick_smallest_surface(gmsh_module, solid_branch2_candidates)

        excluded_surface_tags = interface_tags | {
            surface[1]
            for surface in (
                main_inlet_surface
                + branch1_outlet_surface
                + branch2_outlet_surface
                + solid_main_inlet_end
                + solid_branch1_end
                + solid_branch2_end
            )
        }
        outer_wall = [surface for surface in solid_surfaces if surface[1] not in excluded_surface_tags]

        groups = [
            (3, volumes["fluid"][1], "fluid"),
            (3, volumes["solid"][1], "solid"),
            (2, [surface[1] for surface in main_inlet_surface], "main_inlet"),
            (2, [surface[1] for surface in branch1_outlet_surface], "branch1_outlet"),
            (2, [surface[1] for surface in branch2_outlet_surface], "branch2_outlet"),
            (2, [surface[1] for surface in interface_surfaces], "fsi_interface"),
            (2, [surface[1] for surface in outer_wall], "outer_wall"),
            (2, [surface[1] for surface in solid_main_inlet_end], "solid_main_inlet_end"),
            (2, [surface[1] for surface in solid_branch1_end], "solid_branch1_end"),
            (2, [surface[1] for surface in solid_branch2_end], "solid_branch2_end"),
        ]

        physical_groups = []
        boundary_dimtags: Dict[str, List[DimTag]] = {}
        for dim, tags, name in groups:
            if isinstance(tags, int):
                tags = [tags]
            tags = [int(tag) for tag in tags]
            if not tags:
                raise RuntimeError(f"expected at least one surface for {name}")
            boundary_dimtags[name] = [(dim, tag) for tag in tags]
            ptag = len(physical_groups) + 1
            gmsh_module.model.addPhysicalGroup(dim, tags, ptag)
            gmsh_module.model.setPhysicalName(dim, ptag, name)
            physical_groups.append((dim, ptag, name))

        for name in (
            "main_inlet",
            "branch1_outlet",
            "branch2_outlet",
            "solid_main_inlet_end",
            "solid_branch1_end",
            "solid_branch2_end",
        ):
            tags = [tag for _, tag in boundary_dimtags[name]]
            if len(tags) != 1:
                raise ValueError(f"{name} must contain exactly one surface")

        for name in ("fsi_interface", "outer_wall"):
            tags = [tag for _, tag in boundary_dimtags[name]]
            if len(tags) == 0:
                raise ValueError(f"{name} must contain at least one surface")

        unique_surface_tags = []
        for name in (
            "main_inlet",
            "branch1_outlet",
            "branch2_outlet",
            "fsi_interface",
            "outer_wall",
            "solid_main_inlet_end",
            "solid_branch1_end",
            "solid_branch2_end",
        ):
            unique_surface_tags.extend([tag for _, tag in boundary_dimtags[name]])
        if len(unique_surface_tags) != len(set(unique_surface_tags)):
            raise ValueError("boundary surface tags must be disjoint across groups")

        physical_name_to_dimtag, physical_dimtag_to_name = build_dimtag_maps(physical_groups)
        return {
            "physical_groups": physical_groups,
            "physical_name_to_dimtag": physical_name_to_dimtag,
            "physical_dimtag_to_name": physical_dimtag_to_name,
            "volume_dimtags": {"fluid": volumes["fluid"], "solid": volumes["solid"]},
            "boundary_dimtags": boundary_dimtags,
        }

    def set_mesh_fields(self, gmsh_module, params, boundary_info):
        gmsh_module.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh_module.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh_module.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh_module.option.setNumber("Mesh.MeshSizeMax", float(params["mesh_size_global"]))
        gmsh_module.option.setNumber(
            "Mesh.MeshSizeMin",
            min(float(params["mesh_size_junction"]), float(params["mesh_size_interface"])) * 0.6,
        )

        interface_surfaces = [surface[1] for surface in boundary_info["boundary_dimtags"]["fsi_interface"]]
        distance_field = gmsh_module.model.mesh.field.add("Distance")
        gmsh_module.model.mesh.field.setNumbers(distance_field, "FacesList", interface_surfaces)

        interface_field = gmsh_module.model.mesh.field.add("Threshold")
        gmsh_module.model.mesh.field.setNumber(interface_field, "InField", distance_field)
        gmsh_module.model.mesh.field.setNumber(interface_field, "SizeMin", float(params["mesh_size_interface"]))
        gmsh_module.model.mesh.field.setNumber(interface_field, "SizeMax", float(params["mesh_size_global"]))
        gmsh_module.model.mesh.field.setNumber(interface_field, "DistMin", 0.0)
        gmsh_module.model.mesh.field.setNumber(
            interface_field, "DistMax", 2.0 * float(params["inner_radius"])
        )

        junction_span = 2.5 * float(params["outer_radius"])
        junction_field = gmsh_module.model.mesh.field.add("Box")
        gmsh_module.model.mesh.field.setNumber(junction_field, "XMin", -junction_span)
        gmsh_module.model.mesh.field.setNumber(junction_field, "XMax", junction_span)
        gmsh_module.model.mesh.field.setNumber(junction_field, "YMin", -junction_span)
        gmsh_module.model.mesh.field.setNumber(junction_field, "YMax", junction_span)
        gmsh_module.model.mesh.field.setNumber(junction_field, "ZMin", -junction_span)
        gmsh_module.model.mesh.field.setNumber(junction_field, "ZMax", junction_span)
        gmsh_module.model.mesh.field.setNumber(junction_field, "VIn", float(params["mesh_size_junction"]))
        gmsh_module.model.mesh.field.setNumber(junction_field, "VOut", float(params["mesh_size_global"]))

        background_field = gmsh_module.model.mesh.field.add("Min")
        gmsh_module.model.mesh.field.setNumbers(
            background_field, "FieldsList", [interface_field, junction_field]
        )
        gmsh_module.model.mesh.field.setAsBackgroundMesh(background_field)
