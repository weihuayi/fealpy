import math
import sys
from dataclasses import dataclass, field

import gmsh

@dataclass
class WingModelConfig:
    """Parametric shell model for the UAV wing described in the reference."""

    model_name: str = "uav_wing_naca2412_shell"
    output_basename: str = "uav_wing_naca2412_shell"

    # Units are millimeters. Axes: x chord, y span, z airfoil thickness.
    span: float = 1200.0
    sweep_deg: float = 12.4
    sweep_reference: str = "leading_edge"
    sweep_start_y: float = 400.0

    # Piecewise-linear chord distribution: (spanwise y, chord length).
    chord_stations: tuple = (
        (0.0, 200.0),
        (400.0, 150.0),
        (1200.0, 50.0),
    )

    # Rib stations. The first one is root and the last one is tip.
    rib_y: tuple = (
        0.0,
        200.0,
        400.0,
        600.0,
        800.0,
        975.0,
        1200.0,
    )

    # Spar positions in chord fraction x / c.
    spar_xc: dict = field(
        default_factory=lambda: {
            "front_spar": 0.20,
            "rear_spar": 0.75,
        }
    )

    # NACA 2412 by default.
    airfoil_m: float = 0.02
    airfoil_p: float = 0.40
    airfoil_t: float = 0.12
    n_profile: int = 81
    profile_spacing: str = "uniform"

    # Thickness is not modeled geometrically. It is kept for FE/CalculiX setup.
    shell_thickness: dict = field(
        default_factory=lambda: {
            "skin": 2.0,
            "root_rib": 15.0,
            "ribs": 5.0,
            "front_spar": 5.0,
            "rear_spar": 3.0,
        }
    )

    mesh_size: float = 15.0
    recombine: bool = True
    structured_span_mesh: bool = True
    gmsh_terminal: bool = False
    gmsh_verbosity: int = 2


class WingShellMesher:
    def __init__(self, config=None):
        self.cfg = config if config is not None else WingModelConfig()
        self.occ = None

        self.x_profile = []
        self.z_profile = []
        self.n_upper = 0

        self.point_tags = []
        self.point_coords = {}
        self.line_cache = {}
        self.line_points = {}

        self.skin_surfaces = []
        self.rib_surfaces = []
        self.spar_surfaces = {}
        self.spar_lines_on_ribs = {}
        self.physical_groups = {}

        self.geometry_built = False
        self.mesh_generated = False

    @staticmethod
    def from_naca4(code, **kwargs):
        if len(code) != 4 or not code.isdigit():
            raise ValueError("NACA code must be a 4-digit string, e.g. '2412'.")

        kwargs["airfoil_m"] = int(code[0]) / 100.0
        kwargs["airfoil_p"] = int(code[1]) / 10.0
        kwargs["airfoil_t"] = int(code[2:]) / 100.0

        return WingModelConfig(**kwargs)

    def validate_config(self):
        cfg = self.cfg

        if cfg.mesh_size <= 0.0:
            raise ValueError("mesh_size must be positive.")

        if cfg.n_profile < 11:
            raise ValueError("n_profile is too small. Use at least 11.")

        if abs(cfg.rib_y[0]) > 1.0e-9:
            raise ValueError("The first rib station must be y = 0.0.")

        if abs(cfg.rib_y[-1] - cfg.span) > 1.0e-9:
            raise ValueError("The last rib station must equal span.")

        if cfg.sweep_start_y < -1.0e-9 or cfg.sweep_start_y > cfg.span + 1.0e-9:
            raise ValueError("sweep_start_y must lie within the wing span.")

        valid_refs = {"leading_edge", "quarter_chord", "trailing_edge"}
        if cfg.sweep_reference not in valid_refs:
            raise ValueError(
                "sweep_reference must be one of: "
                "leading_edge, quarter_chord, trailing_edge."
            )

        valid_profile_spacing = {"uniform", "cosine"}
        if cfg.profile_spacing not in valid_profile_spacing:
            raise ValueError(
                "profile_spacing must be one of: uniform, cosine."
            )

        for name, xc in cfg.spar_xc.items():
            if not 0.0 < xc < 1.0:
                raise ValueError(f"{name} x/c must be between 0 and 1.")

    def initialize_gmsh(self):
        try:
            if gmsh.isInitialized():
                gmsh.clear()
            else:
                gmsh.initialize()
        except AttributeError:
            gmsh.initialize()

        gmsh.model.add(self.cfg.model_name)
        gmsh.option.setNumber("General.Terminal", int(self.cfg.gmsh_terminal))
        gmsh.option.setNumber("General.Verbosity", self.cfg.gmsh_verbosity)
        self.occ = gmsh.model.occ

    def chord_length(self, y):
        stations = sorted(self.cfg.chord_stations, key=lambda item: item[0])

        if y < stations[0][0] - 1.0e-9 or y > stations[-1][0] + 1.0e-9:
            raise ValueError(f"y={y} is outside chord station range.")

        for i in range(len(stations) - 1):
            y0, c0 = stations[i]
            y1, c1 = stations[i + 1]

            if y0 - 1.0e-9 <= y <= y1 + 1.0e-9:
                if abs(y1 - y0) < 1.0e-14:
                    return c0

                ratio = (y - y0) / (y1 - y0)
                return c0 + ratio * (c1 - c0)

        return stations[-1][1]

    def leading_edge_x(self, y):
        cfg = self.cfg
        sweep = math.tan(math.radians(cfg.sweep_deg))
        c_root = self.chord_length(0.0)
        c_y = self.chord_length(y)
        swept_y = max(0.0, y - cfg.sweep_start_y)

        if cfg.sweep_reference == "leading_edge":
            return swept_y * sweep

        if cfg.sweep_reference == "quarter_chord":
            x_q_y = 0.25 * c_root + swept_y * sweep
            return x_q_y - 0.25 * c_y

        if cfg.sweep_reference == "trailing_edge":
            x_te_y = c_root + swept_y * sweep
            return x_te_y - c_y

        raise RuntimeError("Invalid sweep_reference.")

    def naca4_closed_profile(self):
        cfg = self.cfg
        extra_x = list(cfg.spar_xc.values())

        if cfg.profile_spacing == "cosine":
            beta = [
                math.pi * i / (cfg.n_profile - 1)
                for i in range(cfg.n_profile)
            ]
            x_values = [0.5 * (1.0 - math.cos(b)) for b in beta]
        else:
            x_values = [
                i / (cfg.n_profile - 1)
                for i in range(cfg.n_profile)
            ]

        x_values.extend(extra_x)
        x_values = sorted(set(round(x, 14) for x in x_values))

        upper = []
        lower = []

        for x in x_values:
            yt = 5.0 * cfg.airfoil_t * (
                0.2969 * math.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1015 * x**4
            )

            if abs(cfg.airfoil_m) < 1.0e-14 or abs(cfg.airfoil_p) < 1.0e-14:
                yc = 0.0
                dyc_dx = 0.0
            elif x < cfg.airfoil_p:
                yc = cfg.airfoil_m / cfg.airfoil_p**2 * (
                    2.0 * cfg.airfoil_p * x - x**2
                )
                dyc_dx = 2.0 * cfg.airfoil_m / cfg.airfoil_p**2 * (
                    cfg.airfoil_p - x
                )
            else:
                yc = cfg.airfoil_m / (1.0 - cfg.airfoil_p) ** 2 * (
                    (1.0 - 2.0 * cfg.airfoil_p)
                    + 2.0 * cfg.airfoil_p * x
                    - x**2
                )
                dyc_dx = 2.0 * cfg.airfoil_m / (1.0 - cfg.airfoil_p) ** 2 * (
                    cfg.airfoil_p - x
                )

            theta = math.atan(dyc_dx)

            xu = x - yt * math.sin(theta)
            zu = yc + yt * math.cos(theta)
            xl = x + yt * math.sin(theta)
            zl = yc - yt * math.cos(theta)

            upper.append((xu, zu))
            lower.append((xl, zl))

        self.n_upper = len(upper)

        closed = upper + lower[-1:0:-1]
        self.x_profile = [p[0] for p in closed]
        self.z_profile = [p[1] for p in closed]

    def station_point(self, y, xbar, zbar):
        chord = self.chord_length(y)
        x = self.leading_edge_x(y) + xbar * chord
        z = zbar * chord

        return x, y, z

    def nearest_profile_index(self, x_target):
        upper_indices = range(0, self.n_upper)
        lower_indices = range(self.n_upper, len(self.x_profile))

        iu = min(
            upper_indices,
            key=lambda i: abs(self.x_profile[i] - x_target),
        )
        il = min(
            lower_indices,
            key=lambda i: abs(self.x_profile[i] - x_target),
        )

        return iu, il

    def line_between(self, pa, pb):
        key = (pa, pb)
        rkey = (pb, pa)

        if key in self.line_cache:
            return self.line_cache[key]

        if rkey in self.line_cache:
            return -self.line_cache[rkey]

        line = self.occ.addLine(pa, pb)
        self.line_cache[key] = line
        self.line_points[line] = (pa, pb)

        return line

    def add_filling_surface(self, curves):
        wire = self.occ.addWire(curves, checkClosed=True)
        return self.occ.addSurfaceFilling(wire)

    def add_plane_surface(self, curves):
        wire = self.occ.addWire(curves, checkClosed=True)
        return self.occ.addPlaneSurface([wire])

    def build_geometry(self):
        self.validate_config()
        self.initialize_gmsh()
        self.naca4_closed_profile()

        cfg = self.cfg
        station_y = list(cfg.rib_y)
        n_station = len(station_y)
        n_profile = len(self.x_profile)

        self.point_tags = []
        self.point_coords = {}
        self.line_cache = {}
        self.line_points = {}
        self.skin_surfaces = []
        self.rib_surfaces = []
        self.spar_surfaces = {}
        self.spar_lines_on_ribs = {}

        for y in station_y:
            row = []

            for xbar, zbar in zip(self.x_profile, self.z_profile):
                x, yy, z = self.station_point(y, xbar, zbar)
                point = self.occ.addPoint(x, yy, z, cfg.mesh_size)
                row.append(point)
                self.point_coords[point] = (x, yy, z)

            self.point_tags.append(row)

        for i in range(n_station - 1):
            for j in range(n_profile):
                j2 = (j + 1) % n_profile

                p00 = self.point_tags[i][j]
                p01 = self.point_tags[i][j2]
                p11 = self.point_tags[i + 1][j2]
                p10 = self.point_tags[i + 1][j]

                curves = [
                    self.line_between(p00, p01),
                    self.line_between(p01, p11),
                    self.line_between(p11, p10),
                    self.line_between(p10, p00),
                ]

                self.skin_surfaces.append(self.add_filling_surface(curves))

        for i in range(n_station):
            curves = []

            for j in range(n_profile):
                j2 = (j + 1) % n_profile
                curves.append(
                    self.line_between(
                        self.point_tags[i][j],
                        self.point_tags[i][j2],
                    )
                )

            rib_surface = self.add_plane_surface(curves)
            self.rib_surfaces.append(rib_surface)
            self.spar_lines_on_ribs[rib_surface] = []

        for spar_name, xc in cfg.spar_xc.items():
            iu, il = self.nearest_profile_index(xc)
            surfaces = []

            for i in range(n_station - 1):
                p_u0 = self.point_tags[i][iu]
                p_u1 = self.point_tags[i + 1][iu]
                p_l1 = self.point_tags[i + 1][il]
                p_l0 = self.point_tags[i][il]

                vertical_0 = self.line_between(p_u0, p_l0)
                vertical_1 = self.line_between(p_u1, p_l1)

                curves = [
                    self.line_between(p_u0, p_u1),
                    vertical_1,
                    self.line_between(p_l1, p_l0),
                    -vertical_0,
                ]

                surfaces.append(self.add_filling_surface(curves))
                self.spar_lines_on_ribs[self.rib_surfaces[i]].append(vertical_0)
                self.spar_lines_on_ribs[self.rib_surfaces[i + 1]].append(vertical_1)

            self.spar_surfaces[spar_name] = surfaces

        self.occ.synchronize()

        for rib_surface, lines in self.spar_lines_on_ribs.items():
            line_tags = sorted(set(abs(line) for line in lines))

            if line_tags:
                gmsh.model.mesh.embed(1, line_tags, 2, rib_surface)

        self.create_physical_groups()
        self.geometry_built = True

        return self

    def create_physical_groups(self):
        self.physical_groups = {}

        skin_group = gmsh.model.addPhysicalGroup(2, self.skin_surfaces)
        gmsh.model.setPhysicalName(2, skin_group, "skin")
        self.physical_groups["skin"] = skin_group

        root_group = gmsh.model.addPhysicalGroup(2, [self.rib_surfaces[0]])
        gmsh.model.setPhysicalName(2, root_group, "root_rib")
        self.physical_groups["root_rib"] = root_group

        if len(self.rib_surfaces) > 1:
            rib_group = gmsh.model.addPhysicalGroup(2, self.rib_surfaces[1:])
            gmsh.model.setPhysicalName(2, rib_group, "ribs")
            self.physical_groups["ribs"] = rib_group

        for spar_name, surfaces in self.spar_surfaces.items():
            group = gmsh.model.addPhysicalGroup(2, surfaces)
            gmsh.model.setPhysicalName(2, group, spar_name)
            self.physical_groups[spar_name] = group

    def all_surfaces(self):
        surfaces = []
        surfaces.extend(self.skin_surfaces)
        surfaces.extend(self.rib_surfaces)

        for spar_surfaces in self.spar_surfaces.values():
            surfaces.extend(spar_surfaces)

        return surfaces

    def set_mesh_options(self):
        cfg = self.cfg

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cfg.mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cfg.mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm", 8)

        if cfg.structured_span_mesh:
            for line, (pa, pb) in self.line_points.items():
                p0 = self.point_coords[pa]
                p1 = self.point_coords[pb]
                if abs(p0[1] - p1[1]) > 1.0e-7:
                    length = math.dist(p0, p1)
                    n_points = max(2, int(round(length / cfg.mesh_size)) + 1)
                    gmsh.model.mesh.setTransfiniteCurve(line, n_points)

        if cfg.recombine:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)

            for surface in self.all_surfaces():
                gmsh.model.mesh.setRecombine(2, surface)

    def generate_mesh(self):
        if not self.geometry_built:
            raise RuntimeError("Call build_geometry() before generate_mesh().")

        self.set_mesh_options()
        gmsh.model.mesh.generate(2)
        self.mesh_generated = True

        return self

    def write_msh(self, filename=None):
        if not self.mesh_generated:
            raise RuntimeError("Call generate_mesh() before write_msh().")

        if filename is None:
            filename = self.cfg.output_basename + ".msh"

        gmsh.write(filename)
        return filename

    def collect_surface_elements(self):
        element_type_map = {
            2: ("S3", 3),
            3: ("S4", 4),
        }
        result = {}

        for dim, group_tag in gmsh.model.getPhysicalGroups(2):
            group_name = gmsh.model.getPhysicalName(dim, group_tag)
            blocks_by_type = {}

            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, group_tag):
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(
                    dim,
                    entity,
                )

                for gmsh_type, tags, nodes in zip(
                    elem_types,
                    elem_tags,
                    elem_node_tags,
                ):
                    if gmsh_type not in element_type_map:
                        raise RuntimeError(
                            f"Unsupported element type {gmsh_type} for CalculiX."
                        )

                    calculix_type, nodes_per_elem = element_type_map[gmsh_type]
                    block = blocks_by_type.setdefault(calculix_type, ([], []))
                    block_tags, block_conn = block

                    for i, element_tag in enumerate(tags):
                        start = i * nodes_per_elem
                        end = start + nodes_per_elem
                        conn = [int(n) for n in nodes[start:end]]

                        block_tags.append(int(element_tag))
                        block_conn.append(conn)

            result[group_name] = [
                (calculix_type, tags, conn)
                for calculix_type, (tags, conn) in blocks_by_type.items()
            ]

        return result

    def write_files(self):
        return self.write_msh()

    def view(self):
        gmsh.fltk.run()

    def close(self):
        try:
            gmsh.finalize()
        except Exception:
            pass

    def run(self, view=False, write_files=True, finalize=True):
        try:
            self.build_geometry()
            self.generate_mesh()

            outputs = None
            if write_files:
                outputs = self.write_files()

            if view:
                self.view()

            return outputs
        finally:
            if finalize:
                self.close()


def build_wing(config=None, view=False, write_files=True, finalize=True):
    mesher = WingShellMesher(config)
    mesher.run(view=view, write_files=write_files, finalize=finalize)
    return mesher


if __name__ == "__main__":
    # Smaller mesh_size gives a finer mesh and a larger CalculiX input file.
    cfg = WingModelConfig(
        span=1200.0,
        sweep_deg=12.4,
        sweep_start_y=400.0,
        chord_stations=((0.0, 200.0), (400.0, 150.0), (1200.0, 50.0)),
        rib_y=(0.0, 200.0, 400.0, 600.0, 800.0, 975.0, 1200.0),
        mesh_size=15.0,
        gmsh_terminal=True,
        gmsh_verbosity=4
    )

    show_gui = "-nopopup" not in sys.argv
    build_wing(cfg, view=show_gui)
