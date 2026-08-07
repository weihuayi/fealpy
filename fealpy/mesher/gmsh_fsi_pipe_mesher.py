from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TetrahedronMesh, TriangleMesh

try:
    import gmsh
except ImportError:
    raise ImportError(
        "The gmsh package is required for FSI pipe remeshers. "
        "Please install it via 'pip install gmsh'."
    )


DimTag = Tuple[int, int]


def build_dimtag_maps(groups: Iterable[Tuple[int, int, str]]):
    name_to_dimtag: Dict[str, DimTag] = {}
    dimtag_to_name: Dict[DimTag, str] = {}
    for dim, tag, name in groups:
        dimtag = (int(dim), int(tag))
        if name in name_to_dimtag:
            raise ValueError(f"duplicate physical group name: {name}")
        if dimtag in dimtag_to_name:
            raise ValueError(f"duplicate physical group dimtag: {dimtag}")
        name_to_dimtag[name] = dimtag
        dimtag_to_name[dimtag] = name
    return name_to_dimtag, dimtag_to_name


def _reshape_triples(values: List[float]):
    return [[float(values[i]), float(values[i + 1]), float(values[i + 2])] for i in range(0, len(values), 3)]


def _reshape_tetra(values: List[int]):
    return [[int(values[i]), int(values[i + 1]), int(values[i + 2]), int(values[i + 3])] for i in range(0, len(values), 4)]


def _reshape_tri(values: List[int]):
    return [[int(values[i]), int(values[i + 1]), int(values[i + 2])] for i in range(0, len(values), 3)]


def _reshape_pairs(values: List[int]):
    return [[int(values[i]), int(values[i + 1])] for i in range(0, len(values), 2)]


def extract_tetra_data(gmsh_module):
    node_tags, node_coords, _ = gmsh_module.model.mesh.getNodes()
    tetra_elem_tags, tetra_connectivity = gmsh_module.model.mesh.getElementsByType(4)

    node_tags_list = [int(v) for v in node_tags]
    node_coords_list = _reshape_triples([float(v) for v in node_coords])
    tetra_elem_tags_list = [int(v) for v in tetra_elem_tags]
    tetra_connectivity_list = _reshape_tetra([int(v) for v in tetra_connectivity])

    used_node_tags = {tag for tet in tetra_connectivity_list for tag in tet}
    filtered_tags = []
    filtered_coords = []
    for tag, coord in zip(node_tags_list, node_coords_list):
        if tag in used_node_tags:
            filtered_tags.append(tag)
            filtered_coords.append(coord)

    node_index_map = {tag: idx for idx, tag in enumerate(filtered_tags)}
    tetra = [[node_index_map[tag] for tag in tet] for tet in tetra_connectivity_list]

    tetra_region = [-1] * len(tetra_elem_tags_list)
    tet_tag_to_index = {tag: idx for idx, tag in enumerate(tetra_elem_tags_list)}

    for dim, phys_tag in gmsh_module.model.getPhysicalGroups(3):
        for entity_tag in gmsh_module.model.getEntitiesForPhysicalGroup(dim, phys_tag):
            entity_types, entity_elem_tags, _ = gmsh_module.model.mesh.getElements(3, entity_tag)
            for etype, elem_tags in zip(entity_types, entity_elem_tags):
                if int(etype) != 4:
                    continue
                for elem_tag in elem_tags:
                    tetra_region[tet_tag_to_index[int(elem_tag)]] = int(phys_tag)

    if any(tag < 0 for tag in tetra_region):
        raise ValueError("tetrahedra are missing physical region assignments")

    return {
        "node": bm.array(filtered_coords, dtype=bm.float64),
        "tetra": bm.array(tetra, dtype=bm.int64),
        "tetra_region": bm.array(tetra_region, dtype=bm.int64),
        "node_index_map": node_index_map,
    }


def extract_boundary_triangles(gmsh_module, node_index_map: Dict[int, int]):
    triangles: List[List[int]] = []
    markers: List[int] = []

    for dim, phys_tag in gmsh_module.model.getPhysicalGroups(2):
        for entity_tag in gmsh_module.model.getEntitiesForPhysicalGroup(dim, phys_tag):
            entity_types, _, entity_node_tags = gmsh_module.model.mesh.getElements(2, entity_tag)
            for etype, node_tags in zip(entity_types, entity_node_tags):
                if int(etype) != 2:
                    continue
                for tri in _reshape_tri([int(v) for v in node_tags]):
                    triangles.append([node_index_map[int(tag)] for tag in tri])
                    markers.append(int(phys_tag))

    if triangles:
        tri_array = bm.array(triangles, dtype=bm.int64)
        marker_array = bm.array(markers, dtype=bm.int64)
    else:
        tri_array = bm.zeros((0, 3), dtype=bm.int64)
        marker_array = bm.zeros((0,), dtype=bm.int64)

    return {
        "boundary_tri": tri_array,
        "boundary_tri_marker": marker_array,
    }


def extract_interface_triangles(gmsh_module, node_index_map, fluid_tag, solid_tag):
    interface_tag = None
    for dim, phys_tag in gmsh_module.model.getPhysicalGroups(2):
        if gmsh_module.model.getPhysicalName(dim, phys_tag) == "fsi_interface":
            interface_tag = int(phys_tag)
            break
    if interface_tag is None:
        raise ValueError("missing fsi_interface physical group")

    tetra_data = extract_tetra_data(gmsh_module)
    tetra = bm.to_numpy(tetra_data["tetra"])
    tetra_region = bm.to_numpy(tetra_data["tetra_region"])

    face_templates = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    face_to_tets: Dict[Tuple[int, int, int], List[int]] = {}
    for tet_index, tet_nodes in enumerate(tetra):
        for face in face_templates:
            key = tuple(sorted(int(tet_nodes[i]) for i in face))
            face_to_tets.setdefault(key, []).append(tet_index)

    interface_tri = []
    interface_adjacent_tet = []
    interface_adjacent_region = []

    for entity_tag in gmsh_module.model.getEntitiesForPhysicalGroup(2, interface_tag):
        entity_types, _, entity_node_tags = gmsh_module.model.mesh.getElements(2, entity_tag)
        for etype, node_tags in zip(entity_types, entity_node_tags):
            if int(etype) != 2:
                continue
            for tri in _reshape_tri([int(v) for v in node_tags]):
                tri_nodes = [node_index_map[int(tag)] for tag in tri]
                neighbours = face_to_tets.get(tuple(sorted(tri_nodes)), [])
                if len(neighbours) != 2:
                    raise ValueError("fsi_interface face must have exactly two adjacent tetrahedra")

                region_a = int(tetra_region[neighbours[0]])
                region_b = int(tetra_region[neighbours[1]])
                if {region_a, region_b} != {int(fluid_tag), int(solid_tag)}:
                    raise ValueError("fsi_interface face must touch one fluid tet and one solid tet")

                if region_a == int(fluid_tag):
                    ordered_tets = [neighbours[0], neighbours[1]]
                else:
                    ordered_tets = [neighbours[1], neighbours[0]]

                interface_tri.append(tri_nodes)
                interface_adjacent_tet.append(ordered_tets)
                interface_adjacent_region.append([int(fluid_tag), int(solid_tag)])

    if interface_tri:
        tri_array = bm.array(interface_tri, dtype=bm.int64)
        adj_tet_array = bm.array(interface_adjacent_tet, dtype=bm.int64)
        adj_region_array = bm.array(interface_adjacent_region, dtype=bm.int64)
    else:
        tri_array = bm.zeros((0, 3), dtype=bm.int64)
        adj_tet_array = bm.zeros((0, 2), dtype=bm.int64)
        adj_region_array = bm.zeros((0, 2), dtype=bm.int64)

    return {
        "interface_tri": tri_array,
        "interface_adjacent_tet": adj_tet_array,
        "interface_adjacent_region": adj_region_array,
    }


def extract_triangle_data(gmsh_module):
    node_tags, node_coords, _ = gmsh_module.model.mesh.getNodes()
    tri_elem_tags, tri_connectivity = gmsh_module.model.mesh.getElementsByType(2)

    node_tags_list = [int(v) for v in node_tags]
    node_coords_list = _reshape_triples([float(v) for v in node_coords])
    tri_elem_tags_list = [int(v) for v in tri_elem_tags]
    tri_connectivity_list = _reshape_tri([int(v) for v in tri_connectivity])

    used_node_tags = {tag for tri in tri_connectivity_list for tag in tri}
    filtered_tags = []
    filtered_coords = []
    for tag, coord in zip(node_tags_list, node_coords_list):
        if tag in used_node_tags:
            filtered_tags.append(tag)
            filtered_coords.append(coord[:2])

    node_index_map = {tag: idx for idx, tag in enumerate(filtered_tags)}
    triangle = [[node_index_map[tag] for tag in tri] for tri in tri_connectivity_list]
    triangle = bm.array(triangle, dtype=bm.int64)
    filtered_coords_np = bm.array(filtered_coords, dtype=bm.float64)

    p0 = filtered_coords_np[triangle[:, 0]]
    p1 = filtered_coords_np[triangle[:, 1]]
    p2 = filtered_coords_np[triangle[:, 2]]

    area2 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - \
            (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])

    mask = area2 < 0
    tmp = triangle[mask, 1].copy()
    triangle[mask, 1] = triangle[mask, 2]
    triangle[mask, 2] = tmp

    triangle_region = [-1] * len(tri_elem_tags_list)
    tri_tag_to_index = {tag: idx for idx, tag in enumerate(tri_elem_tags_list)}
    for dim, phys_tag in gmsh_module.model.getPhysicalGroups(2):
        for entity_tag in gmsh_module.model.getEntitiesForPhysicalGroup(dim, phys_tag):
            entity_types, entity_elem_tags, _ = gmsh_module.model.mesh.getElements(2, entity_tag)
            for etype, elem_tags in zip(entity_types, entity_elem_tags):
                if int(etype) != 2:
                    continue
                for elem_tag in elem_tags:
                    triangle_region[tri_tag_to_index[int(elem_tag)]] = int(phys_tag)

    if any(tag < 0 for tag in triangle_region):
        raise ValueError("triangles are missing physical region assignments")

    return {
        "node": filtered_coords_np,
        "triangle": triangle,
        "triangle_region": bm.array(triangle_region, dtype=bm.int64),
        "node_index_map": node_index_map,
    }


def _orient_triangle_cells(node, triangle):
    triangle_np = bm.to_numpy(triangle)
    if triangle_np.size == 0:
        return triangle
    node_np = bm.to_numpy(node)
    p0 = node_np[triangle_np[:, 0], :2]
    p1 = node_np[triangle_np[:, 1], :2]
    p2 = node_np[triangle_np[:, 2], :2]
    signed_area = 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    flip_mask = signed_area < 0.0
    if not bool(np.any(flip_mask)):
        return triangle
    triangle_np = triangle_np.copy()
    triangle_np[flip_mask] = triangle_np[flip_mask][:, [0, 2, 1]]
    return bm.array(triangle_np, dtype=bm.int64)


def extract_boundary_edges(gmsh_module, node_index_map: Dict[int, int]):
    edges: List[List[int]] = []
    markers: List[int] = []

    for dim, phys_tag in gmsh_module.model.getPhysicalGroups(1):
        for entity_tag in gmsh_module.model.getEntitiesForPhysicalGroup(dim, phys_tag):
            entity_types, _, entity_node_tags = gmsh_module.model.mesh.getElements(1, entity_tag)
            for etype, node_tags in zip(entity_types, entity_node_tags):
                if int(etype) != 1:
                    continue
                edge_nodes = [int(v) for v in node_tags]
                for edge in _reshape_pairs(edge_nodes):
                    edges.append([node_index_map[int(tag)] for tag in edge])
                    markers.append(int(phys_tag))

    if edges:
        edge_array = bm.array(edges, dtype=bm.int64)
        marker_array = bm.array(markers, dtype=bm.int64)
    else:
        edge_array = bm.zeros((0, 2), dtype=bm.int64)
        marker_array = bm.zeros((0,), dtype=bm.int64)

    return {
        "boundary_edge": edge_array,
        "boundary_edge_marker": marker_array,
    }


def extract_interface_edges(gmsh_module, node_index_map, fluid_tag, solid_tag, triangle, triangle_region):
    interface_tag = None
    for dim, phys_tag in gmsh_module.model.getPhysicalGroups(1):
        if gmsh_module.model.getPhysicalName(dim, phys_tag) == "fsi_interface":
            interface_tag = int(phys_tag)
            break
    if interface_tag is None:
        raise ValueError("missing fsi_interface physical group")

    edge_to_triangles: Dict[Tuple[int, int], List[int]] = {}
    triangle_np = bm.to_numpy(triangle)
    for tri_index, tri_nodes in enumerate(triangle_np):
        for edge in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted(int(tri_nodes[i]) for i in edge))
            edge_to_triangles.setdefault(key, []).append(tri_index)

    triangle_region_np = bm.to_numpy(triangle_region)
    interface_edge = []
    interface_adjacent_triangle = []
    interface_adjacent_region = []

    for entity_tag in gmsh_module.model.getEntitiesForPhysicalGroup(1, interface_tag):
        entity_types, _, entity_node_tags = gmsh_module.model.mesh.getElements(1, entity_tag)
        for etype, node_tags in zip(entity_types, entity_node_tags):
            if int(etype) != 1:
                continue
            edge_nodes_raw = [int(v) for v in node_tags]
            for edge in _reshape_pairs(edge_nodes_raw):
                edge_nodes = [node_index_map[int(tag)] for tag in edge]
                neighbours = edge_to_triangles.get(tuple(sorted(edge_nodes)), [])
                if len(neighbours) != 2:
                    raise ValueError("fsi_interface edge must have exactly two adjacent triangles")

                region_a = int(triangle_region_np[neighbours[0]])
                region_b = int(triangle_region_np[neighbours[1]])
                if {region_a, region_b} != {int(fluid_tag), int(solid_tag)}:
                    raise ValueError("fsi_interface edge must touch one fluid triangle and one solid triangle")

                if region_a == int(fluid_tag):
                    ordered_triangles = [neighbours[0], neighbours[1]]
                else:
                    ordered_triangles = [neighbours[1], neighbours[0]]

                interface_edge.append(edge_nodes)
                interface_adjacent_triangle.append(ordered_triangles)
                interface_adjacent_region.append([int(fluid_tag), int(solid_tag)])

    if interface_edge:
        edge_array = bm.array(interface_edge, dtype=bm.int64)
        adj_triangle_array = bm.array(interface_adjacent_triangle, dtype=bm.int64)
        adj_region_array = bm.array(interface_adjacent_region, dtype=bm.int64)
    else:
        edge_array = bm.zeros((0, 2), dtype=bm.int64)
        adj_triangle_array = bm.zeros((0, 2), dtype=bm.int64)
        adj_region_array = bm.zeros((0, 2), dtype=bm.int64)

    return {
        "interface_edge": edge_array,
        "interface_adjacent_triangle": adj_triangle_array,
        "interface_adjacent_region": adj_region_array,
    }


class BaseGmshFSIPipeMesher:
    interface_region_names = ("fluid", "solid")

    def __init__(self, params, gmsh_module=None):
        self.params = dict(params)
        self.gmsh = gmsh if gmsh_module is None else gmsh_module
        self._mesh_data_cache = None

    def model_name(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_remesher_parameters(cls, params=None, gmsh_module=None):
        return cls(params=params, gmsh_module=gmsh_module)

    def geo_dimension(self) -> int:
        return self.mesh_topology_dimension()

    def mesh_topology_dimension(self) -> int:
        return 3

    def cell_mesh_class(self):
        return TriangleMesh if self.mesh_topology_dimension() == 2 else TetrahedronMesh

    def cell_data_key(self) -> str:
        return "triangle" if self.mesh_topology_dimension() == 2 else "tetra"

    def cell_region_key(self) -> str:
        return "triangle_region" if self.mesh_topology_dimension() == 2 else "tetra_region"

    def extract_cell_data(self, gmsh_module):
        if self.mesh_topology_dimension() == 2:
            return extract_triangle_data(gmsh_module)
        return extract_tetra_data(gmsh_module)

    def extract_boundary_data(self, gmsh_module, node_index_map):
        if self.mesh_topology_dimension() == 2:
            return extract_boundary_edges(gmsh_module, node_index_map)
        return extract_boundary_triangles(gmsh_module, node_index_map)

    def extract_interface_data(self, gmsh_module, node_index_map, fluid_tag, solid_tag, cell_data):
        if self.mesh_topology_dimension() == 2:
            return extract_interface_edges(
                gmsh_module,
                node_index_map,
                fluid_tag,
                solid_tag,
                cell_data["triangle"],
                cell_data["triangle_region"],
            )
        return extract_interface_triangles(gmsh_module, node_index_map, fluid_tag, solid_tag)

    def build_fsi_volumes(self, gmsh_module, params):
        raise NotImplementedError

    def classify_boundaries(self, gmsh_module, volumes, params):
        raise NotImplementedError

    def set_mesh_fields(self, gmsh_module, params, boundary_info):
        gmsh_module.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh_module.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh_module.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh_module.option.setNumber("Mesh.MeshSizeMax", float(params["mesh_size_global"]))
        gmsh_module.option.setNumber("Mesh.MeshSizeMin", float(params["mesh_size_global"]) * 0.5)

    def generate_mesh(self, gmsh_module):
        gmsh_module.model.mesh.generate(self.mesh_topology_dimension())

    def generate_tetra_mesh(self, gmsh_module):
        self.generate_mesh(gmsh_module)

    def get_interface_region_names(self):
        return self.interface_region_names

    def extract_mesh_data(self, gmsh_module):
        cell_data = self.extract_cell_data(gmsh_module)
        boundary_data = self.extract_boundary_data(gmsh_module, cell_data["node_index_map"])

        physical_groups: Dict[str, DimTag] = {}
        for dim, tag in gmsh_module.model.getPhysicalGroups():
            physical_groups[gmsh_module.model.getPhysicalName(dim, tag)] = (int(dim), int(tag))

        fluid_name, solid_name = self.get_interface_region_names()
        fluid_tag = physical_groups[fluid_name][1]
        solid_tag = physical_groups[solid_name][1]
        interface_data = self.extract_interface_data(
            gmsh_module, cell_data["node_index_map"], fluid_tag, solid_tag, cell_data
        )

        mesh_data = {key: value for key, value in cell_data.items() if key != "node_index_map"}
        mesh_data.update(boundary_data)
        mesh_data.update(interface_data)
        return mesh_data

    def run(self, visualize=False, write_path=None):
        gmsh_module = self.gmsh
        owns_session = not gmsh_module.isInitialized()
        if owns_session:
            gmsh_module.initialize()

        try:
            gmsh_module.model.add(self.model_name())
            volumes = self.build_fsi_volumes(gmsh_module, self.params)
            gmsh_module.model.occ.synchronize()
            boundary_info = self.classify_boundaries(gmsh_module, volumes, self.params)
            self.set_mesh_fields(gmsh_module, self.params, boundary_info)
            self.generate_mesh(gmsh_module)

            mesh = self.extract_mesh_data(gmsh_module)
            mesh["physical_name_to_dimtag"] = boundary_info["physical_name_to_dimtag"]
            mesh["physical_dimtag_to_name"] = boundary_info["physical_dimtag_to_name"]

            if write_path is not None:
                gmsh_module.write(str(Path(write_path)))
            if visualize:
                gmsh_module.fltk.run()
            self._mesh_data_cache = mesh
            return mesh
        finally:
            if owns_session and gmsh_module.isInitialized():
                gmsh_module.finalize()

    @variantmethod("tet")
    def init_mesh(self, visualize=False, write_path=None):
        mesh_data = self.run(visualize=visualize, write_path=write_path)
        mesh_cls = self.cell_mesh_class()
        mesh = mesh_cls(mesh_data["node"], mesh_data[self.cell_data_key()])
        region_key = self.cell_region_key()
        if region_key in mesh_data:
            mesh.celldata["region"] = mesh_data[region_key]
        mesh_data = self.postprocess_mesh_data(mesh, mesh_data)
        self._mesh_data_cache = mesh_data
        return mesh

    def mesh_data(self, visualize=False, write_path=None):
        if self._mesh_data_cache is None:
            return self.run(visualize=visualize, write_path=write_path)
        return self._mesh_data_cache

    def postprocess_mesh_data(self, mesh, mesh_data):
        return mesh_data
