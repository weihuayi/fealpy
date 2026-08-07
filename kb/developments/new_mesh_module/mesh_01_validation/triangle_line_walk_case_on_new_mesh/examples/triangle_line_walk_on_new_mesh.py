from __future__ import annotations

from dataclasses import dataclass

from fealpy.backend import Tensor, bm
from fealpy.mesh import Mesh
from fealpy.mesh.schema import TriangleSchema
from fealpy.mesher.box import Box2d


@dataclass
class TriangleMeshData:
    positions: Tensor
    tri: Tensor
    tri_to_edge: Tensor
    edge_to_node: Tensor
    boundary_edge_mask: Tensor


@dataclass
class LineWalkStep:
    cell: int
    lambdas: list[float]
    walk_edge_index: int | None
    schema_local_edge: int | None
    global_edge: int | None
    next_cell: int | None


@dataclass
class LineWalkResult:
    point: Tensor
    start_cell: int
    located_cell: int | None
    status: str
    lambdas: list[float] | None
    path: list[int]
    steps: list[LineWalkStep]
    message: str


def extract_triangle_mesh_data(mesh: Mesh) -> TriangleMeshData:
    tri_view = mesh.sector("tri")
    edge_view = mesh.sector("edge")

    return TriangleMeshData(
        positions=mesh.block.positions,
        tri=tri_view.indices,
        tri_to_edge=tri_view.to("edge").tgt_indices,
        edge_to_node=edge_view.to("node").tgt_indices,
        boundary_edge_mask=edge_view.boundary().mask,
    )


def build_edge_to_cells(tri_to_edge: Tensor, nedge: int) -> list[list[tuple[int, int]]]:
    edge_to_cells: list[list[tuple[int, int]]] = [[] for _ in range(int(nedge))]

    for cell_index in range(tri_to_edge.shape[0]):
        for local_edge_index in range(tri_to_edge.shape[1]):
            edge_index = int(tri_to_edge[cell_index, local_edge_index])
            edge_to_cells[edge_index].append((cell_index, local_edge_index))

    for edge_index, cells in enumerate(edge_to_cells):
        if len(cells) > 2:
            raise ValueError(f"non-manifold edge {edge_index} has {len(cells)} cells")

    return edge_to_cells


def build_cell_neighbors(
    tri_to_edge: Tensor,
    edge_to_cells: list[list[tuple[int, int]]],
) -> Tensor:
    neighbors = bm.full(tri_to_edge.shape, -1, dtype=bm.int32)

    for cells in edge_to_cells:
        if len(cells) != 2:
            continue

        (cell0, local0), (cell1, local1) = cells
        neighbors[cell0, local0] = cell1
        neighbors[cell1, local1] = cell0

    return bm.asarray(neighbors, dtype=bm.int32)


def orient2d(a: Tensor, b: Tensor, p: Tensor) -> Tensor:
    return (
        (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1])
        - (b[..., 1] - a[..., 1]) * (p[..., 0] - a[..., 0])
    )


def triangle_lambdas(verts: Tensor, point: Tensor) -> list[float]:
    p0 = verts[0]
    p1 = verts[1]
    p2 = verts[2]

    den0 = float(orient2d(p1, p2, p0))
    den1 = float(orient2d(p2, p0, p1))
    den2 = float(orient2d(p0, p1, p2))

    if min(abs(den0), abs(den1), abs(den2)) == 0.0:
        return [float("nan"), float("nan"), float("nan")]

    lam0 = float(orient2d(p1, p2, point)) / den0
    lam1 = float(orient2d(p2, p0, point)) / den1
    lam2 = float(orient2d(p0, p1, point)) / den2
    return [lam0, lam1, lam2]


def build_walk_edge_to_schema_local_edge(
    schema_edges: list[list[int]] | None = None,
) -> list[int]:
    if schema_edges is None:
        schema_edges = TriangleSchema.local_faces["edge"]

    walk_edges = [(1, 2), (2, 0), (0, 1)]
    mapping: list[int] = []
    for edge in walk_edges:
        edge_set = set(edge)
        for local_edge_index, schema_edge in enumerate(schema_edges):
            if set(schema_edge) == edge_set:
                mapping.append(local_edge_index)
                break
        else:
            raise ValueError(f"walk edge {edge} not found in schema edges")
    return mapping


def _classify_lambdas(lambdas: list[float], tol: float) -> str:
    near_zero = sum(abs(value) <= tol for value in lambdas)
    if near_zero >= 2:
        return "on_vertex"
    if near_zero == 1:
        return "on_edge"
    return "inside"


def line_walk_locate(
    mesh: Mesh,
    point: Tensor,
    *,
    start_cell: int = 0,
    tol: float = 1.0e-12,
    max_steps: int | None = None,
) -> LineWalkResult:
    data = extract_triangle_mesh_data(mesh)
    tri = data.tri
    ncell = tri.shape[0]

    if max_steps is None:
        max_steps = ncell + 1

    if start_cell < 0 or start_cell >= ncell:
        return LineWalkResult(
            point=point,
            start_cell=start_cell,
            located_cell=None,
            status="invalid_input",
            lambdas=None,
            path=[],
            steps=[],
            message="start_cell is outside the valid cell range",
        )

    edge_to_cells = build_edge_to_cells(
        data.tri_to_edge,
        nedge=data.edge_to_node.shape[0],
    )
    neighbors = build_cell_neighbors(data.tri_to_edge, edge_to_cells)
    tri_to_edge = data.tri_to_edge
    walk_to_schema_le = build_walk_edge_to_schema_local_edge()

    current = int(start_cell)
    visited: set[int] = set()
    path: list[int] = []
    steps: list[LineWalkStep] = []

    for _ in range(max_steps):
        if current in visited:
            return LineWalkResult(
                point=point,
                start_cell=start_cell,
                located_cell=None,
                status="cycle",
                lambdas=None,
                path=path,
                steps=steps,
                message="cell walk visited the same cell twice",
            )

        visited.add(current)
        path.append(current)

        verts = data.positions[tri[current]]
        lambdas = triangle_lambdas(verts, point)
        if bool(bm.any(bm.isnan(bm.asarray(lambdas, dtype=bm.float64)))):
            return LineWalkResult(
                point=point,
                start_cell=start_cell,
                located_cell=None,
                status="degenerate_cell",
                lambdas=lambdas,
                path=path,
                steps=steps,
                message="encountered a degenerate triangle",
            )

        if min(lambdas) >= -tol:
            status = _classify_lambdas(lambdas, tol)
            return LineWalkResult(
                point=point,
                start_cell=start_cell,
                located_cell=current,
                status=status,
                lambdas=lambdas,
                path=path,
                steps=steps,
                message="point located",
            )

        walk_edge_index = int(bm.argmin(bm.asarray(lambdas, dtype=bm.float64)))
        schema_local_edge = walk_to_schema_le[walk_edge_index]
        global_edge = int(tri_to_edge[current, schema_local_edge])
        next_cell = int(neighbors[current, schema_local_edge])

        steps.append(
            LineWalkStep(
                cell=current,
                lambdas=lambdas,
                walk_edge_index=walk_edge_index,
                schema_local_edge=schema_local_edge,
                global_edge=global_edge,
                next_cell=next_cell,
            )
        )

        if next_cell < 0:
            return LineWalkResult(
                point=point,
                start_cell=start_cell,
                located_cell=None,
                status="outside",
                lambdas=lambdas,
                path=path,
                steps=steps,
                message="point leaves the mesh through a boundary edge",
            )

        current = next_cell

    return LineWalkResult(
        point=point,
        start_cell=start_cell,
        located_cell=None,
        status="max_steps",
        lambdas=None,
        path=path,
        steps=steps,
        message="maximum number of line-walk steps reached",
    )


def brute_force_locate(
    positions: Tensor,
    tri: Tensor,
    point: Tensor,
    tol: float = 1.0e-12,
) -> list[int]:
    cells: list[int] = []
    for cell_index in range(tri.shape[0]):
        verts = positions[tri[cell_index]]
        lambdas = triangle_lambdas(verts, point)
        has_nan = bool(bm.any(bm.isnan(bm.asarray(lambdas, dtype=bm.float64))))
        if not has_nan and min(lambdas) >= -tol:
            cells.append(cell_index)
    return cells


def main() -> None:
    mesh = Box2d(nx=10, ny=10).triangulate()
    data = extract_triangle_mesh_data(mesh)
    points = [
        bm.asarray([0.23, 0.37], dtype=bm.float64),
        bm.asarray([0.83, 0.71], dtype=bm.float64),
        bm.asarray([-0.1, 0.5], dtype=bm.float64),
        bm.asarray([0.0, 0.1111], dtype=bm.float64),
        bm.asarray([0.0, 1.0], dtype=bm.float64),
    ]

    print(f"nodes: {data.positions.shape[0]}")
    print(f"triangles: {data.tri.shape[0]}")
    print(f"edges: {data.edge_to_node.shape[0]}")

    for point in points:
        result = line_walk_locate(mesh, point)
        point_list = [float(point[0]), float(point[1])]
        print(
            f"point={point_list} "
            f"status={result.status} cell={result.located_cell} path={result.path}"
        )


if __name__ == "__main__":
    main()
