from typing import Mapping, Sequence

import numpy as np

from ...backend import TensorLike
from ...backend import backend_manager as bm
from ...decorator import cartesian, variantmethod
from ...mesh import TriangleMesh
from ...mesher import BoxWithCircularHoleMesher2D


class Exp0013(BoxWithCircularHoleMesher2D):
    """
    2D Poisson problem on a box with a circular hole:

        -Delta u(x, y) = f(x, y),  (x, y) in (-1, 1)^2 \\ B(0, 0.3)
         u(x, y) = g(x, y),        on boundary

    with the exact solution:

        u(x, y) = exp(-(x^2 + y^2)/2)

    The corresponding source term is:

        f(x, y) = (2 - x^2 - y^2) exp(-(x^2 + y^2)/2)

    Dirichlet boundary conditions are applied on both the outer box and the
    inner circular boundary using the exact solution.
    """

    def __init__(self, params: Mapping | None = None):
        defaults = {
            "box": (-1.0, 1.0, -1.0, 1.0),
            "center": (0.0, 0.0),
            "radius": 0.3,
            "h": 0.25,
            "mesh_size_profile": "uniform",
            "mesh_size_inner": None,
            "mesh_size_outer": None,
            "mesh_size_transition": None,
        }
        if params is not None:
            defaults.update(dict(params))
        super().__init__(defaults)

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.params["box"]

    @variantmethod("uniform_tri")
    def init_mesh(self, nx=10, ny=10):
        params = dict(self.params)
        if nx is not None and ny is not None:
            xmin, xmax, ymin, ymax = params["box"]
            params["h"] = min((xmax - xmin) / float(nx), (ymax - ymin) / float(ny))
        return BoxWithCircularHoleMesher2D(params).init_mesh()

    @init_mesh.register("complex_tri")
    def init_mesh(self):
        params = dict(self.params)
        params.update(
            {
                "mesh_size_profile": "graded",
                "mesh_size_inner": params.get("mesh_size_inner") or 0.035,
                "mesh_size_outer": params.get("mesh_size_outer") or max(params["h"], 0.30),
                "mesh_size_transition": params.get("mesh_size_transition") or 0.75,
            }
        )
        mesh = BoxWithCircularHoleMesher2D(params).init_mesh()
        self._distort_interior_nodes(mesh, amplitude=0.10)
        return mesh

    @init_mesh.register("bad_tri")
    def init_mesh(self):
        a = 0.01
        y = 1.0
        node = bm.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [-a, -y],
                [a, y],
            ],
            dtype=bm.float64,
        )
        cell = bm.array([[0, 1, 2], [0, 3, 1]], dtype=bm.int32)
        return TriangleMesh(node, cell)

    @staticmethod
    def _distort_interior_nodes(mesh, amplitude: float = 0.10):
        """Apply a deterministic interior-node warp to expose non-orthogonal corrections."""
        node = np.asarray(mesh.entity("node"), dtype=float).copy()
        is_boundary = np.asarray(mesh.boundary_node_flag(), dtype=bool)
        x = node[:, 0]
        y = node[:, 1]
        xmin, xmax, ymin, ymax = mesh.box
        cx, cy = mesh.obstacle_center
        radius = mesh.obstacle_radius

        box_envelope = (x - xmin) * (xmax - x) * (y - ymin) * (ymax - y)
        box_envelope = np.maximum(box_envelope, 0.0)
        max_envelope = float(np.max(box_envelope))
        if max_envelope > 0.0:
            box_envelope /= max_envelope
        radial_distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        outer_radius = max(0.5 * (xmax - xmin), 0.5 * (ymax - ymin))
        hole_envelope = np.clip((radial_distance - radius) / (outer_radius - radius), 0.0, 1.0)
        envelope = box_envelope * hole_envelope

        displacement = np.zeros_like(node)
        displacement[:, 0] = amplitude * envelope * np.sin(9.0 * x + 4.0 * y) * np.cos(5.0 * y)
        displacement[:, 1] = amplitude * envelope * np.cos(6.0 * x - 7.0 * y) * np.sin(5.0 * x)
        node[~is_boundary] += displacement[~is_boundary]
        mesh.node = bm.array(node, dtype=mesh.ftype)
        return mesh

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.exp(-0.5 * (x**2 + y**2))

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        u = self.solution(p)
        return bm.stack([-x * u, -y * u], axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        u = self.solution(p)
        return (2.0 - x**2 - y**2) * u

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        xmin, xmax, ymin, ymax = self.params["box"]
        cx, cy = self.params["center"]
        radius = self.params["radius"]
        atol = 1.0e-10
        r = bm.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return (
            (bm.abs(x - xmin) < atol)
            | (bm.abs(x - xmax) < atol)
            | (bm.abs(y - ymin) < atol)
            | (bm.abs(y - ymax) < atol)
            | (bm.abs(r - radius) < atol)
        )
