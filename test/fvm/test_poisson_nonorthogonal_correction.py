import logging

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesh import (
    HexahedronMesh,
    QuadrangleMesh,
    TetrahedronMesh,
    TriangleMesh,
)
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.topology.builder import TopologyBuilder
from fealpy.mesh.view import Mesh

from fealpy.fvm import (
    FVMLinearSolver,
    PoissonFVMModel,
    PoissonSolverControls,
)


def _bad_two_triangle_mesh():
    bm.set_backend("numpy")
    node = bm.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [-0.01, -1.0],
            [0.01, 1.0],
        ],
        dtype=bm.float64,
    )
    cell = bm.array([[0, 1, 2], [0, 3, 1]], dtype=bm.int32)
    return TriangleMesh(node, cell)


class AffinePDE:
    init_mesh = {
        "skew": lambda nx, ny: _bad_two_triangle_mesh(),
    }

    @staticmethod
    def solution(points):
        return 1.0 + 2.0 * points[..., 0] - 3.0 * points[..., 1]

    dirichlet = solution

    @staticmethod
    def source(points):
        return bm.zeros(points.shape[:-1], dtype=points.dtype)


class MeshBoundAffinePDE:
    """Affine harmonic field bound to one mesh factory for model-level tests."""

    def __init__(self, mesh):
        self.dimension = mesh.geo_dimension()
        self.init_mesh = {"test": lambda **kwargs: mesh}

    @cartesian
    def solution(self, points):
        coefficients = bm.array([2.0, -3.0, 0.5][: self.dimension])
        return 1.0 + bm.einsum("...d,d->...", points, coefficients)

    dirichlet = solution

    @cartesian
    def source(self, points):
        return bm.zeros(points.shape[:-1], dtype=points.dtype)


def _mixed_tri_quad_mesh():
    node = np.array(
        [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [1.0, 1.0], [2.0, 0.0], [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
    quadrilaterals = np.array([[1, 4, 5, 3]], dtype=np.int32)

    block = MeshBlock(positions=node)
    block.add_sector(EntitySector("tri", triangles), root=True)
    block.add_sector(EntitySector("quad", quadrilaterals), root=True)
    TopologyBuilder.construct(block)
    return Mesh(block).fealpy_api()


@pytest.mark.parametrize(
    "mesh_factory",
    [
        lambda: TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2),
        lambda: QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2),
        _mixed_tri_quad_mesh,
        lambda: TetrahedronMesh.from_box(
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], nx=2, ny=2, nz=2
        ),
        lambda: HexahedronMesh.from_box(
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], nx=2, ny=2, nz=2
        ),
    ],
    ids=["triangle", "quadrangle", "mixed_tri_quad", "tetrahedron", "hexahedron"],
)
def test_poisson_affine_solution_is_exact_across_mesh_families(mesh_factory):
    bm.set_backend("numpy")
    mesh = mesh_factory()
    pde = MeshBoundAffinePDE(mesh)
    model = PoissonFVMModel(
        {
            "pde": pde,
            "mesh_type": "test",
                "linear_solver": FVMLinearSolver("scipy"),
            "nonorthogonal_max_iter": 100,
            "nonorthogonal_rtol": 1.0e-11,
            "nonorthogonal_atol": 1.0e-12,
        }
    )

    solution = model.solve()
    exact = pde.solution(model.fvm_geometry.cell_center)

    np.testing.assert_allclose(
        bm.to_numpy(solution),
        bm.to_numpy(exact),
        rtol=1.0e-9,
        atol=1.0e-10,
    )
    assert model.nonorthogonal_diagnostics["converged"] is True


def _poisson_model(diffusion_method="over_relaxed"):
    from fealpy.fvm import PoissonFVMModel

    return PoissonFVMModel(
        {
            "pde": AffinePDE(),
            "nx": 1,
            "ny": 1,
            "space_degree": 0,
            "mesh_type": "skew",
            "diffusion_method": diffusion_method,
        }
    )


@pytest.mark.parametrize("method", ["over_relaxed", "bounded_over_relaxed"])
def test_poisson_affine_cross_diffusion_uses_boundary_faces(method):
    model = _poisson_model(method)
    geometry = model.fvm_geometry
    gradient = np.array([2.0, -3.0])
    values = AffinePDE.solution(geometry.cell_center)

    actual = np.asarray(model.compute_cross_diffusion(values))
    if method == "over_relaxed":
        T_f = geometry.diffusion_face_decomposition("over_relaxed").T_f
    else:
        T_f = geometry.diffusion_face_decomposition(
            "bounded_over_relaxed"
        ).T_f
    face_flux = np.einsum("fd,d->f", np.asarray(T_f), gradient)
    expected = np.asarray(geometry.scatter_face_flux_to_cells(face_flux))

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


class IdentityLinearSolver:
    def solve(self, matrix, rhs):
        from fealpy.fvm import LinearSolveDiagnostics, LinearSolveResult

        return LinearSolveResult(
            solution=np.asarray(rhs),
            diagnostics=LinearSolveDiagnostics(
                provider="test",
                solver="identity",
                iterations=None,
                converged=True,
                provider_code=None,
                relative_residual=0.0,
            ),
        )


def test_poisson_nonorthogonal_solve_stops_on_full_residual():
    model = object.__new__(PoissonFVMModel)
    model.logger = logging.getLogger("poisson-test")
    model.assemble_base_system = lambda: (np.eye(1), np.array([1.0]))
    model.compute_cross_diffusion = lambda u: 0.25 * np.asarray(u)
    model.controls = PoissonSolverControls(
        nonorthogonal_rtol=1.0e-12,
        nonorthogonal_atol=1.0e-14,
        nonorthogonal_max_iter=100,
    )
    model.linear_solver = IdentityLinearSolver()

    solution = model.solve()

    np.testing.assert_allclose(solution, np.array([4.0 / 3.0]), rtol=1.0e-10)
    np.testing.assert_allclose(model.solution, solution)
    assert model.nonorthogonal_diagnostics["converged"] is True
    assert model.nonorthogonal_diagnostics["iterations"] > 1


def test_poisson_nonorthogonal_solve_raises_at_safety_limit():
    model = object.__new__(PoissonFVMModel)
    model.logger = logging.getLogger("poisson-test")
    model.assemble_base_system = lambda: (np.eye(1), np.array([1.0]))
    model.compute_cross_diffusion = lambda u: 2.0 * np.asarray(u)
    model.controls = PoissonSolverControls(
        nonorthogonal_rtol=1.0e-12,
        nonorthogonal_atol=1.0e-14,
        nonorthogonal_max_iter=4,
    )
    model.linear_solver = IdentityLinearSolver()

    with pytest.raises(
        RuntimeError,
        match="non-orthogonal correction did not converge",
    ):
        model.solve()
