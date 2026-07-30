import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm
from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.mesh import TriangleMesh


def _box_space(nx=2, ny=1):
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=nx, ny=ny)
    return mesh, ScaledMonomialSpace2d(mesh, 0)


def _bad_two_triangle_space():
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
    mesh = TriangleMesh(node, cell)
    return mesh, ScaledMonomialSpace2d(mesh, 0)


def test_scalar_diffusion_local_matrix_matches_orthogonal_flux_formula():
    from fealpy.fvm import FVMGeometry
    from fealpy.fvm.scalar_diffusion_integrator import scalar_diffusion_local_matrix

    mesh, space = _box_space(nx=2, ny=1)
    geometry = FVMGeometry(mesh)
    coef = np.linspace(0.8, 1.4, mesh.number_of_faces())
    orthogonal_factor = geometry.diffusion_face_decomposition(
        "over_relaxed"
    ).orthogonal_factor

    local = scalar_diffusion_local_matrix(
        space,
        space.number_of_local_dofs(),
        coef=coef,
        orthogonal_factor=orthogonal_factor,
    )

    expected_coef = np.asarray(orthogonal_factor) * coef
    expected = expected_coef[:, None, None] * np.array(
        [[1.0, -1.0], [-1.0, 1.0]]
    )
    np.testing.assert_allclose(np.asarray(local), expected, rtol=1.0e-13, atol=1.0e-13)


def test_scalar_diffusion_rejects_cell_wise_coefficient():
    from fealpy.fvm import FVMGeometry
    from fealpy.fvm.scalar_diffusion_integrator import scalar_diffusion_local_matrix

    mesh, space = _box_space(nx=2, ny=1)
    geometry = FVMGeometry(mesh)
    with pytest.raises(ValueError, match="face-wise"):
        scalar_diffusion_local_matrix(
            space,
            space.number_of_local_dofs(),
            coef=np.ones(mesh.number_of_cells()),
            orthogonal_factor=geometry.diffusion_face_decomposition(
                "over_relaxed"
            ).orthogonal_factor,
        )


def test_scalar_diffusion_integrator_expands_face_stencil_for_tensor_space():
    from fealpy.fvm import ScalarDiffusionIntegrator

    _, scalar_space = _box_space(nx=1, ny=1)
    vector_space = TensorFunctionSpace(scalar_space, shape=(2, -1))

    matrix = BilinearForm(vector_space).add_integrator(
        ScalarDiffusionIntegrator()
    ).assembly()

    assert matrix.shape == (2 * scalar_space.number_of_global_dofs(),) * 2


def test_scalar_diffusion_variants_use_selected_Ef():
    from fealpy.fem import BilinearForm
    from fealpy.fvm import FVMGeometry, ScalarDiffusionIntegrator

    mesh, space = _bad_two_triangle_space()
    geometry = FVMGeometry(mesh)

    ordinary = BilinearForm(space).add_integrator(
        ScalarDiffusionIntegrator(
            geometry=geometry,
            method="over_relaxed",
        )
    ).assembly()
    bounded = BilinearForm(space).add_integrator(
        ScalarDiffusionIntegrator(
            geometry=geometry,
            method="bounded_over_relaxed",
            nonorthogonal_eps=0.05,
        )
    ).assembly()

    assert not np.allclose(
        np.asarray(ordinary.to_dense()),
        np.asarray(bounded.to_dense()),
    )


def test_scalar_diffusion_rejects_unknown_method():
    from fealpy.fvm import ScalarDiffusionIntegrator

    with pytest.raises(ValueError, match="unknown diffusion method"):
        ScalarDiffusionIntegrator(method="misspelled")
