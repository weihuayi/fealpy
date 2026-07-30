import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm
from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fvm import (
    CrossDiffusionRHSAssembler,
    FVMGeometry,
    ScalarCrossDiffusionIntegrator,
)
from fealpy.mesh import TriangleMesh


def _box_space(nx=1, ny=1):
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=nx, ny=ny)
    return mesh, ScaledMonomialSpace2d(mesh, 0)


def _bad_two_triangle_space():
    bm.set_backend("numpy")
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
    mesh = TriangleMesh(node, cell)
    return mesh, ScaledMonomialSpace2d(mesh, 0)


def _expected_scalar_scatter(mesh, face_flux):
    geometry = FVMGeometry(mesh)
    face_to_cell = np.asarray(geometry.face_to_cell, dtype=np.int64)
    expected = np.zeros(geometry.NC, dtype=float)
    internal = face_to_cell[:, 0] != face_to_cell[:, 1]
    np.add.at(expected, face_to_cell[:, 0], face_flux)
    np.add.at(expected, face_to_cell[internal, 1], -face_flux[internal])
    return expected


def _expected_vector_scatter(mesh, face_flux):
    geometry = FVMGeometry(mesh)
    face_to_cell = np.asarray(geometry.face_to_cell, dtype=np.int64)
    expected = np.zeros((geometry.NC, face_flux.shape[1]), dtype=float)
    internal = face_to_cell[:, 0] != face_to_cell[:, 1]
    np.add.at(expected, face_to_cell[:, 0], face_flux)
    np.add.at(expected, face_to_cell[internal, 1], -face_flux[internal])
    return expected


def test_face_flux_correction_scatter_scalar():
    mesh, space = _box_space(nx=1, ny=1)
    face_flux = np.linspace(0.2, 1.2, mesh.number_of_edges())

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(face_flux_correction=face_flux)
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs),
        _expected_scalar_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_face_flux_correction_scatter_reuses_fvm_geometry(monkeypatch):
    mesh, space = _box_space(nx=2, ny=1)
    face_flux = np.linspace(0.2, 1.2, mesh.number_of_edges())
    calls = []
    original_scatter = FVMGeometry.scatter_face_flux_to_cells

    def counted_scatter(self, selected_face_flux):
        calls.append(np.asarray(selected_face_flux).copy())
        return original_scatter(self, selected_face_flux)

    monkeypatch.setattr(FVMGeometry, "scatter_face_flux_to_cells", counted_scatter)

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(face_flux_correction=face_flux)
    ).assembly()

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], face_flux, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(
        np.asarray(rhs),
        _expected_scalar_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_face_flux_correction_scatter_vector():
    mesh, scalar_space = _box_space(nx=1, ny=1)
    space = TensorFunctionSpace(scalar_space, shape=(2, -1))
    face_flux = np.stack(
        [
            np.linspace(0.2, 1.2, mesh.number_of_edges()),
            np.linspace(-0.4, 0.6, mesh.number_of_edges()),
        ],
        axis=1,
    )

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(face_flux_correction=face_flux)
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs).reshape(2, mesh.number_of_cells()).T,
        _expected_vector_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_default_cross_diffusion_matches_over_relaxed_scatter():
    mesh, space = _box_space(nx=2, ny=1)
    grad_f = np.stack(
        [
            np.linspace(-0.3, 0.5, mesh.number_of_edges()),
            np.linspace(0.1, 0.7, mesh.number_of_edges()),
        ],
        axis=1,
    )
    coef = np.linspace(0.8, 1.4, mesh.number_of_edges())
    correction = np.asarray(
        FVMGeometry(mesh).diffusion_face_decomposition("over_relaxed").T_f
    )
    face_flux = coef * np.einsum("ij,ij->i", correction, grad_f)

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(np.zeros(mesh.number_of_cells()), grad_f, coef=coef)
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs),
        _expected_scalar_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_cross_diffusion_rhs_assembler_matches_linear_form_scalar():
    mesh, space = _box_space(nx=2, ny=1)
    geometry = FVMGeometry(mesh)
    grad_f = np.stack(
        [
            np.linspace(-0.3, 0.5, mesh.number_of_edges()),
            np.linspace(0.1, 0.7, mesh.number_of_edges()),
        ],
        axis=1,
    )
    coef = np.linspace(0.8, 1.4, mesh.number_of_edges())

    reference = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            coef=coef,
            geometry=geometry,
        )
    ).assembly()
    fast = CrossDiffusionRHSAssembler(space, geometry=geometry).assembly(
        uh=np.zeros(mesh.number_of_cells()),
        grad_f=grad_f,
        coef=coef,
    )

    np.testing.assert_allclose(np.asarray(fast), np.asarray(reference), rtol=1.0e-13, atol=1.0e-13)


def test_cross_diffusion_rhs_assembler_matches_linear_form_vector_boundary_all():
    mesh, scalar_space = _box_space(nx=2, ny=1)
    space = TensorFunctionSpace(scalar_space, shape=(2, -1))
    geometry = FVMGeometry(mesh)
    grad_f = np.stack(
        [
            np.stack([
                np.linspace(-0.3, 0.5, mesh.number_of_edges()),
                np.linspace(0.1, 0.7, mesh.number_of_edges()),
            ], axis=1),
            np.stack([
                np.linspace(0.4, 0.9, mesh.number_of_edges()),
                np.linspace(-0.2, 0.3, mesh.number_of_edges()),
            ], axis=1),
        ],
        axis=1,
    )
    coef = np.linspace(0.8, 1.4, mesh.number_of_edges())

    reference = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros((mesh.number_of_cells(), 2)),
            grad_f,
            coef=coef,
            geometry=geometry,
            boundary_policy="all",
        )
    ).assembly()
    fast = CrossDiffusionRHSAssembler(space, geometry=geometry).assembly(
        uh=np.zeros((mesh.number_of_cells(), 2)),
        grad_f=grad_f,
        coef=coef,
        boundary_policy="all",
    )

    np.testing.assert_allclose(np.asarray(fast), np.asarray(reference), rtol=1.0e-13, atol=1.0e-13)


def test_cross_diffusion_rhs_assembler_matches_linear_form_limited():
    mesh, space = _bad_two_triangle_space()
    geometry = FVMGeometry(mesh)
    face_to_cell = np.asarray(geometry.face_to_cell)
    is_internal = face_to_cell[:, 0] != face_to_cell[:, 1]
    uh = np.array([0.0, 1.0])
    grad_f = np.zeros((mesh.number_of_edges(), mesh.geo_dimension()))
    grad_f[is_internal, 0] = 100.0

    reference = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            uh,
            grad_f,
            geometry=geometry,
            method="bounded_over_relaxed",
            cross_flux_limiter="orthogonal_flux_ratio",
            limit_coeff=0.5,
        )
    ).assembly()
    fast = CrossDiffusionRHSAssembler(
        space,
        geometry=geometry,
        method="bounded_over_relaxed",
        cross_flux_limiter="orthogonal_flux_ratio",
    ).assembly(
        uh=uh,
        grad_f=grad_f,
        limit_coeff=0.5,
    )

    np.testing.assert_allclose(np.asarray(fast), np.asarray(reference), rtol=1.0e-13, atol=1.0e-13)


def test_explicit_bounded_cross_diffusion_reuses_fvm_geometry_decomposition(monkeypatch):
    mesh, space = _box_space(nx=2, ny=1)
    grad_f = np.stack(
        [
            np.linspace(-0.3, 0.5, mesh.number_of_edges()),
            np.linspace(0.1, 0.7, mesh.number_of_edges()),
        ],
        axis=1,
    )
    calls = []
    original_decomposition = FVMGeometry.diffusion_face_decomposition

    def counted_decomposition(self, method="over_relaxed", *, eps=0.05):
        calls.append((method, eps))
        return original_decomposition(self, method, eps=eps)

    monkeypatch.setattr(
        FVMGeometry,
        "diffusion_face_decomposition",
        counted_decomposition,
    )

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            method="bounded_over_relaxed",
            boundary_policy="zero",
        )
    ).assembly()

    T_f = original_decomposition(
        FVMGeometry(mesh), "bounded_over_relaxed", eps=0.05
    ).T_f
    face_flux = np.einsum("ij,ij->i", np.asarray(T_f), grad_f)
    face_flux[np.asarray(FVMGeometry(mesh).is_boundary)] = 0.0

    assert calls == [("bounded_over_relaxed", 0.05)]
    np.testing.assert_allclose(
        np.asarray(rhs),
        _expected_scalar_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_bounded_over_relaxed_Tf_is_stabilized_on_bad_internal_face():
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
    mesh = TriangleMesh(node, cell)
    geometry = FVMGeometry(mesh)
    eps = 0.05

    face_to_cell = np.asarray(geometry.face_to_cell)
    is_internal = face_to_cell[:, 0] != face_to_cell[:, 1]
    delta = np.asarray(geometry.d_f)
    normal = np.asarray(geometry.n_f)
    face_area = np.asarray(geometry.mag_S_f)
    ratio = np.einsum("ij,ij->i", normal, delta) / np.linalg.norm(delta, axis=1)
    T_f = np.asarray(
        geometry.diffusion_face_decomposition(
            "bounded_over_relaxed", eps=eps
        ).T_f
    )
    stabilized_ratio = np.linalg.norm(T_f, axis=1) / face_area

    assert np.any(ratio[is_internal] <= eps)
    assert np.max(stabilized_ratio[is_internal]) <= 1.0 + 1.0 / eps


def test_uncorrected_method_returns_zero_rhs():
    mesh, space = _box_space(nx=2, ny=1)
    grad_f = np.ones((mesh.number_of_edges(), mesh.geo_dimension()))

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            method="uncorrected",
        )
    ).assembly()

    np.testing.assert_allclose(np.asarray(rhs), 0.0, atol=0.0)


def test_limited_correction_with_zero_limit_coeff_returns_zero_internal_rhs():
    mesh, space = _bad_two_triangle_space()
    geometry = FVMGeometry(mesh)
    face_to_cell = np.asarray(geometry.face_to_cell)
    is_internal = face_to_cell[:, 0] != face_to_cell[:, 1]
    grad_f = np.zeros((mesh.number_of_edges(), mesh.geo_dimension()))
    grad_f[is_internal, 0] = 1.0

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.array([0.0, 1.0]),
            grad_f,
            geometry=geometry,
            method="bounded_over_relaxed",
            cross_flux_limiter="orthogonal_flux_ratio",
            limit_coeff=0.0,
        )
    ).assembly()

    np.testing.assert_allclose(np.asarray(rhs), 0.0, atol=1.0e-13)


def test_limited_correction_matches_manual_limiter_on_bad_internal_face():
    mesh, space = _bad_two_triangle_space()
    geometry = FVMGeometry(mesh)
    face_to_cell = np.asarray(geometry.face_to_cell)
    is_internal = face_to_cell[:, 0] != face_to_cell[:, 1]
    uh = np.array([0.0, 1.0])
    grad_f = np.zeros((mesh.number_of_edges(), mesh.geo_dimension()))
    grad_f[is_internal, 0] = 100.0
    decomposition = geometry.diffusion_face_decomposition(
        "bounded_over_relaxed", eps=0.05
    )
    correction_vector = np.asarray(decomposition.T_f)
    full_flux = np.einsum("ij,ij->i", correction_vector, grad_f)
    orthogonal_coeff = np.asarray(decomposition.orthogonal_factor)
    owner = face_to_cell[:, 0]
    neighbour = face_to_cell[:, 1]
    orthogonal_flux = np.zeros(mesh.number_of_edges())
    orthogonal_flux[is_internal] = (
        orthogonal_coeff[is_internal]
        * np.abs(uh[neighbour[is_internal]] - uh[owner[is_internal]])
    )
    limit_coeff = 0.5
    limiter = np.ones(mesh.number_of_edges())
    limiter[is_internal] = np.minimum(
        limit_coeff * orthogonal_flux[is_internal]
        / ((1.0 - limit_coeff) * np.abs(full_flux[is_internal]) + 1.0e-30),
        1.0,
    )
    expected = _expected_scalar_scatter(mesh, full_flux * limiter)

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            uh,
            grad_f,
            geometry=geometry,
            method="bounded_over_relaxed",
            cross_flux_limiter="orthogonal_flux_ratio",
            limit_coeff=limit_coeff,
        )
    ).assembly()

    assert np.max(np.abs(full_flux[is_internal] * limiter[is_internal])) < np.max(
        np.abs(full_flux[is_internal])
    )
    np.testing.assert_allclose(np.asarray(rhs), expected, rtol=1.0e-13, atol=1.0e-13)


def test_correction_vector_matches_equivalent_face_flux_correction():
    mesh, space = _box_space(nx=2, ny=1)
    correction_vector = np.stack(
        [
            np.linspace(0.1, 0.5, mesh.number_of_edges()),
            np.linspace(-0.2, 0.4, mesh.number_of_edges()),
        ],
        axis=1,
    )
    grad_f = np.stack(
        [
            np.linspace(-0.3, 0.7, mesh.number_of_edges()),
            np.linspace(0.4, 0.9, mesh.number_of_edges()),
        ],
        axis=1,
    )
    face_flux = np.einsum("ij,ij->i", correction_vector, grad_f)

    from_vector = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            correction_vector=correction_vector,
        )
    ).assembly()
    from_flux = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(face_flux_correction=face_flux)
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(from_vector),
        np.asarray(from_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


@pytest.mark.parametrize("method", ["over_relaxed", "bounded_over_relaxed"])
def test_cross_diffusion_method_selects_matching_Tf(method):
    mesh, space = _bad_two_triangle_space()
    geometry = FVMGeometry(mesh)
    grad_f = np.ones((mesh.number_of_faces(), mesh.geo_dimension()))
    T_f = np.asarray(geometry.diffusion_face_decomposition(method).T_f)
    expected_flux = np.einsum("fd,fd->f", T_f, grad_f)

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            grad_f=grad_f,
            geometry=geometry,
            method=method,
            boundary_policy="all",
        )
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs),
        _expected_scalar_scatter(mesh, expected_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_cross_diffusion_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown diffusion method"):
        ScalarCrossDiffusionIntegrator(method="misspelled")


def test_bounded_over_relaxed_boundary_policy_zero_masks_boundary_flux():
    mesh, space = _box_space(nx=1, ny=1)
    geometry = FVMGeometry(mesh)
    face_to_cell = np.asarray(geometry.face_to_cell)
    is_internal = face_to_cell[:, 0] != face_to_cell[:, 1]
    grad_f = np.stack(
        [
            np.linspace(0.2, 1.0, mesh.number_of_edges()),
            np.linspace(-0.4, 0.8, mesh.number_of_edges()),
        ],
        axis=1,
    )
    correction_vector = np.asarray(
        geometry.diffusion_face_decomposition("bounded_over_relaxed").T_f
    )
    full_flux = np.einsum("ij,ij->i", correction_vector, grad_f)
    expected_flux = full_flux.copy()
    expected_flux[~is_internal] = 0.0
    expected = _expected_scalar_scatter(mesh, expected_flux)

    assert np.max(np.abs(full_flux[~is_internal])) > 0.0

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            geometry=geometry,
            method="bounded_over_relaxed",
            boundary_policy="zero",
        )
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )

def test_bounded_over_relaxed_boundary_policy_all_keeps_raw_boundary_flux():
    mesh, space = _box_space(nx=1, ny=1)
    geometry = FVMGeometry(mesh)
    grad_f = np.stack(
        [
            np.linspace(0.2, 1.0, mesh.number_of_edges()),
            np.linspace(-0.4, 0.8, mesh.number_of_edges()),
        ],
        axis=1,
    )
    face_flux = np.einsum(
        "ij,ij->i",
        np.asarray(
            geometry.diffusion_face_decomposition("bounded_over_relaxed").T_f
        ),
        grad_f,
    )
    expected = _expected_scalar_scatter(mesh, face_flux)

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            geometry=geometry,
            method="bounded_over_relaxed",
            boundary_policy="all",
        )
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
