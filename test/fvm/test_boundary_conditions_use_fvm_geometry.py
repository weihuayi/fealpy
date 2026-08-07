import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, TriangleMesh
from fealpy.sparse import spdiags


class ShiftedBoundaryGeometry:
    def __init__(self, mesh):
        from fealpy.fvm.fvm_geometry import FVMGeometry

        real = FVMGeometry(mesh)
        self._real = real
        is_boundary = np.asarray(real.is_boundary)
        face_center = np.asarray(real.face_center).copy()
        face_center[is_boundary, 0] += 5.0
        face_center[is_boundary, 1] += 7.0

        S_f = np.asarray(real.S_f).copy()
        S_f[is_boundary] *= 3.0

        self.owner = real.owner
        self.neighbour = real.neighbour
        self.face_to_cell = real.face_to_cell
        self.is_internal = real.is_internal
        self.is_boundary = real.is_boundary
        self.d_f = real.d_f
        self.face_center = bm.array(face_center)
        self.S_f = bm.array(S_f)
        self.mag_S_f = bm.linalg.norm(self.S_f, axis=1)
        self.mag_d_f = real.mag_d_f
        self._mag_E_f = 2.0 * real.mag_d_f

    def __getattr__(self, name):
        return getattr(self._real, name)

    def diffusion_face_decomposition(self, method="over_relaxed", *, eps=0.05):
        from fealpy.fvm.fvm_geometry import DiffusionFaceDecomposition

        return DiffusionFaceDecomposition(
            E_f=self.S_f,
            mag_E_f=self._mag_E_f,
            T_f=bm.zeros_like(self.S_f),
            orthogonal_factor=self._mag_E_f / self.mag_d_f,
        )


def _mesh():
    bm.set_backend("numpy")
    return QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)


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


def _boundary_faces(geometry):
    return np.flatnonzero(np.asarray(geometry.is_boundary))


def _dirichlet(geometry, value, **kwargs):
    from fealpy.fvm import DirichletBC

    faces = _boundary_faces(geometry)
    values = value(geometry.face_center[faces])
    return DirichletBC(geometry, faces, values, **kwargs)


def test_dirichlet_diffusion_uses_selected_decomposition():
    from fealpy.fvm import DirichletBC, FVMGeometry

    mesh = _bad_two_triangle_mesh()
    geometry = FVMGeometry(mesh)
    bc = _dirichlet(
        geometry,
        lambda p: p[:, 0] + 2.0 * p[:, 1],
        diffusion_method="bounded_over_relaxed",
        nonorthogonal_eps=0.05,
    )
    _, coefficient = bc.diffusion_boundary_data()
    boundary = np.asarray(geometry.is_boundary)
    expected = np.asarray(
        geometry.diffusion_face_decomposition(
            "bounded_over_relaxed", eps=0.05
        ).orthogonal_factor
    )[boundary]
    np.testing.assert_allclose(np.asarray(coefficient), expected)


def test_dirichlet_diffusion_rejects_unknown_method():
    from fealpy.fvm import FVMGeometry

    mesh = _bad_two_triangle_mesh()
    geometry = FVMGeometry(mesh)
    with pytest.raises(ValueError, match="unknown diffusion method"):
        _dirichlet(
            geometry,
            lambda p: p[:, 0],
            diffusion_method="misspelled",
        )


@pytest.mark.parametrize("method", ["over_relaxed", "bounded_over_relaxed"])
def test_affine_dirichlet_full_flux_is_exact_on_skew_boundary(method):
    from fealpy.fvm import FVMGeometry

    mesh = _bad_two_triangle_mesh()
    geometry = FVMGeometry(mesh)
    boundary = np.asarray(geometry.is_boundary)
    owner = np.asarray(geometry.owner)[boundary]
    cell_center = np.asarray(geometry.cell_center)
    face_center = np.asarray(geometry.face_center)[boundary]
    gradient = np.array([2.0, -3.0])
    cell_value = 1.0 + cell_center @ gradient
    boundary_value = 1.0 + face_center @ gradient

    decomposition = geometry.diffusion_face_decomposition(method, eps=0.05)

    numerical_flux = (
        np.asarray(decomposition.orthogonal_factor)[boundary]
        * (boundary_value - cell_value[owner])
        + np.einsum(
            "fd,d->f", np.asarray(decomposition.T_f)[boundary], gradient
        )
    )
    exact_flux = np.einsum(
        "fd,d->f", np.asarray(geometry.S_f)[boundary], gradient
    )
    np.testing.assert_allclose(numerical_flux, exact_flux, atol=1.0e-12)


def test_neumann_diffusion_uses_explicit_boundary_face_data():
    from fealpy.fvm import NeumannBC

    mesh = _mesh()
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    gd = lambda points: points[:, 0] - 0.5 * points[:, 1]
    face_values = gd(geometry.face_center[boundary_faces])

    actual = NeumannBC(
        geometry,
        boundary_faces,
        face_values,
    ).apply_diffusion(bm.zeros(mesh.number_of_cells()))

    expected = np.zeros(mesh.number_of_cells())
    np.add.at(
        expected,
        np.asarray(geometry.owner)[boundary_faces],
        np.asarray(gd(geometry.face_center[boundary_faces]))
        * np.asarray(geometry.mag_S_f)[boundary_faces],
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1.0e-13, atol=1.0e-13)

def test_neumann_diffusion_applies_only_explicit_selected_faces():
    from fealpy.fvm import NeumannBC

    mesh = _mesh()
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    face_centers = geometry.face_center[boundary_faces]
    selected_faces = boundary_faces[np.asarray(face_centers[:, 0] < 0.2)]
    face_values = bm.ones(
        selected_faces.shape[0],
        dtype=geometry.cell_center.dtype,
    )

    actual = NeumannBC(
        geometry,
        selected_faces,
        face_values,
    ).apply_diffusion(
        bm.zeros(mesh.number_of_cells())
    )

    expected = np.zeros(mesh.number_of_cells())
    np.add.at(
        expected,
        np.asarray(geometry.owner)[selected_faces],
        np.asarray(geometry.mag_S_f)[selected_faces],
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1.0e-13, atol=1.0e-13)


def test_dirichlet_diffusion_rejects_boundary_face_wise_coef():
    from fealpy.fvm import DirichletBC, FVMGeometry

    mesh = _mesh()
    geometry = FVMGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    A = spdiags(
        bm.zeros(mesh.number_of_cells()),
        0,
        mesh.number_of_cells(),
        mesh.number_of_cells(),
    )

    with pytest.raises(ValueError, match="scalar or face-wise"):
        _dirichlet(geometry, lambda p: p[:, 0]).apply_diffusion(
            A,
            bm.zeros(mesh.number_of_cells()),
            coef=bm.ones(boundary_faces.shape[0]),
            components=1,
        )


def test_dirichlet_convection_uses_fvm_geometry_for_boundary_flux(monkeypatch):
    import fealpy.fvm.dirichlet_bc as dirichlet_module
    from fealpy.fvm import DirichletBC

    mesh = _mesh()
    monkeypatch.setattr(dirichlet_module, "FVMGeometry", ShiftedBoundaryGeometry)
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    coef = bm.ones((mesh.number_of_faces(), 2))

    def gd(points):
        return bm.stack([points[:, 0] - 2.0, 0.5 * points[:, 1]], axis=1)

    actual = _dirichlet(geometry, gd).apply_convection(
        bm.zeros(2 * mesh.number_of_cells()),
        coef,
        components=2,
    )

    expected = np.zeros(2 * mesh.number_of_cells())
    owner = np.asarray(geometry.owner)[boundary_faces]
    value = np.asarray(gd(geometry.face_center[boundary_faces]))
    flux = np.einsum(
        "ij,ij->i",
        np.asarray(coef)[boundary_faces],
        np.asarray(geometry.S_f)[boundary_faces],
    )
    np.add.at(expected, owner, -flux * value[:, 0])
    np.add.at(expected, owner + mesh.number_of_cells(), -flux * value[:, 1])
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1.0e-13, atol=1.0e-13)


def test_dirichlet_convection_rejects_boundary_face_wise_coef():
    from fealpy.fvm import DirichletBC, FVMGeometry

    mesh = _mesh()
    geometry = FVMGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)

    with pytest.raises(ValueError, match="face-wise"):
        _dirichlet(geometry, lambda p: p[:, 0]).apply_convection(
            bm.zeros(mesh.number_of_cells()),
            bm.ones(boundary_faces.shape[0]),
            components=1,
        )


def test_pde_boundary_conditions_use_fvm_geometry_for_boundary_velocity():
    from fealpy.fvm import PDEBoundaryConditions

    mesh = _mesh()
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    shifted_bc = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda p: bm.ones_like(p),
        dirichlet_velocity_selector=lambda p: p[..., 0] > 4.0,
        geometry=ShiftedBoundaryGeometry(mesh),
    )

    selected_faces = shifted_bc.velocity.dirichlet_faces
    points = shifted_bc.geometry.face_center[selected_faces]
    selected_values = shifted_bc.velocity.dirichlet_value(
        points
    )

    np.testing.assert_array_equal(np.asarray(selected_faces), boundary_faces)
    np.testing.assert_allclose(
        np.asarray(selected_values),
        np.ones((boundary_faces.shape[0], mesh.geo_dimension())),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
