import numpy as np

from fealpy.backend import backend_manager as bm


def _right_outlet(points):
    return bm.abs(points[..., 0] - 1.0) < 1.0e-12


def _explicit_traction(mesh, value):
    from fealpy.fvm import FVMGeometry, TractionBC

    geometry = FVMGeometry(mesh)
    boundary_faces = bm.nonzero(geometry.is_boundary)[0]
    faces = boundary_faces[
        _right_outlet(geometry.face_center[boundary_faces])
    ]
    values = value(geometry.face_center[faces])
    source = bm.index_add(
        bm.zeros(
            (geometry.NC, geometry.GD),
            dtype=geometry.cell_center.dtype,
        ),
        geometry.owner[faces],
        geometry.mag_S_f[faces, None] * values,
        axis=0,
    )
    return TractionBC(
        geometry=geometry,
        faces=faces,
        face_average_values=values,
        cell_source=source,
    )


def test_traction_bc_is_public_fvm_boundary_operator():
    bm.set_backend("numpy")
    from fealpy import fvm

    assert fvm.TractionBC.__name__ == "TractionBC"


def test_traction_bc_consumes_explicit_patch_and_face_average():
    bm.set_backend("numpy")
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    bc = _explicit_traction(
        mesh,
        lambda points: bm.stack(
            [points[..., 1], -points[..., 1]],
            axis=-1,
        ),
    )

    centers = mesh.entity_barycenter("face")[bc.faces]

    np.testing.assert_allclose(bm.to_numpy(centers[:, 0]), 1.0)
    np.testing.assert_allclose(
        bm.to_numpy(bc.face_average_values),
        [[0.25, -0.25], [0.75, -0.75]],
        atol=1.0e-14,
    )


def test_traction_bc_cell_source_is_owner_oriented_face_integral():
    bm.set_backend("numpy")
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    bc = _explicit_traction(
        mesh,
        lambda points: bm.broadcast_to(
            bm.array([2.0, -3.0], dtype=points.dtype),
            points.shape,
        ),
    )
    geometry = bc.geometry
    faces = bc.faces
    values = bc.face_average_values
    expected = bm.zeros(
        (geometry.NC, geometry.face_center.shape[1]),
        dtype=geometry.face_center.dtype,
    )
    expected = bm.index_add(
        expected,
        geometry.owner[faces],
        geometry.mag_S_f[faces, None] * values,
        axis=0,
    )

    actual = bc.source(expected)

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected),
        atol=1.0e-14,
    )


def test_traction_bc_removes_reconstructed_pressure_face_force():
    bm.set_backend("numpy")
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    bc = _explicit_traction(
        mesh,
        lambda points: bm.zeros(points.shape, dtype=points.dtype),
    )
    geometry = bc.geometry
    faces = bc.faces
    owner = geometry.owner[faces]
    pressure_value = 2.5
    pressure = pressure_value * bm.ones(
        geometry.NC,
        dtype=geometry.face_center.dtype,
    )
    pressure_gradient = bm.zeros(
        (geometry.NC, geometry.face_center.shape[1]),
        dtype=geometry.face_center.dtype,
    )
    cell_force = bm.zeros_like(pressure_gradient)
    expected = bm.index_add(
        cell_force,
        owner,
        pressure_value * geometry.S_f[faces],
        axis=0,
        alpha=-1,
    )

    actual = bc.apply_pressure_force(
        cell_force,
        pressure,
        pressure_gradient,
    )

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected),
        atol=1.0e-14,
    )


def test_traction_bc_returns_deferred_face_convection_source():
    bm.set_backend("numpy")
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    bc = _explicit_traction(
        mesh,
        lambda points: bm.zeros(points.shape, dtype=points.dtype),
    )
    geometry = bc.geometry
    faces = bc.faces
    owner = geometry.owner[faces]
    cell_velocity = bm.copy(geometry.cell_center)
    reconstructed_face_velocity = bm.zeros_like(geometry.face_center)
    delta = bm.broadcast_to(
        bm.array([0.4, -0.2], dtype=geometry.face_center.dtype),
        (faces.shape[0], geometry.face_center.shape[1]),
    )
    reconstructed_face_velocity = bm.set_at(
        reconstructed_face_velocity,
        faces,
        cell_velocity[owner] + delta,
    )
    convection_face_velocity = bm.zeros_like(geometry.face_center)
    convection_face_velocity = bm.set_at(
        convection_face_velocity,
        faces,
        bm.broadcast_to(
            bm.array([1.5, 0.0], dtype=geometry.face_center.dtype),
            delta.shape,
        ),
    )
    flux = bm.einsum(
        "ij,ij->i",
        convection_face_velocity[faces],
        geometry.S_f[faces],
    )
    expected = bm.index_add(
        bm.zeros_like(cell_velocity),
        owner,
        -flux[:, None] * delta,
        axis=0,
    )

    actual = bc.convection_source(
        convection_face_velocity,
        cell_velocity,
        reconstructed_face_velocity,
    )

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected),
        atol=1.0e-14,
    )


def test_traction_bc_only_masks_selected_cross_diffusion_flux():
    bm.set_backend("numpy")
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    bc = _explicit_traction(
        mesh,
        lambda points: bm.zeros(points.shape, dtype=points.dtype),
    )
    geometry = bc.geometry
    faces = bc.faces
    face_flux = bm.reshape(
        bm.arange(
            geometry.NF * geometry.face_center.shape[1],
            dtype=geometry.face_center.dtype,
        )
        + 1.0,
        geometry.face_center.shape,
    )
    expected = bm.set_at(
        bm.copy(face_flux),
        faces,
        bm.zeros_like(face_flux[faces]),
    )

    actual = bc.apply_cross_diffusion_flux(face_flux)

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected),
        atol=1.0e-14,
    )
