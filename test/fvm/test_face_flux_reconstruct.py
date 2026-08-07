import inspect

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fvm import FVMGeometry
from fealpy.mesh import HexahedronMesh, TetrahedronMesh
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.topology.builder import TopologyBuilder
from fealpy.mesh.view import Mesh


def test_face_flux_correction_requires_explicit_boundary_tensors():
    from fealpy.fvm.face_flux_reconstruct import FaceFluxReconstruct

    variants = FaceFluxReconstruct.__dict__["correction"].virtual_table
    for variant in variants.values():
        parameters = inspect.signature(variant).parameters
        assert (
            parameters["boundary_face_average"].default
            is inspect.Parameter.empty
        )
        assert (
            parameters["boundary_faces"].default
            is inspect.Parameter.empty
        )


def _vector_quadratic(points):
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    return bm.stack(
        [
            1.0 + 2.0 * x - y + x * y + z**2,
            -0.5 + x + 3.0 * z + x**2 - 2.0 * y * z,
            2.0 - y + z + y**2 + x * z,
        ],
        axis=-1,
    )


def _empty_boundary_values(geometry):
    return (
        bm.zeros(
            (0, geometry.GD),
            dtype=geometry.cell_center.dtype,
            device=bm.get_device(geometry.cell_center),
        ),
        geometry.boundary_faces[:0],
    )


def _divergence_free_quadratic(points):
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    return bm.stack(
        [x**2 + y**2, z**2 - 2.0 * x * y, x**2],
        axis=-1,
    )


def _entity_average(mesh, entity, function, q=5):
    views = mesh.Entities(entity)
    integral = bm.concatenate(
        [view.integral(function, q=q) for view in views],
        axis=0,
    )
    measure = bm.concatenate(
        [view.measure() for view in views],
        axis=0,
    )
    return integral / measure[:, None]


def _vector_quadratic_2d(points):
    x = points[..., 0]
    y = points[..., 1]
    return bm.stack(
        [
            1.0 + 2.0 * x - y + x * y + y**2,
            -0.5 + x + 3.0 * y + x**2 - 2.0 * x * y,
        ],
        axis=-1,
    )


def _mixed_tri_quad_mesh():
    node = bm.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
    ])
    quads = bm.array([[0, 1, 4, 3], [3, 4, 7, 6]], dtype=bm.int32)
    triangles = bm.array(
        [[1, 2, 5], [1, 5, 4], [4, 5, 8], [4, 8, 7]],
        dtype=bm.int32,
    )
    block = MeshBlock(positions=node)
    block.add_sector(EntitySector("quad", quads), root=True)
    block.add_sector(EntitySector("tri", triangles), root=True)
    TopologyBuilder.construct(block)
    return Mesh(block).fealpy_api()


def _cell_average(geometry, function, q=5):
    integral = geometry.cell_integral(lambda points, _: function(points), q=q)
    return integral / geometry.cell_measure[:, None]


def _face_average(geometry, function, q=5):
    integral = geometry.face_integral(lambda points, _: function(points), q=q)
    return integral / geometry.face_measure[:, None]


def test_cell_anchored_quadratic_face_flux_recovers_quadratic_face_average():
    bm.set_backend("numpy")
    from fealpy.fvm import CellAnchoredQuadraticFaceFluxReconstruct

    mesh = TetrahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
        nz=2,
    )
    geometry = FVMGeometry(mesh)
    cell_average = _entity_average(mesh, "cell", _vector_quadratic)
    exact_face_average = _entity_average(mesh, "face", _vector_quadratic)

    reconstruct = CellAnchoredQuadraticFaceFluxReconstruct(
        geometry=geometry,
    )
    face_average = reconstruct.face_average(
        cell_average,
        boundary_face_average=exact_face_average[geometry.is_boundary],
        boundary_faces=geometry.boundary_faces,
    )
    flux = reconstruct.reconstruct(
        cell_average,
        boundary_face_average=exact_face_average[geometry.is_boundary],
        boundary_faces=geometry.boundary_faces,
    )

    np.testing.assert_allclose(
        bm.to_numpy(face_average),
        bm.to_numpy(exact_face_average),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        bm.to_numpy(flux),
        bm.to_numpy(bm.einsum("fi,fi->f", exact_face_average, geometry.S_f)),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    diagnostics = reconstruct.diagnostics()
    assert diagnostics["minimum_rank"] == 9
    assert diagnostics["condition_limit"] == 100.0
    assert diagnostics["rank_deficient_cell_count"] == 0
    assert diagnostics["ill_conditioned_cell_count"] == 0
    assert diagnostics["fallback_cell_count"] == 0
    assert diagnostics["failed_cell_count"] == 0
    assert diagnostics["fallback_reason_counts"] == {
        "rank_deficient": 0,
        "ill_conditioned": 0,
    }
    assert diagnostics["maximum_accepted_condition"] is not None
    assert diagnostics["maximum_accepted_condition"] <= 100.0
    assert diagnostics["maximum_stencil_layer"] <= 4


@pytest.mark.parametrize("max_condition", [0.0, float("nan"), float("inf")])
def test_face_flux_reconstruct_rejects_invalid_condition_limit(max_condition):
    bm.set_backend("numpy")
    from fealpy.fvm import (
        CellAnchoredQuadraticFaceFluxReconstruct,
        FaceFluxReconstruct,
    )

    mesh = TetrahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=1,
        ny=1,
        nz=1,
    )
    geometry = FVMGeometry(mesh)

    with pytest.raises(ValueError, match="max_condition"):
        CellAnchoredQuadraticFaceFluxReconstruct(
            geometry=geometry,
            max_condition=max_condition,
        )
    with pytest.raises(ValueError, match="max_condition"):
        FaceFluxReconstruct(
            geometry=geometry,
            method="none",
            max_condition=max_condition,
        )


def test_cell_anchored_quadratic_falls_back_when_full_rank_stencil_remains_ill_conditioned():
    bm.set_backend("numpy")
    from fealpy.fvm import FaceFluxReconstruct

    mesh = TetrahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
        nz=2,
    )
    geometry = FVMGeometry(mesh)
    cell_velocity = bm.copy(geometry.cell_center)
    base_face_velocity = 0.5 * (
        cell_velocity[geometry.owner] + cell_velocity[geometry.neighbour]
    )
    reconstruct = FaceFluxReconstruct(
        geometry=geometry,
        method="cell_anchored_quadratic",
        max_condition=1.0,
    )

    empty_values, empty_faces = _empty_boundary_values(geometry)
    correction = reconstruct.correction(
        cell_velocity,
        base_face_velocity,
        empty_values,
        empty_faces,
    )
    diagnostics = reconstruct.diagnostics()

    np.testing.assert_allclose(bm.to_numpy(correction), 0.0)
    assert diagnostics["condition_limit"] == 1.0
    assert diagnostics["rank_deficient_cell_count"] == 0
    assert diagnostics["ill_conditioned_cell_count"] == mesh.number_of_cells()
    assert diagnostics["fallback_cell_count"] == mesh.number_of_cells()
    assert diagnostics["failed_cell_count"] == mesh.number_of_cells()
    assert diagnostics["fallback_reason_counts"] == {
        "rank_deficient": 0,
        "ill_conditioned": mesh.number_of_cells(),
    }
    assert diagnostics["maximum_accepted_condition"] is None
    assert diagnostics["fallback_face_count"] == mesh.number_of_faces()
    assert sum(diagnostics["stencil_layer_counts"].values()) == (
        mesh.number_of_cells()
    )
    assert diagnostics["maximum_stencil_layer"] == 4


def test_cell_anchored_quadratic_expands_full_rank_stencil_until_condition_is_acceptable():
    bm.set_backend("numpy")
    from fealpy.fvm import CellAnchoredQuadraticFaceFluxReconstruct

    mesh = TetrahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
        nz=2,
    )
    geometry = FVMGeometry(mesh)
    cell_average = _entity_average(mesh, "cell", _vector_quadratic)
    exact_face_average = _entity_average(mesh, "face", _vector_quadratic)
    reconstruct = CellAnchoredQuadraticFaceFluxReconstruct(
        geometry=geometry,
        max_condition=40.0,
    )

    face_average = reconstruct.face_average(
        cell_average,
        boundary_face_average=exact_face_average[geometry.is_boundary],
        boundary_faces=geometry.boundary_faces,
    )
    diagnostics = reconstruct.diagnostics()

    np.testing.assert_allclose(
        bm.to_numpy(face_average),
        bm.to_numpy(exact_face_average),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    assert diagnostics["fallback_cell_count"] == 0
    assert diagnostics["maximum_accepted_condition"] <= 40.0
    assert diagnostics["maximum_stencil_layer"] == 4
    assert diagnostics["stencil_layer_counts"].get(4, 0) > 0


def test_face_flux_reconstruct_closes_divergence_free_quadratic_field():
    bm.set_backend("numpy")
    from fealpy.fvm import FaceFluxReconstruct

    mesh = TetrahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
        nz=2,
    )
    geometry = FVMGeometry(mesh)
    cell_average = _entity_average(mesh, "cell", _divergence_free_quadratic)
    exact_face_average = _entity_average(
        mesh,
        "face",
        _divergence_free_quadratic,
    )
    base_face_velocity = 0.5 * (
        cell_average[geometry.owner] + cell_average[geometry.neighbour]
    )
    reconstruct = FaceFluxReconstruct(
        geometry=geometry,
        method="cell_anchored_quadratic",
    )

    correction = reconstruct.correction(
        cell_average,
        base_face_velocity,
        boundary_face_average=exact_face_average[geometry.is_boundary],
        boundary_faces=geometry.boundary_faces,
    )
    base_flux = bm.einsum("fi,fi->f", base_face_velocity, geometry.S_f)
    divergence = geometry.scatter_face_flux_to_cells(base_flux + correction)

    np.testing.assert_allclose(
        bm.to_numpy(divergence),
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert reconstruct.diagnostics()["failed_cell_count"] == 0


def test_face_flux_reconstruct_none_variant_returns_zero_correction():
    bm.set_backend("numpy")
    from fealpy.fvm import FaceFluxReconstruct

    mesh = TetrahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=1,
        ny=1,
        nz=1,
    )
    geometry = FVMGeometry(mesh)
    cell_velocity = bm.zeros(
        (mesh.number_of_cells(), mesh.geo_dimension()),
        dtype=geometry.cell_center.dtype,
    )
    face_velocity = bm.ones(
        (mesh.number_of_faces(), mesh.geo_dimension()),
        dtype=geometry.cell_center.dtype,
    )
    reconstruct = FaceFluxReconstruct(
        geometry=geometry,
        method="none",
    )

    empty_values, empty_faces = _empty_boundary_values(geometry)
    correction = reconstruct.correction(
        cell_velocity,
        face_velocity,
        empty_values,
        empty_faces,
    )

    np.testing.assert_allclose(bm.to_numpy(correction), 0.0)


def test_cell_anchored_quadratic_falls_back_to_zero_defect_when_stencil_is_rank_deficient():
    bm.set_backend("numpy")
    from fealpy.fvm import FaceFluxReconstruct

    mesh = HexahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
        nz=2,
    )
    geometry = FVMGeometry(mesh)
    cell_velocity = bm.copy(geometry.cell_center)
    base_face_velocity = 0.5 * (
        cell_velocity[geometry.owner] + cell_velocity[geometry.neighbour]
    )
    reconstruct = FaceFluxReconstruct(
        geometry=geometry,
        method="cell_anchored_quadratic",
    )

    empty_values, empty_faces = _empty_boundary_values(geometry)
    correction = reconstruct.correction(
        cell_velocity,
        base_face_velocity,
        empty_values,
        empty_faces,
    )
    diagnostics = reconstruct.diagnostics()

    np.testing.assert_allclose(bm.to_numpy(correction), 0.0)
    assert diagnostics["failed_cell_count"] == mesh.number_of_cells()
    assert diagnostics["fallback_face_count"] == mesh.number_of_faces()


def test_cell_anchored_quadratic_recovers_mixed_tri_quad_face_averages():
    bm.set_backend("numpy")
    from fealpy.fvm import CellAnchoredQuadraticFaceFluxReconstruct

    mesh = _mixed_tri_quad_mesh()
    geometry = FVMGeometry(mesh)
    cell_average = _cell_average(geometry, _vector_quadratic_2d)
    exact_face_average = _face_average(geometry, _vector_quadratic_2d)
    reconstruct = CellAnchoredQuadraticFaceFluxReconstruct(
        geometry=geometry,
    )

    face_average = reconstruct.face_average(
        cell_average,
        boundary_face_average=exact_face_average[geometry.is_boundary],
        boundary_faces=geometry.boundary_faces,
    )

    np.testing.assert_allclose(
        bm.to_numpy(face_average),
        bm.to_numpy(exact_face_average),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    assert reconstruct.diagnostics()["failed_cell_count"] == 0
