import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fvm import FVMGeometry, ResolvedGradientBoundary
from fealpy.fvm.face_gradient import reconstruct_face_gradient
from fealpy.mesh import TriangleMesh


def test_face_gradient_rejects_full_face_wise_patch_sn_grad():
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    geometry = FVMGeometry(mesh)
    boundary_faces = bm.nonzero(geometry.is_boundary)[0][:2]
    cell_gradient = bm.ones((mesh.number_of_cells(), mesh.geo_dimension()))

    empty_faces = boundary_faces[:0]
    with pytest.raises(ValueError, match="one value per neumann face"):
        ResolvedGradientBoundary(
            dirichlet_faces=empty_faces,
            dirichlet_values=bm.zeros(0),
            neumann_faces=boundary_faces,
            neumann_sn_grad=bm.ones(mesh.number_of_faces()),
        )


def test_face_gradient_rejects_broadcast_patch_dirichlet_value():
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    geometry = FVMGeometry(mesh)
    boundary_faces = bm.nonzero(geometry.is_boundary)[0][:2]
    cell_gradient = bm.ones((mesh.number_of_cells(), mesh.geo_dimension()))
    cell_values = bm.ones(mesh.number_of_cells())

    empty_faces = boundary_faces[:0]
    with pytest.raises(ValueError, match="one value per dirichlet face"):
        ResolvedGradientBoundary(
            dirichlet_faces=boundary_faces,
            dirichlet_values=bm.array(1.0),
            neumann_faces=empty_faces,
            neumann_sn_grad=bm.zeros(0),
        )


def test_face_gradient_does_not_copy_normalized_tensor_inputs(monkeypatch):
    mesh = TriangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    geometry = FVMGeometry(mesh)
    boundary = ResolvedGradientBoundary.empty(geometry)
    cell_gradient = bm.ones((geometry.NC, geometry.GD))
    cell_values = bm.ones(geometry.NC)

    def reject_array_copy(*_args, **_kwargs):
        raise AssertionError("normalized tensor input must not be copied")

    monkeypatch.setattr(
        "fealpy.fvm.face_gradient.bm.array",
        reject_array_copy,
    )

    reconstruct_face_gradient(
        geometry,
        cell_gradient,
        cell_values,
        boundary=boundary,
    )
