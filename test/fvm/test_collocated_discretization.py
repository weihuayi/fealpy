import pytest

from fealpy.backend import backend_manager as bm


def _discretization():
    from fealpy.fvm import FVMGeometry
    from fealpy.fvm.collocated_discretization import CollocatedDiscretization
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    geometry = FVMGeometry(mesh)
    return CollocatedDiscretization(geometry=geometry), geometry


def test_collocated_discretization_owns_p0_layout_and_reuses_geometry():
    discretization, geometry = _discretization()

    assert discretization.geometry is geometry
    assert discretization.degree == 0
    assert discretization.NC == 4
    assert discretization.GD == 2


def test_collocated_discretization_is_unique_velocity_dof_boundary():
    discretization, _ = _discretization()
    velocity = bm.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )

    dofs = discretization.cell_vector_to_dofs(velocity)

    assert bm.to_numpy(dofs).tolist() == [
        1.0,
        3.0,
        5.0,
        7.0,
        2.0,
        4.0,
        6.0,
        8.0,
    ]
    assert bm.to_numpy(discretization.dofs_to_cell_vector(dofs)).tolist() == (
        bm.to_numpy(velocity).tolist()
    )


def test_collocated_discretization_rejects_noncanonical_velocity_shapes():
    discretization, _ = _discretization()

    with pytest.raises(ValueError, match=r"cell velocity must have shape \(NC, GD\)"):
        discretization.cell_vector_to_dofs(bm.zeros(discretization.NC))
    with pytest.raises(ValueError, match=r"velocity dofs must have shape \(GD\*NC,\)"):
        discretization.dofs_to_cell_vector(
            bm.zeros((discretization.GD, discretization.NC))
        )


def test_collocated_discretization_exposes_one_scalar_cell_response():
    discretization, geometry = _discretization()
    diagonal = 2.0 * geometry.cell_measure

    response = discretization.cell_response(diagonal)

    assert response.shape == (discretization.NC,)
    assert bm.to_numpy(response).tolist() == [0.5] * discretization.NC
