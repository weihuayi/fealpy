import numpy as np


def _solver():
    from test.fvm.test_collocated_simple_solver_structure import _cavity_solver

    return _cavity_solver(convection_coef=0.0)


def test_pressure_response_consumes_one_scalar_cell_diagonal():
    from fealpy.backend import backend_manager as bm

    solver = _solver()
    equation = solver.pressure_equation
    diagonal = 2.0 * bm.ones(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    response = equation.face_response_coefficient(diagonal)

    assert response.shape == (solver.discretization.NF,)
    np.testing.assert_allclose(
        bm.to_numpy(equation.discretization.cell_response(diagonal)),
        bm.to_numpy(solver.discretization.geometry.cell_measure / diagonal),
    )


def test_pressure_sparse_pattern_is_owned_by_pressure_equation():
    from fealpy.backend import backend_manager as bm

    equation = _solver().pressure_equation
    response = bm.ones(
        equation.discretization.NF,
        dtype=equation.discretization.geometry.cell_measure.dtype,
    )
    first = equation.diffusion_matrix(response)
    assembler = equation._diffusion_matrix_assembler
    second = equation.diffusion_matrix(response)

    assert assembler is equation._diffusion_matrix_assembler
    np.testing.assert_allclose(
        bm.to_numpy(first.values),
        bm.to_numpy(second.values),
    )


def test_pressure_equation_caches_whether_cross_flux_can_exist():
    equation = _solver().pressure_equation

    first = equation.has_nonorthogonal_cross_flux()
    cached = equation._has_nonorthogonal_cross_flux
    second = equation.has_nonorthogonal_cross_flux()

    assert isinstance(first, bool)
    assert second is first
    assert equation._has_nonorthogonal_cross_flux is cached
