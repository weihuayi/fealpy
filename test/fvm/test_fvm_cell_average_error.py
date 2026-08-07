import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh


def test_cell_average_uses_control_volume_mean_not_cell_center_value():
    from fealpy.fvm import cell_average

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=1, ny=1)

    def exact(points):
        return points[..., 0] ** 2 + points[..., 1] ** 2

    average = cell_average(mesh, exact, q=4)
    center_value = exact(mesh.entity_barycenter("cell"))

    np.testing.assert_allclose(np.asarray(average), np.array([2.0 / 3.0]))
    assert not np.allclose(np.asarray(average), np.asarray(center_value))


def test_cell_average_l2_error_compares_with_control_volume_mean():
    from fealpy.fvm import cell_average_l2_error

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=1, ny=1)

    def exact(points):
        return points[..., 0] ** 2 + points[..., 1] ** 2

    error, average = cell_average_l2_error(mesh, exact, bm.array([2.0 / 3.0]), q=4)

    np.testing.assert_allclose(np.asarray(average), np.array([2.0 / 3.0]))
    assert float(error) < 1.0e-13


def test_poisson_compute_error_compares_against_cell_average():
    from fealpy.fvm import PoissonFVMModel, cell_average

    bm.set_backend("numpy")

    class PDE:
        init_mesh = {
            "uniform_quad": lambda nx, ny: QuadrangleMesh.from_box(
                [0.0, 1.0, 0.0, 1.0], nx=nx, ny=ny
            )
        }

        @staticmethod
        def solution(points):
            return points[..., 0] ** 2 + points[..., 1] ** 2

        source = solution
        dirichlet = solution

    model = PoissonFVMModel(
        {
            "pde": PDE(),
            "nx": 1,
            "ny": 1,
            "space_degree": 0,
            "mesh_type": "uniform_quad",
        }
    )
    model.solution = cell_average(model.mesh, model.pde.solution, q=4)

    assert float(model.compute_error()) < 1.0e-13
