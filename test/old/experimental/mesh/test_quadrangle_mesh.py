from fealpy.backend import backend_manager as bm
from fealpy.mesh.quadrangle_mesh import QuadrangleMesh


# from fealpy.mesh import QuadrangleMesh

# bm.set_backend('pytorch')
# bm.backend_name

def test_quad_mesh_init():
    node = bm.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32)

    mesh = QuadrangleMesh(node, cell)

    assert mesh.node.shape == (4, 2)
    assert mesh.cell.shape == (1, 4)

    assert mesh.number_of_nodes() == 4
    assert mesh.number_of_cells() == 1
    assert mesh.number_of_edges() == 4
    assert mesh.number_of_faces() == 4

    print('init test done!')


def test_quad_mesh_entity_measure():
    node = bm.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32)

    mesh = QuadrangleMesh(node, cell)

    assert mesh.entity_measure(0) == bm.tensor([0.0], dtype=bm.float64)
    assert all(mesh.entity_measure(1) == bm.tensor([1.0, 1.0, 1.0, 1.0], dtype=bm.float64))
    assert all(mesh.entity_measure('cell') == bm.tensor([1.0], dtype=bm.float64))

    print('entity measure test done!')


def test_quad_mesh_quadrature_and_bc_to_point():
    node = bm.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32)

    mesh = QuadrangleMesh(node, cell)
    q = 2
    integrator = mesh.quadrature_formula(q)
    bcs, ws = integrator.get_quadrature_points_and_weights()

    for bc in bcs:
        assert all((bc[0] - bm.tensor([0.78867513, 0.21132487], dtype=bm.float64)) < 1e-7)
        assert all((bc[1] - bm.tensor([0.21132487, 0.78867513], dtype=bm.float64)) < 1e-7)
    assert all(ws == bm.tensor([0.25, 0.25, 0.25, 0.25], dtype=bm.float64))

    point = mesh.bc_to_point(bcs)
    assert all(point.reshape(-1) - bm.tensor([[[0.21132487, 0.21132487]],
                                              [[0.21132487, 0.78867513]],
                                              [[0.78867513, 0.21132487]],
                                              [[0.78867513, 0.78867513]]], dtype=bm.float64).reshape(-1) < 1e-7)

    print('quadrature formula and bc to point test done!')


def test_shape_function_and_grad_shape_function():
    node = bm.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32)
    bcs = (bm.tensor([[0.78867513, 0.21132487], [0.21132487, 0.78867513]], dtype=bm.float64),
           bm.tensor([[0.78867513, 0.21132487], [0.21132487, 0.78867513]], dtype=bm.float64))
    mesh = QuadrangleMesh(node, cell)

    shape_function = mesh.shape_function(bcs)
    assert all(shape_function.reshape(-1) - bm.tensor([[0.62200847, 0.16666667, 0.16666667, 0.0446582],
                                                       [0.16666667, 0.62200847, 0.0446582, 0.16666667],
                                                       [0.16666667, 0.0446582, 0.62200847, 0.16666667],
                                                       [0.0446582, 0.16666667, 0.16666667, 0.62200847]],
                                                      dtype=bm.float64).reshape(-1) < 1e-7)
    grad_shape_function = mesh.grad_shape_function(bcs)
    assert all(grad_shape_function.reshape(-1) - bm.tensor(
        [[[[-0.78867513, -0.78867513], [-0.21132487, 0.78867513], [0.78867513, -0.21132487], [0.21132487, 0.21132487]]],
         [[[-0.21132487, -0.78867513], [-0.78867513, 0.78867513], [0.21132487, -0.21132487], [0.78867513, 0.21132487]]],
         [[[-0.78867513, -0.21132487], [-0.21132487, 0.21132487], [0.78867513, -0.78867513], [0.21132487, 0.78867513]]],
         [[[-0.21132487, -0.21132487], [-0.78867513, 0.21132487], [0.21132487, -0.78867513],
           [0.78867513, 0.78867513]]]],
        dtype=bm.float64).reshape(-1) < 1e-7)
    grad_shape_function_x = mesh.grad_shape_function(bcs, variables='x')
    assert all(grad_shape_function_x.reshape(-1) - bm.tensor(
        [[[[-0.78867513, - 0.78867513], [-0.21132487, 0.78867513], [0.78867513, - 0.21132487],
           [0.21132487, 0.21132487]]],
         [[[-0.21132487, - 0.78867513], [-0.78867513, 0.78867513], [0.21132487, -0.21132487],
           [0.78867513, 0.21132487]]],
         [[[-0.78867513, - 0.21132487], [-0.21132487, 0.21132487], [0.78867513, - 0.78867513],
           [0.21132487, 0.78867513]]],
         [[[-0.21132487, - 0.21132487], [-0.78867513, 0.21132487], [0.21132487, - 0.78867513],
           [0.78867513, 0.78867513]]]],
        dtype=bm.float64).reshape(-1) < 1e-7)
    print('shape function and grad shape function test done!')


def test_jacobi_matrix_rel():
    node = bm.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32)
    bcs = (bm.tensor([[0.78867513, 0.21132487], [0.21132487, 0.78867513]], dtype=bm.float64),
           bm.tensor([[0.78867513, 0.21132487], [0.21132487, 0.78867513]], dtype=bm.float64))

    mesh = QuadrangleMesh(node, cell)

    jacobi_matrix = mesh.jacobi_matrix(bcs)
    assert all(jacobi_matrix.reshape(-1) - bm.tensor(
        [[[[1., 0.], [0., 1.]]],
         [[[1., 0.], [0., 1.]]],
         [[[1., 0.], [0., 1.]]],
         [[[1., 0.], [0., 1.]]]],
        dtype=bm.float64).reshape(-1) < 1e-7)
    first_fundamental_form = mesh.first_fundamental_form(jacobi_matrix)
    assert all(first_fundamental_form.reshape(-1) - bm.tensor(
        [[[[1., 0.],[0., 1.]]],
         [[[1., 0.],[0., 1.]]],
         [[[1., 0.],[0., 1.]]],
         [[[1., 0.],[0., 1.]]]],
        dtype=bm.float64).reshape(-1) < 1e-7)


def test_edge_geo():
    node = bm.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32)
    mesh = QuadrangleMesh(node, cell)

    edge_frame = mesh.edge_frame()
    edge_unit_normal = mesh.edge_unit_normal()

    assert all(
        edge_frame[0].reshape(-1) - bm.tensor([[0., -1.], [-1., 0.], [1., 0.], [0., 1.]], dtype=bm.float64).reshape(
            -1) < 1e-7)
    assert all(
        edge_frame[1].reshape(-1) - bm.tensor([[1., 0.], [0., -1.], [0., 1.], [-1., 0.]], dtype=bm.float64).reshape(
            -1) < 1e-7)
    assert all(
        edge_unit_normal.reshape(-1) - bm.tensor([[0., -1.], [-1., 0.], [1., 0.], [0., 1.]], dtype=bm.float64).reshape(
            -1) < 1e-7)


if __name__ == '__main__':
    test_quad_mesh_init()
    test_quad_mesh_entity_measure()
    test_quad_mesh_quadrature_and_bc_to_point()
    test_shape_function_and_grad_shape_function()
    test_edge_geo()
    test_jacobi_matrix_rel()
