from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.hexahedron_mesh import HexahedronMesh

#bm.set_backend('pytorch')

def test_bc_to_point():
    mesh = HexahedronMesh.from_one_hexahedron()
    integrator = mesh.quadrature_formula(2, 'cell')

    bcs, ws = integrator.get_quadrature_points_and_weights()
    point = mesh.bc_to_point(bcs)

def test_jacobi_matrix():
    mesh = HexahedronMesh.from_one_hexahedron()
    integrator = mesh.quadrature_formula(2, 'cell')

    bcs, ws = integrator.get_quadrature_points_and_weights()
    jacobi = mesh.jacobi_matrix(bcs)

def test_first_fundamental_form():
    mesh = HexahedronMesh.from_one_hexahedron()
    integrator = mesh.quadrature_formula(2, 'cell')

    bcs, ws = integrator.get_quadrature_points_and_weights()
    jacobi = mesh.jacobi_matrix(bcs)

    fff = mesh.first_fundamental_form(jacobi)



if __name__ == "__main__":
    #test_bc_to_point()
    #test_jacobi_matrix()
    test_first_fundamental_form()



