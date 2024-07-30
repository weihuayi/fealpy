
from fealpy.mesh import HexahedronMesh as HexahedronMesh_old
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.hexahedron_mesh import HexahedronMesh

bm.set_backend('pytorch')

def test_bc_to_point():
    mesh = HexahedronMesh.from_one_hexahedron(twist=True)
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

def test_cell_to_ip():
    mesh = HexahedronMesh.from_one_hexahedron()
    ip = mesh.cell_to_ipoint(2)
    print(ip)

def test_face_to_ip():
    mesh = HexahedronMesh.from_one_hexahedron()
    ip = mesh.face_to_ipoint(2)
    print(ip)

def test_interpolation_points():
    mesh = HexahedronMesh.from_one_hexahedron()
    integrator = mesh.quadrature_formula(2, 'cell')

    ip = mesh.interpolation_points(2)
    print(ip)

def test_uniform_refine():
    mesh = HexahedronMesh.from_one_hexahedron()
    mesh.uniform_refine(2)

def test_entity_measure():
    mesh = HexahedronMesh.from_one_hexahedron()
    mesh.entity_measure('cell')
    



if __name__ == "__main__":
    #test_bc_to_point()
    #test_jacobi_matrix()
    #test_first_fundamental_form()
    #test_cell_to_ip()
    test_uniform_refine()



