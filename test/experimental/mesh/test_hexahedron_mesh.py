
from fealpy.mesh import HexahedronMesh as HexahedronMesh_old
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.hexahedron_mesh import HexahedronMesh

#bm.set_backend('pytorch')

def test_bc_to_point():
    mesh = HexahedronMesh.from_one_hexahedron(twist=True)
    mesh_old = HexahedronMesh_old.from_one_hexahedron(twist=True)

    integrator = mesh.quadrature_formula(2, 'cell')
    integrator_old = mesh_old.integrator(2, 'cell')

    bcs, ws = integrator.get_quadrature_points_and_weights()
    bcs_old, ws_old = integrator_old.get_quadrature_points_and_weights()
    print(bcs)

    point = mesh.bc_to_point(bcs)
    point_old = mesh_old.bc_to_point(bcs_old)
    print(point)

    #print("result of bc_to_point : ", bm.max(bm.abs(point - point_old)))


def test_jacobi_matrix_and_first_fundamental_form():
    mesh = HexahedronMesh.from_one_hexahedron(twist=True)
    mesh_old = HexahedronMesh_old.from_one_hexahedron(twist=True)

    integrator = mesh.quadrature_formula(2, 'cell')
    integrator_old = mesh_old.integrator(2, 'cell')

    bcs, ws = integrator.get_quadrature_points_and_weights()
    bcs_old, ws_old = integrator_old.get_quadrature_points_and_weights()

    jacobi = mesh.jacobi_matrix(bcs)
    jacobi_old = mesh_old.jacobi_matrix(bcs_old)

    print("result of jacobi_matrix : ", bm.max(bm.abs(jacobi - jacobi_old))<1e-12)

    fff = mesh.first_fundamental_form(jacobi)
    fff_old = mesh_old.first_fundamental_form(jacobi_old)

    print("result of first_fundamental_form : ", bm.max(bm.abs(fff - fff_old))<1e-12)

def test_cell_and_face_to_ipoint():
    mesh = HexahedronMesh.from_one_hexahedron(twist=True)
    mesh_old = HexahedronMesh_old.from_one_hexahedron(twist=True)

    cip = mesh.cell_to_ipoint(2)
    cip_old = mesh_old.cell_to_ipoint(2)

    print("result of cell_to_ipoint : ", bm.max(bm.abs(cip - cip_old))<1e-12)

    fip = mesh.face_to_ipoint(2)
    fip_old = mesh_old.face_to_ipoint(2)

    print("result of face_to_ipoint : ", bm.max(bm.abs(fip - fip_old))<1e-12)

def test_interpolation_points():
    mesh = HexahedronMesh.from_one_hexahedron(twist=True)
    mesh_old = HexahedronMesh_old.from_one_hexahedron(twist=True)

    ip = mesh.interpolation_points(2)
    ip_old = mesh_old.interpolation_points(2)

    print("result of interpolation_points : ", bm.max(bm.abs(ip - ip_old))<1e-12)

def test_uniform_refine():
    mesh = HexahedronMesh.from_one_hexahedron(twist=True)
    mesh_old = HexahedronMesh_old.from_one_hexahedron(twist=True)

    mesh.uniform_refine(2)
    mesh_old.uniform_refine(2)

    node = mesh.entity('node')
    cell = mesh.entity('cell')

    node_old = mesh_old.entity('node')
    cell_old = mesh_old.entity('cell')

    print("result of uniform_refine : ", bm.max(bm.abs(node - node_old))<1e-12, bm.max(bm.abs(cell - cell_old))<1e-12)

def test_entity_measure():
    mesh = HexahedronMesh.from_one_hexahedron(twist=True)
    mesh_old = HexahedronMesh_old.from_one_hexahedron(twist=True)

    cm = mesh.entity_measure('cell')
    fm = mesh.entity_measure('face')
    em = mesh.entity_measure('edge')

    cm_old = mesh_old.entity_measure('cell')
    fm_old = mesh_old.entity_measure('face')
    em_old = mesh_old.entity_measure('edge')

    print("result of entity_measure : ", bm.max(bm.abs(cm - cm_old))<1e-12, bm.max(bm.abs(fm - fm_old))<1e-12, bm.max(bm.abs(em - em_old))<1e-12)
    
if __name__ == "__main__":
    test_bc_to_point()
    test_jacobi_matrix_and_first_fundamental_form()
    test_cell_and_face_to_ipoint()
    test_interpolation_points()
    test_uniform_refine()
    test_entity_measure()



