import pickle
from app.gearx.gear import ExternalGear, InternalGear
from app.gearx.utils import *
from fealpy.mesh import HexahedronMesh, QuadrangleMesh



# nx, ny, nz = 2, 2, 2
# domain_hex = [0, 1, 0, 1, 0, 1]
# mesh_hex = HexahedronMesh.from_box(box=domain_hex, nx=nx, ny=ny, nz=nz)

with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/external_gear.pkl', 'rb') as f:
    data = pickle.load(f)
quad_mesh = data['quad_mesh']
hex_mesh = data['hex_mesh']
node_hex = hex_mesh.node
face_hex = hex_mesh.face
cell2face = hex_mesh.cell_to_face()

# hex_mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/hexahedron.vtu')
external_gear = data['gear']

n = 15
helix_d = np.linspace(external_gear.d, external_gear.effective_da, n)
helix_width = np.linspace(0, external_gear.tooth_width, n)
helix_node = cylindrical_to_cartesian(helix_d, helix_width, external_gear)


target_cell_idx = np.zeros(n, dtype=np.int32)
for i, t_node in enumerate(helix_node):
    target_cell_idx[i] = find_node_location_kd_tree(t_node, hex_mesh)

bd_face = hex_mesh.boundary_face_flag()

target_cell_face = bd_face[cell2face[target_cell_idx]]
print(target_cell_face)




print("-----------")