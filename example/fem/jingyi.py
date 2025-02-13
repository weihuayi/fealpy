
from fealpy.backend import backend_manager as bm


from fealpy.mesh import HexahedronMesh, TetrahedronMesh 
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import LinearElasticityLFEMSolver
from fealpy.material.elastic_material import LinearElasticMaterial


def is_bottom_bd_node(node):
    return node[:, 2] < 1e-13

def is_right_bd_node(node):
    return node[:, 0] > 1 - 1e-13

def is_right_center_bad_node(node):
    return (node[:, 0] > 1 - 1e-13) & (bm.abs(node[:, 1] - 0.5) <  1e-13) & (bm.abs(node[:, 2] - 0.5) < 1e-13)

def is_bd_node(node):
    return (node[:, 0] > 1 - 1e-13) & ((bm.abs(node[:, 1] - 0.5) <  1e-13) | (bm.abs(node[:, 2] - 0.5) < 1e-13))

mesh = HexahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=10, ny=10, nz=10)
#mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=10, ny=10, nz=10)
material = LinearElasticMaterial('hexmesh', elastic_modulus=2.06e5, poisson_ratio=0.3)
s0 = LinearElasticityLFEMSolver(material, mesh, p=1)


node = mesh.entity('node')
isBdNode = mesh.boundary_node_flag()
isRightBdNode = is_right_bd_node(node)
isBottomBdNode = is_bottom_bd_node(node)
isRightCenterBdNode = is_right_center_bad_node(node)

isNode = is_bd_node(node)

#fload = {'nset': [1270], 'value': [0, -1, 0]}
#mesh.meshdata['load'] = {'face': {'fem': fload}}

mesh.meshdata['load'] = {'node': { 
                                  'c0': {'nset': [1270], 'value': [-100, 0, 0]},
                                  #'c1': {'nset': [1271, 1269, 1281, 1259], 'value': [0, -20, 0]}
                                  }}
s0.apply_node_load()
s0.apply_dirichlet_bc(bm.repeat(isBottomBdNode, 3))
du = s0.solve()

print("1273:", du[1273][0])
print("1272:", du[1272][0])
print("1271:", du[1271][0])
print("1270:", du[1270][0])
print("1269:", du[1269][0])
print("1268:", du[1268][0])
print("1267:", du[1267][0])


print("1292:", du[1292][0])
print("1281:", du[1281][0])
print("1270:", du[1270][0])
print("1259:", du[1259][0])
print("1248:", du[1248][0])

mesh.nodedata['displacement'] = du
mesh.to_vtk(fname='tetmesh.vtu')


#s0.show_mesh(nindex=isNode)
s0.show_displacement(alpha=10)

