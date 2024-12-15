from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh, HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
nx, ny, nz = 1, 1, 1 
mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], 
                            nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
p = 1
q = p+1
space = LagrangeFESpace(mesh, p=p, ctype='C')
qf2 = mesh.quadrature_formula(2)
bcs_q2, ws = qf2.get_quadrature_points_and_weights()
gphix_q2 = space.grad_basis(bcs_q2, variable='x')  # (NC, NQ, LDOF, GD)

gphix_q2_cart = mesh.bc_to_point(bcs_q2) # (NC, LDOF, GD)
print(f"gphix_q2_cart:\n {gphix_q2_cart}")
node = mesh.entity('node')
cell = mesh.entity('cell')
print(f"node(cell):\n {node[cell[0]]}")
print("-----------")
