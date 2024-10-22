from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh

bm.set_backend('numpy')
nx, ny, nz = 2, 2, 2
mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))

q = 1
qf = mesh.quadrature_formula(q)
bcs, ws = qf.get_quadrature_points_and_weights()
sf = mesh.shape_function(bcs=bcs, p=1)
gsf = mesh.grad_shape_function(bcs=bcs, p=1)
print("-----------------------")
