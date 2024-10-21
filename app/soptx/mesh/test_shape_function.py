from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh

bm.set_backend('numpy')
bm.set_default_device('cpu')
nx, ny, nz = 2, 2, 2
mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz)

sf = mesh.shape_function(bcs=[0, 1], p=1)
gsf = mesh.grad_shape_function(bcs=[0, 1], p=1)
print("-----------------------")
