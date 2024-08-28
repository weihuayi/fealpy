from fealpy.experimental.backend import backend_manager as bm
# bm.set_backend('numpy')
# bm.set_backend('pytorch')
bm.set_backend('jax')

from fealpy.experimental.mesh import TriangleMesh

nx = 2
ny = 2
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
