from fealpy.experimental.mesh import UniformMesh3d, HexahedronMesh

from fealpy.experimental.backend import backend_manager as bm


bm.set_backend('pytorch')
# bm.set_backend('numpy')



extent = [0, 1, 0, 1, 0, 1]
h = [1, 1, 1]
origin = [0, 0, 0]
mesh = UniformMesh3d(extent, h, origin)