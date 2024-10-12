from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.functionspace.tensor_space import TensorFunctionSpace
from fealpy.experimental.functionspace.lagrange_fe_space import LagrangeFESpace
from fealpy.experimental.mesh.uniform_mesh_2d import UniformMesh2d

bm.set_backend('numpy')
nx, ny = 1, 1
extent = [0, nx, 0, ny]
h = [1.0, 1.0]
origin = [0.0, 0.0]

mesh = UniformMesh2d(extent=extent, h=h, origin=origin, device='cpu')
qf = mesh.quadrature_formula(q=1)
bcs, ws = qf.get_quadrature_points_and_weights()
space_C = LagrangeFESpace(mesh=mesh, p=1, ctype='C')
tensor_space = TensorFunctionSpace(scalar_space=space_C, shape=(-1, 2))
basis = tensor_space.basis(p=bcs)
print("----------------")