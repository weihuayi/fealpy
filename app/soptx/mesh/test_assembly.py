from fealpy.experimental.backend import backend_manager as bm

from app.soptx.soptx.cases.material_properties import ElasticMaterialProperties, SIMPInterpolation
from fealpy.experimental.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.experimental.fem.bilinear_form import BilinearForm
from app.soptx.soptx.pde.mbb_beam import MBBBeamOneData
from fealpy.experimental.fem import DirichletBC
from fealpy.experimental.sparse import COOTensor
from fealpy.experimental.solver import cg
from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.functionspace.tensor_space import TensorFunctionSpace

bm.set_backend('numpy')

volfrac = 0.5
nx, ny = 3, 2

rho = volfrac * bm.ones(nx * ny, dtype=bm.float64)
rho[0] = 0.8
rho[-1] = 1.5

material_properties = ElasticMaterialProperties(
            E0=1.0, Emin=1e-9, nu=0.3, penal=3.0, 
            hypo="plane_stress", rho=rho,
            interpolation_model=SIMPInterpolation())
pde = MBBBeamOneData(nx=nx, ny=ny)

extent = pde.domain()
h = [(extent[1] - extent[0]) / nx, (extent[3] - extent[2]) / ny]
origin = [extent[0], extent[2]]

mesh = UniformMesh2d(extent=extent, h=h, origin=origin, flip_direction=True)

# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()
p_C = 1
space_C = LagrangeFESpace(mesh, p=p_C, ctype='C')
tensor_space = TensorFunctionSpace(space_C, shape=(-1, 2))

cell2tldof = tensor_space.cell_to_dof()

uh = tensor_space.function()
# uh[:] = bm.ones(uh.shape, dtype=uh.dtype)

integrator = LinearElasticIntegrator(material=material_properties, 
                                    q=tensor_space.p+3)
KE = integrator.assembly(space=tensor_space).round(4)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator)
K = bform.assembly()
KFull = K.to_dense().round(4)


F = tensor_space.interpolate(pde.force)


# K, F = DirichletBC(space=tensor_space, gd=pde.dirichlet, 
#                    threshold=(pde.is_dirichlet_boundary_edge, 
#                             pde.is_dirichlet_node, 
#                             pde.is_dirichlet_direction)).apply(A=K, f=F)


dbc = DirichletBC(space=tensor_space, gd=pde.dirichlet, left=False)
F = dbc.check_vector(F)

# isDDof = tensor_space.is_boundary_dof(threshold=(pde.is_dirichlet_boundary_edge, 
#                                                 pde.is_dirichlet_node, 
#                                                 pde.is_dirichlet_direction))

uh, isDDof = tensor_space.boundary_interpolate(gD=pde.dirichlet, uh=uh, 
                                    threshold=(pde.is_dirichlet_boundary_edge, 
                                               pde.is_dirichlet_node, 
                                               pde.is_dirichlet_direction))
KFull_internal = KFull[~isDDof][:, ~isDDof]

F = F - K.matmul(uh[:])
F[isDDof] = uh[isDDof]

K = dbc.check_matrix(K)
indices = K.indices()
new_values = bm.copy(K.values())
IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
new_values[IDX] = 0

K = COOTensor(indices, new_values, K.sparse_shape)
index, = bm.nonzero(isDDof)
one_values = bm.ones(len(index), **K.values_context())
one_indices = bm.stack([index, index], axis=0)
K1 = COOTensor(one_indices, one_values, K.sparse_shape)
K = K.add(K1).coalesce()
KFull2 = K.to_dense()


uh[:] = cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)

print("--------------------------")