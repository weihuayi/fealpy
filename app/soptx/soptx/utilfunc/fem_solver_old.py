from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.mesh import UniformMesh2d
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.fem.bilinear_form import BilinearForm
from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor
from fealpy.experimental.solver import cg

from app.soptx.soptx.cases.mbb_beam_cases import MBBBeamCase
from app.soptx.soptx.cases.material_properties import MaterialProperties, SIMPInterpolation


mbb_case = MBBBeamCase(case_name="top88")

nx = mbb_case.nx
ny = mbb_case.ny

extent = [0, nx, 0, ny]
h = [1, 1]
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)

p_C = 1
space_C = LagrangeFESpace(mesh, p=p_C, ctype='C')
tensor_space = TensorFunctionSpace(space_C, shape=(-1, 2))
uh = tensor_space.function()

p_D = 0
space_D = LagrangeFESpace(mesh, p=p_D, ctype='D')        
rho = space_D.function()
rho[:] = mbb_case.rho
isotropic_material = MaterialProperties(E0=1.0, Emin=1e-9, nu=0.3, 
                                    penal=3.0, hypo="plane_stress", 
                                    rho=mbb_case.rho, interpolation_model=SIMPInterpolation())

integrator = LinearElasticIntegrator(material=isotropic_material, q=p_C+3)

KK = integrator.assembly(space=tensor_space)

bform = BilinearForm(tensor_space)
bform.add_integrator(integrator)
K = bform.assembly()

force = mbb_case.boundary_conditions.force
dirichlet = mbb_case.boundary_conditions.dirichlet
is_dirichlet_boundary_edge = mbb_case.boundary_conditions.is_dirichlet_boundary_edge
is_dirichlet_node = mbb_case.boundary_conditions.is_dirichlet_node
is_dirichlet_direction = mbb_case.boundary_conditions.is_dirichlet_direction

F = tensor_space.interpolate(force)

dbc = DBC(space=tensor_space, gd=dirichlet, left=False)
F = dbc.check_vector(F)
isDDof = tensor_space.is_boundary_dof(threshold=(is_dirichlet_boundary_edge, 
                                                is_dirichlet_node,
                                                is_dirichlet_direction))

uh = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh, 
                                    threshold=is_dirichlet_boundary_edge)

F = F - K.matmul(uh[:])
F[isDDof] = uh[isDDof]

K = dbc.check_matrix(K)
kwargs = K.values_context()
indices = K.indices()
new_values = bm.copy(K.values())
IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
new_values[IDX] = 0

K = COOTensor(indices, new_values, K.sparse_shape)
index, = bm.nonzero(isDDof)
one_values = bm.ones(len(index), **kwargs)
one_indices = bm.stack([index, index], axis=0)
K1 = COOTensor(one_indices, one_values, K.sparse_shape)
K = K.add(K1).coalesce()

uh[:] = cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)
print("jhhhhhhhhhhhhhhhhhh")