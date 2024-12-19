from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh, TetrahedronMesh
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.sparse import COOTensor, CSRTensor
from fealpy.solver import cg, spsolve
# 刚度矩阵
E = 2.1e5
nu = 0.3
# E = 1e6
# nu = 0.25
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=1, ny=1, nz=1)
p = 1
q = p+1
space = LagrangeFESpace(mesh, p=p, ctype='C')
sgdof = space.number_of_global_dofs()
print(f"sgdof: {sgdof}")
tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K_sri = LinearElasticIntegrator(material=linear_elastic_material,
                                           q=q, method='C3D8_SRI')
qf2 = mesh.quadrature_formula(q)
bcs2, ws2 = qf2.get_quadrature_points_and_weights()
gphi2 = space.grad_basis(bcs2, variable='x')

# B0_q1 = linear_elastic_material._normal_strain_sri(gphi=gphi1)
KE_sri_yz_xz_xy = integrator_K_sri.c3d8_sri_assembly(space=tensor_space)
print("---------------------")