from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.decorator import cartesian
from fealpy.experimental.mesh import UniformMesh2d, QuadrangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.material.elastic_material import LinearElasticMaterial
from fealpy.experimental.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.experimental.fem.bilinear_form import BilinearForm
from fealpy.experimental.fem.linear_form import LinearForm
from fealpy.experimental.sparse import COOTensor
from fealpy.experimental.solver import cg

class BoxDomainData2D():

    def domain(self):
        return [0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike, index=None) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype)
        val[..., 0] = 35/13 * y - 35/13 * y**2 + 10/13 * x - 10/13 * x**2
        val[..., 1] = -25/26 * (-1 + 2 * y) * (-1 + 2 * x)
        
        return val
    
    @cartesian
    def solution(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype)
        val[..., 0] = x * (1 - x) * y * (1 - y)
        val[..., 1] = 0
        
        return val
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)
pde = BoxDomainData2D()

bm.set_backend('numpy')
nx, ny = 2, 2
extent = pde.domain()
h = [(extent[1] - extent[0]) / nx, (extent[3] - extent[2]) / ny]
origin = [extent[0], extent[2]]
mesh_nec = UniformMesh2d(extent=[0, 1, 0, 1], h=h, origin=origin, 
                        ipoints_ordering='nec')
mesh_yx = UniformMesh2d(extent=[0, 1, 0, 1], h=h, origin=origin,
                        ipoints_ordering='yx')


p = 2
space_nec = LagrangeFESpace(mesh_nec, p=p, ctype='C')
tensor_space_nec = TensorFunctionSpace(space_nec, shape=(-1, 2))
gdof_nec = space_nec.number_of_global_dofs()
tgdof_nec = tensor_space_nec.number_of_global_dofs()
cell2dof_nec = space_nec.cell_to_dof()
cell2tdof_nec = tensor_space_nec.cell_to_dof()

space_yx = LagrangeFESpace(mesh_yx, p=p, ctype='C')
tensor_space_yx = TensorFunctionSpace(space_yx, shape=(-1, 2))
gdof_yx = space_yx.number_of_global_dofs()
tgdof_yx = tensor_space_yx.number_of_global_dofs()
cell2dof_yx = space_yx.cell_to_dof()
cell2tdof_yx = tensor_space_yx.cell_to_dof()


linear_elastic_material = LinearElasticMaterial(name='E1nu0.3', 
                                            elastic_modulus=1, poisson_ratio=0.3, 
                                            hypo='plane_strain')

integrator_K_nec = LinearElasticIntegrator(material=linear_elastic_material, 
                                        q=tensor_space_nec.p+3)
KE_nec = integrator_K_nec.assembly(space=tensor_space_nec)
bform_nec = BilinearForm(tensor_space_nec)
bform_nec.add_integrator(integrator_K_nec)
K_nec = bform_nec.assembly()
K_nec_test = K_nec.to_dense().round(4)

integrator_K_yx = LinearElasticIntegrator(material=linear_elastic_material,
                                        q=tensor_space_yx.p+3)
KE_yx = integrator_K_yx.assembly(space=tensor_space_yx)
bform_yx = BilinearForm(tensor_space_yx)
bform_yx.add_integrator(integrator_K_yx)
K_yx = bform_yx.assembly()
K_yx_test = K_yx.to_dense().round(4)


integrator_F_nec = VectorSourceIntegrator(source=pde.source, 
                                        q=tensor_space_nec.p+3)
lform_nec = LinearForm(tensor_space_nec)    
lform_nec.add_integrator(integrator_F_nec)
F_nec = lform_nec.assembly()
F_nec_test = F_nec.round(4)

integrator_F_yx = VectorSourceIntegrator(source=pde.source,
                                        q=tensor_space_yx.p+3)
lform_yx = LinearForm(tensor_space_yx)
lform_yx.add_integrator(integrator_F_yx)
F_yx = lform_yx.assembly()
F_yx_test = F_yx.round(4)


uh_bd_nec = bm.zeros(tensor_space_nec.number_of_global_dofs(), dtype=bm.float64)
uh_bd_nec, isDDof_nec = tensor_space_nec.boundary_interpolate(gD=pde.dirichlet, 
                                                            uh=uh_bd_nec, threshold=None)
F_nec = F_nec - K_nec.matmul(uh_bd_nec)
F_nec[isDDof_nec] = uh_bd_nec[isDDof_nec]
indices = K_nec.indices()
new_values = bm.copy(K_nec.values())
IDX = isDDof_nec[indices[0, :]] | isDDof_nec[indices[1, :]]
new_values[IDX] = 0
K_nec = COOTensor(indices, new_values, K_nec.sparse_shape)
index, = bm.nonzero(isDDof_nec)
one_values = bm.ones(len(index), **K_nec.values_context())
one_indices = bm.stack([index, index], axis=0)
K1_nec = COOTensor(one_indices, one_values, K_nec.sparse_shape)
K_nec = K_nec.add(K1_nec).coalesce()

uh_nec = tensor_space_nec.function()
uh_nec[:] = cg(K_nec, F_nec, maxiter=5000, atol=1e-14, rtol=1e-14)
uh_nec_test = uh_nec.round(4)

uh_bd_yx = bm.zeros(tensor_space_yx.number_of_global_dofs(), dtype=bm.float64)
uh_bd_yx, isDDof_yx = tensor_space_yx.boundary_interpolate(gD=pde.dirichlet, 
                                                        uh=uh_bd_yx, threshold=None)
F_yx = F_yx - K_yx.matmul(uh_bd_yx)
F_yx[isDDof_yx] = uh_bd_yx[isDDof_yx]
indices = K_yx.indices()
new_values = bm.copy(K_yx.values())
IDX = isDDof_yx[indices[0, :]] | isDDof_yx[indices[1, :]]
new_values[IDX] = 0
K_yx = COOTensor(indices, new_values, K_yx.sparse_shape)
index, = bm.nonzero(isDDof_yx)
one_values = bm.ones(len(index), **K_yx.values_context())
one_indices = bm.stack([index, index], axis=0)
K1_yx = COOTensor(one_indices, one_values, K_yx.sparse_shape)
K_yx = K_yx.add(K1_yx).coalesce()

uh_yx = tensor_space_yx.function() 
uh_yx[:] = cg(K_yx, F_yx, maxiter=5000, atol=1e-14, rtol=1e-14)
uh_yx_test = uh_yx.round(4)


u_exact_yx = tensor_space_yx.interpolate(pde.solution)
u_exact_yx_test = u_exact_yx.round(4)

error_yx = bm.max(bm.abs(uh_yx - u_exact_yx))

u_exact_nec = tensor_space_nec.interpolate(pde.solution)
u_exact_nec_test = u_exact_nec.round(4)

error_nec = bm.max(bm.abs(uh_nec - u_exact_nec))
print("------------")