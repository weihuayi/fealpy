'''
用于说明对于 bilinear_form，调用同一个积分子时，需要清除缓存
'''
from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator

from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian

from fealpy.sparse import COOTensor, CSRTensor

class BoxDomainPolyLoaded3d():
    def domain(self):
        return [0, 1, 0, 1, 0, 1]
    
    @cartesian
    def source(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        mu = 1
        factor1 = -400 * mu * (2 * y - 1) * (2 * z - 1)
        term1 = 3 * (x ** 2 - x) ** 2 * (y ** 2 - y + z ** 2 - z)
        term2 = (1 - 6 * x + 6 * x ** 2) * (y ** 2 - y) * (z ** 2 - z)
        val[..., 0] = factor1 * (term1 + term2)

        factor2 = 200 * mu * (2 * x - 1) * (2 * z - 1)
        term1 = 3 * (y ** 2 - y) ** 2 * (x ** 2 - x + z ** 2 - z)
        term2 = (1 - 6 * y + 6 * y ** 2) * (x ** 2 - x) * (z ** 2 - z)
        val[..., 1] = factor2 * (term1 + term2)

        factor3 = 200 * mu * (2 * x - 1) * (2 * y - 1)
        term1 = 3 * (z ** 2 - z) ** 2 * (x ** 2 - x + y ** 2 - y)
        term2 = (1 - 6 * z + 6 * z ** 2) * (x ** 2 - x) * (y ** 2 - y)
        val[..., 2] = factor3 * (term1 + term2)

        return val

    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))

        mu = 1
        val[..., 0] = 200*mu*(x-x**2)**2 * (2*y**3-3*y**2+y) * (2*z**3-3*z**2+z)
        val[..., 1] = -100*mu*(y-y**2)**2 * (2*x**3-3*x**2+x) * (2*z**3-3*z**2+z)
        val[..., 2] = -100*mu*(z-z**2)**2 * (2*y**3-3*y**2+y) * (2*x**3-3*x**2+x)

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, 
                        dtype=points.dtype, device=bm.get_device(points))

pde = BoxDomainPolyLoaded3d()

bm.set_backend('numpy')
nx, ny, nz = 2, 2, 2 
mesh_dof = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
mesh_gd = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
NC = mesh_dof.number_of_cells()

space_dof = LagrangeFESpace(mesh_dof, p=1, ctype='C')
space_gd = LagrangeFESpace(mesh_gd, p=1, ctype='C')

q = 2
qf = mesh_dof.quadrature_formula(q)
bcs, ws = qf.get_quadrature_points_and_weights()
phi = space_dof.basis(bcs)

linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='3D', device=bm.get_device(mesh_dof))
tensor_space_dof = TensorFunctionSpace(space_dof, shape=(3, -1))
tensor_space_gd = TensorFunctionSpace(space_gd, shape=(-1, 3))
integrator_K_dof = LinearElasticIntegrator(material=linear_elastic_material, q=2)
integrator_K_gd = LinearElasticIntegrator(material=linear_elastic_material, q=2)
KE_dof = integrator_K_dof.assembly(space=tensor_space_dof)
KE_gd = integrator_K_gd.assembly(space=tensor_space_gd)

cell = mesh_dof.entity('cell')
cell2dof_dof = tensor_space_dof.cell_to_dof() # (NC, tldof)
cell2dof_gd = tensor_space_gd.cell_to_dof() # (NC, tldof)
tgdof = tensor_space_dof.number_of_global_dofs()

I_dof = bm.broadcast_to(cell2dof_dof[:, :, None], shape=KE_dof.shape)
J_dof = bm.broadcast_to(cell2dof_dof[:, None, :], shape=KE_dof.shape)
K_dof_me = COOTensor(
            indices = bm.empty((2, 0), dtype=bm.int32, device=bm.get_device(space_dof)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space_dof)),
            spshape = (tgdof, tgdof))
indices = bm.stack([I_dof.ravel(), J_dof.ravel()], axis=0)
K_dof_me = K_dof_me.add(COOTensor(indices, KE_dof.reshape(-1), (tgdof, tgdof))).to_dense()

I_gd = bm.broadcast_to(cell2dof_gd[:, :, None], shape=KE_gd.shape)
J_gd = bm.broadcast_to(cell2dof_gd[:, None, :], shape=KE_gd.shape)
K_gd_me = COOTensor(
            indices = bm.empty((2, 0), dtype=bm.int32, device=bm.get_device(space_gd)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space_gd)),
            spshape = (tgdof, tgdof))
indices = bm.stack([I_gd.ravel(), J_gd.ravel()], axis=0)
K_gd_me = K_gd_me.add(COOTensor(indices, KE_gd.reshape(-1), (tgdof, tgdof))).to_dense()

bform_gd = BilinearForm(tensor_space_gd)
bform_gd.add_integrator(integrator_K_gd)
K_gd_fealpy = bform_gd.assembly(format='coo').to_dense()


bform_dof = BilinearForm(tensor_space_dof)
integrator_K_gd.clear()
bform_dof.add_integrator(integrator_K_gd)

K_dof_fealpy = bform_dof.assembly(format='coo').to_dense()
print("-----------------------------------")

