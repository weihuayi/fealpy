'''
用于说明对于 linear_form，调用同一个积分子时，需要清除缓存
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
nx, ny, nz = 3, 3, 3 
mesh_dof = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
mesh_gd = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
NC = mesh_dof.number_of_cells()

space_dof = LagrangeFESpace(mesh_dof, p=1, ctype='C')
space_gd = LagrangeFESpace(mesh_gd, p=1, ctype='C')
q = 2
qf = mesh_dof.quadrature_formula(q)
bcs, ws = qf.get_quadrature_points_and_weights()
phi = space_dof.basis(bcs)

tensor_space_dof = TensorFunctionSpace(space_dof, shape=(3, -1))
tensor_space_gd = TensorFunctionSpace(space_gd, shape=(-1, 3))

cell = mesh_dof.entity('cell')
cell2dof_dof = tensor_space_dof.cell_to_dof() # (NC, tldof)
cell2dof_gd = tensor_space_gd.cell_to_dof() # (NC, tldof)
tgdof = tensor_space_dof.number_of_global_dofs()


integrator_F_dof = VectorSourceIntegrator(source=pde.source, q=2)
integrator_F_gd = VectorSourceIntegrator(source=pde.source, q=2)

FE_dof = integrator_F_dof.assembly(space=tensor_space_dof)
FE_gd = integrator_F_gd.assembly(space=tensor_space_gd)

lform_dof = LinearForm(tensor_space_dof)
lform_dof.add_integrator(integrator_F_gd)
F_dof_fealpy = lform_dof.assembly() # (tgdof)

lform_gd = LinearForm(tensor_space_gd)
integrator_F_gd.clear()
lform_gd.add_integrator(integrator_F_gd)
F_gd_fealpy = lform_gd.assembly()

F_dof_me = COOTensor(
            indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space_gd)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space_gd)),
            spshape = (tgdof, ))
indices = cell2dof_dof.reshape(1, -1)
F_dof_me = F_dof_me.add(COOTensor(indices, FE_dof.reshape(-1), (tgdof, ))).to_dense()

F_gd_me = COOTensor(
            indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space_gd)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space_gd)),
            spshape = (tgdof, ))
indices = cell2dof_gd.reshape(1, -1)
test = FE_gd.reshape(-1)
F_gd_me = F_gd_me.add(COOTensor(indices, FE_gd.reshape(-1), (tgdof, ))).to_dense()
print("-------------------------")
