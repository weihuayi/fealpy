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
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.solver import cg, spsolve

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

bm.set_backend('numpy')
nx, ny, nz = 4, 4, 4 
mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz)
NC = mesh.number_of_cells()

# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()


space = LagrangeFESpace(mesh, p=1, ctype='C')
q = 3
qf = mesh.quadrature_formula(q)
bcs, ws = qf.get_quadrature_points_and_weights()
phi = space.basis(bcs)


def save_2d_array_to_txt(array, filename):
    """
    保存二维数组到txt文件，每个数字保留6位小数
    array: 要保存的numpy数组
    filename: 文件名
    """
    with open(filename, 'w') as f:
        # 写入数组的形状信息
        f.write(f"# Array shape: {array.shape}\n")
        
        # 逐行保存数据
        for i in range(array.shape[0]):
            # 将每行数据格式化为保留6位小数的字符串，使用制表符分隔
            row_data = '\t'.join(f"{x:.6f}" for x in array[i, :])
            f.write(row_data + '\n')


linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='3D', device=bm.get_device(mesh))

tensor_space_dof = TensorFunctionSpace(space, shape=(3, -1))

gphi = space.grad_basis(bcs) # (NC, NQ, ldof, 3)
gphi_dof = tensor_space_dof.grad_basis(bcs) # (NC, NQ, ldof, 3)

cm = mesh.cell_volume()

cell = mesh.entity('cell')
cel2dof = space.cell_to_dof()
cell2dof_dof = tensor_space_dof.cell_to_dof() # (NC, tldof)
tgdof = tensor_space_dof.number_of_global_dofs()

integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=2)

bform_dof = BilinearForm(tensor_space_dof)
bform_dof.add_integrator(integrator_K)
K_dof = bform_dof.assembly(format='coo')
K_dof_dense = K_dof.to_dense()


def is_dirichlet_boundary_dof(points: TensorLike) -> TensorLike:
        domain = [0, 1, 0, 1, 0, 1]
        eps = 1e-12
        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < eps
        
        return coord

node = mesh.entity('node')
isDDof = space.is_boundary_dof(threshold=is_dirichlet_boundary_dof, method='interp')
isDDof_dof = tensor_space_dof.is_boundary_dof(threshold=is_dirichlet_boundary_dof, method='interp')

pde = BoxDomainPolyLoaded3d()
dbc_dof = DirichletBC(space=tensor_space_dof, 
                    gd=pde.dirichlet, 
                    threshold=isDDof_dof, 
                    method='interp')
K_after_dof_me = dbc_dof.apply_matrix(matrix=K_dof, check=True)
K_after_dof_me_dense = K_after_dof_me.to_dense()
save_2d_array_to_txt(K_after_dof_me_dense, 
                    "/home/heliang/FEALPy_Development/fealpy/app/soptx/mesh/K_after_dof_me_dense.txt")



kwargs = K_dof.values_context()
indices = K_dof.indices()
remove_flag = bm.logical_or(
                isDDof_dof[indices[0, :]], isDDof_dof[indices[1, :]]
            )
retain_flag = bm.logical_not(remove_flag)
new_indices = indices[:, retain_flag]
new_values = K_dof.values()[..., retain_flag]
K1_after_dof_fealpy = COOTensor(new_indices, new_values, K_dof.sparse_shape)

index = bm.nonzero(isDDof_dof)[0]
shape = new_values.shape[:-1] + (len(index), )
one_values = bm.ones(shape, **kwargs)
one_indices = bm.stack([index, index], axis=0)
K2_after_dof_fealpy = COOTensor(one_indices, one_values, K_dof.sparse_shape)
K1_after_dof_fealpy = K1_after_dof_fealpy.add(K2_after_dof_fealpy).coalesce()


integrator_F = VectorSourceIntegrator(source=pde.source, q=3)
FE = integrator_F.assembly(tensor_space_dof)
lform_dof = LinearForm(tensor_space_dof)    
lform_dof.add_integrator(integrator_F)
F_dof = lform_dof.assembly()

uh_bd_dof = bm.zeros(tensor_space_dof.number_of_global_dofs(), 
                    dtype=bm.float64, device=bm.get_device(mesh))
uh_bd_dof, _ = tensor_space_dof.boundary_interpolate(gd=pde.dirichlet, uh=uh_bd_dof, 
                                                threshold=None, method='interp')
F_dof = F_dof - K_dof.matmul(uh_bd_dof)
F_dof = bm.set_at(F_dof, isDDof_dof, uh_bd_dof[isDDof_dof])

print("-------------------------")



# bform = BilinearForm(tensor_space)
# bform.add_integrator(integrator_K)
# K = bform.assembly(format='csr')

