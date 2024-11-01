from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator

from fealpy.fem.bilinear_form import BilinearForm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian

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
nx, ny, nz = 2, 2, 2 
mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
NC = mesh.number_of_cells()

mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/mesh/hexahedron.vtu')


space = LagrangeFESpace(mesh, p=1, ctype='C')

q = 2
qf = mesh.quadrature_formula(q)
bcs, ws = qf.get_quadrature_points_and_weights()
phi = space.basis(bcs)
gphi = space.grad_basis(bc=bcs) # (NC, NQ, 8, 3)



def save_array_to_txt(array, filename):
    """
    保存三维数组到txt文件，保持二维数组格式，每个数字保留8位小数
    array: 要保存的numpy数组
    filename: 文件名
    """
    with open(filename, 'w') as f:
        # 写入数组的形状信息
        f.write(f"# Array shape: {array.shape}\n")
        
        # 逐层保存数据
        for i in range(array.shape[0]):
            f.write(f"# Layer {i}\n")
            for j in range(array.shape[1]):
                # 将每行数据格式化为保留8位小数的字符串，使用制表符分隔
                row_data = '\t'.join(f"{x:.8f}" for x in array[i, j, :])
                f.write(row_data + '\n')
            f.write('\n')  # 每层之间添加空行

tensor_space_dof = TensorFunctionSpace(space, shape=(3, -1))
tensor_space_gd = TensorFunctionSpace(space, shape=(-1, 3))

linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='3D', device=bm.get_device(mesh))

B_dof = linear_elastic_material.strain_matrix(dof_priority=True, gphi=gphi, shear_order=['xy', 'yz', 'zx'])
B_dof_0 = B_dof[0, :, :, :]
save_array_to_txt(B_dof_0, "/home/heliang/FEALPy_Development/fealpy/app/soptx/mesh/B_dof_0.txt")

B_gd = linear_elastic_material.strain_matrix(dof_priority=False, gphi=gphi, shear_order=['xy', 'yz', 'zx'])
B_gd_0 = B_gd[0, :, :, :]
save_array_to_txt(B_gd_0, "/home/heliang/FEALPy_Development/fealpy/app/soptx/mesh/B_gd_0.txt")

# D = linear_elastic_material.elastic_matrix(bcs)
cm = mesh.cell_volume()

# KK_dof = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B_dof, D, B_dof)
# save_array_to_txt(KK_dof, "/home/heliang/FEALPy_Development/fealpy/app/soptx/mesh/KK_dof.txt")

# KK_gd = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B_gd, D, B_gd)
# save_array_to_txt(KK_gd, "/home/heliang/FEALPy_Development/fealpy/app/soptx/mesh/KK_gd.txt")

integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=2)
KE_dof = integrator_K.assembly(space=tensor_space_dof)

bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')

KE_gd = integrator_K.assembly(space=tensor_space_gd)


pde = BoxDomainPolyLoaded3d()

ps = mesh.bc_to_point(bcs)
coef_val = pde.source(ps) # (NC, NQ, GD)

phi_dof = tensor_space_dof.basis(bcs) # (1, NQ, tldof, GD)
phi_dof_test = phi_dof.squeeze(0)
phi_gd = tensor_space_gd.basis(bcs) # (1, NQ, tldof, GD)
phi_gd_test = phi_gd.squeeze(0)

FE_dof = bm.einsum('q, c, cqid, cqd -> ci', ws, cm, phi_dof, coef_val) # (NC, tldof)

FE_gd = bm.einsum('q, c, cqid, cqd -> ci', ws, cm, phi_gd, coef_val) # (NC, tldof)

integrator_F = VectorSourceIntegrator(source=pde.source, q=2)
FF_dof = integrator_F.assembly(space=tensor_space_dof) # (NC, tldof)

FF_gd = integrator_F.assembly(space=tensor_space_gd)


lform = LinearForm(tensor_space)    
# lform.add_integrator(integrator_F



# bform = BilinearForm(tensor_space)
# bform.add_integrator(integrator_K)
# K = bform.assembly(format='csr')

