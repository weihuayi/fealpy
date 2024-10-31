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




linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='3D', device=bm.get_device(mesh))

tensor_space_dof = TensorFunctionSpace(space, shape=(3, -1))
tensor_space_gd = TensorFunctionSpace(space, shape=(-1, 3))

gphi = space.grad_basis(bcs) # (NC, NQ, ldof, 3)
gphi_dof = tensor_space_dof.grad_basis(bcs) # (NC, NQ, ldof, 3)

cm = mesh.cell_volume()
B_dof = linear_elastic_material.strain_matrix(dof_priority=True, gphi=gphi, shear_order=['xy', 'yz', 'zx'])
B_gd = linear_elastic_material.strain_matrix(dof_priority=False, gphi=gphi, shear_order=['xy', 'yz', 'zx'])

D = linear_elastic_material.elastic_matrix(bcs)

KK_dof = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B_dof, D, B_dof) # (NC, tldof, tldof)
KK_gd = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B_gd, D, B_gd)

cell = mesh.entity('cell')
cell2dof_dof = tensor_space_dof.cell_to_dof() # (NC, tldof)
cell2dof_gd = tensor_space_gd.cell_to_dof() # (NC, tldof)
tgdof = tensor_space_dof.number_of_global_dofs()


I_dof = bm.broadcast_to(cell2dof_dof[:, :, None], shape=KK_dof.shape)
J_dof = bm.broadcast_to(cell2dof_dof[:, None, :], shape=KK_dof.shape)
K_dof_me = COOTensor(
            indices = bm.empty((2, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, tgdof))
indices = bm.stack([I_dof.ravel(), J_dof.ravel()], axis=0)
K_dof_me = K_dof_me.add(COOTensor(indices, KK_dof.reshape(-1), (tgdof, tgdof))).to_dense()

I_gd = bm.broadcast_to(cell2dof_gd[:, :, None], shape=KK_dof.shape)
J_gd = bm.broadcast_to(cell2dof_gd[:, None, :], shape=KK_dof.shape)
K_gd_me = COOTensor(
            indices = bm.empty((0, 2), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, tgdof))
indices = bm.stack([I_gd.ravel(), J_gd.ravel()], axis=0)
K_gd_me = K_gd_me.add(COOTensor(indices, KK_gd.reshape(-1), (tgdof, tgdof))).to_dense()




integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=2)

bform_gd = BilinearForm(tensor_space_gd)
bform_gd.add_integrator(integrator_K)
K_gd_fealpy = bform_gd.assembly(format='csr').to_dense()


KE_dof = integrator_K.assembly(space=tensor_space_dof)
KE_gd = integrator_K.assembly(space=tensor_space_gd)

bform_dof = BilinearForm(tensor_space_dof)
bform_dof.add_integrator(integrator_K)
K_dof = bform_dof.assembly(format='csr').to_dense()
# save_2d_array_to_txt(K_dof, "/home/heliang/FEALPy_Development/fealpy/app/soptx/mesh/K_dof.txt")

bform_gd = BilinearForm(tensor_space_gd)
bform_gd.add_integrator(integrator_K)
K_gd = bform_gd.assembly(format='csr').to_dense()
# save_2d_array_to_txt(K_gd, "/home/heliang/FEALPy_Development/fealpy/app/soptx/mesh/K_gd.txt")

pde = BoxDomainPolyLoaded3d()

ps = mesh.bc_to_point(bcs)
coef_val = pde.source(ps) # (NC, NQ, GD)

phi_dof = tensor_space_dof.basis(bcs) # (1, NQ, tldof, GD)
phi_dof_test = phi_dof.squeeze(0)
phi_gd = tensor_space_gd.basis(bcs) # (1, NQ, tldof, GD)
phi_gd_test = phi_gd.squeeze(0)

# FE_dof = bm.einsum('q, c, cqid, cqd -> ci', ws, cm, phi_dof, coef_val) # (NC, tldof)

# M_dof = COOTensor(
#             indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
#             values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
#             spshape = (tgdof, ))
# indices = cell2dof_dof.reshape(1, -1)
# M_dof = M_dof.add(COOTensor(indices, FE_dof.reshape(-1), (tgdof, ))).to_dense()


FE_gd = bm.einsum('q, c, cqid, cqd -> ci', ws, cm, phi_gd, coef_val) # (NC, tldof)

M_gd = COOTensor(
            indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, ))
indices = cell2dof_gd.reshape(1, -1)
M_gd = M_gd.add(COOTensor(indices, FE_gd.reshape(-1), (tgdof, ))).to_dense()

integrator_F = VectorSourceIntegrator(source=pde.source, q=2)
# FF_dof = integrator_F.assembly(space=tensor_space_dof) # (NC, tldof)

FF_gd = integrator_F.assembly(space=tensor_space_gd)


lform_dof = LinearForm(tensor_space_dof)    
lform_dof.add_integrator(integrator_F)
# F_dof = lform_dof.assembly()


lform_gd = LinearForm(tensor_space_gd)
lform_gd.add_integrator(integrator_F)
F_gd = lform_gd.assembly()

# error = bm.sum(bm.abs(M_dof - F_dof))


print("-------------------------")



# bform = BilinearForm(tensor_space)
# bform.add_integrator(integrator_K)
# K = bform.assembly(format='csr')

