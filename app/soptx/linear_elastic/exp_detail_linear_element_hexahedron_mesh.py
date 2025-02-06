from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.sparse import COOTensor
from fealpy.solver import cg

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
nx, ny, nz = 3, 3, 3 
mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], 
                            nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))

NC = mesh.number_of_cells()
cm = mesh.cell_volume()
node = mesh.entity('node')
cell = mesh.entity('cell')

space = LagrangeFESpace(mesh, p=1, ctype='C')

q = 2
qf = mesh.quadrature_formula(q)
bcs, ws = qf.get_quadrature_points_and_weights()
phi = space.basis(bcs) # (1, NQ, ldof)
gphi = space.grad_basis(bc=bcs) # (NC, NQ, ldof, GD)

tensor_space = TensorFunctionSpace(space, shape=(3, -1))
tgdof = tensor_space.number_of_global_dofs()
phi_tensor = tensor_space.basis(bcs) # (1, NQ, tldof, GD)
cell2tdof = tensor_space.cell_to_dof() # (NC, tldof)

linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='3D', device=bm.get_device(mesh))

B = linear_elastic_material.strain_matrix(dof_priority=True, 
                                        gphi=gphi, shear_order=['xy', 'yz', 'zx'])
D = linear_elastic_material.elastic_matrix(bcs)
KE = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B, D, B)


I = bm.broadcast_to(cell2tdof[:, :, None], shape=KE.shape)
J = bm.broadcast_to(cell2tdof[:, None, :], shape=KE.shape)
K = COOTensor(
            indices = bm.empty((2, 0), dtype=bm.int32, device=bm.get_device(mesh)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(mesh)),
            spshape = (tgdof, tgdof))
indices = bm.stack([I.ravel(), J.ravel()], axis=0)
K = K.add(COOTensor(indices, KE.reshape(-1), (tgdof, tgdof)))

pde = BoxDomainPolyLoaded3d()
ps = mesh.bc_to_point(bc=bcs)
f = pde.source(ps) # (NC, NQ, GD)
FE = bm.einsum('q, c, cqid, cqd -> ci', ws, cm, phi_tensor, f) # (NC, tldof)

F = COOTensor(
            indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(mesh)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(mesh)),
            spshape = (tgdof, ))
indices = cell2tdof.reshape(1, -1)
F = F.add(COOTensor(indices, FE.reshape(-1), (tgdof, ))).to_dense()

isDDof = tensor_space.is_boundary_dof(threshold=None, method='interp')
kwargs = K.values_context()
# 1. 移除边界自由度相关的非零元素
indices = K.indices()
remove_flag = bm.logical_or(isDDof[indices[0, :]], isDDof[indices[1, :]])
retain_flag = bm.logical_not(remove_flag)
new_indices = indices[:, retain_flag]
new_values = K.values()[..., retain_flag]
K = COOTensor(new_indices, new_values, K.sparse_shape)

# 2. 在边界自由度位置添加单位对角元素
index = bm.nonzero(isDDof)[0]
shape = new_values.shape[:-1] + (len(index), )
one_values = bm.ones(shape, **kwargs)
one_indices = bm.stack([index, index], axis=0)
K1 = COOTensor(one_indices, one_values, K.sparse_shape)
K = K.add(K1).coalesce()

# 1. 边界插值
uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), 
                dtype=bm.float64, device=bm.get_device(mesh))
uh_bd, isDDof = tensor_space.boundary_interpolate(gd=pde.dirichlet, uh=uh_bd, 
                                                threshold=None, method='interp')
# 2. 修改右端向量
F = F - K.matmul(uh_bd)
F = bm.set_at(F, isDDof, uh_bd[isDDof])

uh = tensor_space.function()
uh[:] = cg(K, F, maxiter=1000, atol=1e-14, rtol=1e-14)



u_exact = tensor_space.interpolate(pde.solution)
error = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)
print("----------------------")
