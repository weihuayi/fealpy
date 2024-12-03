from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.sparse import COOTensor
from fealpy.solver import cg

class BoxDomainPolyLoaded3d():
    def __init__(self):
        """
        flip_direction = True
        0 ------- 3 ------- 6 
        |    0    |    2    |
        1 ------- 4 ------- 7 
        |    1    |    3    |
        2 ------- 5 ------- 8 
        """
        self.eps = 1e-12

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

        result = bm.zeros(points.shape, 
                        dtype=points.dtype, device=bm.get_device(points))

        return result
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]

        tag1 = bm.abs(x - domain[0]) < self.eps
        tag2 = bm.abs(y - domain[0]) < self.eps
        tag3 = bm.abs(z - domain[0]) < self.eps

        return tag1 & tag2 & tag3
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]

        tag1 = bm.abs(x - domain[0]) < self.eps
        tag2 = bm.abs(y - domain[0]) < self.eps
        tag3 = bm.abs(z - domain[0]) < self.eps

        return tag1 & tag2 & tag3
    @cartesian
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]

        tag1 = bm.abs(x - domain[0]) < self.eps
        tag2 = bm.abs(y - domain[0]) < self.eps
        tag3 = bm.abs(z - domain[0]) < self.eps

        return tag1 & tag2 & tag3
    
    @cartesian
    def threshold(self):

        temp = (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y,
                self.is_dirichlet_boundary_dof_z)
        
        return temp


bm.set_backend('numpy')
nx, ny, nz = 1, 1, 1 
mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], 
                            nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
GD = mesh.geo_dimension()
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
cm = mesh.cell_volume()
node = mesh.entity('node')
cell = mesh.entity('cell')

space = LagrangeFESpace(mesh, p=1, ctype='C')
cell2dof = space.cell_to_dof()

q = 2

# tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # gd_priority
tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority
print(f"dofs = {tensor_space.dof_priority}")
E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu',
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))

integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=q)
KE = integrator_K.assembly(space=tensor_space)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')
# print(f"K.shape = {K.shape}:\n {K.to_dense()}, ")

pde = BoxDomainPolyLoaded3d()

ip1 = mesh.interpolation_points(p=1)
integrator_F = VectorSourceIntegrator(source=pde.source, q=q)
FE = integrator_F.assembly(space=tensor_space)
lform = LinearForm(tensor_space)    
lform.add_integrator(integrator_F)
F = lform.assembly()
print(f"F.shape = {F.shape}:\n {F}, ")

from app.gearx.utils import *

if tensor_space.dof_priority == True:
    F_load_nodes = bm.transpose(F.reshape(3, -1))
else:
    F_load_nodes = F.reshape(NN, GD)
print(f"F_load_nodes.shape = {F_load_nodes.shape}:\n {F_load_nodes}, ")

load_node_indices = cell[0]
fixed_node_index = bm.tensor([0])
export_to_inp(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/box.inp', 
              nodes=node, elements=cell, fixed_nodes=fixed_node_index, load_nodes=load_node_indices, loads=F_load_nodes, 
              young_modulus=206e3, poisson_ratio=0.3, density=7.85e-9)


dbc = DirichletBC(space=tensor_space, 
                    gd=pde.dirichlet, 
                    threshold=pde.threshold(), 
                    method='interp')
K = dbc.apply_matrix(matrix=K, check=True)
# print(f"K.shape = {K.shape}:\n {K.to_dense()}, ")

uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64, device=bm.get_device(mesh))
uh_bd, isDDof = tensor_space.boundary_interpolate(gd=pde.dirichlet, uh=uh_bd, threshold=pde.threshold(), method='interp')

# 2. 修改右端向量
F = F - K.matmul(uh_bd)
F = bm.set_at(F, isDDof, uh_bd[isDDof])
print(f"F.shape = {F.shape}:\n {F}, ")

uh = tensor_space.function()
uh[:] = cg(K, F, maxiter=1000, atol=1e-14, rtol=1e-14)
uh_dof_show = uh.reshape(GD, NN).T
print(f"uh_dof_show.shape = {uh_dof_show.shape}:\n {uh_dof_show}, ")
uh_magnitude = bm.linalg.norm(uh_dof_show, axis=1)
print(f"uh_magnitude = {uh_magnitude}")


qf = mesh.quadrature_formula(1)
bcs, ws = qf.get_quadrature_points_and_weights()
phi = space.basis(bcs) # (1, 1, ldof)
gphi = space.grad_basis(bc=bcs) # (NC, 1, ldof, GD)
B = linear_elastic_material.strain_matrix(dof_priority=True, 
                                        gphi=gphi, shear_order=['xy', 'yz', 'zx']) # (NC, 1, 6, tldof)
print(f"B.shape = {B.shape}:\n {B}, ")
cell2tdof = tensor_space.cell_to_dof()  # (NC, tldof)
tldof = tensor_space.number_of_local_dofs()
uh_cell = bm.zeros((NC, tldof))
for c in range(NC):
    uh_cell[c] = uh[cell2tdof[c]]
print(f"uh_cell.shape = {uh_cell.shape}:\n {uh_cell}, ")
strain = bm.einsum('cqij, cj -> ci', B, uh_cell)  # (NC, 6)
print(f"strain.shape = {strain.shape}:\n {strain}, ")


print("----------------------")