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
    
def compute_equivalent_strain(strain, nu):
    """
    Calculate equivalent strain for elements
    
    Parameters
    ----------
    strain : array
        Element strain with shape (NC, 6) [εxx, εyy, εzz, γxy, γyz, γxz]
    nu : float
        Poisson's ratio
        
    Returns
    -------
    equiv_strain : array
        Element equivalent strain (NC, 1)
    """
    # Extract strain components
    exx = strain[..., 0, 0]
    eyy = strain[..., 1, 1]
    ezz = strain[..., 2, 2]
    gamma_xy = strain[..., 0, 1]
    gamma_yz = strain[..., 1, 2]
    gamma_xz = strain[..., 0, 2]
    
    # Normal strain differences
    d1 = exx - eyy
    d2 = eyy - ezz
    d3 = ezz - exx
    
    # Combine all terms
    equiv_strain = (d1**2 + d2**2 + d3**2 + 6.0 * (gamma_xy**2 + gamma_yz**2 + gamma_xz**2))
    
    # Final computation with Poisson's ratio factor and square root
    equiv_strain = bm.sqrt(equiv_strain / 2.0) / (1.0 + nu)
    
    return equiv_strain.reshape(-1, 1)
    
def compute_element_equivalent_strain(strain, nu):
    """
    Calculate equivalent strain for elements
    
    Parameters
    ----------
    strain : array
        Element strain with shape (NC, 6) [εxx, εyy, εzz, γxy, γyz, γxz]
    nu : float
        Poisson's ratio
        
    Returns
    -------
    equiv_strain : array
        Element equivalent strain (NC, 1)
    """
    # Extract strain components
    exx = strain[..., 0]
    eyy = strain[..., 1]
    ezz = strain[..., 2]
    gamma_xy = strain[..., 3]
    gamma_yz = strain[..., 4]
    gamma_xz = strain[..., 5]
    
    # Normal strain differences
    d1 = exx - eyy
    d2 = eyy - ezz
    d3 = ezz - exx
    
    # Combine all terms
    equiv_strain = (d1**2 + d2**2 + d3**2 + 6.0 * (gamma_xy**2 + gamma_yz**2 + gamma_xz**2))
    
    # Final computation with Poisson's ratio factor and square root
    equiv_strain = bm.sqrt(equiv_strain / 2.0) / (1.0 + nu)
    
    return equiv_strain.reshape(-1, 1)


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

###########################333
# bcs1 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64),)
# p1 = mesh.bc_to_point(bc=bcs1)
# bcs2 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64),)
# p2 = mesh.bc_to_point(bc=bcs2)
# bcs3 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64),)
# p3 = mesh.bc_to_point(bc=bcs3)
# bcs4 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64),)
# p4 = mesh.bc_to_point(bc=bcs4)
# bcs5 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64),)
# p5 = mesh.bc_to_point(bc=bcs5)
# bcs6 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64),)
# p6 = mesh.bc_to_point(bc=bcs6)
# bcs7 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64),)
# p7 = mesh.bc_to_point(bc=bcs7)
# bcs8 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64),)
# p8 = mesh.bc_to_point(bc=bcs8)

bcs1 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
p1 = mesh.bc_to_point(bc=bcs1)
bcs5 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
p5 = mesh.bc_to_point(bc=bcs5)
bcs7 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
p7 = mesh.bc_to_point(bc=bcs7)
bcs3 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
p3 = mesh.bc_to_point(bc=bcs3)
bcs2 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
p2 = mesh.bc_to_point(bc=bcs2)
bcs6 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
p6 = mesh.bc_to_point(bc=bcs6)
bcs8 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
p8 = mesh.bc_to_point(bc=bcs8)
bcs4 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
p4 = mesh.bc_to_point(bc=bcs4)

test0 = space.function()
test0[0] = 1
print(f"1:{p1},  {test0(bcs1)}, {test0(bcs2)}, {test0(bcs3)}, {test0(bcs4)}, {test0(bcs5)}, {test0(bcs6)}, {test0(bcs7)}, {test0(bcs8)}")

test1 = space.function()
test1[1] = 1
print(f"2:{p2},  {test1(bcs2)}, {test1(bcs1)}, {test1(bcs3)}, {test1(bcs4)}, {test1(bcs5)}, {test1(bcs6)}, {test1(bcs7)}, {test1(bcs8)}")

test2 = space.function()
test2[2] = 1
print(f"3:{p3},  {test2(bcs3)}, {test2(bcs1)}, {test2(bcs2)}, {test2(bcs4)}, {test2(bcs5)}, {test2(bcs6)}, {test2(bcs7)}, {test2(bcs8)}")

test3 = space.function()
test3[3] = 1
print(f"4:{p4},  {test3(bcs4)}, {test3(bcs1)}, {test3(bcs2)}, {test3(bcs3)}, {test3(bcs5)}, {test3(bcs6)}, {test3(bcs7)}, {test3(bcs8)}")

test4 = space.function()
test4[4] = 1
print(f"5:{p5},  {test4(bcs5)}, {test4(bcs1)}, {test4(bcs2)}, {test4(bcs3)}, {test4(bcs4)}, {test4(bcs6)}, {test4(bcs7)}, {test4(bcs8)}")

test5 = space.function()
test5[5] = 1
print(f"6:{p6},  {test5(bcs6)}, {test5(bcs1)}, {test5(bcs2)}, {test5(bcs3)}, {test5(bcs4)}, {test5(bcs5)}, {test5(bcs7)}, {test5(bcs8)}")

test6 = space.function()
test6[6] = 1
print(f"7:{p7},  {test6(bcs7)}, {test6(bcs1)}, {test6(bcs2)}, {test6(bcs3)}, {test6(bcs4)}, {test6(bcs5)}, {test6(bcs6)}, {test6(bcs8)}")

test7 = space.function()
test7[7] = 1
print(f"8:{p8},  {test7(bcs8)}, {test7(bcs1)}, {test7(bcs2)}, {test7(bcs3)}, {test7(bcs4)}, {test7(bcs5)}, {test7(bcs6)}, {test7(bcs7)}")
##############3

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
export_to_inp(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/box_fealpy.inp', 
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
print(f"uh.shape = {uh.shape}:\n {uh[:]}, ")

if tensor_space.dof_priority:
    uh_show = uh.reshape(GD, NN).T
else:
    uh_show = uh.reshape(NN, GD)

print(f"uh_show.shape = {uh_show.shape}:\n {uh_show}, ")
uh_magnitude = bm.linalg.norm(uh_show, axis=1)
print(f"uh_magnitude = {uh_magnitude}")

uh_ansys = tensor_space.function()
uh_ansys2 = tensor_space.function()
ux = bm.array([0.0, 1.1348e-5, 2.0691e-6, 1.3418e-5, 
               -4.345e-6, 7.0035e-6, 4.2098e-7, 1.1769e-5])

uy = bm.array([0.0, -8.7815e-6, 7.2734e-6, -1.4257e-5,
               2.9569e-6, -1.8573e-5, -2.5185e-6, -1.13e-5])

uz = bm.array([0.0, 5.3937e-6, 1.9651e-5, 1.6055e-5,
               -1.1348e-5, -5.9547e-6, 8.3022e-6, 4.7064e-6])

uh_ansys_show = bm.stack([ux, uy, uz], axis=1)  # (NN, GD)
map = [0, 4, 6, 2, 1, 5, 7, 3]
uh_ansys_show2 = uh_ansys_show[map, ] # (NN, GD)

if tensor_space.dof_priority:
    uh_ansys[:] = uh_ansys_show.T.flatten() 
    uh_ansys2[:] = uh_ansys_show2.T.flatten()
else:
    uh_ansys[:] = uh_ansys_show2.flatten() 
    uh_ansys2[:] = uh_ansys_show.flatten()

print(f"uh_ansys_show = {uh_ansys_show.shape}:\n {uh_ansys_show}, ")
print(f"uh_ansys_show2 = {uh_ansys_show2.shape}:\n {uh_ansys_show2}, ")
print(f"uh_ansys = {uh_ansys.shape}:\n {uh_ansys[:]}, ")
print(f"uh_ansys2 = {uh_ansys2.shape}:\n {uh_ansys2[:]}, ")

bcs1 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
p1 = mesh.bc_to_point(bc=bcs1)
bcs2 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
p2 = mesh.bc_to_point(bc=bcs2)
bcs3 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
p3 = mesh.bc_to_point(bc=bcs3)
bcs4 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
p4 = mesh.bc_to_point(bc=bcs4)
bcs5 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
p5 = mesh.bc_to_point(bc=bcs5)
bcs6 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
p6 = mesh.bc_to_point(bc=bcs6)
bcs7 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
p7 = mesh.bc_to_point(bc=bcs7)
bcs8 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
p8 = mesh.bc_to_point(bc=bcs8)

gphi1 = tensor_space.grad_basis(bcs1) # (NC, 1, tldof, GD, GD)
gphi2 = tensor_space.grad_basis(bcs2) # (NC, 1, ldof, GD)
gphi3 = tensor_space.grad_basis(bcs3) # (NC, 1, ldof, GD)
gphi4 = tensor_space.grad_basis(bcs4) # (NC, 1, ldof, GD)
gphi5 = tensor_space.grad_basis(bcs5) # (NC, 1, ldof, GD)
gphi6 = tensor_space.grad_basis(bcs6) # (NC, 1, ldof, GD)
gphi7 = tensor_space.grad_basis(bcs7) # (NC, 1, ldof, GD)
gphi8 = tensor_space.grad_basis(bcs8) # (NC, 1, ldof, GD)

cell2tdof = tensor_space.cell_to_dof()  # (NC, tldof)
tldof = tensor_space.number_of_local_dofs()

uh_ansys_cell = bm.zeros((NC, tldof))
uh_ansys_cell2 = bm.zeros((NC, tldof))
for c in range(NC):
    uh_ansys_cell[c] = uh_ansys[cell2tdof[c]]
    uh_ansys_cell2[c] = uh_ansys2[cell2tdof[c]]
print(f"uh_ansys_cell.shape = {uh_ansys_cell.shape}:\n {uh_ansys_cell}, ")
print(f"uh_ansys_cell2.shape = {uh_ansys_cell2.shape}:\n {uh_ansys_cell2}, ")

uh_cell = bm.zeros((NC, tldof))
for c in range(NC):
    uh_cell[c] = uh[cell2tdof[c]]
print(f"uh_cell.shape = {uh_cell.shape}:\n {uh_cell}, ")



grad1 = bm.einsum('cqimn, ci -> cqmn', gphi1, uh_cell)  # (1, 1, GD, GD)
strain1 = (grad1 + bm.transpose(grad1, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)
# grad1_map = bm.einsum('cqimn, ci -> cqmn', gphi1, uh_cell2)  # (1, 1, GD, GD)
# strain1_map = (grad1_map + bm.transpose(grad1_map, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)

grad2 = bm.einsum('cqimn, ci -> cqmn', gphi2, uh_cell)  # (1, 1, GD, GD)
strain2 = (grad2 + bm.transpose(grad2, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)
# grad2_map = bm.einsum('cqimn, ci -> cqmn', gphi1, uh_cell2)  # (1, 1, GD, GD)
# strain2_map = (grad2_map + bm.transpose(grad2_map, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)

grad3 = bm.einsum('cqimn, ci -> cqmn', gphi3, uh_cell)  # (1, 1, GD, GD)
strain3 = (grad3 + bm.transpose(grad3, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)
# grad3_map = bm.einsum('cqimn, ci -> cqmn', gphi3, uh_cell2)  # (1, 1, GD, GD)
# strain3_map = (grad3_map + bm.transpose(grad3_map, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)

grad4 = bm.einsum('cqimn, ci -> cqmn', gphi4, uh_cell)  # (1, 1, GD, GD)
strain4 = (grad4 + bm.transpose(grad4, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)
# grad4_map = bm.einsum('cqimn, ci -> cqmn', gphi4, uh_cell2)  # (1, 1, GD, GD)
# strain4_map = (grad4_map + bm.transpose(grad4_map, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)

grad5 = bm.einsum('cqimn, ci -> cqmn', gphi5, uh_cell)  # (1, 1, GD, GD)
strain5 = (grad5 + bm.transpose(grad5, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)
# grad5_map = bm.einsum('cqimn, ci -> cqmn', gphi5, uh_cell2)  # (1, 1, GD, GD)
# strain5_map = (grad5_map + bm.transpose(grad5_map, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)

grad6 = bm.einsum('cqimn, ci -> cqmn', gphi6, uh_cell)  # (1, 1, GD, GD)
strain6 = (grad6 + bm.transpose(grad6, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)
# grad6_map = bm.einsum('cqimn, ci -> cqmn', gphi6, uh_cell2)  # (1, 1, GD, GD)
# strain6_map = (grad6_map + bm.transpose(grad6_map, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)

grad7 = bm.einsum('cqimn, ci -> cqmn', gphi7, uh_cell)  # (1, 1, GD, GD)
strain7 = (grad7 + bm.transpose(grad7, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)
# grad7_map = bm.einsum('cqimn, ci -> cqmn', gphi7, uh_cell2)  # (1, 1, GD, GD)
# strain7_map = (grad7_map + bm.transpose(grad7_map, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)

grad8 = bm.einsum('cqimn, ci -> cqmn', gphi8, uh_cell)  # (1, 1, GD, GD)
strain8 = (grad8 + bm.transpose(grad8, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)
# grad8_map = bm.einsum('cqimn, ci -> cqmn', gphi8, uh_cell2)  # (1, 1, GD, GD)
# strain8_map = (grad8_map + bm.transpose(grad8_map, (0, 1, 3, 2))) / 2  # (1, 1, GD, GD)

# strain3 = bm.einsum('cqimn, ci -> cqmn', gphi3, uh_cell)  # (1, 1, GD, GD)
# strain4 = bm.einsum('cqimn, ci -> cqmn', gphi4, uh_cell)  # (1, 1, GD, GD)
# strain5 = bm.einsum('cqimn, ci -> cqmn', gphi5, uh_cell)  # (1, 1, GD, GD)
# strain6 = bm.einsum('cqimn, ci -> cqmn', gphi6, uh_cell)  # (1, 1, GD, GD)
# strain7 = bm.einsum('cqimn, ci -> cqmn', gphi7, uh_cell)  # (1, 1, GD, GD)
# strain8 = bm.einsum('cqimn, ci -> cqmn', gphi8, uh_cell)  # (1, 1, GD, GD)

strian_test = bm.stack([strain1, strain2, strain3, strain4, strain5, strain6, strain7, strain8], axis=1)  # (1, 8, GD, GD)

equiv_strain1= compute_equivalent_strain(strian_test, nu)
print(f"equiv_strain1.shape = {equiv_strain1.shape}:\n {equiv_strain1}, ")

B = linear_elastic_material.strain_matrix(dof_priority=True, 
                                        gphi=gphi, shear_order=['xy', 'yz', 'zx']) # (NC, 1, 6, tldof)
print(f"B.shape = {B.shape}:\n {B}, ")


uh_cell = bm.zeros((NC, tldof))
for c in range(NC):
    uh_cell[c] = uh[cell2tdof[c]]
print(f"uh_cell.shape = {uh_cell.shape}:\n {uh_cell}, ")
strain = bm.einsum('cqij, cj -> ci', B, uh_cell)  # (NC, 6)
print(f"strain.shape = {strain.shape}:\n {strain}, ")


uh_ansys_cell = bm.zeros((NC, tldof))
for c in range(NC):
    uh_ansys_cell[c] = uh_ansys[cell2tdof[c]]
print(f"uh_ansys_cell.shape = {uh_ansys_cell.shape}:\n {uh_ansys_cell}, ")
strain_ansys = bm.einsum('cqij, cj -> ci', B, uh_ansys_cell)  # (NC, 6)
print(f"strain_ansys.shape = {strain_ansys.shape}:\n {strain_ansys}, ")

nodal_strain_ansys = bm.zeros((NN, 6))
weights_ansys = bm.zeros((NN, 1))

for c in range(NC):
    for n in range(8):
        node_id = cell[c, n] 
        for i in range(6):
            nodal_strain_ansys[node_id, i] += strain_ansys[c, i]
        
        weights_ansys[node_id, 0] += 1.0

# Compute averages at nodes
for n in range(NN):
    if weights_ansys[n, 0] > 0:
        for i in range(6):
            nodal_strain_ansys[n, i] /= weights_ansys[n, 0]

print(f"ANSYS nodal strain shape = {nodal_strain_ansys.shape}")
print(f"ANSYS nodal strain = \n{nodal_strain_ansys}")

element_equiv_strain_ansys = compute_element_equivalent_strain(strain_ansys, nu)
print(f"ANSYS element equivalent strain shape = {element_equiv_strain_ansys.shape}")
print(f"ANSYS element equivalent strain : \n{element_equiv_strain_ansys}")

nodal_equiv_strain_ansys = compute_element_equivalent_strain(nodal_strain_ansys, nu)
print(f"ANSYS nodal equivalent strain shape = {nodal_equiv_strain_ansys.shape}")
print(f"ANSYS nodal equivalent strain : \n{nodal_equiv_strain_ansys}")


print("----------------------")

