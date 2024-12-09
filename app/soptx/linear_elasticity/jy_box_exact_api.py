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
from fealpy.solver import cg, spsolve

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

    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:

        result = bm.zeros(points.shape, 
                        dtype=points.dtype, device=bm.get_device(points))

        return result
    
bm.set_backend('numpy')

def compute_equivalent_strain(strain, nu):
    exx = strain[..., 0, 0]
    eyy = strain[..., 1, 1]
    ezz = strain[..., 2, 2]
    gamma_xy = strain[..., 0, 1]
    gamma_yz = strain[..., 1, 2]
    gamma_xz = strain[..., 0, 2]
    
    d1 = exx - eyy
    d2 = eyy - ezz
    d3 = ezz - exx
    
    equiv_strain = (d1**2 + d2**2 + d3**2 + 6.0 * (gamma_xy**2 + gamma_yz**2 + gamma_xz**2))
    
    # equiv_strain = bm.sqrt(equiv_strain / 2.0) / (1.0 + nu)
    equiv_strain = bm.sqrt(equiv_strain / 2.0) / (1.0)
    
    return equiv_strain

def compute_equivalent_stress(stress, nu):
    sxx = stress[..., 0, 0]
    syy = stress[..., 1, 1]
    szz = stress[..., 2, 2]
    sxy = stress[..., 0, 1]
    syz = stress[..., 1, 2]
    sxz = stress[..., 0, 2]
    
    d1 = sxx - syy
    d2 = syy - szz
    d3 = szz - sxx
    
    equiv_stress = (d1**2 + d2**2 + d3**2 + 6.0 * (sxy**2 + syz**2 + sxz**2))

    equiv_stress = bm.sqrt(equiv_stress / 2.0)
    
    return equiv_stress

nx, ny, nz = 4, 4, 4 
mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], 
                            nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
GD = mesh.geo_dimension()
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
cm = mesh.cell_volume()
node = mesh.entity('node')
cell = mesh.entity('cell')

p = 1
# space = LagrangeFESpace(mesh, p=p, ctype='C')
# sgdof = space.number_of_global_dofs()
# print(f"sgdof: {sgdof}")
# cell2dof = space.cell_to_dof()

# q = p+1

# # tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # gd_priority
# tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority
# tgdof = tensor_space.number_of_global_dofs()
# print(f"tgdof: {tgdof}")

E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu',
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=q)

bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
print(f"K.shape = {K.shape}")
print(f"Matrix norm before dc: {K_norm:.6f}")

pde = BoxDomainPolyLoaded3d()
integrator_F = VectorSourceIntegrator(source=pde.source, q=q)
lform = LinearForm(tensor_space)    
lform.add_integrator(integrator_F)
F = lform.assembly()
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"F.shape = {F.shape}")
print(f"Load vector norm before dc: {F_norm:.6f}")

from app.gearx.utils import *

if tensor_space.dof_priority == True:
    F_load_nodes = bm.transpose(F.reshape(3, -1))
else:
    F_load_nodes = F.reshape(NN, GD)
print(f"F_load_nodes.shape = {F_load_nodes.shape}:\n {F_load_nodes}, ")

load_node_indices = cell[0]
isBdDof = tensor_space.is_boundary_dof(threshold=None)
fixed_node_index = bm.where(isBdDof)[0]
export_to_inp(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/box_fealpy.inp', 
              nodes=node, elements=cell, 
              fixed_nodes=fixed_node_index, load_nodes=load_node_indices, loads=F_load_nodes, 
              young_modulus=206e3, poisson_ratio=0.3, density=7.85e-9)


p = 1
E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
errorType = ['$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$']
maxit = 4
errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
NDof = bm.zeros(maxit, dtype=bm.int32)
for i in range(maxit):
    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority
    tgdof = tensor_space.number_of_global_dofs()
    print(f"tgdof: {tgdof}")

    NDof[i] = tensor_space.number_of_global_dofs()

    linear_elastic_material = LinearElasticMaterial(name='E_nu',
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
    tmr.send('material')

    integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=tensor_space.p+3)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_K)
    K = bform.assembly(format='csr')
    # tmr.send('stiffness assembly')

    integrator_F = VectorSourceIntegrator(source=pde.source, q=tensor_space.p+3)
    lform = LinearForm(tensor_space)    
    lform.add_integrator(integrator_F)
    F = lform.assembly()
    tmr.send('source assembly')

    dbc = DirichletBC(space=tensor_space, 
                    gd=pde.dirichlet, 
                    threshold=None, 
                    method='interp')


dbc = DirichletBC(space=tensor_space, 
                    gd=pde.dirichlet, 
                    threshold=None, 
                    method='interp')
K = dbc.apply_matrix(matrix=K, check=True)
print(f"K.shape = {K.shape}")
print(f"Matrix norm after dc: {K_norm:.6f}")
uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64, device=bm.get_device(mesh))
uh_bd, isDDof = tensor_space.boundary_interpolate(gd=pde.dirichlet, uh=uh_bd, threshold=None, method='interp')
F = F - K.matmul(uh_bd)
F = bm.set_at(F, isDDof, uh_bd[isDDof])
print(f"F.shape = {F.shape}")
print(f"Load vector norm after dc: {F_norm:.6f}")

from fealpy import logger
logger.setLevel('INFO')

uh = tensor_space.function()
uh[:] = cg(K, F, maxiter=1000, atol=1e-8, rtol=1e-8)
# uh[:] = spsolve(K, F, solver="mumps")

u_exact = tensor_space.interpolate(pde.solution)

errorMatrix[0, 0] = bm.sqrt(bm.sum(bm.abs(uh[:] - u_exact)**2 * (1 / tgdof)))
errorMatrix[1, 0] = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)
print("errorMatrix:\n", errorType, "\n", errorMatrix)

if tensor_space.dof_priority:
    uh_show = uh.reshape(GD, NN).T
else:
    uh_show = uh.reshape(NN, GD)

uh_magnitude = bm.linalg.norm(uh_show, axis=1)

mesh.nodedata['uh'] = uh_show[:]
mesh.nodedata['uh_magnitude'] = uh_magnitude[:]

mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear_box_fealpy.vtu')

print("----------------------")

