from fealpy.backend import backend_manager as bm
from fealpy.cfd.equation import IncompressibleNS
from fealpy.cfd.equation import CahnHilliard
from fealpy.cfd.simulation.fem.incompressible_ns import BDF2
from fealpy.cfd.simulation.fem import CahnHilliardModel 
from fealpy.mmesh.pde import RayleignTaylor
from fealpy.decorator import barycentric
from fealpy.fem import DirichletBC
from fealpy.functionspace import TensorFunctionSpace,LagrangeFESpace

from fealpy.mmesh import MMesher,Config

from fealpy.solver import spsolve
import psutil
import time 

class two_phase_phield_solver:
    def __init__(self, pde):
        self.pde = pde

    def update_space(self, mesh, up=2, pp=1):
        '''
        更新空间
        '''
        usspace = LagrangeFESpace(mesh, p=up)
        uspace = TensorFunctionSpace(usspace, (mesh.GD,-1))
        pspace = LagrangeFESpace(mesh, p=pp)
        return pspace, usspace, uspace
    
    def update_function(self, phi):
        epsilon = self.epsilon
        H = self.heaviside(phi)
        rho = phi.space.function()
        eta = phi.space.function()
        rho[:] = self.pde.gas.rho/self.pde.liquid.rho + \
                (1 - self.pde.gas.rho/self.pde.liquid.rho) * H[:]
        eta[:] = self.pde.gas.eta/self.pde.liquid.eta + \
                (1 - self.pde.gas.eta/self.pde.liquid.eta) * H[:]
        return rho, eta
    
    def show_mesh(self,mesh):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        plt.show()

    
    def rho(self, phi):
        result = phi.space.function()
        result[:] = (self.pde.rho_up - self.pde.rho_down)/2 * phi[:]
        result[:] += (self.pde.rho_up + self.pde.rho_down)/2 
        return result

bm.set_backend('numpy')
# bm.set_default_device('cpu')

# dt = 0.00125*bm.sqrt(bm.array(2))
dt = 0.005 * 0.16 * bm.sqrt(bm.array(2))
print(dt)
pde = RayleignTaylor()
domain = pde.domain()

mesh = pde.init_moving_mesh(nx=40, ny=80)

ns_eqaution = IncompressibleNS(pde,init_variables=False)
ns_solver = BDF2(ns_eqaution,mesh)
ns_solver.dt = dt

phispace = ns_solver.uspace.scalar_space
# phispace = LagrangeFESpace(mesh, p=2)

ch_equation = CahnHilliard(pde, init_variables=False)
ch_solver = CahnHilliardModel(ch_equation, phispace)
ch_solver.dt = dt

solver = two_phase_phield_solver(pde) 


phi0 = phispace.interpolate(pde.init_interface)
phi1 = phispace.interpolate(pde.init_interface)
phi2 = phispace.function()
mu1 = phispace.function()
mu2 = phispace.function()

u0 = ns_solver.uspace.function()
u1 = ns_solver.uspace.function()
u2 = ns_solver.uspace.function()
p1 = ns_solver.pspace.function()
p2 = ns_solver.pspace.function()

config = Config()
config.active_method = 'MetricTensorAdaptive'
config.is_pre = True
config.pde = pde
config.mol_times = 8
config.pre_steps = 10
config.alpha = 0.5
config.tau = 0.005
config.t_max = 0.5
config.gamma = 1.25
mm = MMesher(mesh,uh=phi0, space=phispace,beta=0.9, config=config)
mm.initialize()
mm.set_interpolation_method('solution')
mm.set_monitor('matrix_normal')
mm.set_mol_method('constant_projector')
smspace = mm.instance.mspace
mspace = TensorFunctionSpace(smspace, (mesh.GD,-1))
move_vector = mspace.function()

mm.run()
mm.set_interpolation_method('linear')
x0 = mesh.node.copy()
x1 = mesh.node.copy()
x2 = mesh.node.copy()
phi0 = phispace.interpolate(pde.init_interface)
phi1 = phispace.interpolate(pde.init_interface)


mesh.nodedata['phi'] = phi1
mesh.nodedata['velocity'] = u1.reshape(2,-1).T  
fname = './' + 'test_'+ str(1).zfill(10) + '.vtu'
mesh.to_vtk(fname=fname)

ugdof = ns_solver.uspace.number_of_global_dofs()
phigdof = phispace.number_of_global_dofs()
pgdof = ns_solver.pspace.number_of_global_dofs()

ns_BForm = ns_solver.BForm()
ns_LForm = ns_solver.LForm()
ch_BFrom = ch_solver.BForm()
ch_LForm = ch_solver.LForm()
nsx = None
ch_x = None
is_bd = ns_solver.uspace.is_boundary_dof((pde.is_ux_boundary, pde.is_uy_boundary), method='interp')
is_bd = bm.concatenate((is_bd, bm.zeros(pgdof, dtype=bm.bool)))
gd = bm.concatenate((bm.zeros(ugdof, dtype=bm.float64), bm.zeros(pgdof, dtype=bm.float64)))
BC = DirichletBC((ns_solver.uspace, ns_solver.pspace), gd=gd, threshold=is_bd, method='interp')

#BC = DirichletBC((ns_solver.uspace,ns_solver.pspace), gd=(pde.velocity_boundary, pde.pressure_boundary), 
#                      threshold=(pde.is_u_boundary, pde.is_p_boundary), method='interp')

#设置参数
ns_eqaution.set_coefficient('viscosity', 1/pde.Re)
ns_eqaution.set_coefficient('pressure', 1)
ch_equation.set_coefficient('mobility', 1/pde.Pe)
ch_equation.set_coefficient('interface', pde.epsilon**2)
ch_equation.set_coefficient('free_energy', 1)

print(ns_solver.uspace.number_of_global_dofs(), ns_solver.pspace.number_of_global_dofs())
#mgr = DirectSolverManager()
for i in range(1,2500):
    if i>1:
        x0 = x1.copy()
        x1 = x2.copy()
        mm.run()
        x2 = mesh.node.copy()
    move_vector[:] = ((3 * x2 - 4 * x1 + x0)/(2*dt)).T.flatten()
    # 设置参数
    print("iteration:", i)
    print("内存占用",psutil.Process().memory_info().rss / 1024 ** 2, "MB")  # RSS内存(MB)    
    
    t0 = time.time()
    ch_solver.update(u0, u1, phi0, phi1,mv=move_vector)
    ch_A = ch_BFrom.assembly()
    ch_b = ch_LForm.assembly()
    t1 = time.time()
    # ch_x = cg(ch_A, ch_b,x0 = ch_x, atol=1e-10)
    ch_x = spsolve(ch_A, ch_b, 'scipy')
    t2 = time.time()

    phi2[:] = ch_x[:phigdof]
    mu2[:] = ch_x[phigdof:]  
    
    # 更新NS方程参数
    t3 = time.time()
    rho = solver.rho(phi1) 
    @barycentric
    def body_force(bcs, index):
        result = rho(bcs, index)
        result = bm.stack((result, result), axis=-1)
        result[..., 0] = (1/pde.Fr) * result[..., 0] * 0
        result[..., 1] = (1/pde.Fr) * result[..., 1] * -1
        return result
    
    ns_eqaution.set_coefficient('time_derivative', rho)
    ns_eqaution.set_coefficient('convection', rho)
    ns_eqaution.set_coefficient('body_force', body_force)

    ns_solver.update(u0, u1,mv=move_vector)
     
    ns_A = ns_BForm.assembly()
    ns_b = ns_LForm.assembly()
    ns_A,ns_b = BC.apply(ns_A, ns_b)
    t4 = time.time() 
    # ns_x = cg(ns_A , ns_b,x0=nsx, atol=1e-10)
    ns_x = spsolve(ns_A, ns_b,'scipy')
    t5 = time.time()

    print("CH组装时间:", t1-t0)
    print("求解CH方程时间:", t2-t1)
    print("NS组装时间:", t4-t3)
    print("求解NS方程时间:", t5-t4)
    u2[:] = ns_x[:ugdof]
    p2[:] = ns_x[ugdof:]
    nsx = ns_x
    u0[:] = u1[:]
    u1[:] = u2[:]
    phi0[:] = phi1[:]
    phi1[:] = phi2[:]
    mu1[:] = mu2[:]
    p1[:] = p2[:]
    
    mm.update_solution(phi2)
    mesh.nodedata['phi'] = phi2
    mesh.nodedata['velocity'] = u2.reshape(2,-1).T  
    mesh.nodedata['pressure'] = p2 
    mesh.nodedata['rho'] = rho
    fname = './' + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.to_vtk(fname=fname)