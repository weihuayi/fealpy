from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_bend_turbulent_flow import PipeBendTurbulentFlow
from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_geo_mesh import PipeGeometry, PipeMesh
from fealpy.cfd.equation import StationaryIncompressibleRANS, StationaryTurbulentKineticEnergy, StationarySpecificDissipationRate
from fealpy.cfd.simulation.fem.stationary_sst_k_omega.stationary_incompressible_rans import Ossen
from fealpy.cfd.simulation.fem.stationary_sst_k_omega.stationary_turbulent_kinetic_energy import StationaryTurbulentKineticEnergyPicard
from fealpy.cfd.simulation.fem.stationary_sst_k_omega.stationary_specific_dissipation_rate import StationarySpecificDissipationRatePicard
import matplotlib.pyplot as plt
from fealpy.functionspace import LagrangeFESpace
from fealpy.solver import cg, spsolve, minres, gmres, bicgstab
from fealpy.backend import backend_manager as bm
from fealpy.fem import DirichletBC, LinearForm, BlockForm, SourceIntegrator
from fealpy.sparse import COOTensor
from fealpy.mesher import ElbowPipeMesher

# geom = PipeGeometry()
# geom.build()
# mesher = PipeMesh(geom, mesh_size=0.3)
# mesh = mesher.generate_mesh()

params = {
    "D": 1.0,                     # 管道内径 1.0 m (对应半径 0.5 m)
    "bend_angle": 90.0,           # 90度弯曲
    "R_bend_inner": 2.3,          # 使得中心曲率半径 Rc = (2.3 + 0.5) * D = 2.8D
    "L_in_ratio": 10.0,           # 上游直管段 10m / 1m = 10.0
    "L_out_ratio": 15.0,          # 下游直管段 15m / 1m = 15.0
    "wall_thickness": 0.05,       # 报告未给定，基于1m管径假定一个合理值 (如 50mm)
    "mesh_size_global": 0.3,     # 使用默认网格大小策略
    "mesh_size_bend": 0.3,
    "mesh_size_interface": 0.3,
}
mesher = ElbowPipeMesher(params)
mesh = mesher.init_mesh()
mesh.to_vtk("pipe_bend_mesh.vtu")


# 网格可视化
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh.add_plot(axes)
# plt.savefig("1.png")

pde = PipeBendTurbulentFlow()
eq = StationaryIncompressibleRANS(pde = pde)
eq_k = StationaryTurbulentKineticEnergy(pde = pde)
eq_omega = StationarySpecificDissipationRate(pde = pde)
fem = Ossen(equation = eq, mesh = mesh)
fem_k = StationaryTurbulentKineticEnergyPicard(equation = eq_k, mesh = mesh)
fem_omega = StationarySpecificDissipationRatePicard(equation = eq_omega, mesh = mesh)

uspace = fem.uspace
pspace = fem.pspace
kspace = LagrangeFESpace(mesh, p=2)
omegaspace = LagrangeFESpace(mesh, p=2)

u0 = uspace.function()
u1 = uspace.function()
p0 = pspace.function()
p1 = pspace.function()
k0 = kspace.function()
k1 = kspace.function()
omega0 = omegaspace.function()
omega1 = omegaspace.function()

k0[:] = 0.00292 * bm.ones(k0.shape)
omega0[:] = 1.41 * bm.ones(omega0.shape)
ugdof = uspace.number_of_global_dofs()

# 加入亚松弛因子 alpha
alpha_u = 0.5  # 速度松弛因子 (通常 0.3 - 0.7)
alpha_p = 0.3  # 压力松弛因子 (通常 0.2 - 0.5)
alpha_k_omega = 0.5 # 湍流变量松弛因子

for i in range(1000):
    print(f"第{i}步")
    # rans 方程求解
    BForm = fem.BForm()
    LForm = fem.LForm()
    fem.update(u0=u0, k0=k0, omega0=omega0)
    A = BForm.assembly()
    b = LForm.assembly()
    A, b = fem.apply_bc(A, b, pde=pde)
    # A, b = fem.lagrange_multiplier(A, b)
    x = spsolve(A, b)
    u1[:] = x[:ugdof]
    # p1[:] = x[ugdof:-1]
    p1[:] = x[ugdof:]

    res_u = mesh.error(u0, u1)
    res_p = mesh.error(p0, p1)
    print(f"res_u", res_u)
    print(f"res_p", res_p)
    if res_u + res_p < 1e-8:
        break   

    # 亚松弛更新
    u1[:] = alpha_u * u1[:] + (1 - alpha_u) * u0[:]
    p1[:] = alpha_p * p1[:] + (1 - alpha_p) * p0[:]
    mesh.nodedata["uh"] = u1.reshape(3, -1).T
    mesh.nodedata["ph"] = p1
    
    # 湍动能方程求解
    BForm_k = fem_k.BForm()
    LForm_k = fem_k.LForm()
    fem_k.update(u1=u1, k0 = k0, omega0=omega0, mu_t=fem.mu_t)
    A_k = BForm_k.assembly()
    b_k = LForm_k.assembly()
    BC_k = DirichletBC(
        fem_k.kspace,
        gd=pde.k_dirichlet,
        threshold=pde.is_k_boundary,
        method="interp"
    )
    A_k, b_k = BC_k.apply(A_k, b_k)
    k1[:] = spsolve(A_k, b_k)
    k1[:] = bm.maximum(k1[:], 1e-8)
    k1[:] = alpha_k_omega * k1[:] + (1 - alpha_k_omega) * k0[:]
    res_k = mesh.error(k0, k1)
    print(f"res_k", res_k)
    k0[:] = k1

    mesh.nodedata["kh"] = k1
    print("max_mut", bm.max(fem.mu_t))
    mesh.nodedata["mu_t"] = fem.mu_t

    # 湍流耗散率方程求解
    BForm_omega = fem_omega.BForm()
    LForm_omega = fem_omega.LForm()
    fem_omega.update(u1, k1, omega0, mu_t = fem.mu_t)
    A_omega = BForm_omega.assembly()
    b_omega = LForm_omega.assembly()
    BC_omega = DirichletBC(
        fem_omega.omegaspace,
        gd=pde.omega_dirichlet,
        threshold=pde.is_omega_boundary,
        method="interp"
    )
    A_omega, b_omega = BC_omega.apply(A_omega, b_omega)
    omega1[:] = spsolve(A_omega, b_omega)
    omega1[:] = bm.minimum(bm.maximum(omega1[:], 1e-8), 2e6)
    omega1[:] = alpha_k_omega * omega1[:] + (1 - alpha_k_omega) * omega0[:]
    res_omega = mesh.error(omega0, omega1)
    print(f"res_omega", res_omega)
    omega0[:] = omega1

    # 更新初始值
    u0[:] = u1
    p0[:] = p1
    # k0[:] = k1
    # omega0[:] = omega1

    mesh.to_vtk(f"stationary_sst_k_omega_{i+1}.vtu")

