from fealpy.backend import backend_manager as bm
bm.set_backend("numpy")
from fealpy.mesh.node_mesh import NodeMesh
from fealpy.cfd.simulation.sph.sph_base import SPHQueryKernel, Kernel
from fealpy.cfd.simulation.sph.equation_solver import EquationSolver
from fealpy.cfd.simulation.sph.particle_solver_new import SPHSolver
from fealpy.cfd.simulation.utils import VTKWriter

dx = 0.03
dy = 0.03
dt = 0.001
H = 0.92 * bm.sqrt(dx**2 + dy**2)
eta = 0.1 * H
rhomin = 995
box_size = bm.array([4.0, 4.0], dtype=bm.float64)
maxstep = 4000
path = "./"

mesh = NodeMesh.from_dam_break_domain_2d(dx, dy)
solver = EquationSolver()
sphsolver = SPHSolver(mesh)
writer = VTKWriter()
kinfo = {
    "type": "wendlandc2",  # 选择核函数类型
    "h": H,  # 设定核函数的平滑长度
    "space": False,  # 是否周期边界空间
}
kernel = Kernel(kinfo, dim=2)
sph_query = SPHQueryKernel(
    mesh=mesh,
    radius=2 * H,
    box_size=box_size,
    mask_self=True,
    kernel_info=kinfo,
    periodic=[False, False, False],
)

for i in range(maxstep):
    print(i)

    self_node, neighbors = sph_query.find_node()
    dr = kernel.compute_displacement(self_node, neighbors, mesh.nodedata["position"], box_size)
    dist = kernel.compute_distance(self_node, neighbors, mesh.nodedata["position"], box_size)
    w = sph_query.compute_kernel_value(self_node, neighbors)
    grad_w, grad_w_norm = sph_query.compute_kernel_gradient(self_node, neighbors)

    if i % 30 == 1 and i != 1:
        sphsolver.rein_rho_2d(mesh.nodedata, self_node, neighbors, w)
        mesh.nodedata["mass"] = mesh.nodedata["rho"] * dx * dy

    mesh.nodedata["drhodt"] = sphsolver.change_rho_dam(mesh.nodedata, self_node, neighbors, grad_w)
    mesh.nodedata["rho"] += dt * mesh.nodedata["drhodt"]
    mesh.nodedata["rho"] = bm.maximum(mesh.nodedata["rho"], rhomin)
    mesh.nodedata["mass"] = mesh.nodedata["rho"] * dx * dy

    mesh.nodedata["pressure"] = solver.state_equation("tait_eos", mesh.nodedata, rho=mesh.nodedata["rho"], c0=mesh.nodedata["c0"], rho0=mesh.nodedata["rho0"],gamma=mesh.nodedata["gamma"])
    mesh.nodedata["sound"] = sphsolver.sound_dam(mesh.nodedata, mesh.nodedata["c0"], mesh.nodedata["rho0"], mesh.nodedata["gamma"])
    

    mesh.nodedata["dudt"] = sphsolver.change_u_dam(mesh.nodedata, self_node, neighbors, dr, dist, grad_w)
    mesh.nodedata["u"] = bm.where((mesh.nodedata["tag"] == 0)[:, None], mesh.nodedata["u"] + dt * mesh.nodedata["dudt"], mesh.nodedata["u"],)
    mesh.nodedata["dxdt"] = sphsolver.change_r_dam(mesh.nodedata, self_node, neighbors, w)
    mesh.nodedata["position"] = bm.where((mesh.nodedata["tag"] == 0)[:, None], mesh.nodedata["position"] + dt * mesh.nodedata["dxdt"], mesh.nodedata["position"],)
    
    current_data = {
        "position": mesh.nodedata["position"].tolist(),  # ndarray -> list
        "velocity": mesh.nodedata["u"].tolist(),  # ndarray -> list
        "pressure": mesh.nodedata["pressure"].tolist(),  # ndarray -> list
    }

    # zfname = path + "test_" + str(i + 1).zfill(10) + ".vtk"
    # writer.write_vtk(current_data, zfname)

    kernel = Kernel(kinfo, dim=2)
    sph_query = SPHQueryKernel(
        mesh=mesh,
        radius=2 * H,
        box_size=box_size,
        mask_self=True,
        kernel_info=kinfo,
        periodic=[False, False, False],
    )
