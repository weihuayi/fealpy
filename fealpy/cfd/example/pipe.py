from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
from fealpy.mesh.node_mesh import NodeMesh, NeighborManager
from fealpy.cfd.simulation.sph.sph_base import SPHQueryKernel, Kernel
from fealpy.cfd.simulation.sph.equation_solver import EquationSolver
from fealpy.cfd.simulation.sph.processing_technology import ProcessingTechnology
from fealpy.cfd.simulation.utils import VTKWriter

#打印
import torch
torch.set_printoptions(threshold=float('inf'))

dx = 1.25e-4
H = 1.5 * dx
dt = bm.array([1e-7], dtype=bm.float64)
u_in = bm.array([5.0, 0.0], dtype=bm.float64)
domain = [0.0, 0.05, 0.0, 0.005]
init_domain = [0.0, 0.005, 0, 0.005]
side = bm.array([0.0508, 0.0058], dtype=bm.float64)

#粘性本构方程参数
n = 0.3083
tau_star = 16834.4
mu_0 = 938.4118
#控制方程参数
eta = 0.5
#Tait 模型状态方程参数
c_1 = 0.0894
B = 5.914e7
rho_0 = 737.54
path = "frames/"

mesh = NodeMesh.from_pipe_domain(domain, init_domain, H, dx)
solver = EquationSolver()
writer = VTKWriter()
tech = ProcessingTechnology(mesh)
n_manager = NeighborManager()
kinfo = {
    "type": "wendlandc2",  # 选择核函数类型
    "h": 1.5*dx,           # 设定核函数的平滑长度
    "periodic": False      # 是否周期边界空间
}
kernel = Kernel(kinfo, dim=2)

#wall-virtual
v_node_self, w_neighbors = n_manager.wall_virtual(mesh.nodedata["position"], mesh.nodedata["tag"])

for i in range(1):
    print(i)
    #阀门粒子更新
    state = mesh.nodedata
    state = tech.gate_change(state, dx, domain, H, u_in, rho_0, dt)
    
    r = state["position"]
    rho = state["rho"]
    u  =state["u"]
    sph_query = SPHQueryKernel(
       mesh=mesh,
       radius=3*dx,
       box_size=side,
       mask_self=True,
       kernel_info=kinfo,
       periodic=[False, False, False],
    )
    node_self, neighbors = sph_query.find_node() 
    dr_i_j = kernel.compute_displacement(node_self, neighbors, mesh.nodedata["position"], side)    
    dist = kernel.compute_distance(node_self, neighbors, mesh.nodedata["position"], side)
    w_dist = sph_query.compute_kernel_value(node_self, neighbors)
    grad_w_dist = sph_query.compute_kernel_gradient(node_self, neighbors)[0]
    
    # fuild-wall-virtual
    f_node, fwvg_neighbors, dr, dis, w, dw = n_manager.fuild_fwvg(state, node_self, neighbors, dr_i_j, dist, w_dist, grad_w_dist) 
    
    # wall-fuild
    w_node, fg_neighbors, fg_w = n_manager.wall_fg(state, node_self, neighbors, w_dist)
    state["u"] = tech.vtag_u(state, v_node_self, w_neighbors, w_node, fg_neighbors, fg_w)
    
    # 更新半步压强和声速
    state["p"] = solver.state_equation("injection_molding", state, rho0=rho_0)
    state["p"] = tech.wall_p(state, w_node, fg_neighbors, fg_w)
    state["p"] = tech.virtual_p(state, v_node_self, w_neighbors)
    state["sound"] = solver.sound(state)
    
    # 更新半步密度
    A_s = tech.A_matrix(state, f_node, fwvg_neighbors, dr, dw)
    print(bm.sum(A_s))
    exit()
    
    state["drhodt"] = solver.mass_equation_solve(1, state, self_node=f_node, neighbors=fwvg_neighbors,\
         dr=dr, dist=dis, dw=dw, A_s=A_s)
    drho_0 = state["drhodt"]
    state["rho"] = state["rho"] + 0.5 * dt * state["drhodt"]
    
    
