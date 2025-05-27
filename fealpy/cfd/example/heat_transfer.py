from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
from fealpy.mesh.node_mesh import NodeMesh, Space
from fealpy.cfd.simulation.sph.sph_base import SPHQueryKernel, Kernel
from fealpy.cfd.simulation.sph.equation_solver import EquationSolver
from fealpy.cfd.simulation.sph.processing_technology import ProcessingTechnology
from fealpy.cfd.simulation.utils import VTKWriter

dx = 0.02
dt = 0.00045454545454545455
h = dx
box_size = bm.array([1.0, 0.2 + dx*3*2], dtype=bm.float64) #模拟区域
path = "./"

mesh = NodeMesh.from_heat_transfer_domain(dx=dx,dy=dx)
solver = EquationSolver()
tech = ProcessingTechnology(mesh)
space = Space()
writer = VTKWriter()
displacement, shift = space.periodic(side=box_size)
kinfo = {
    "type": "quintic",  # 选择核函数类型
    "h": dx,           # 设定核函数的平滑长度
    "space": True      # 是否周期边界空间
}
kernel = Kernel(kinfo, dim=2)
sph_query = SPHQueryKernel(
       mesh=mesh,
       radius=3*dx,
       box_size=box_size,
       mask_self=True,
       kernel_info=kinfo,
       periodic=[True, True, True],
    )
self_node, neighbors = sph_query.find_node()
dr = kernel.compute_displacement(self_node, neighbors, mesh.nodedata["position"], box_size)
dist = kernel.compute_distance(self_node, neighbors, mesh.nodedata["position"], box_size)
w = sph_query.compute_kernel_value(self_node, neighbors)
grad_w, grad_w_norm = sph_query.compute_kernel_gradient(self_node, neighbors)

mesh.nodedata["p"] = solver.state_equation("tait_eos", mesh.nodedata, X=5.0)
mesh.nodedata = tech.boundary_conditions(mesh.nodedata, box_size, dx=dx)

for i in range(1000):
    print("i:", i)
    mesh.nodedata['mv'] += 1.0*dt*mesh.nodedata["dmvdt"]
    mesh.nodedata['tv'] = mesh.nodedata['mv']
    mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])

    sph_query = SPHQueryKernel(
        mesh=mesh,
        radius=3*dx,
        box_size=box_size,
        mask_self=True,
        kernel_info=kinfo,
        periodic=[True, True, True],
    )
    self_node, neighbors = sph_query.find_node()
    dr = kernel.compute_displacement(self_node, neighbors, mesh.nodedata["position"], box_size)
    dist = kernel.compute_distance(self_node, neighbors, mesh.nodedata["position"], box_size)
    w = sph_query.compute_kernel_value(self_node, neighbors)
    grad_w, grad_w_norm = sph_query.compute_kernel_gradient(self_node, neighbors)

    g_ext = tech.external_acceleration(mesh.nodedata["position"], box_size, dx=dx)
    wall_mask = bm.where(bm.isin(mesh.nodedata["tag"], bm.array([1, 3])), 1.0, 0.0)
    fluid_mask = bm.where(mesh.nodedata["tag"] == 0, 1.0, 0.0) > 0.5
    rho_summation = solver.mass_equation_solve(0, mesh.nodedata, neighbors, w)
    rho = bm.where(fluid_mask, rho_summation, mesh.nodedata["rho"])
    
    p = solver.state_equation("tait_eos", mesh.nodedata, rho=rho, X=5.0)
    pb = solver.state_equation("tait_eos", mesh.nodedata, rho=bm.zeros_like(p), X=5.0)
    p, rho, mv, tv, T = tech.enforce_wall_boundary(mesh.nodedata, p, g_ext, neighbors, self_node, w, dr, with_temperature=True)
    mesh.nodedata["rho"] = rho
    mesh.nodedata["mv"] = mv
    mesh.nodedata["tv"] = tv
    
    T += dt * mesh.nodedata["dTdt"]
    mesh.nodedata["T"] = T
    mesh.nodedata["dTdt"] = solver.heat_equation_solve(0, mesh.nodedata, dr, dist, neighbors, self_node, grad_w)
    
    mesh.nodedata["dmvdt"] = solver.momentum_equation_solve(0,\
        mesh.nodedata, neighbors, self_node, dr, dist, grad_w_norm, p)
    mesh.nodedata["dmvdt"] = mesh.nodedata["dmvdt"] + g_ext
    mesh.nodedata["p"] = p
    mesh.nodedata["dtvdt"] = solver.momentum_equation_solve(1,\
        mesh.nodedata, neighbors, self_node, dr, dist, grad_w_norm, pb)
    mesh.nodedata = tech.boundary_conditions(mesh.nodedata, box_size, dx=dx)

    #zfname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
    #writer.write_vtk(mesh.nodedata, zfname)