from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
from fealpy.mesh.node_mesh import NodeMesh, Space
from fealpy.cfd.simulation.sph.sph_base import SPHQueryKernel, Kernel
from fealpy.cfd.simulation.sph.equation_solver import EquationSolver
from fealpy.cfd.simulation.utils import VTKWriter

dx = 0.02
dt = 0.0004 
rho0 = 1.0 
box_size = bm.array([1.0,1.0], dtype=bm.float64) 
path = "./"

mesh = NodeMesh.from_tgv_domain(box_size, dx)
solver = EquationSolver()
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

for i in range(1000):
   print(i)
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
   
   mesh.nodedata['rho'] = solver.mass_equation_solve(0, mesh.nodedata, neighbors, w)
   p = solver.state_equation("tait_eos", mesh.nodedata)
   mesh.nodedata["dmvdt"] = solver.momentum_equation_solve(0,\
         mesh.nodedata, neighbors, self_node, dr, dist, grad_w_norm, p)
   
   #zfname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
   #writer.write_vtk(mesh.nodedata, zfname)