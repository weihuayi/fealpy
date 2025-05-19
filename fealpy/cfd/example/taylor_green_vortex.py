from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
from fealpy.mesh.node_mesh import NodeMesh, Space
from fealpy.cfd.simulation.sph_base import SPHQueryKernel
from fealpy.cfd.simulation.sph.equation_solver import EquationSolver
from fealpy.cfd.simulation.utils import VTKWriter

dx = 0.02
dt = 0.0004 
rho0 = 1.0 
box_size = bm.array([1.0,1.0], dtype=bm.float64) 
path = "./"

mesh = NodeMesh.from_tgv_domain(box_size, dx)
sph_query = SPHQueryKernel(
       mesh=mesh,
       radius=3*dx,
       box_size=box_size,
       periodic=[True, True, True]
    )
self_node, neighbors, w, grad_w , dr, dist, grad_w_norm = sph_query.compute()
solver = EquationSolver()
space = Space()
writer = VTKWriter()
displacement, shift = space.periodic(side=box_size)

for i in range(1):
    print(i)
    mesh.nodedata['mv'] += 1.0*dt*mesh.nodedata["dmvdt"]
    mesh.nodedata['tv'] = mesh.nodedata['mv']
    mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])

    sph_query = SPHQueryKernel(
       mesh=mesh,
       radius=3*dx,
       box_size=box_size,
       periodic=[True, True, True]
    )
    self_node, neighbors, w, grad_w , dr, dist , grad_w_norm = sph_query.compute()

    mesh.nodedata['rho'] = solver.mass_equation_solve(False, mesh.nodedata, neighbors, w)
    p = solver.constitutive_equation("tait_eos", mesh.nodedata)
    mesh.nodedata["dmvdt"] = solver.momentum_equation_solve(False,\
            mesh.nodedata, neighbors, self_node, dr, dist, grad_w_norm, p)
    
    #zfname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
    #writer.write_vtk(mesh.nodedata, zfname)