from fealpy.backend import backend_manager as bm
from fealpy.cfd.sph.kdtree import Neighbor 
from fealpy.mesh.node_mesh import NodeMesh
from fealpy.cfd.sph.particle_solver import SPHSolver,TimeLine
from fealpy.cfd.sph.particle_kernel_function import QuinticKernel
from jax_md import space
from jax import vmap
import matplotlib.pyplot as plt

EPS = bm.finfo(float).eps
dx = 0.02
dy = 0.02
h = dx 
Vmax = 1.0 #预期最大速度
c0 =10 * Vmax #声速
rho0 = 1.0 #参考密度
eta0 = 0.01
nu = 1.0 #运动粘度
T = 2 #终止时间
dt = 0.0004 #时间间隔
t_num = int(T / dt)
dim = 2 #维数
box_size = bm.array([1.0,1.0]) #模拟区域
path = "./"

#初始化
mesh = NodeMesh.from_tgv_domain(box_size, dx)
solver = SPHSolver(mesh)
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(side=box_size)

for i in range(300):
    print(i)
    mesh.nodedata['mv'] += 1.0*dt*mesh.nodedata["dmvdt"]
    mesh.nodedata['tv'] = mesh.nodedata['mv']
    mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])

    r = mesh.nodedata["position"]
    i_s, j_s = Neighbor.find_neighbors(mesh.nodedata, h) 
    #i_s0 = bm.array([(len(mesh.nodedata["position"])-1) if x == len(mesh.nodedata["position"]) else x for x in i_s])
    #j_s0 = bm.array([(len(mesh.nodedata["position"])-1) if x == len(mesh.nodedata["position"]) else x for x in j_s])
    #r_i_s, r_j_s = mesh.nodedata["position"][bm.array(i_s0)], mesh.nodedata["position"][bm.array(j_s0)]
    
    r_i_s, r_j_s = r[i_s], r[j_s]
    dr_i_j = vmap(displacement)(r_i_s, r_j_s)
    dist = space.distance(dr_i_j)
    
    w_dist = vmap(kernel.value)(dist)  
    e_s = dr_i_j / (dist[:, None] + EPS) 
    grad_w_dist_norm = vmap(kernel.grad_value)(dist)
    grad_w_dist = grad_w_dist_norm[:, None] * e_s

    mesh.nodedata['rho'] = solver.compute_rho(mesh.nodedata['mass'], i_s, w_dist)
    p = solver.tait_eos(mesh.nodedata['rho'],c0,rho0)
    background_pressure_tvf = solver.tait_eos(bm.zeros_like(p), c0, rho0)
    
    mesh.nodedata["dmvdt"] = solver.compute_mv_acceleration(\
            mesh.nodedata, i_s, j_s, dr_i_j, dist, grad_w_dist_norm, p)

    #fname = path + 'test_'+ str(i+1).zfill(10) + '.h5'
    #solver.write_h5(mesh.nodedata, fname)
    #fname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
    #solver.write_vtk(mesh.nodedata, fname)
