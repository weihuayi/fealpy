#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: tgv_test_backend.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 14 Jan 2025 04:07:32 PM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.mesh.node_mesh import NodeMesh, KDTree
from fealpy.cfd.sph.particle_solver_new import SPHSolver, Space, VmapBackend
from fealpy.cfd.sph.particle_kernel_function import QuinticKernel
import time

bm.set_backend('pytorch')

EPS = bm.finfo(float).eps
dx = 0.25
dy = 0.25
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

mesh = NodeMesh.from_tgv_domain(box_size, dx)
solver = SPHSolver(mesh)
space = Space()
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(box_size, True)
kdtree = KDTree(mesh.nodedata["position"],box_size)
vmap_backend = VmapBackend()


#start = time.time()
for i in range(1):
    print(i)
    mesh.nodedata['mv'] += 1.0*dt*mesh.nodedata["dmvdt"]
    mesh.nodedata['tv'] = mesh.nodedata['mv']
    mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])
    
    r = mesh.nodedata["position"]
    i_s, j_s = kdtree.range_query(mesh.nodedata["position"], 3*h, include_self=True)
    r_i_s, r_j_s = r[i_s], r[j_s]
    dr_i_j = vmap_backend.apply(displacement, r_i_s, r_j_s)
    #print(dr_i_j)
    
	

#end = time.time()
#print(end-start)