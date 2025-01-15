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

bm.set_backend('numpy')

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


mesh = NodeMesh.from_tgv_domain(box_size, dx)
solver = SPHSolver(mesh)
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(side=box_size)
