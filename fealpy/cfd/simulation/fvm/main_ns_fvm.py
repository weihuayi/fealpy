from fealpy.old.timeintegratoralg import UniformTimeLine
import matplotlib.pyplot as plt
from fealpy.utils import timer 
from fealpy.backend import backend_manager as bm
from ns_fvm_solver import NSFVMSolver
from ns_fvm_pde import NSFVMPde
from ns_fvm_pde1 import NSFVMPde1
import torch
from fealpy.solver import spsolve

bm.set_backend('pytorch')
torch.set_printoptions(precision=16)
#bm.set_default_device('cuda')

pde = NSFVMPde1()
mesh = pde.mesh(nx = 400, ny = 400)
mesh0 = pde.mesh0()
mesh1 = pde.mesh1()
output = './'
T = 0.5
nt = 1000

timeline = UniformTimeLine(0, T, nt)
ht = timeline.dt

solver = NSFVMSolver(pde, ht)
u = pde.velocity_u(solver.point_u)
v = pde.velocity_v(solver.point_v)
p = pde.pressure(solver.point_p)

u0 = u
v0 = v
p0 = p

for i in range(nt):
    tmr = timer()
    next(tmr)

    t = timeline.next_time_level()
    print(f"第{i+1}步")
    print("time=", t)
    #IPCS算法第一步
    A_u0 = solver.NS_Lform_us()
    A_v0 = solver.NS_Lform_vs()
    tmr.send('第1步矩阵组装时间')
    b_u0, b_v0 = solver.DirichletBC_1(u0, v0, p0)
    tmr.send('第1步边界处理时间')
    
    us = spsolve(A_u0, b_u0)
    vs = spsolve(A_v0, b_v0)
    tmr.send('第1步求解器时间')

    #print('us', us)
    #print('vs', vs)

    #IPCS算法第二步
    A_p = solver.NS_Lform_p()
    b_p = solver.NS_Bform_p(us, vs, p0)
    tmr.send('第2步矩阵组装时间')
    p1 = spsolve(A_p, b_p)
    tmr.send('第2步求解器时间')

    #IPCS算法第三步
    A_u1 = solver.NS_Lform_u1()
    A_v1 = solver.NS_Lform_v1()
    b_u1 = solver.NS_Bform_u1(us, p0, p1)
    b_v1 = solver.NS_Bform_v1(vs, p0, p1) 
    tmr.send('第3步矩阵组装时间')
    
    u1 = spsolve(A_u1, b_u1)
    v1 = spsolve(A_v1, b_v1)
    tmr.send('第3步求解器时间')

    u0 = u1
    v0 = v1
    p0 = p1
    eu = u - u1
    ev = v - v1
    ep = p - p1
    h = 1/pde.nx
    print('u-L2', bm.sqrt(bm.sum(h**2 * eu**2)))
    print('v-L2', bm.sqrt(bm.sum(h**2 * ev**2)))
    print('p-L2', bm.sqrt(bm.sum(h**2 * ep**2)))
    print('u-max', bm.max(bm.abs(eu)))
    print('v-max', bm.max(bm.abs(ev)))
    print('p-max', bm.max(bm.abs(ep)))
    
    next(tmr)

    timeline.advance()


fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ipoint = mesh0.entity_barycenter('cell')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(pde.nx+1, pde.ny)
Y = yy.reshape(pde.nx+1, pde.ny)
Z = u1.reshape(pde.nx+1, pde.ny)
surf = ax1.plot_surface(X, Y, Z, cmap='rainbow')
ax1.set_zlabel('Z (numerical solution of u)', fontsize = 16)
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=20)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ipoint = mesh1.entity_barycenter('cell')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(pde.nx, pde.ny+1)
Y = yy.reshape(pde.nx, pde.ny+1)
Z = v1.reshape(pde.nx, pde.ny+1)
surf = ax1.plot_surface(X, Y, Z, cmap='rainbow')
ax1.set_zlabel('Z (numerical solution of v)', fontsize = 16)
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=20)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ipoint = mesh.entity_barycenter('cell')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(pde.nx, pde.ny)
Y = yy.reshape(pde.ny, pde.ny)
Z = p1.reshape(pde.nx, pde.ny)
surf = ax1.plot_surface(X, Y, Z, cmap='rainbow')
ax1.set_zlabel('Z (numerical solution of p)', fontsize = 16)
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=20)
plt.show()

