from fealpy.old.timeintegratoralg import UniformTimeLine
import matplotlib.pyplot as plt
from fealpy.utils import timer 
from fealpy.backend import backend_manager as bm
from scipy.sparse.linalg import spsolve
from ns_fvm_solver import NSFVMSolver
from ns_fvm_pde import NSFVMPde

pde = NSFVMPde()
mesh = pde.mesh(nx = 5, ny = 5)
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
    t = timeline.next_time_level()
    #print(f"第{i+1}步")
    #print("time=", t)

    #IPCS算法第一步
    A_u0 = solver.NS_Lform_us()
    A_v0 = solver.NS_Lform_vs()
    b_u0, b_v0 = solver.DirichletBC_1(u0, v0, p0)

    us = spsolve(A_u0, b_u0)
    vs = spsolve(A_v0, b_v0)

    #print('us', us)
    #print('vs', vs)

    #IPCS算法第二步
    A_p = solver.NS_Lform_p()
    b_p = solver.NS_Bform_p(us, vs, p0)
    p1 = spsolve(A_p, b_p)

    #IPCS算法第三步
    A_u1 = solver.NS_Lform_u1()
    A_v1 = solver.NS_Lform_v1()
    b_u1 = solver.NS_Bform_u1(us, p0, p1)
    b_v1 = solver.NS_Bform_v1(vs, p0, p1) 
    u1 = spsolve(A_u1, b_u1)
    v1 = spsolve(A_v1, b_v1)

    u0 = u1
    v0 = v1
    p0 = p1
    #print( mesh0.error(u, u1))
    #print( mesh1.error(v, v1))
    #print( mesh.error(p, p1))
    #print( bm.max(bm.abs(u - u1)))
    #print( bm.max(bm.abs(v - v1)))
    #print( bm.max(bm.abs(p - p1)))

    timeline.advance()

fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ipoint = mesh0.entity_barycenter('cell')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(pde.nx+1, pde.ny)
Y = yy.reshape(pde.nx+1, pde.ny)
Z = u1.reshape(pde.nx+1, pde.ny)
ax1.plot_surface(X, Y, Z, cmap='rainbow')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ipoint = mesh1.entity_barycenter('cell')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(pde.nx, pde.ny+1)
Y = yy.reshape(pde.nx, pde.ny+1)
Z = v1.reshape(pde.nx, pde.ny+1)
ax1.plot_surface(X, Y, Z, cmap='rainbow')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ipoint = mesh.entity_barycenter('cell')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(pde.nx, pde.ny)
Y = yy.reshape(pde.ny, pde.ny)
Z = p1.reshape(pde.nx, pde.ny)
ax1.plot_surface(X, Y, Z, cmap='rainbow')
plt.show()

print('A_u0', A_u0.toarray())
print('A_v0', A_v0.toarray())
print('A_p', A_p.toarray())
print('A_u1', A_u1.toarray())
print('A_v1', A_v1.toarray())
print('b_u0', b_u0)
print('b_v0', b_v0)
print('b_p', b_p)
print('b_u1', b_u1)
print('b_v1', b_v1)


'''
print(mesh0.error(u, u1))
print(mesh1.error(v, v1))
print(mesh.error(p, p1))

print("最终误差", errorMatrix)
print("order : ", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
print("order : ", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))
print("order : ", bm.log2(errorMatrix[2, :-1] / errorMatrix[2, 1:]))
'''

