from fealpy.backend import backend_manager as bm
from fealpy.pde.poisson_2d import CosCosData
from poisson_fvm_solver import PoissonFvmSolver
from poisson_fvm_pde import PoissonFvmPde
from poisson_fvm_pde1 import PoissonFvmPde1
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from fealpy.utils import timer

maxit =6
tmr = timer()
next(tmr)
errorType = ['$|| u - u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros(maxit, dtype=bm.float64)
x = bm.zeros(maxit, dtype=bm.float64)
pde = PoissonFvmPde1()
mesh = pde.mesh(nx=10, ny=10)

for i in range(maxit):

    solver = PoissonFvmSolver(pde)
    A0 = solver.Poisson_LForm()
    b = solver.Poisson_BForm()
    tmr.send(f'第{i}次矩阵组装时间')

    A0, b = solver.DirichletBC(A0, b)
    tmr.send(f'第{i}次边界处理时间')

    uh = spsolve(A0, b)
    tmr.send(f'第{i}次求解器时间')

    ipoint = pde.mesh.entity_barycenter('cell')
    u = pde.solution(ipoint)
    errorMatrix[i] = mesh.error(u, uh)
    x[i] = pde.mesh.number_of_cells()

    if i < maxit-1:
        pde.mesh.uniform_refine(n=1)
    tmr.send(f'第{i}次误差计算及网格加密时间')


next(tmr)
print("最终误差",errorMatrix)
print("order : ", bm.log2(errorMatrix[:-1]/errorMatrix[1:]))
print("网格数目", x)

order = bm.log2(errorMatrix[:-1]/errorMatrix[1:])

'''
fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ipoint = pde.mesh.entity_barycenter('cell')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(pde.nx, pde.ny)
Y = yy.reshape(pde.nx, pde.ny)
Z = uh.reshape(pde.nx, pde.ny)
surf = ax1.plot_surface(X, Y, Z, cmap='rainbow')
ax1.set_zlabel('Z (numerical solution of u)', fontsize = 16)
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=20)
plt.show()
'''

plt.figure(figsize=(10, 6))
#plt.plot(x, errorMatrix, marker='o', linestyle='-', color='black')
plt.plot(x[1:], order, marker='o', linestyle='-', color='black')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('X (Number of cells)', fontsize=14)
plt.ylabel('Y (max order)', fontsize=14)
plt.show()
