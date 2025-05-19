from fealpy.backend import backend_manager as bm
from fealpy.pde.poisson_2d import CosCosData
from fealpy.fvm.poisson_fvm_solver import PoissonFvmSolver
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from fealpy.utils import timer


maxit = 6
tmr = timer()
next(tmr)
errorType = ['$|| u - u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros(maxit, dtype=bm.float64)
x = bm.zeros(maxit, dtype=bm.float64)
pde = CosCosData()
mesh = pde.mesh(nx=20, ny=20)

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

plt.figure(figsize=(10, 6))
#plt.plot(x, errorMatrix, marker='o', linestyle='-', color='black')
plt.plot(x[1:], order, marker='o', linestyle='-', color='black')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('X (Number of cells)', fontsize=14)
plt.ylabel('Y (order)', fontsize=14)
plt.show()