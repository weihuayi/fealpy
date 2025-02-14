from fealpy.backend import backend_manager as bm
from fealpy.pde.poisson_2d import CosCosData
from fealpy.fvm.poisson_fvm_solver import PoissonFvmSolver
from scipy.sparse.linalg import spsolve
from fealpy.utils import timer


maxit = 4
tmr = timer()
next(tmr)
errorType = ['$|| u - u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
tmr.send('网格和pde生成时间')
pde = CosCosData()
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
    errorMatrix[0, i] = mesh.error(u, uh)

    if i < maxit-1:
        pde.mesh.uniform_refine(n=1)    
    tmr.send(f'第{i}次误差计算及网格加密时间')
    

next(tmr)
print("最终误差",errorMatrix)
print("order : ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))





    
