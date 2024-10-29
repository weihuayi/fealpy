#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: poisson_lfem_robin.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 28 Oct 2024 08:01:29 PM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.utils import timer
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import ScalarNeumannBCIntegrator
from fealpy.fem import ScalarRobinBCIntegrator
from fealpy.fem import DirichletBC
from fealpy.pde.poisson_2d import CosCosData 
from fealpy.mesh import TriangleMesh
from fealpy.solver import cg

bm.set_backend('numpy')
p = 1 
n = 10 
maxit = 5
pde = CosCosData()

tmr = timer()
next(tmr)

mesh = TriangleMesh.from_box(pde.domain(), n, n)
errorType = ['$|| u - u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
tmr.send('网格和pde生成时间')

for i in range(maxit):
    space= LagrangeFESpace(mesh, p=p)
    tmr.send(f'第{i}次空间时间') 

    uh = space.function() # 建立一个有限元函数

    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator())
    bform.add_integrator(ScalarRobinBCIntegrator(coef=pde.kappa, threshold=pde.is_robin_boundary))
    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(pde.source))
    lform.add_integrator(ScalarNeumannBCIntegrator(gn=pde.robin, threshold=pde.is_robin_boundary))
    lform.add_integrator(ScalarNeumannBCIntegrator(gn=pde.neumann, threshold=pde.is_neumann_boundary))

    A = bform.assembly()
    F = lform.assembly()
    tmr.send(f'第{i}次矩组装时间')

    gdof = space.number_of_global_dofs()

    A, F = DirichletBC(space, gd=pde.dirichlet, threshold=pde.is_dirichlet_boundary).apply(A, F)
    tmr.send(f'第{i}次边界处理时间')

    uh[:] = cg(A, F, maxiter=5000, atol=1e-14, rtol=1e-14)
    tmr.send(f'第{i}次求解器时间')

    errorMatrix[0, i] = mesh.error(pde.solution, uh)

    if i < maxit-1:
        mesh.uniform_refine(n=1)
    tmr.send(f'第{i}次误差计算及网格加密时间')

next(tmr)
print("最终误差",errorMatrix)
print("order : ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))
