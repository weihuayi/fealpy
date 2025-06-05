from fealpy.backend import backend_manager as bm
from fealpy.utils import timer
from fealpy import logger
from fealpy.pde.semilinear_2d import SinSinData
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import NonlinearForm, ScalarNonlinearMassIntegrator
from fealpy.fem import ScalarDiffusionIntegrator, ScalarSourceIntegrator
from fealpy.solver import cg
from fealpy.decorator import barycentric
from fealpy.fem import DirichletBC


backend = 'pytorch'
device = 'cuda'
bm.set_backend(backend)
bm.set_default_device(device)


logger.setLevel('WARNING')
tmr = timer()
next(tmr)


pde = SinSinData()


mesh = TriangleMesh.from_box(pde.domain(), nx=8, ny=8, device=device)
maxit = 4
tol = 1e-14

errorType = ['$|| u - u_h||_{//Omega,0}$']
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64, device=device)

def diffusion_coef(p, **args):
    return pde.diffusion_coefficient(p)

def reaction_coef(p, **args):
    return pde.reaction_coefficient(p)

reaction_coef.kernel_func = pde.kernel_func_reaction
reaction_coef.grad_kernel_func = pde.grad_kernel_func_reaction


for i in range(maxit):
    space = LagrangeFESpace(mesh, p=1)
    tmr.send(f'第{i}次空间时间')
    u0 = space.function()
    du = space.function()
    diffusion_coef.uh = u0
    reaction_coef.uh = u0
    isDDof = space.set_dirichlet_bc(pde.dirichlet, u0)


    D = ScalarDiffusionIntegrator(diffusion_coef, q=3)
    M = ScalarNonlinearMassIntegrator(reaction_coef, q=3)
    f = ScalarSourceIntegrator(pde.source, q=3)
    
    sform = NonlinearForm(space)
    sform.add_integrator([D, M])
    sform.add_integrator(f)
    bc = DirichletBC(space, gd=0.0, threshold=isDDof)


    while True:
        A, F = sform.assembly()
        tmr.send(f'第{i}次矩组装时间')
        A, F = bc.apply(A, F)
        du = cg(A, F)
        u0 += du
        tmr.send(f'第{i}次求解器时间')
        M.clear()
        
        err = bm.max(bm.abs(du))
        if err < tol:
            break


    @barycentric
    def ugval(p):
        return space.grad_value(u0, p)
    
    errorMatrix[0, i] = mesh.error(pde.solution, u0, q=3)
    errorMatrix[1, i] = mesh.error(pde.gradient, ugval, q=3)
    if i < maxit-1:
        mesh.uniform_refine()
    tmr.send(f'第{i}次误差计算及网格加密时间')

next(tmr)
print("最终误差：",errorMatrix)
print("收敛阶 : ", errorMatrix[:, 0:-1]/errorMatrix[:, 1:])