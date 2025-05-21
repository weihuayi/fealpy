from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.pde.poisson_2d import CosCosData 
from fealpy.solver import IterativeSolverManager    
p = 1
n = 100
pde = CosCosData()
domain = pde.domain()
mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
space = LagrangeFESpace(mesh, p=p)
uh = space.function() 
bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator(method='fast'))
lform = LinearForm(space)
lform.add_integrator(ScalarSourceIntegrator(pde.source))
A = bform.assembly()
b = lform.assembly()
gdof = space.number_of_global_dofs()
A, b = DirichletBC(space, gd=pde.solution).apply(A, b)
ism = IterativeSolverManager()
ism.set_matrix(A, matrix_type='SP')  # SPD matrix
ism.set_solver('cg')
ism.set_pc('jacobi')
ism.set_tolerances(rtol=1e-6, atol=1e-8, maxit=1000)
x_sol = ism.solve(b)