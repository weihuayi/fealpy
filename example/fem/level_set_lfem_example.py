from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.fem import LevelSetLFEModel 
from fealpy.functionspace import LagrangeFESpace
from fealpy.decorator import cartesian
from fealpy.functionspace import TensorFunctionSpace
from fealpy.solver import spsolve

@cartesian
def velocity_field(p):
    x = p[..., 0]
    y = p[..., 1]
    u = bm.zeros(p.shape)
    u[..., 0] = bm.sin((bm.pi*x))**2 * bm.sin(2*bm.pi*y)
    u[..., 1] = -bm.sin((bm.pi*y))**2 * bm.sin(2*bm.pi*x)
    return u

@cartesian
def pic(p):
    x = p[...,0]
    y = p[...,1]
    val = bm.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    return val


domain = [0, 1, 0, 1]
T=2
nt=100
ns = 20
p = 2
dt = T/nt
output = '/home/wpx/临时文件/result/test_'

mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
space = LagrangeFESpace(mesh,p=p)

solver = LevelSetLFEModel(space, velocity_field, method='CN')
options = solver.options
options.set_evo_params(evo_method='CN', evo_solver='mumps', evo_rein=True)
options.set_reinit_params(re_space=15, re_solver='mumps', re_maxit=100, re_tol=1e-10)
phi = solver.run(T, dt, pic, output = output)





exit()
phi0 = space.interpolate(pic)
Bform = solver.Bform()
Lform = solver.Lform()

solver.output(phi0, solver.u, 0, output)
for i in range(10):
    print("t=",i*dt)

    solver.update(dt, phi0)
    A = Bform.assembly()
    b = Lform.assembly()
    
    x = spsolve(A, b, solver='mumps')
    phi0[:] = x
    
    solver.output(phi0, solver.u, 0, output)

rephi = space.function()
rephi[:] = solver.reinit_run(phi0)
remeasure = solver.compute_zero_level_set_area(rephi)
measure = solver.compute_zero_level_set_area(phi0)
renorm,_ = solver.check_gradient_norm_at_interface(rephi)
norm,_ = solver.check_gradient_norm_at_interface(phi0)
print(renorm)
print(norm)
print(measure)
print(remeasure)
