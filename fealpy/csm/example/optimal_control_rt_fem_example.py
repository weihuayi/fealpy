from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.decorator import barycentric, cartesian
bm.set_backend('numpy')

from fealpy.fem import OPCRTFEMModel
n = 128
mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=n, ny=n)
model = OPCRTFEMModel(mesh,c=1)
maxit = 20  # Maximum number of iterations for the optimization process
space1,space2 = model.space()
pdof = space2.dof.number_of_global_dofs()
ydof = space1.dof.number_of_global_dofs()
qdof = space2.dof.number_of_global_dofs()
zdof = space1.dof.number_of_global_dofs()
y0 = space1.function()
y1 = space1.function()
p0 = space2.function()
q0 = space2.function()
q1 = space2.function()
z0 = space1.function()
z1 = space1.function()
p1 = space2.function()
u0 = space1.function()
u1 = space1.function()
for i in range(maxit):
    pde = model.pde
    A, b_forward = model.linear_system(source1=0, source2=0, source3=pde.f_fun, source4=u0)
    A, b_forward = model.boundary_apply(A, b_forward, gd=pde.y_solution)
    p1,y1 = model.solve(A, b_forward)
    @barycentric
    def coef_p(bcs, index=None):
        result = - p1(bcs, index)
        return result
    @cartesian
    def coef_pd(p, index=None):
        result = pde.pd_fun(p)
        return result
    @barycentric
    def coef_y(bcs, index=None):
        return y1(bcs) 
    @cartesian
    def coef_yd(p, index=None):
        return -pde.y_d_fun(p)
    A,b_backward = model.linear_system(source1=coef_p, source2=coef_pd, source3=coef_y, source4=coef_yd)
    A, b_backward = model.boundary_apply(A, b_backward, gd=pde.z_solution)
    q1, z1 = model.solve(A, b_backward)
    # 假设控制系数为1
    nu = 1
    u1[:] = bm.maximum(0,-z1/nu)
    p_error = mesh.error(p0,p1)
    q_error = mesh.error(q0,q1)
    y_error = mesh.error(y0,y1)
    z_error = mesh.error(z0,z1)
    p0[:] = p1[:]
    q0[:] = q1[:]
    y0[:] = y1[:]
    z0[:] = z1[:]
    u0[:] = u1[:]
    if p_error < 1e-14:
        print('p收敛',p_error)
        print('q收敛',q_error)
        print('y收敛',y_error)
        print('z收敛',z_error)
        break
    
errory, errorp = model.L2_error(y1, p1, solution1=pde.y_solution, solution2=pde.p_solution)
print('L2 error of y:', errory)
print('L2 error of p:', errorp)
errorz , errorq = model.L2_error(z1, q1, solution1=pde.z_solution, solution2=pde.q_solution)
print('L2 error of z:', errorz)
print('L2 error of q:', errorq)
errory_max, errorp_max = model.max_error(y1, p1, solution1=pde.y_solution, solution2=pde.p_solution)
print('Max error of y:', errory_max)
print('Max error of p:', errorp_max)
errorz_max, errorq_max = model.max_error(z1, q1, solution1=pde.z_solution, solution2=pde.q_solution)
print('Max error of z:', errorz_max)
print('Max error of q:', errorq_max)
