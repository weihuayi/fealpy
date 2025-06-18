import argparse

from fealpy.backend import backend_manager as bm

## 参数解析
parser = argparse.ArgumentParser(description=
    """
    最低阶Raviart-Thomas元和分片常数空间混合元求解椭圆方程
    其中, 刚度矩阵带转移算子项
    """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy. 还可以选择 pytorch, jax, tensorflow 等")


args = parser.parse_args()
bm.set_backend(args.backend)

from fealpy.csm.fem import OPCMixedFEMModel
from fealpy.decorator import barycentric, cartesian
model = OPCMixedFEMModel()
model.set_pde()
model.set_init_mesh(nx=40, ny=40)
model.set_order(p=0)
maxit = 20  # Maximum number of iterations for the optimization process
space1,space2 = model.set_space(p=0)
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
xh = bm.zeros((pdof + ydof), dtype=bm.float64)
hx = bm.zeros((qdof + zdof), dtype=bm.float64)
for i in range(maxit):
    pde = model.pde
    A, b_forward = model.linear_system(model.mesh, p=0, s1=0, s2=0, s3=pde.f_fun, s4=u0)
    A, b_forward = model.apply_bc(A, b_forward, gd=pde.y_solution)
    xh[:] = model.solve(A, b_forward)
    p1[:] = xh[:pdof]
    y1[:] = xh[pdof:]
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
        return -pde.yd_fun(p)
    A,b_backward = model.linear_system(model.mesh, p=0 ,s1=coef_p, s2=coef_pd, s3=coef_y, s4=coef_yd)
    A, b_backward = model.apply_bc(A, b_backward, gd=pde.z_solution)
    hx[:]= model.solve(A, b_backward)
    q1[:] = hx[:qdof]
    z1[:] = hx[qdof:]
    # 假设控制系数为1
    nu = 1
    u1[:] = bm.maximum(0,-z1/nu)
    mesh = model.mesh
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
    
errory, errorp = model.postprocess(y1, p1, solution1=pde.y_solution, solution2=pde.p_solution)
print('L2 error of y:', errory)
print('L2 error of p:', errorp)
errorz , errorq = model.postprocess(z1, q1, solution1=pde.z_solution, solution2=pde.q_solution)
print('L2 error of z:', errorz)
print('L2 error of q:', errorq)

