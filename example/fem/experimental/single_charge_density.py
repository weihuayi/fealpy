from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh
from fealpy.model import PDEDataManager
from fealpy.fem import ScalarSourceIntegrator, ScalarMassIntegrator, ScalarDiffusionIntegrator
from fealpy.fem import ScalarNonlinearMassAndDiffusionIntegrator
from fealpy.fem import BilinearForm, LinearForm, NonlinearForm
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DirichletBC
from fealpy.sparse import COOTensor
from fealpy.solver import spsolve
from fealpy.utils import timer


backend = 'pytorch'
device = 'cpu'
bm.set_backend(backend)
bm.set_default_device(device)
tmr = timer()
next(tmr)


def node_project_to_sphere(node, radius):
    """将节点投影到球面上"""
    node_radius = bm.linalg.norm(node, axis=-1, keepdims=True)
    return node * (radius / node_radius)

def coeff1(p, **args):
    return pde.coeff1(p)

def coeff2(p, **args):
    return pde.coeff2(p)


n = 1
p = 2
tol = 1e-4
maxit = 2
h = 0.1
mesh = TetrahedronMesh.from_spherical_shell(r1=0.05, r2=0.5, h=h, device=device)
pde = PDEDataManager('nonlinear').get_example('single')
# mesh = TetrahedronMesh(pde.node, pde.cell)
# mesh = TetrahedronMesh.from_vtu("test_tet_mesh.vtu")
pde.mesh = mesh
tmr.send("网格生成与 PDE 构造")

phi_error_matrix = bm.zeros((3, maxit), dtype=bm.float64, device=device)
rho_error_matrix = bm.zeros((3, maxit), dtype=bm.float64, device=device)


for i in range(maxit):
    print("========================================")
    print("第{}次加密".format(i))
    space = LagrangeFESpace(mesh, p=p)
    # 输出自由度
    print("自由度数量:", space.number_of_global_dofs())
    # 刚度矩阵
    bform = BilinearForm(space)
    integrator1 = ScalarDiffusionIntegrator(coef=1, q=p + 3)
    bform.add_integrator(integrator1)
    A = bform.assembly()
    # 质量矩阵
    bform = BilinearForm(space)
    integrator2 = ScalarMassIntegrator(coef=1, q=p + 3)
    bform.add_integrator(integrator2)
    B = bform.assembly()

    # 右端项
    lform = LinearForm(space)
    integrator3 = ScalarSourceIntegrator(pde.source_f1, q=p + 3)
    lform.add_integrator(integrator3)
    f1 = lform.assembly()

    lform = LinearForm(space)
    integrator4 = ScalarSourceIntegrator(pde.source_f2, q=p + 3)
    lform.add_integrator(integrator4)
    f2 = lform.assembly()

    # 初始解
    phi0 = space.interpolate(pde.init_phi)
    rho0 = space.interpolate(pde.init_rho)

    BC = DirichletBC((space, space), gd=(pde.dirichlet_zero, pde.dirichlet_zero),
                     threshold=(None, None), method='interp')
    gdof = space.number_of_global_dofs()
    tmr.send("线性项矩阵组装")

    while(True):
        # 右端项 F_phi（F1） 计算
        f3 = A @ phi0[:]
        f4 = B @ rho0[:]
        F1 = -(f3 - f4 - f1)
        # 自动微分计算 F_rho（F2） 及其雅可比矩阵（C、D）
        # 上一步解（初始解）以及相关系数
        coeff1.phi_h = phi0
        coeff2.rho_h = rho0
        # 非线性型
        sform = NonlinearForm(space)
        # 针对当前模型的非线性积分子，需要改成更通用形式
        # grad_var=0 表示对第一个变量（phi）求梯度
        integrator5 = ScalarNonlinearMassAndDiffusionIntegrator(coef1=coeff1, coef2=coeff2, grad_var=0, q=p + 3)
        sform.add_integrator(integrator5)
        C, _ = sform.assembly()

        coeff1.phi_h = phi0
        coeff2.rho_h = rho0
        sform = NonlinearForm(space)
        # grad_var=1 表示对第二个变量（rho）求梯度
        integrator6 = ScalarNonlinearMassAndDiffusionIntegrator(coef1=coeff1, coef2=coeff2, grad_var=1, q=p + 3)
        sform.add_integrator(integrator6)
        D, F2 = sform.assembly()
        F2 = F2 + f2
        tmr.send("非线性项矩阵组装")

        # ============================================================
        # 矩阵转换成 COO，并拼接成一个大矩阵
        A = A.tocoo()
        B = B.tocoo()
        C = C.tocoo()
        D = D.tocoo()
        temp_line1 = COOTensor.concat([A, B], axis=1)
        temp_line2 = COOTensor.concat([C, D], axis=1)
        F_jacobi = COOTensor.concat([temp_line1, temp_line2], axis=0)
        F_jacobi = F_jacobi.tocsr()
        # ============================================================
        # 右端项拼接
        F = bm.concat([F1, F2], axis=0)

        # 边界条件处理
        F_jacobi, F = BC.apply(F_jacobi, F)
        tmr.send("矩阵拼接与边界条件")
        # 求解
        deltax = spsolve(F_jacobi, F,solver='mumps')
        tmr.send("线性方程组求解")
        # 解向量更新
        phi0[:] += deltax[:gdof]
        rho0[:] += deltax[gdof:]

        print('迭代误差：', bm.max(bm.abs(deltax)))
        if bm.max(bm.abs(deltax)) < tol:
            break

    phiso = space.interpolate(pde.solution_phi)
    rhoso = space.interpolate(pde.solution_rho)
    max_phi_error = bm.max(bm.abs(phiso[:] - phi0[:]))
    max_rho_error = bm.max(bm.abs(rhoso[:] - rho0[:]))
    phi_error_matrix[0, i] = max_phi_error
    rho_error_matrix[0, i] = max_rho_error
    phierror = mesh.error(pde.solution_phi, phi0.value)
    rhoerror = mesh.error(pde.solution_rho, rho0.value)
    phi_error_matrix[1, i] = phierror
    rho_error_matrix[1, i] = rhoerror
    gphierror = mesh.error(pde.gradient_phi, phi0.grad_value)
    grhoerror = mesh.error(pde.gradient_rho, rho0.grad_value)
    phi_error_matrix[2, i] = gphierror
    rho_error_matrix[2, i] = grhoerror

    # print("第{}次迭代误差：".format(i))
    # print("max error:")
    # print("phi error:", max_phi_error)
    # print("rho error:", max_rho_error)
    # print("L2 error:")
    # print(phierror)
    # print(rhoerror)
    # print("L2 gradient error:")
    # print(gphierror)
    # print(grhoerror)

    mesh.nodedata['phi_h'] = phi0[:]
    mesh.nodedata['phi'] = phiso[:]
    mesh.nodedata['rho_h'] = rho0[:]
    mesh.nodedata['rho'] = rhoso[:]

    # 将结果保存为VTU文件
    # mesh.to_vtk(fname=f"single_charge_density_refine{i}_U{pde.U0}_projection_mesh2_h{h}_p{p}.vtu")

    # 加密网格并将边界投影到对应球面上
    if i < maxit - 1:
        mesh.uniform_refine()
        node = mesh.node
        bd_node_idx = mesh.boundary_node_index()
        bd_node = node[bd_node_idx]
        bd_node_radius = bm.linalg.norm(bd_node, axis=-1, keepdims=True)
        is_inner_node = (bd_node_radius < (pde.r1+pde.r2)/2).reshape(-1)
        is_outer_node = (bd_node_radius > (pde.r1+pde.r2)/2).reshape(-1)
        bd_node[is_inner_node] = node_project_to_sphere(bd_node[is_inner_node], pde.r1)
        bd_node[is_outer_node] = node_project_to_sphere(bd_node[is_outer_node], pde.r2)
        mesh.node[bd_node_idx] = bd_node
        pde.mesh = mesh
        tmr.send("网格加密")

next(tmr)
# 输出误差矩阵
print("phi error matrix:")
print(phi_error_matrix)
print("rho error matrix:")
print(rho_error_matrix)
# 收敛阶
phi_order = bm.log2(phi_error_matrix[:, :-1] / phi_error_matrix[:, 1:])
rho_order = bm.log2(rho_error_matrix[:, :-1] / rho_error_matrix[:, 1:])
print("phi order:")
print(phi_order)
print("rho order:")
print(rho_order)




