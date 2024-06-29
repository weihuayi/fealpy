import numpy as np

from fealpy.decorator import cartesian

class BoxDomainData2d:
    """
    @brief Dirichlet 边界条件的线弹性问题模型
    @note 本模型假设在二维方形区域 [0,1] x [0,1] 内的线性弹性问题
    """
    def __init__(self, E=1.0, nu=0.3):
        """
        @brief 构造函数
        @param[in] E 弹性模量，默认值为 1.0
        @param[in] nu 泊松比，默认值为 0.3
        """
        self.E = E 
        self.nu = nu

        self.lam = self.nu * self.E / ((1 + self.nu) * (1 - 2*self.nu))
        self.mu = self.E / (2 * (1+self.nu))


    @cartesian
    def source(self, p):
        """
        @brief 模型的源项值 f
        """

        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
        val[..., 1] = -25/26*(-1+2*y) * (-1+2*x)

        return val

    @cartesian
    def solution(self, p):
        """
        @brief 模型真解
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = x*(1-x)*y*(1-y)
        val[..., 1] = 0

        return val


    @cartesian
    def dirichlet(self, p):
        """
        @brief Dirichlet 边界条件
        """

        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief 判断给定点是否在 Dirichlet 边界上
        @param[in] p 一个表示空间点坐标的数组
        @return 如果在 Dirichlet 边界上，返回 True，否则返回 False
        """
        x = p[..., 0]
        y = p[..., 1]
        flag1 = np.abs(x) < 1e-13
        flag2 = np.abs(x - 1) < 1e-13
        flagx = np.logical_or(flag1, flag2)
        flag3 = np.abs(y) < 1e-13
        flag4 = np.abs(y - 1) < 1e-13
        flagy = np.logical_or(flag3, flag4)
        flag = np.logical_or(flagx, flagy)

        return flag

from fealpy.np.functionspace import LagrangeFESpace as Space
from fealpy.np.mesh.triangle_mesh import TriangleMesh

from fealpy.np.fem import (
    LinearElasticityPlaneStrainOperatorIntegrator, BilinearForm, LinearForm
    )

from fealpy.utils import timer
from matplotlib import pyplot as plt

tmr = timer()

mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=2, ny=2)
next(tmr)

fig = plt.figure(figsize=(12, 9))
fig.tight_layout()
fig.suptitle('Solving Poisson equation on 2D Triangle mesh')

axes = fig.add_subplot(1, 1, 1)
mesh.add_plot(axes, cmap='jet', linewidths=0, showaxis=True)

plt.show()


space = Space(mesh, p=1, ctype='C')
GD = 2
vspace = GD*(space, )
tmr.send('mesh_and_vspace')

uh = vspace[0].function(dim=GD)
print("uh:", uh.shape, "\n", uh)
gdof = vspace[0].number_of_global_dofs()
vgdof = gdof * GD
ldof = vspace[0].number_of_local_dofs()
vldof = ldof * GD
cell2dof = space.cell_to_dof()


# 材料参数
E0 = 1.0  # 弹性模量
nu = 0.3  # 泊松比
lam = (E0 * nu) / ((1 + nu) * (1 - 2 * nu))
mu = E0 / (2 * (1 + nu))
integrator = LinearElasticityPlaneStrainOperatorIntegrator(lam=lam, mu=mu)

bform = BilinearForm(vspace)
bform.add_integrator(integrator)
tmr.send('forms')

K = bform.assembly()
tmr.send('assembly')
