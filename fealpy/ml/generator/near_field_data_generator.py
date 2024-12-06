
import os
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
from typing import Sequence, Callable

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, QuadrangleMesh, UniformMesh2d
from fealpy.pde.pml_2d import PMLPDEModel2d
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    ScalarDiffusionIntegrator,
    ScalarMassIntegrator,
    ScalarSourceIntegrator,
    ScalarConvectionIntegrator,
    BilinearForm,
    LinearForm,
    DirichletBC
)
from fealpy.solver import spsolve


class NearFieldDataFEMGenerator2d:
    def __init__(self, 
                 domain: Sequence[float],
                 mesh: str,
                 nx: int,
                 ny: int,
                 p: int,
                 q: int,
                 u_inc: str,
                 levelset: Callable[[NDArray], NDArray],
                 d: Sequence[float],
                 k: Sequence[float],
                 reciever_points: NDArray):

        self.domain = domain
        self.nx = nx
        self.ny = ny
        self.p = p
        self.q = q
        self.u_inc = u_inc
        self.levelset = levelset

        # 验证并创建网格
        if mesh not in ['InterfaceMesh', 'QuadrangleMesh', 'UniformMesh']:
            raise ValueError("Invalid value for 'mesh'. Choose from 'InterfaceMesh', 'QuadrangleMesh' or 'UniformMesh'.")
        
        if mesh == 'InterfaceMesh':
            self.mesh = TriangleMesh.interfacemesh_generator(box=self.domain, nx=self.nx, ny=self.ny, phi=self.levelset)  
            self.meshtype = 'InterfaceMesh'
        elif mesh == 'QuadrangleMesh':
            self.mesh = QuadrangleMesh.from_box(box=self.domain, nx=self.nx, ny=self.ny)
            self.meshtype = 'QuadrangleMesh'
        else:
            EXTC_1 = self.nx
            EXTC_2 = self.ny
            HC_1 = 1 / EXTC_1 * (self.domain[1] - self.domain[0])
            HC_2 = 1 / EXTC_2 * (self.domain[3] - self.domain[2])
            self.mesh = UniformMesh2d((0, EXTC_1, 0, EXTC_2), (HC_1, HC_2), origin=(self.domain[0], self.domain[2]))
            self.meshtype = 'UniformMesh'
        
        self.d = d
        self.k = k
        self.reciever_points = reciever_points
        qf = self.mesh.quadrature_formula(self.q, 'cell')
        self.bc, _ = qf.get_quadrature_points_and_weights()

    def get_nearfield_data(self, k: float, d: Sequence[float]):
        """
        获取近场数据。

        参数:
        - k: 波数
        - d: 波矢量方向

        返回:
        - uh: 近场数据
        """
        k_index = self.k.index(k)
        d_index = self.d.index(d)
        pde = PMLPDEModel2d(
            levelset=self.levelset, 
            domain=self.domain,  
            u_inc=self.u_inc,
            A=1,
            k=self.k[k_index],
            d=self.d[d_index],
            refractive_index=[1, 1 + 1 / self.k[k_index]**2],
            absortion_constant=1.79,
            lx=1.0,
            ly=1.0
        )

        space = LagrangeFESpace(self.mesh, p=self.p)

        # 定义积分子
        D = ScalarDiffusionIntegrator(pde.diffusion_coefficient, q=self.q)
        C = ScalarConvectionIntegrator(pde.convection_coefficient, q=self.q)
        M = ScalarMassIntegrator(pde.reaction_coefficient, q=self.q)
        f = ScalarSourceIntegrator(pde.source, q=self.q)

        # 组装双线性形式和线性形式
        b = BilinearForm(space)
        b.add_integrator([D, C, M])

        l = LinearForm(space)
        l.add_integrator(f)

        A = b.assembly()
        F = l.assembly()

        bc = DirichletBC(space, pde.dirichlet) 
        uh = space.function(dtype=bm.complex128)
        A, F = bc.apply(A, F)
        uh[:] = spsolve(A, F, solver='scipy')
        return uh

    def points_location_and_bc(self, p: NDArray, domain: Sequence[float], nx: int, ny: int):
        """
        计算接收点的位置和重心坐标。

        参数:
        - p: 接收点坐标
        - domain: 计算域
        - nx: x方向的网格数
        - ny: y方向的网格数

        返回:
        - location: 接收点所在单元的索引
        - bc: 重心坐标
        """
        x = p[..., 0]
        y = p[..., 1]
        cell_length_x = (domain[1] - domain[0]) / nx
        cell_length_y = (domain[3] - domain[2]) / ny
        index_x = (x - domain[0]) // cell_length_x
        index_y = (y - domain[2]) // cell_length_y
        location = int(index_x * ny + index_y)

        bc_x_ = ((x - domain[0]) / cell_length_x) % 1
        bc_y_ = ((y - domain[2]) / cell_length_y) % 1
        bc_x = bm.array([[bc_x_, 1 - bc_x_]], dtype=bm.float64)
        bc_y = bm.array([[bc_y_, 1 - bc_y_]], dtype=bm.float64)
        bc = (bc_x, bc_y)
        return location, bc

    def data_for_dsm(self, k: float, d: Sequence[float]):
        """
        获取用于DSM的数据。

        参数:
        - k: 波数
        - d: 波矢量方向

        返回:
        - data: DSM数据
        """
        reciever_points = self.reciever_points
        data_length = reciever_points.shape[0]
        data = bm.zeros((data_length,), dtype=bm.complex128)
        uh = self.get_nearfield_data(k=k, d=d)

        if self.meshtype == 'InterfaceMesh':
            b = self.mesh.point_to_bc(reciever_points)
            location = self.mesh.location(reciever_points)
            for i in range(data_length):
                data[i] = uh(b[i])[location[i]]
        elif self.meshtype == 'QuadrangleMesh':
            for i in range(data_length):
                location, b = self.points_location_and_bc(reciever_points[i], self.domain, self.nx, self.ny)
                u = uh(b).reshape(-1)
                data[i] = u[location]
        else:
            for i in range(data_length):
                cell_location = self.mesh.cell_location(reciever_points[i])
                location_column = cell_location[0]
                location_row = cell_location[1]
                location = location_column * self.ny + location_row
                b = self.mesh.point_to_bc(reciever_points[i])
                u = uh(b).reshape(-1)
                data[i] = u[location]
        return data

    def save(self, save_path: str, scatterer_index: int):
        """
        保存数据。

        参数:
        - save_path: 保存路径
        - scatterer_index: 散射体索引
        """
        k_values = self.k
        d_values = self.d
        data_dict = {}
        for i in range(len(k_values)):
            for j in range(len(d_values)):
                k_name = f'k={k_values[i]}'
                d_name = d_values[j]
                name = f"{k_name}, d={d_name}"
                data_dict[name] = self.data_for_dsm(k=k_values[i], d=d_values[j])
        filename = os.path.join(save_path, f"data_for_dsm_{scatterer_index}.npz")
        np.savez(filename, **data_dict)

    def visualization_of_nearfield_data(self, k: float, d: Sequence[float]):
        """
        可视化近场数据。

        参数:
        - k: 波数
        - d: 波矢量方向
        """
        uh = self.get_nearfield_data(k=k, d=d)
        value = uh(self.bc)
        if self.meshtype == 'UniformMesh':
            self.mesh.ftype = bm.float64
        self.mesh.add_plot(plt, cellcolor=value[..., 0].real, linewidths=0)
        self.mesh.add_plot(plt, cellcolor=value[..., 0].imag, linewidths=0)
        
        # TODO: 添加更多可视化选项
        # fig = plt.figure()
        # axes = fig.add_subplot(1, 3, 1)
        # self.mesh.add_plot(axes)
        # if self.meshtype == 'UniformMesh':
        #     uh = uh.view(bm.ndarray)
        # axes = fig.add_subplot(1, 3, 2, projection='3d')
        # self.mesh.show_function(axes, bm.real(uh))
        # axes = fig.add_subplot(1, 3, 3, projection='3d')
        # self.mesh.show_function(axes, bm.imag(uh))
        plt.show()
        