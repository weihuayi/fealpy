import time
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

from ..backend import bm
from typing import Union
from ..mesh import TriangleMesh
from ..typing import TensorLike
from ..model import ComputationalModel, PDEDataManager
from ..model.poisson import PoissonPDEDataT
from . import gradient


from .modules import Solution
from .sampler import BoxBoundarySampler, ISampler


class PoissonPINNModel(ComputationalModel):
    def __init__(self, options):
        self.options = options
        self.set_pde(self.options['pde'])

        self.gd = self.pde.geo_dimension()
        self.domain = self.pde.domain()

        # 采样器
        self.sampler_pde = ISampler(self.domain, requires_grad=True)
        self.sampler_bc = BoxBoundarySampler(self.domain, requires_grad=True)

        # 网络超参数、采样点数、激活函数
        self.lr = self.options['lr']
        self.epochs = self.options['epochs']
        self.hidden_sizes = self.options['hidden_sizes']
        self.npde = self.options['npde']
        self.nbc = self.options['nbc']
        self.activation = self.options['activation']

        # 损失函数
        self.loss = self.options['loss']
        

        

    def set_pde(self, pde: Union[PoissonPDEDataT, str]='sinsin'):
        """
        """
        if isinstance(pde, str):
            self.pde = PDEDataManager('poisson').get_example(pde)
        else:
            self.pde = pde 

    def set_optimizer(self, net, optimizer=None,):
        """
        """
        if optimizer==None:
            self.optimizer = Adam(params=net.parameters(), lr=self.lr)
        # else:
        #     self.optimizer = optimizer(params=self.net.parameters(), lr=self.lr)

    def set_steplr(self, step_size=0, gamma=0.1):
        """
        """
        if step_size == 0:
            self.scheduler = None
        else:
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def create_mesh(self, mesh=None):
        """
        """
        if mesh is None:
            self.mesh = TriangleMesh.from_box(self.domain, nx=64, ny=64)
        return self.mesh


    def create_network(self, net=None):
        """ """
        if net == None:
            layers = []
            sizes = (self.gd,) + self.hidden_sizes + (1,)
            for i in range(len(sizes)-1):
                layers.append(nn.Linear(sizes[i], sizes[i+1], dtype=bm.float64))
                if i < len(sizes)-2:  # 不在最后一层后加激活函数
                    layers.append(self.activation)
            net = nn.Sequential(*layers)
        return Solution(net)

    import torch

    def pde_residual(self, p: TensorLike, net) -> TensorLike:
        """
        适用于任意维度(1D/2D/3D)的PDE残差计算
        Args:
            p: 输入坐标点 (npde, dim), dim=1,2,3
            pde: 需实现source()方法返回源项
            net: PINN网络模型
        Returns:
            PDE残差值 (batch_size, 1)
        """
        u = net(p)
        f = self.pde.source(p)
    
        u_x, u_y = gradient(u, p, create_graph=True, split=True)
        u_xx, _ = gradient(u_x, p, create_graph=True, split=True)
        _, u_yy = gradient(u_y, p, create_graph=True, split=True)
        # print(bm.sum(bm.abs(u_xx+u_yy)<1e-3))
        return u_xx + u_yy + f
        # dim = p.shape[-1]  # 自动获取空间维度
        
        # # 一阶导数计算
        # grad_u = gradient(u, p, create_graph=True)  # (npde, dim)
        
        # # 拉普拉斯项初始化
        # laplacian = bm.zeros_like(u)
        
        # # 特殊处理1D情况（不需要split）
        # if dim == 1:
        #     u_xx = gradient(grad_u, p, create_graph=True)
        #     laplacian = u_xx
        # else:
        #     # 高维情况计算各方向二阶导
        #     for i in range(dim):
        #         # 计算∂²u/∂x_i²
        #         grad_u_i = grad_u[..., i:i+1]  # 保持维度 (npde, 1)
        #         u_ii = gradient(grad_u_i, p, create_graph=True, split=True)[0]
        #         laplacian += u_ii
        
        # return laplacian + self.pde.source(p)

    def bc_residual(self, p: TensorLike, net) -> TensorLike:
        """
        适用于任意维度(1D/2D/3D)的边界条件残差计算
        Args:
            p: 输入坐标点 (nbc, dim), dim=1,2,3
            pde: 需实现bc()方法返回边界条件
            net: PINN网络模型
        Returns:
            边界条件残差值 (nbc, 1)
        """
        u = net(p)
        return u - self.pde.dirichlet(p)

    def train(self):
        """
        """
        def print_network_parameters(net):
            """
            打印网络中所有可训练参数的名称和值
            """
            for name, param in net.named_parameters():
                print(f"Layer: {name} | Shape: {param.shape}")
                print(param.data)  # 输出参数值（不包含梯度）
                print("-" * 50)
        net = self.create_network()
        # print_network_parameters(net)
        # 优化器与学习率调度器
        self.set_optimizer(net =net, optimizer=None)
        # self.set_steplr(self.options['step_size'], self.options['gamma'])
        start_time = time.time()
        Loss = []
        Error = []
        e = []
        mesh = self.create_mesh()
        node = mesh.entity('node')

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # 采样点
            spde = self.sampler_pde.run(self.npde)
            sbc = self.sampler_bc.run(self.nbc, self.nbc)

            # 计算残差
            pde_res = self.pde_residual(spde, net)
            bc_res = self.bc_residual(sbc, net)

            # 计算损失
            mse_pde = self.loss(pde_res, bm.zeros_like(pde_res))
            mse_bc = self.loss(bc_res, bm.zeros_like(bc_res))

            loss = 0.5 * mse_pde + 0.5 * mse_bc
            loss.backward()
            # 更新参数
            self.optimizer.step()
            # if self.scheduler:
            #     self.scheduler.step()

            if epoch % 10 == 0:
                error = net.estimate_error(self.pde.solution, mesh, coordtype='c')
                e0 = bm.mean((net(node)-self.pde.solution(node))**2)
                e.append(e0.detach().numpy())
                Error.append(error.detach().numpy())
                Loss.append(loss.detach().numpy())
               

        end_time = time.time()
        print(f'Training completed in {end_time - start_time:.2f} seconds.')
        # print_network_parameters(net)
        return Loss, Error, e
    
    # def show(self):
    #     """
    #     在同一三维坐标系中可视化真解与PINN预测解
    #     """
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

    #     mesh = self.mesh
    #     node = mesh.entity('node')
        
    #     # 获取预测解和真解
    #     u_pred = self.net(node).detach().numpy().flatten()  # PINN预测解
    #     u_true = self.pde.solution(node).detach().numpy()   # 解析解

    #     # 创建三维图形
    #     fig = plt.figure(figsize=(12, 6))
        
    #     # 子图1：PINN预测解
    #     ax1 = fig.add_subplot(121, projection='3d')
    #     surf1 = ax1.plot_trisurf(
    #         node[:, 0], node[:, 1], u_pred, 
    #         cmap='viridis', edgecolor='none', alpha=0.8
    #     )
    #     ax1.set_title('PINN Predicted Solution')
    #     ax1.set_xlabel('X-axis')
    #     ax1.set_ylabel('Y-axis')
    #     ax1.set_zlabel('u(x,y)')
    #     fig.colorbar(surf1, ax=ax1, shrink=0.5, label='Value')

    #     # 子图2：真解
    #     ax2 = fig.add_subplot(122, projection='3d')
    #     surf2 = ax2.plot_trisurf(
    #         node[:, 0], node[:, 1], u_true, 
    #         cmap='plasma', edgecolor='none', alpha=0.8
    #     )
    #     ax2.set_title('Analytical True Solution')
    #     ax2.set_xlabel('X-axis')
    #     ax2.set_ylabel('Y-axis')
    #     ax2.set_zlabel('u(x,y)')
    #     fig.colorbar(surf2, ax=ax2, shrink=0.5, label='Value')

    #     plt.tight_layout()
    #     plt.show()


