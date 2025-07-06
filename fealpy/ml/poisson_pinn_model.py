import time
import torch.nn as nn
# from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

from ..backend import bm
from typing import Union
from ..typing import TensorLike
from ..model import ComputationalModel, PDEDataManager
from ..model.poisson import PoissonPDEDataT
from . import gradient
from ..mesh import TriangleMesh, UniformMesh
from .modules import Solution
from .sampler import BoxBoundarySampler, ISampler


class PoissonPINNModel(ComputationalModel):
    """A Physics-Informed Neural Network (PINN) model for solving Poisson equations.
    
    This class implements a PINN framework to solve Poisson PDE problems using neural networks.
    It handles PDE residual calculation, boundary condition enforcement, and training process.
    The model supports both uniform and random sampling strategies for collocation points.
    
    Parameters
        options : dict
            Configuration dictionary containing:
            - pde: PDE definition (str or PoissonPDEDataT)
            - meshtype: Type of mesh ('tri' or 'uni')
            - lr: Learning rate
            - epochs: Number of training epochs
            - hidden_sizes: List of hidden layer sizes
            - npde: Number of PDE collocation points
            - nbc: Number of boundary collocation points
            - activation: Activation function
            - loss: Loss function
            - optimizer: Optimization algorithm
            - step_size: Learning rate scheduler step size
            - gamma: Learning rate decay factor
            - sampling_mode: Sampling strategy ('linspace' or 'random')
        
    Attributes
        pde : PoissonPDEDataT
            The Poisson PDE problem definition
        gd : int
            Geometric dimension of the problem
        domain : tuple
            Computational domain boundaries
        mesh : TriangleMesh or UniformMesh
            Discretization mesh for error estimation
        sampler_pde : ISampler
            Sampler for PDE collocation points
        sampler_bc : BoxBoundarySampler
            Sampler for boundary points
        net : Solution
            Neural network approximator
        optimizer : torch.optim.Optimizer
            Training optimizer
        scheduler : torch.optim.lr_scheduler.StepLR
            Learning rate scheduler
        Loss : list
            Training loss history
        error_fem : list
            FEM error history
        error_true : list
            True solution error history
    
    Methods
        set_pde(pde)
            Initialize the PDE problem
        set_network(net)
            Configure the neural network architecture
        set_steplr(step_size, gamma)
            Setup learning rate scheduler
        set_mesh(type)
            Initialize computational mesh
        pde_residual(p, net)
            Compute PDE residual
        bc_residual(p, net)
            Compute boundary condition residual
        train()
            Execute training process
        predict(p)
            Make predictions at given points
        show()
            Visualize results
    
    Notes
        The implementation uses automatic differentiation for:
        1. PDE residual calculation (Laplacian computation)
        2. Backpropagation during training
        The loss function combines PDE residual and boundary condition terms.
    
    Examples
        Basic usage:
        
        >>> options = {
        ...     'pde': 'sinsin',
        ...     'meshtype': 'tri',
        ...     'lr': 0.001,
        ...     'epochs': 1000,
        ...     'hidden_sizes': [20, 20],
        ...     'npde': 100,
        ...     'nbc': 50,
        ...     'activation': nn.Tanh(),
        ...     'loss': nn.MSELoss(),
        ...     'optimizer': torch.optim.Adam,
        ...     'sampling_mode': 'random'
        ... }
        >>> model = PoissonPINNModel(options)
        >>> model.train()
        >>> model.show()
    """
    def __init__(self, options):
        self.options = options
        self.set_pde(self.options['pde'])

        self.gd = self.pde.geo_dimension()
        self.domain = self.pde.domain()
        self.set_mesh(self.options['meshtype'])

        # 采样器
        self.sampler_pde = ISampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])
        self.sampler_bc = BoxBoundarySampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])

        # 网络超参数、采样点数、激活函数
        self.lr = self.options['lr']
        self.epochs = self.options['epochs']
        self.hidden_sizes = self.options['hidden_sizes']
        self.npde = self.options['npde']
        self.nbc = self.options['nbc']
        self.activation = self.options['activation']

        # 损失函数
        self.loss = self.options['loss']

        # 网络
        self.set_network()

        # 优化器与学习率调度器
        self.optimizer = self.options["optimizer"](params=self.net.parameters(), lr=self.lr)
        self.set_steplr(self.options['step_size'], self.options['gamma'])
        
    def set_pde(self, pde: Union[PoissonPDEDataT, str]='sinsin'):
        """Initialize the PDE problem definition.
        
        Parameters
            pde : Union[PoissonPDEDataT, str]
                Either a predefined PDE object or string identifier for built-in examples.
                Defaults to 'sinsin' example problem.
        """
        if isinstance(pde, str):
            self.pde = PDEDataManager('poisson').get_example(pde)
        else:
            self.pde = pde 

    def set_network(self, net=None):
        """Configure the neural network architecture.
        
        Parameters
            net : torch.nn.Module, optional
                Custom network architecture. If None, creates default MLP.
                Defaults to None (auto-create network).
        """
        if net == None:
            layers = []
            sizes = (self.gd,) + self.hidden_sizes + (1,)
            for i in range(len(sizes)-1):
                layers.append(nn.Linear(sizes[i], sizes[i+1], dtype=bm.float64))
                if i < len(sizes)-2:  
                    layers.append(self.activation)
            net = nn.Sequential(*layers)
        self.net = Solution(net)

    def set_steplr(self, step_size=0, gamma=0.1):
        """
        """
        if step_size == 0:
            self.scheduler = None
        else:
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def set_mesh(self, type='tri'):
        if self.gd > 1:
            if type == "tri":
                self.mesh = TriangleMesh.from_box(self.domain, nx=30, ny=30)
            elif type == 'uni':
                 self.mesh = UniformMesh(self.domain, (0, 30)*self.gd)
        else:
            self.mesh = UniformMesh(self.domain, (0, 30)*self.gd)
    



    def pde_residual(self, p: TensorLike) -> TensorLike:
        """Compute PDE residual (Laplacian(u) + f).
        
        Parameters
            p : TensorLike
                Collocation points where residual is evaluated
            net : torch.nn.Module
                Neural network approximator
            
        Returns
            TensorLike
                PDE residual values at input points
        """
        u = self.net(p)
        f = self.pde.source(p)
        
        # 一阶导数计算
        grad_u = gradient(u, p, create_graph=True)  ## 一阶导数计算(npde, dim)
        laplacian = bm.zeros(u.shape[0])    # 拉普拉斯项初始化
        
        for i in range(p.shape[-1]):
            u_ii = gradient(grad_u[..., i], p, create_graph=True, split=True)[i]   # 计算 ∂²u/∂x_i²
            laplacian += u_ii.flatten()

        assert f.shape == laplacian.shape, \
            f"Shape mismatch: f.shape={f.shape}, laplacian.shape={laplacian.shape}."
        val = laplacian + f
        return val

    def bc_residual(self, p: TensorLike) -> TensorLike:
        """

        """
        u = self.net(p).flatten()
        bc = self.pde.dirichlet(p)
        # print("p ", p, "bc ", bc, "u ", u, sep='\n')
        assert u.shape == bc.shape, \
            f"Shape mismatch: u.shape={u.shape}, bc.shape={bc.shape}."
        val = u - self.pde.dirichlet(p)
        # print("bc:", "u.shape", u.shape, " bc.shape",  bc.shape, "val.shape", val.shape)
        return val

    def train(self):
        """Execute the training process.
        
        Notes
            Implements:
            1. Collocation point sampling
            2. PDE and BC residual computation
            3. Loss backpropagation
            4. Periodic error evaluation
            5. Learning rate scheduling
       """
        start_time = time.time()
        self.Loss = []
        self.error_fem = []
        self.error_true = []

        mesh = self.mesh

        for epoch in range(self.epochs+1):
            self.optimizer.zero_grad()

            # 采样点
            if (self.options['sampling_mode'] == 'linspace') :
                if epoch == 0:
                    ''' 均匀采样只采一次 '''
                    spde = self.sampler_pde.run(self.npde)
                    sbc = self.sampler_bc.run(self.nbc)
                else:
                    spde = spde
                    sbc = sbc
            else:
                spde = self.sampler_pde.run(self.npde)
                sbc = self.sampler_bc.run(self.nbc)

            # 计算残差
            pde_res = self.pde_residual(spde)
            bc_res = self.bc_residual(sbc)

            # 计算损失
            mse_pde = self.loss(pde_res, bm.zeros_like(pde_res))
            mse_bc = self.loss(bc_res, bm.zeros_like(bc_res))

            loss = 0.7 * mse_pde + 0.3 * mse_bc
            loss.backward()
            # 更新参数
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step() 

            if epoch % 100 == 0:
                error = self.net.estimate_error(self.pde.solution, mesh, coordtype='c')
                # e0 = self.net.estimate_error_func(self.pde.solution, mesh)
                # self.error_true.append(e0.detach().numpy())
                self.error_fem.append(error.detach().numpy())
                self.Loss.append(loss.detach().numpy())
                print(f"epoch: {epoch}, loss: {loss.item():.4f}, error_fem: {error.item():.4f}")
               

        end_time = time.time()
        print(f'Training completed in {end_time - start_time:.2f} seconds.')

    def prefict(self, p: TensorLike) -> TensorLike:
        return self.net(p)

    def show(self):
        """
        在同一三维坐标系中可视化真解与PINN预测解
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

        # 绘制损失曲线 (顶部子图)
        axes[0].plot(self.Loss, 'r-', linewidth=2)
        axes[0].set_title('Training Loss', fontsize=12)
        axes[0].set_ylabel('Loss Value', fontsize=10)
        axes[0].grid(True)

        # 绘制PINN vs FEM误差 (中间子图)
        axes[1].plot(self.error_fem, 'b--', linewidth=2)
        axes[1].set_title('Error between PINN and FEM Solution', fontsize=12)
        axes[1].set_ylabel('Error', fontsize=10)
        axes[1].grid(True)

        # 绘制PINN vs 真实解误差 (底部子图)
        # axes[2].plot(self.error_true, 'g-.', linewidth=2)
        # axes[2].set_title('Error between PINN and True Solution', fontsize=12)
        # axes[2].set_xlabel('Epoch/Iteration', fontsize=10)
        # axes[2].set_ylabel('Error', fontsize=10)
        # axes[2].grid(True)

        
        if self.gd <= 2:
            mesh = self.mesh
            node = mesh.entity('node')
            # 获取预测解和真解
            u_pred = self.net(node).detach().numpy().flatten()  # PINN预测解
            u_true = self.pde.solution(node).detach().numpy()   # 解析解
            
            fig = plt.figure()
            if self.gd == 1:
                # 绘制真实解和预测解
                plt.plot(node, u_true, 'b-', linewidth=3, label='Analytical Solution')
                plt.plot(node, u_pred, 'r--', linewidth=2, label='PINN Prediction')

                # 图形修饰
                plt.xlabel('x', fontsize=12)
                plt.ylabel('u(x)', fontsize=12)
                plt.title('Comparison between PINN and Analytical Solution', fontsize=14)
                plt.legend(fontsize=12)
                plt.grid(True, linestyle=':')
            else:
                # 子图1：PINN预测解
                ax1_3d = fig.add_subplot(121, projection='3d')
                surf1 = ax1_3d.plot_trisurf(
                    node[:, 0], node[:, 1], u_pred,
                    cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.8)
                ax1_3d.set_title('PINN Predicted Solution')
                ax1_3d.set_xlabel('X')
                ax1_3d.set_ylabel('Y')
                ax1_3d.set_zlabel('u(x,y)')
                fig.colorbar(surf1, ax=ax1_3d, shrink=0.5, label='Value')

                # 子图2：真解
                ax2_3d = fig.add_subplot(122, projection='3d')
                surf2 = ax2_3d.plot_trisurf(
                    node[:, 0], node[:, 1], u_true,
                    cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
                ax2_3d.set_title('Analytical True Solution')
                ax2_3d.set_xlabel('X')
                ax2_3d.set_ylabel('Y')
                ax2_3d.set_zlabel('u(x,y)')
                fig.colorbar(surf2, ax=ax2_3d, shrink=0.5, label='Value')
                plt.suptitle('Solution Comparison: PINN vs Analytical')
        plt.tight_layout()      
        plt.show()  # 显示第二个Figure（3D图）


