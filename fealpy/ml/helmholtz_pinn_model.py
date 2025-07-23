
import torch.nn as nn
# from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

from ..backend import bm
from ..utils import timer

from typing import Union, Optional
from ..typing import TensorLike
from ..model import ComputationalModel, PDEModelManager
from ..model.poisson import PoissonPDEDataT
from . import gradient
from ..mesh import TriangleMesh, UniformMesh, Mesh
from .modules import Solution
from .sampler import BoxBoundarySampler, ISampler


class HelmholtzPINNModel(ComputationalModel):
    """Physics-Informed Neural Network (PINN) model for solving Helmholtz equations.
    
    Implements a PINN framework to solve Helmholtz PDE problems using neural networks.
    Handles PDE residual calculation, boundary condition enforcement, and training process.
    Supports both uniform and random sampling strategies for collocation points.
    Specialized for complex-valued solutions (real + imaginary components).
    
    Parameters
        options : dict
            Configuration dictionary containing:
            - pde: PDE definition (str or PoissonPDEDataT);
            - meshtype: Type of mesh ('tri' or 'uni');
            - lr: Learning rate;
            - epochs: Number of training epochs;
            - hidden_sizes: List of hidden layer sizes;
            - npde: Number of PDE collocation points;
            - nbc: Number of boundary collocation points;
            - activation: Activation function;
            - loss: Loss function;
            - optimizer: Optimization algorithm;
            - sampling_mode: Sampling strategy ('linspace' or 'random');
            - complex: Boolean flag for complex-valued solutions;
            - wave: Wave number k for Helmholtz equation.
        
    Attributes
        pde : PoissonPDEDataT
            Helmholtz PDE problem definition
        gd : int
            Geometric dimension
        domain : list
            Computational domain boundaries
        mesh : TriangleMesh or UniformMesh
            Discretization mesh for error estimation
        sampler_pde : ISampler
            Sampler for PDE collocation points
        sampler_bc : BoxBoundarySampler
            Sampler for boundary points
        net : torch.nn.Module
            Neural network model
        optimizer : torch.optim.Optimizer
            Training optimizer
        Loss : list
            Training loss history
        error_real : list
            Real part error history (vs FEM solution)
        error_imag : list
            Imaginary part error history (vs FEM solution, only when complex=True)
        options : dict
            Initial configuration dictionary
        complex : bool
            Flag indicating complex-valued solutions
        k : float
            Wave number for Helmholtz equation
    
    Reference
        https://wnesm678i4.feishu.cn/wiki/U219wwT18iH4v7kNTOacxl8cnXb?from=from_copylink
    
    Methods
        set_pde(pde)
            Initialize PDE problem
        set_network(net)
            Configure neural network architecture
        set_mesh(type)
            Initialize computational mesh
        set_n(p)
            Compute normal vectors at boundary points
        pde_residual(p)
            Compute PDE residual (Δu + k²u + f)
        bc_residual(p)
            Compute boundary condition residual
        run()
            Execute training process
        predict(p)
            Make predictions at given points
        show()
            Visualize results
    
    Notes
        Key features:
        1. Supports complex-valued solutions: When complex=True, network outputs two channels (real + imaginary)
        2. Helmholtz residual: Δu + k²u + f = 0
        3. Supports multiple boundary conditions: Dirichlet and Robin
        4. Uses automatic differentiation for higher-order derivatives
        5. Loss function handles real/imaginary components separately
        
        Complex-valued solution handling:
        - Network output layer: 2 channels when complex=True (real + imaginary), else 1 channel
        - Residual calculation: Separate computation for real and imaginary components
        - Loss function: Weighted sum of real and imaginary PDE/boundary losses
        - Error evaluation: Separate error calculation for real and imaginary components
        
    Examples
        >>> options = {
        ...     'pde': 1,  # Built-in example ID
        ...     'meshtype': 'uniform_tri',
        ...     'lr': 0.001,
        ...     'epochs': 1000,
        ...     'hidden_sizes': (32, 32, 16),
        ...     'npde': 400,
        ...     'nbc': 100,
        ...     'activation': nn.Tanh(),
        ...     'loss': nn.MSELoss(),
        ...     'optimizer': torch.optim.Adam,
        ...     'sampling_mode': 'random',
        ...     'complex': True,  # Enable complex-valued solution
        ...     'wave': 1.0  # Wave number k
        ... }
        >>> model = HelmholtzPINNModel(options)
        >>> model.run()
        >>> model.show()
    """
    def __init__(self, options):
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.options = options
        self.complex = self.options['complex']
        self.k = self.options['wave']

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
        self.steplr = StepLR(self.optimizer, self.options['step_size'], self.options['gamma'])
        
    def set_pde(self, pde: Union[PoissonPDEDataT, int]=1):
        """Initialize the PDE problem definition.
        
        Parameters
            pde : Union[PoissonPDEDataT, int]
                PDE object or built-in example ID.
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('helmholtz').get_example(pde, k=self.k)
        else:
            self.pde = pde 


    def set_network(self, net=None):
        """Configure the neural network architecture.
        
        Parameters
            net : torch.nn.Module, optional
                Custom network architecture. If None, creates default MLP.
                
        Notes
            When complex=True:
            - Output layer dimension is 2 (real + imaginary)
            - Solution wrapper handles complex-valued output
        """
        if net == None:
            layers = []
            sizes = (self.gd,) + self.hidden_sizes
            for i in range(len(sizes)-1):
                layers.append(nn.Linear(sizes[i], sizes[i+1], dtype=bm.float64))
                if i < len(sizes)-1:  
                    layers.append(self.activation)
                    
            if self.complex:
                layers.append(nn.Linear(sizes[-1], 2, dtype=bm.float64))
            else:
                layers.append(nn.Linear(sizes[-1], 1, dtype=bm.float64))
            net = nn.Sequential(*layers)
            
        self.net = Solution(net, self.complex)

    def set_mesh(self, mesh: Union[Mesh, str] = "uniform_tri"):
        """Create computational mesh.
        
        Args:
            mesh: Mesh instance or mesh type name string
        """
        if isinstance(mesh, str):
            if self.gd == 2:
                self.mesh = self.pde.init_mesh[mesh](30, 30)
            elif self.gd ==3:
                self.mesh = self.pde.init_mesh[mesh](30, 30, 30)
        else:
            self.mesh = mesh
            
    def set_n(self, p: TensorLike) -> TensorLike:
        """Compute normal vectors at boundary points (for rectangular domains)
        
        Parameters
            p : TensorLike
                Boundary point coordinates
                
        Returns
            TensorLike
                Unit normal vectors
        """
        n = bm.zeros_like(p)
        tol = 1e-4
        dim = self.gd
        coords = [p[..., i] for i in range(dim)]  # 分解各维度坐标
        
        # 边界处理优先级顺序（按维度从低到高）
        for axis in range(dim):
            min_val, max_val = self.domain[2*axis], self.domain[2*axis+1]
            
            # 当前维度边界掩码
            min_mask = bm.abs(coords[axis] - min_val) <= tol
            max_mask = bm.abs(coords[axis] - max_val) <= tol
            
            # 生成active_mask：如果是第一个维度（axis=0），则全部点为active；否则，只选择未被标记的点
            if axis == 0:
                active_mask = bm.ones_like(min_mask, dtype=bool)  # 全部为True
            else:
                active_mask = ~bm.any(n != 0, axis=-1)  # 未被更高优先级标记的点
            
            # 设置法向量分量
            n[min_mask & active_mask, axis] = -1.0  # 负向边界
            n[max_mask & active_mask, axis] = 1.0   # 正向边界
        
        return n

    def pde_residual(self, p: TensorLike) -> TensorLike:
        """Compute PDE residual (Δu + k²u + f)
        
        Parameters
            p : TensorLike
                Collocation point coordinates
                
        Returns
            TensorLike
                PDE residual values
                
        Notes
            Helmholtz equation form: Δu + k²u + f = 0
            Uses automatic differentiation to compute Laplacian
        """
        u = self.net(p)
        f = self.pde.source(p).flatten()
        
        # 一阶导数计算
        grad_u = gradient(u, p, create_graph=True)  ## (npde, dim)
        laplacian = bm.zeros(u.shape[0])    # 拉普拉斯项初始化
        
        for i in range(p.shape[-1]):
            u_ii = gradient(grad_u[..., i], p, create_graph=True, split=True)[i]   # 计算 ∂²u/∂x_i²
            laplacian += u_ii.flatten()

        assert f.shape == laplacian.shape, \
            f"Shape mismatch: f.shape={f.shape}, laplacian.shape={laplacian.shape}."
        val = laplacian + self.pde.k * u.flatten() + f
        return val

    def bc_residual(self, p: TensorLike) -> TensorLike:
        """Compute boundary condition residual
        
        Parameters
            p : TensorLike
                Boundary point coordinates
                
        Returns
            TensorLike
                Boundary condition residual values
                
        Notes
            Supported boundary conditions:
            1. Dirichlet: u - g = 0
            2. Robin: αu + β∂u/∂n - γ = 0
        """
        u = self.net(p).flatten()
        if hasattr(self.pde, 'dirichlet'):
            bc = self.pde.dirichlet(p)
        elif hasattr(self.pde, 'robin'):
            n = self.set_n(p)
            bc = self.pde.robin(p, n)

        assert u.shape == bc.shape, \
            f"Shape mismatch: u.shape={u.shape}, bc.shape={bc.shape}."
        val = u - bc
        return val

    def run(self):
        """Execute training process.
        
        Notes
            Training workflow:
            1. Sample collocation points (domain + boundary)
            2. Compute PDE and boundary residuals
            3. Separate real/imaginary loss computation (when complex=True)
            4. Backpropagation and parameter update
            5. Periodic error evaluation
            
            Complex-valued solution handling:
            - Separate computation of real/imaginary PDE and boundary residuals
            - Total loss = real_PDE_loss + imag_PDE_loss + real_BC_loss + imag_BC_loss
        """
        tmr = timer()
        next(tmr)
        self.Loss = []
        self.error_real= []
        self.error_imag = []

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
                spde = self.sampler_pde.run(self.npde)
                sbc = self.sampler_bc.run(self.nbc)


            # 计算残差与损失
            pde_res = self.pde_residual(spde)
            bc_res = self.bc_residual(sbc)
            pde_r = bm.real(pde_res)
            bc_r = bm.real(bc_res)
            mse_pde_r = self.loss(pde_r, bm.zeros_like(pde_r))
            mse_bc_r = self.loss(bc_r, bm.zeros_like(bc_r))

            if self.complex:
                pde_i =  bm.imag(pde_res)
                bc_i = bm.imag(bc_res)
                mse_pde_i = self.loss(pde_i, bm.zeros_like(pde_i))
                mse_bc_i = self.loss(bc_i, bm.zeros_like(bc_i))
                loss = 1.0 * mse_pde_r + 1.0 * mse_pde_i + 1.0 * mse_bc_r + 1.0 * mse_bc_i
            else:
                loss = 1.0 * mse_pde_r + 1.0 * mse_bc_r

            loss.backward()            
            self.optimizer.step()    # 更新参数
            self.steplr.step()


            if epoch % 100 == 0:
                error = self.net.estimate_error(self.pde.solution, mesh, coordtype='c', compare='real')
                self.error_real.append(error.detach().numpy())
                self.Loss.append(loss.detach().numpy())

                if self.complex:
                    error_i = self.net.estimate_error(self.pde.solution, mesh, coordtype='c', compare='imag')
                    self.error_imag.append(error_i.detach().numpy()) 
                    self.logger.info(f"epoch: {epoch}, mse_pde_r: {mse_pde_r:.6f}, mse_bc_r: {mse_bc_r:.6f}, "
                                    f"mse_pde_i: {mse_pde_i:.6f}, mse_bc_i: {mse_bc_i:.6f}, "
                                    f"loss: {loss.item():.6f}, error_real: {error.item():.4f}, error_imag: {error_i.item():.4f}")
                else:
                    self.logger.info(f"epoch: {epoch}, mse_pde_r: {mse_pde_r:.6f}, mse_bc_r: {mse_bc_r:.6f}, "                                    
                                    f"loss: {loss.item():.6f}, error_real: {error.item():.4f}")
        
        tmr.send(f'Training completed time')
        next(tmr)

    def predict(self, p: TensorLike) -> TensorLike:
        """Make predictions using trained network.
        
        Parameters
            p : TensorLike
                Input point coordinates
                
        Returns
            TensorLike
                Network predictions (complex tensor when complex=True)
        """
        return self.net(p)

    def show(self):
        """Visualize training results and solution comparisons.
        
        Notes
            Visualizations include:
            1. Training loss history
            2. Real/imaginary error history
            3. 1D/2D solution comparisons
            
            Complex-valued solution handling:
            - For 2D: Separate plots for real/imaginary components of true/predicted solutions
            - Error plots show both real and imaginary components
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

        # 绘制损失曲线 (顶部子图)
        axes[0].plot(self.Loss, 'r-', linewidth=2)
        axes[0].set_title('Training Loss', fontsize=12)
        axes[0].set_ylabel('Loss Value', fontsize=10)
        axes[0].grid(True)

        # 绘制实部上PINN vs FEM误差 (中间子图)
        axes[1].plot(self.error_real, 'b--', linewidth=2)
        axes[1].set_title('Error of Real between PINN and FEM Solution', fontsize=12)
        axes[1].set_ylabel('Error Real', fontsize=10)
        axes[1].grid(True)

        # 2. 绘制实部和虚部误差曲线 (右子图)
        axes[1].plot(self.error_real, 'b-', linewidth=2, label='Real Part Error')
        if self.error_imag != []:
            axes[1].plot(self.error_imag, 'g--', linewidth=2, label='Imaginary Part Error')
        axes[1].set_title('PINN vs FEM Solution Errors', fontsize=12)
        axes[1].set_ylabel('Error Value', fontsize=10)
        axes[1].grid(True)
        axes[1].legend()

        if (self.gd <= 2) and (self.complex):
            mesh = self.mesh
            node = mesh.entity('node')

            # 获取预测解和真解
            u_pred = self.net(node)  # PINN预测解
            u_true = self.pde.solution(node)   # 解析解
            node = node.detach().numpy()

            # fig = plt.figure()
            if self.gd == 1:
                fig = plt.figure()
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
                u_pred_r = bm.real(u_pred).detach().numpy().flatten()
                u_pred_i = bm.imag(u_pred).detach().numpy().flatten()
                u_true_r = bm.real(u_true).detach().numpy().flatten()
                u_true_i = bm.imag(u_true).detach().numpy().flatten()
                fig, axes = plt.subplots(2, 2)
                axes[0, 0].set_title('Real Part of True Solution')
                axes[0, 1].set_title('Imag Part of True Solution')
                axes[1, 0].set_title("Real Part of Pinn Module's Solution")
                axes[1, 1].set_title("Imag Part of Pinn Module's Solution")

                mesh.add_plot(axes[0, 0], cellcolor=u_true_r, linewidths=0, aspect=1)
                mesh.add_plot(axes[0, 1], cellcolor=u_true_i, linewidths=0, aspect=1)
                mesh.add_plot(axes[1, 0], cellcolor=u_pred_r, linewidths=0, aspect=1)
                mesh.add_plot(axes[1, 1], cellcolor=u_pred_i, linewidths=0, aspect=1)

                # # 子图1：PINN预测解
                fig = plt.figure()
                ax1_3d = fig.add_subplot(121, projection='3d')
                surf1 = ax1_3d.plot_trisurf(
                    node[:, 0], node[:, 1], u_pred_r,
                    cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.8)
                ax1_3d.set_title('PINN Predicted Solution of Real')
                ax1_3d.set_xlabel('X')
                ax1_3d.set_ylabel('Y')
                ax1_3d.set_zlabel('u(x,y)')
                fig.colorbar(surf1, ax=ax1_3d, shrink=0.5, label='Value')

                # 子图2：真解
                ax2_3d = fig.add_subplot(122, projection='3d')
                surf2 = ax2_3d.plot_trisurf(
                    node[:, 0], node[:, 1], u_true_r,
                    cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
                ax2_3d.set_title('Analytical True Solution of Real')
                ax2_3d.set_xlabel('X')
                ax2_3d.set_ylabel('Y')
                ax2_3d.set_zlabel('u(x,y)')
                fig.colorbar(surf2, ax=ax2_3d, shrink=0.5, label='Value')
                plt.suptitle('Solution Comparison: PINN vs Analytical')

        plt.tight_layout()      
        plt.show()  # 显示第二个Figure

