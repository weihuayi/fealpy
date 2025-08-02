import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from typing import Union, Optional

from fealpy.backend import bm
from fealpy.utils import timer
from fealpy.typing import TensorLike
from fealpy.model import ComputationalModel, PDEModelManager
from fealpy.model.helmholtz import HelmholtzPDEDataT

from fealpy.ml import gradient, optimizers, activations

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, ISampler


class HelmholtzPINNModel(ComputationalModel):
    """Physics-Informed Neural Network (PINN) model for solving Helmholtz equations.
    
    Implements a PINN framework to solve Helmholtz PDE problems using neural networks.
    Handles PDE residual calculation, boundary condition enforcement, and training process.
    Supports both uniform and random sampling strategies for collocation points.
    Specialized for complex-valued solutions (real + imaginary components).
    
    Parameters
        options : dict
            If None, default parameters from get_options() will be used.
            Configuration dictionary containing:
            - pde: PDE definition (int or HelmholtzPDEDataT);
            - lr: Learning rate (float);
            - epochs: Number of training epochs (int);
            - weights: Weight for the equation loss and boundary loss (tuple);
            - hidden_size: Tuple of hidden layer sizes (tuple);
            - npde: Number of PDE collocation points (int);
            - nbc: Number of boundary collocation points (int);
            - activation: Activation function (str, options: 'Tanh', 'ReLU', 'LeakyReLU', 'Sigmoid', 'LogSigmoid', 'Softmax', 'LogSoftmax');
            - optimizer: Optimization algorithm (str, options: 'Adam', 'SGD');
            - sampling_mode: Sampling strategy (str, options: 'linspace' or 'random');
            - complex: Boolean flag for complex-valued solutions (bool);
            - wave: Wave number k for Helmholtz equation (float).
            - step_size: Period of learning rate decay (int);
            - gamma: Multiplicative factor of learning rate decay (float).
            - pbar_log: Whether to use progress bar for logging (bool);
            - log_level: Logging level (str, options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        
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
    
    Reference
        https://wnesm678i4.feishu.cn/wiki/U219wwT18iH4v7kNTOacxl8cnXb?from=from_copylink
        
    Examples
        >>> from fealpy.backend import bm  
        >>> bm.set_backend('pytorch')  # Set the backend to PyTorch  
        >>> from fealpy.ml import HelmholtzPINNModel  
        >>> options = HelmholtzPINNModel.get_options()  # Get the default options of the network  
        >>> model = HelmholtzPINNModel(options=options)  
        >>> model.run()   # Train the network  
        >>> model.show()   # Show the results of the network training  
    """
    def __init__(self, options: Optional[dict] = None):
        if options is None:
            self.options = self.get_options()
        else:
            self.options = options
        
        self.pbar_log = self.options.get('pbar_log', True)
        self.log_level = self.options.get('log_level', 'INFO')
        super().__init__(pbar_log=self.pbar_log, log_level=self.log_level)
        
        self.complex = self.options.get('complex', False)
        self.k = self.options.get('wave', 1.0)

        self.set_pde(self.options.get('pde', 1))
        self.gd = self.pde.geo_dimension()
        self.domain = self.pde.domain()
        self.set_mesh(self.options.get('mesh_size', 30))
        

        # 采样器
        self.sampler_pde = ISampler(self.domain, requires_grad=True, mode=self.options.get('sampling_mode', 'random'))
        self.sampler_bc = BoxBoundarySampler(self.domain, requires_grad=True, mode=self.options.get('sampling_mode', 'random'))

        # 网络超参数、激活函数、采样点数、权重
        self.lr = self.options.get('lr', 0.005)
        self.epochs = self.options.get('epochs', 1000)
        self.hidden_size = self.options.get('hidden_size', (50, 50, 50, 50))
        self.activation = activations[self.options.get('activation', "Tanh")]
        self.npde = self.options.get('npde', 400)
        self.nbc = self.options.get('nbc')
        self.weights = self.options.get('weights', (1, 30))

        # 损失函数
        self.loss = nn.MSELoss(reduction='mean')

        # 网络
        self.set_network()

        # 优化器与学习率调度器
        opt = optimizers[self.options.get('optimizer', 'Adam')]
        self.optimizer = opt(params=self.net.parameters(), lr=self.lr)
        
        # 学习率调度器
        step_size = self.options.get('step_size', 0)
        gamma = self.options.get('gamma', 0.99)
        self.set_steplr(step_size, gamma)

        self.tmr = timer()  # 计时器
    
    @classmethod
    def get_options(cls):
        """Get default configuration parameters for the model.
        
        Defines and returns default configurations for the model through a command-line argument parser,
        including PDE problem number, grid size, network structure, and optimizer parameters.
        
        Returns
            options : dict
                Dictionary containing all configuration parameters with parameter names as keys and default values
        """

        import argparse
        parser = argparse.ArgumentParser(description=
                """
                A simple example of using PINN to solve Poisson equation.
                """)

        parser.add_argument('--pde',default=1, type=int,
                            help="Built-in PDE example ID for different Helmholtz problems, default is 1.")
        
        parser.add_argument('--mesh_size',
                            default=30, type=int,
                            help='Number of grid points along each dimension, default is 30.')

        parser.add_argument('--complex', 
                            default=True, type=bool,
                            help="Enable complex-valued solution modeling, default is True")

        parser.add_argument('--wave', 
                            default=1.0, type=float,
                            help="Wave number k for Helmholtz equation (Δu + k²u + f = 0), default is 1.0")

        parser.add_argument('--sampling_mode', 
                            default='random', type=str,
                            help="Sampling method for collocation points: 'random' or 'linspace', default is 'random'")

        parser.add_argument('--npde',
                            default=1500, type=int,
                            help='Number of PDE samples, default is 400.')

        parser.add_argument('--nbc',
                            default=300, type=int,
                            help='Number of boundary condition samples, default is 100.')
        
        parser.add_argument('--weights',
                            default=(1, 30), type=tuple,
                            help='The first value is the weight for the equation loss, and the second ' \
                            'value is the weight for the boundary loss., default is (1, 30).')

        parser.add_argument('--hidden_size',
                            default=(50, 50, 50, 50), type=tuple,
                            help='Default hidden sizes, default is (50, 50, 50, 50).')

        parser.add_argument('--loss',
                            default=nn.MSELoss(reduction='mean'), type=callable,
                            help='Loss function to use, default is nn.MSELoss')

        parser.add_argument('--optimizer', 
                            default="Adam",  type=str,
                            help='Optimizer to use for training, default is Adam')

        parser.add_argument('--activation',
                            default="Tanh", type=str,
                            help='Activation function, default is Tanh')

        parser.add_argument('--lr',
                            default=0.001, type=float,
                            help='Learning rate for the optimizer, default is 0.001.')

        parser.add_argument('--step_size',
                            default=1000, type=int,
                            help='Period of learning rate decay, default is 0.')

        parser.add_argument('--gamma',
                            default=0.99, type=float,
                            help='Multiplicative factor of learning rate decay. Default: 0.99.')

        parser.add_argument('--epochs',
                            default=3000, type=int,
                            help='Number of training epochs, default is 3000.')

        parser.add_argument('--pbar_log',
                            default=True, type=bool,
                            help='Whether to show progress bar, default is True')

        parser.add_argument('--log_level',
                            default='INFO', type=str,
                            help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')
        options = vars(parser.parse_args())
        return options
        
    def set_pde(self, pde: Union[HelmholtzPDEDataT, int]=1):
        """Initialize the PDE problem definition.
        
        Parameters
            pde : Union[[HelmholtzPDEDataT, int]
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
            sizes = (self.gd,) + self.hidden_size
            for i in range(len(sizes)-1):
                layers.append(nn.Linear(sizes[i], sizes[i+1], dtype=bm.float64))
                if i < len(sizes)-1:  
                    layers.append(self.activation())
                    
            if self.complex:
                layers.append(nn.Linear(sizes[-1], 2, dtype=bm.float64))
            else:
                layers.append(nn.Linear(sizes[-1], 1, dtype=bm.float64))
            net = nn.Sequential(*layers)
            
        self.net = Solution(net, self.complex)

    def set_mesh(self, mesh_size: int):
        """Create computational mesh.
        
        Creates a computational mesh over the domain defined by the PDE based on the specified mesh size.
        
        Parameters
            mesh_size : tuple of int
                Number of nodes in each dimension.
        """
        self.mesh_size = (mesh_size, ) * self.gd
        cell_size = tuple(x - 1 for x in self.mesh_size)
        self.mesh = self.pde.init_mesh(*cell_size)

    def set_steplr(self, step_size: int=0, gamma: float=0.9):
        """Create learning rate scheduler
        
        Initializes a learning rate scheduler for decaying the learning rate periodically during training.
        
        Parameters
            step_size : int, optional, default=0
                Period for learning rate decay, i.e., decay every step_size epochs. No scheduler is used if step_size = 0.
            gamma : float, optional, default=0.9
                Multiplicative factor for learning rate decay, new_lr = current_lr * gamma
        """
        if step_size == 0:
            self.steplr = None
        else:
            self.steplr = StepLR(self.optimizer, step_size, gamma)

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
            2. Robin: i*k*u + ∂u/∂n - g = 0, i serves as the imaginary unit.
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
        w = self.weights

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
                loss = w[0]* (mse_pde_r + mse_pde_i) + w[1] * (mse_bc_r + mse_bc_i)
            else:
                loss = w[0] * mse_pde_r + w[1]* mse_bc_r

            loss.backward()            
            self.optimizer.step()    # 更新参数
            if self.steplr is not None:
                self.steplr.step()

            if epoch % 100 == 0:
                error = self.net.estimate_error(self.pde.solution, mesh, coordtype='c', compare='real')
                self.error_real.append(error.detach().numpy())
                self.Loss.append(loss.item())
                self.logger.info(f"epoch: {epoch}, Loss: {loss.item():.6f}")  

                if self.complex:
                    error_i = self.net.estimate_error(self.pde.solution, mesh, coordtype='c', compare='imag')
                    self.error_imag.append(error_i.detach().numpy()) 
                #     self.logger.info(f"epoch: {epoch}, mse_pde_r: {mse_pde_r:.6f}, mse_bc_r: {mse_bc_r:.6f}, "
                #                     f"mse_pde_i: {mse_pde_i:.6f}, mse_bc_i: {mse_bc_i:.6f}, "
                #                     f"loss: {loss.item():.6f}, error_real: {error.item():.4f}, error_imag: {error_i.item():.4f}")
               
                    # self.logger.info(f"epoch: {epoch}, mse_pde: {mse_pde_r:.6f}, mse_bc: {mse_bc_r:.6f}, "                                    
                    #                 f"loss: {loss.item():.6f}, error_real: {error.item():.4f}")
                    # self.logger.info(f"epoch: {epoch}, Loss: {loss.item():.6f}")        
        tmr.send(f'PINN training time')
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
        Loss = bm.log10(bm.tensor(self.Loss)).numpy()

        # 绘制损失曲线
        axes[0].plot(Loss, 'r-', linewidth=2)
        axes[0].set_title('Training Loss', fontsize=12)
        axes[0].set_xlabel('training epochs*100', fontsize=10)
        axes[0].set_ylabel('log10(Loss)', fontsize=10)
        axes[0].grid(True)

        # 绘制实部和虚部误差曲线
        error_real = bm.log10(bm.tensor(self.error_real)).numpy()
        error_imag = bm.log10(bm.tensor(self.error_imag)).numpy()
        axes[1].plot(error_real, 'b-', linewidth=2, label='Real Part Error')
        if self.error_imag != []:
            axes[1].plot(error_imag, 'g--', linewidth=2, label='Imag Part Error')
        axes[1].set_title('L2 Error between PINN Solution and Exact Solution', fontsize=12)
        axes[1].set_ylabel('log10(Error)', fontsize=10)
        axes[1].set_xlabel('training epochs*100', fontsize=10)
        axes[1].grid(True)
        axes[1].legend()

        if self.gd <= 2:
            mesh = self.mesh
            node = mesh.entity('node')

            # 获取预测解和真解
            u_pred = self.net(node)  # PINN预测解
            u_true = self.pde.solution(node)   # 解析解
            node = node.detach().numpy()

            u_pred_r = bm.real(u_pred).detach().numpy().flatten()
            u_true_r = bm.real(u_true).detach().numpy().flatten()
            if self.error_imag != []:
                u_pred_i = bm.imag(u_pred).detach().numpy().flatten()
                u_true_i = bm.imag(u_true).detach().numpy().flatten()

            # fig = plt.figure()
            if self.gd == 1:
                fig = plt.figure()
                # 绘制真实解和预测解
                plt.plot(node, u_true_r, 'b-', linewidth=3, label='Real Part of Exact  Solution')
                plt.plot(node, u_pred_r, 'r--', linewidth=2, label='Real Part of PINN Prediction')
                if self.error_imag != []:
                    plt.plot(node, u_true_i, 'g-', linewidth=3, label='Imag Part of Exact  Solution')
                    plt.plot(node, u_pred_i, 'y--', linewidth=2, label='Imag Part of PINN Prediction')
                # 图形修饰
                plt.xlabel('x', fontsize=12)
                plt.ylabel('u(x)', fontsize=12)
                plt.title('Comparison between PINN and Exact Solution', fontsize=14)
                plt.legend(fontsize=12)
                plt.grid(True, linestyle=':')
            else:
                # fig, axes = plt.subplots(2, 2)
                # axes[0, 0].set_title('Real Part of Exact Solution')
                # axes[0, 1].set_title('Imag Part of Exact Solution')
                # axes[1, 0].set_title("Real Part of PINN Solution")
                # axes[1, 1].set_title("Imag Part of PINN Solution")

                # mesh.add_plot(axes[0, 0], cellcolor=u_true_r, linewidths=0, aspect=1)
                # mesh.add_plot(axes[0, 1], cellcolor=u_true_i, linewidths=0, aspect=1)
                # mesh.add_plot(axes[1, 0], cellcolor=u_pred_r, linewidths=0, aspect=1)
                # mesh.add_plot(axes[1, 1], cellcolor=u_pred_i, linewidths=0, aspect=1)

                #  PINN预测解的实部
                fig = plt.figure()
                ax1_3d = fig.add_subplot(131, projection='3d')
                surf1 = ax1_3d.plot_trisurf(
                    node[:, 0], node[:, 1], u_pred_r,
                    cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.8)
                ax1_3d.set_title('PINN Solution of Real')
                ax1_3d.set_xlabel('X')
                ax1_3d.set_ylabel('Y')
                ax1_3d.set_zlabel('u(x,y)')
                fig.colorbar(surf1, ax=ax1_3d, shrink=0.5, label='Value')

                # 真解的实部
                ax2_3d = fig.add_subplot(132, projection='3d')
                surf2 = ax2_3d.plot_trisurf(
                    node[:, 0], node[:, 1], u_true_r,
                    cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
                ax2_3d.set_title('Exact Solution of Real')
                ax2_3d.set_xlabel('X')
                ax2_3d.set_ylabel('Y')
                ax2_3d.set_zlabel('u(x,y)')
                fig.colorbar(surf2, ax=ax2_3d, shrink=0.5, label='Value')

                # 实部的误差
                ax3_3d = fig.add_subplot(133, projection='3d')
                surf3 = ax3_3d.plot_trisurf(
                    node[:, 0], node[:, 1], u_pred_r-u_true_r,
                    cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
                ax3_3d.set_title('Error: PINN - Exact')
                ax3_3d.set_xlabel('X')
                ax3_3d.set_ylabel('Y')
                ax3_3d.set_zlabel('u(x,y)')
                fig.colorbar(surf3, ax=ax3_3d, shrink=0.5, label='Value')

                plt.suptitle('Comparison between PINN and Exact Solution of Real')

                if self.error_imag != []:
                    fig = plt.figure()
                    ax1_3d = fig.add_subplot(131, projection='3d')
                    surf1 = ax1_3d.plot_trisurf(
                        node[:, 0], node[:, 1], u_pred_i,
                        cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.8)
                    ax1_3d.set_title('PINN Solution of Imag')
                    ax1_3d.set_xlabel('X')
                    ax1_3d.set_ylabel('Y')
                    ax1_3d.set_zlabel('u(x,y)')
                    fig.colorbar(surf1, ax=ax1_3d, shrink=0.5, label='Value')

                    # 真解的虚部
                    ax2_3d = fig.add_subplot(132, projection='3d')
                    surf2 = ax2_3d.plot_trisurf(
                        node[:, 0], node[:, 1], u_true_i,
                        cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
                    ax2_3d.set_title('Exact   Solution of Imag')
                    ax2_3d.set_xlabel('X')
                    ax2_3d.set_ylabel('Y')
                    ax2_3d.set_zlabel('u(x,y)')
                    fig.colorbar(surf2, ax=ax2_3d, shrink=0.5, label='Value')

                    # 虚部的误差
                    ax3_3d = fig.add_subplot(133, projection='3d')
                    surf3 = ax3_3d.plot_trisurf(
                        node[:, 0], node[:, 1], u_pred_i-u_true_i,
                        cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
                    ax3_3d.set_title('Error: PINN - Exact')
                    ax3_3d.set_xlabel('X')
                    ax3_3d.set_ylabel('Y')
                    ax3_3d.set_zlabel('u(x,y)')
                    fig.colorbar(surf3, ax=ax3_3d, shrink=0.5, label='Value')

                    plt.suptitle('Comparison between PINN and Exact Solution of Imag')

        plt.tight_layout()      
        plt.show()  # 显示第二个Figure

