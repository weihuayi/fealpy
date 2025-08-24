import math
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from typing import Union, Optional


from fealpy.backend import  bm
from fealpy.utils import timer
from fealpy.typing import TensorLike
from fealpy.model import ComputationalModel, PDEModelManager
from fealpy.model.poisson import PoissonPDEDataT

from fealpy.ml import gradient, optimizers, activations
from fealpy.ml.modules import Solution


class PoissonPENNModel(ComputationalModel):
    """Physics embedded neural network (PENN) model for solving Poisson's equation with Dirichlet boundary conditions.
    
    This class implements a PENN approach to solve Poisson's equation.
    It combines neural networks with the governing equations and boundary conditions of Poisson's equation
    for numerical solution of partial differential equations. The model uses a Multi-Layer Perceptron (MLP)
    as the base network structure, computes partial derivatives through automatic differentiation to satisfy
    PDE constraints, and supports comparison with Finite Element Method (FEM) results.

    Parameters
        options : dict, optional
            If None, default parameters from get_options() will be used.
            Configuration dictionary containing:
            - pde: PDE definition (int or HelmholtzPDEDataT);
            - scaling_function: The scaling_function is a function that satisfies the boundary conditions, default is None;
            - mesh_size: Number of grid points along each dimension (int);
            - lr: Learning rate (float);
            - epochs: Number of training epochs (int);
            - weights: Weight for the equation loss and boundary loss (tuple);
            - hidden_size: Tuple of hidden layer sizes (tuple);
            - activation: Activation function (str, options: 'Tanh', 'ReLU', 'LeakyReLU', 'Sigmoid', 'LogSigmoid', 'Softmax', 'LogSoftmax');
            - optimizer: Optimization algorithm (str, options: 'Adam', 'SGD');
            - step_size: Period of learning rate decay (int);
            - gamma: Multiplicative factor of learning rate decay (float);
            - pbar_log: Whether to use progress bar for logging (bool);
            - log_level: Logging level (str, options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    
    Attributes
        options : dict
            Collection of model configuration parameters
        pde : PoissonPDEDataT
            Poisson's equation problem definition object containing equation parameters, boundary conditions, 
            and source terms.
        gd : int
            Geometric dimension (1D, 2D, or 3D) of the problem domain.
        domain : tuple, list
            Computational domain boundaries in the format (xmin, xmax, ymin, ymax, ...)
        mesh_size : tuple
            Grid size specifying the number of nodes in each dimension
        mesh : Mesh
            Computational grid object
        lr : float
            Optimizer learning rate
        epochs : int
            Number of training iterations
        hidden_size : tuple
            Size of neural network hidden layers, e.g., (50, 50) indicates two hidden layers with 50 neurons each
        activation : torch.nn.Module
            Activation function class used for non-linear mapping between network layers
        net : Solution
            Encapsulated neural network model for predicting solution coefficients
        optimizer : torch.optim.Optimizer
            Optimizer object for updating network parameters
        steplr : torch.optim.lr_scheduler.StepLR or None
            Learning rate scheduler, None if step_size is 0
        tmr : timer
            Timer object for recording training and solution process times
        Loss : list
            Record of loss values during training
        error : list
            Record of error values during training
        pred : TensorLike
            Final prediction results

    Reference
        https://wnesm678i4.feishu.cn/wiki/Xc2iw6mDUiOBZCkQtFcczVcZnW9?from=from_copylink

    Examples
        >>> from fealpy.backend import bm
        >>> bm.set_backend('pytorch')  # Set computation backend
        >>> from fealpy.ml.poisson_penn_model import PoissonPENNModel
        >>> options = PoissonPENNModel.get_options()    # Get and modify default options        
        >>> model = PoissonPENNModel(options=options)   # Initialize and run the model
        >>> model.run()  # Start training
        >>> model.show()  # Visualize results
    """
    def __init__(self, options: Optional[dict] = None):
        if options is None:
            self.options = self.get_options()
        else:
            self.options = options
        
        self.pbar_log = self.options.get('pbar_log', True)
        self.log_level = self.options.get('log_level', 'INFO')
        super().__init__(pbar_log=self.pbar_log, log_level=self.log_level)

        self.set_pde(self.options.get('pde', 8))
        self.gd = self.pde.geo_dimension()
        self.domain = self.pde.domain()
        self.set_mesh(self.options.get('mesh_size', 30))

        # 网络超参数、激活函数
        self.lr = self.options.get('lr', 0.005)
        self.epochs = self.options.get('epochs', 1000)
        self.hidden_size = self.options.get('hidden_size', (50, 50))
        self.activation = activations[self.options.get('activation', "LogSigmoid")]

        self.set_network()  # 网络
       
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
        parser = argparse.ArgumentParser(description="Poisson equation solver using PENN.")

        parser.add_argument('--pde',default=8, type=int,
                            help="Built-in PDE example ID (1, 2, 6, 8, 9) for different Poisson problems, default is 8.")

        parser.add_argument('--scaling_function',default=None, 
                            help="The caling_function is a function that satisfies the boundary conditions, default is None.")
        
        parser.add_argument('--mesh_size',
                            default=30, type=int,
                            help='Number of grid points along each dimension, default is 30.')

        parser.add_argument('--hidden_size',
                            default=(50, 50), type=tuple,
                            help='Default hidden sizes, default is (50, 50).')

        parser.add_argument('--optimizer', 
                            default="Adam",  type=str,
                            help="Optimizer to use for training, default is Adam,  options are 'Adam' , 'SGD'.")

        parser.add_argument('--activation',
                            default="LogSigmoid", type=str,
                            help="Activation function, default is 'LogSigmoid', " \
                            "options are 'Tanh', 'ReLU', 'LeakyReLU', 'Sigmoid', 'LogSigmoid', 'Softmax', 'LogSoftmax'.")

        parser.add_argument('--lr',
                            default=0.005, type=float,
                            help='Learning rate for the optimizer, default is 0.005.')

        parser.add_argument('--step_size',
                            default=0, type=int,
                            help='Period of learning rate decay, default is 0.')

        parser.add_argument('--gamma',
                            default=0.99, type=float,
                            help='Multiplicative factor of learning rate decay. Default: 0.99.')

        parser.add_argument('--epochs',
                            default=1000, type=int,
                            help='Number of training epochs, default is 1000.')

        parser.add_argument('--pbar_log',
                            default=True, type=bool,
                            help='Whether to show progress bar, default is True')

        parser.add_argument('--log_level',
                            default='INFO', type=str,
                            help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')
        options = vars(parser.parse_args())
        return options
    
    def set_pde(self, pde: Union[PoissonPDEDataT, int]=1):
        """Initialize the PDE problem definition
        
        Sets up the Poisson's equation problem to be solved by the model based on 
        the input PDE object or built-in example ID.
        
        Parameters
            pde: Union[PoissonPDEDataT, int]
                Either a Poisson's equation problem object or the ID (integer) of a predefined example. 
                If an integer, the corresponding predefined Poisson's equation problem is retrieved from 
                the PDE model manager.
        """  
        if isinstance(pde, int):
            self.pde = PDEModelManager('poisson').get_example(pde)
        else:
            self.pde = pde 


    def set_network(self, net=None):
        """Configure the neural network architecture
        
        Creates or sets the neural network model used for solving the PDE. If no custom network is provided,
        a default Multi-Layer Perceptron (MLP) is created with input dimension equal to the geometric dimension,
        output dimension 1, and hidden layer sizes specified by hidden_size.
        
        Parameters
            net : torch.nn.Module, optional
                Custom neural network architecture. If None, a default MLP network is automatically created
        """
        if net == None:
            layers = []
            sizes = (self.gd,) + self.hidden_size + (1,)
            for i in range(len(sizes)-1):
                layers.append(nn.Linear(sizes[i], sizes[i+1], dtype=bm.float64))
                if i < len(sizes)-2:  
                    layers.append(self.activation())
            net = nn.Sequential(*layers)
        self.net = Solution(net)

    def set_mesh(self, mesh_size: int):
        """Create computational mesh
        
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
        

    def shape_function(self, p: TensorLike) -> TensorLike:
        """Compute shape function values at given points。

        The construction of the shape function must satisfy the condition that its value equals zero 
        at the equation's boundaries. For a two-dimensional rectangular domain where all edges serve 
        as the equation's boundaries, the shape function can be formulated as:

            ψ(x,y) = [e^{-((x-x₀)² + (y-y₀)²)/32} / √(32π)] * (x-x_min)(x-x_max)(y-y_min)(y-y_max)
        
        where (x₀, y₀) represents the center point of the domain.
        
        Parameters
            p : TensorLike
                Tensor of shape (n_points, geo_dimension) representing coordinates of input points
        
        Returns
            val : TensorLike
                Tensor of shape (n_points, ) representing shape function values at each point
        """
        a = 32
        c = math.sqrt(math.pi * a) ** self.gd
        domain = self.domain
       
        for i in range(self.gd):
            x1, x2 = domain[2*i], domain[2*i+1]
            # h = x2 - x1
            x0 = (x1 + x2) / 2
            if i == 0:
                val = bm.exp(-((p[..., i] - x0)**2) / a) / c
            else:
                val = val * bm.exp(-((p[..., i] - x0) **2 ) / a)
        
        if hasattr(self.pde, 'identify_boundary_edge'):
            bc_flag = self.pde.identify_boundary_edge()
            for i in range(len(bc_flag)):
                k = domain[bc_flag[i]]
                j = int(bc_flag[i] / 2)
                if i == 0:
                    v = p[..., j] - k
                else:
                    v = v * (p[..., j] - k)
        else:
            for i in range(self.gd):
                if i == 0:
                    v = (p[..., i] - domain[2*i]) * (p[..., i] - domain[2*i+1])
                else:
                    v = v * (p[..., i] - domain[2*i]) * (p[..., i] - domain[2*i+1])
        
        return val * bm.abs(v) 

    def scaling_function(self, p: TensorLike, func:callable=None) -> TensorLike:
        """Compute scaling function values at given points.The caling_function is a function 
        that satisfies the boundary conditions.
        
        This method evaluates the scaling function at specified points, either using:
        1. The PDE's built-in scaling function (if defined in self.pde)
        2. A user-provided function (if passed via func parameter)
        3. Raises ValueError if neither is available
        
        Args:
            p: TensorLike of shape (n_points, geo_dimension), 
                input points where the scaling function should be evaluated.
            func: Optional callable that takes points as input and returns 
                scaling function values. Used as fallback if no built-in 
                function exists in the PDE.
        
        Returns:
            TensorLike of shape (n_points, n_shape_functions), 
            the computed scaling function values at each point.
        
        Raises:
            ValueError: If neither the PDE has a scaling function 
                    nor a function is provided through func argument.
        """
        if func is not None: 
            return func(p)
        elif hasattr(self.pde, 'scaling_function'):
            return self.pde.scaling_function(p)
        else:
            raise ValueError("No scaling function is defined for this PDE."
                             "Please provide a scaling function through the 'func' argument.")
    
    def pde_residual(self, p: TensorLike, pred: TensorLike) -> TensorLike:
        """Compute PDE residual (Laplacian(u)).
        
        Parameters
            p : TensorLike
                Collocation points where residual is evaluated.
            pred : TensorLike
                Predicted solution values at collocation points by net.
            
        Returns
            val : TensorLike
                PDE residual values at input points.
                
        Notes
            Uses automatic differentiation to compute second derivatives.
            The residual is calculated as Δu where Δ is the Laplacian operator.
        """     
        # 一阶导数计算
        grad_u = gradient(pred, p, create_graph=True)  ## (npde, dim)
        laplacian = bm.zeros(pred.shape[0], 1)    # 拉普拉斯项初始化
        
        for i in range(p.shape[-1]):
            u_ii = gradient(grad_u[..., i], p, create_graph=True, split=True)[i]   # 计算 ∂²u/∂x_i²
            laplacian += u_ii
        val = laplacian 
        return val 
    
    def run(self):
        """Execute training process."""
        next(self.tmr)
        self.Loss = []
        self.error = []
        mesh = self.mesh
        p = mesh.entity('node').requires_grad_(True)

        U = self.shape_function(p).reshape(-1, 1)
        B = self.scaling_function(p=p, func=self.options['scaling_function']).reshape(-1, 1)
        f = self.pde.source(p).reshape(-1, 1)
        f_std = bm.std(f, dim=0) 
        mse = nn.MSELoss(reduction='mean')
        for epoch in range(self.epochs+1):
            self.optimizer.zero_grad()

            coff = self.net(p)
            pred = coff * U  + B
            pde_res = self.pde_residual(p, pred) + f
            loss = mse(pde_res, bm.zeros_like(pde_res)) / f_std

            loss.backward(retain_graph=True)            
            self.optimizer.step()    # 更新参数
            if self.steplr is not None:
                self.steplr.step()

            if epoch % 100  == 0:
                self.Loss.append(loss.item())
                self.logger.info(f"epoch: {epoch}, Loss: {loss.item():.6f}")
        
        self.pred = pred
        self.tmr.send(f'PENN training completed time')

    def predict(self, p: TensorLike) -> TensorLike:
        """Make predictions using the trained network.
        
        Parameters
            p : TensorLike
                Input points where prediction is needed.
                
        Returns
            pred : TensorLike
                Network predictions at input points.
        """
        coff = self.net(p)
        U = self.shape_function(p).reshape(-1, 1)
        func = self.options.get('scaling_function', None)
        B = self.scaling_function(p=p, func=func).reshape(-1, 1)
        pred = coff * U + B
        return pred.detach().numpy()
    
    def fem(self):
        """Solve Poisson's equation using Finite Element Method (FEM) for comparison
        
        Returns
            uh : TensorLike
                FEM solution results
        
        Notes
            q=1, p=q+2, where p is the polynomial degree of the finite element space.
        """
        from fealpy.functionspace import LagrangeFESpace
        from fealpy.fem import BilinearForm, LinearForm
        from fealpy.fem  import ScalarDiffusionIntegrator, ScalarSourceIntegrator
        from fealpy.fem import DirichletBC
        from fealpy.solver import spsolve

        pde = self.pde
        mesh_size = tuple(x - 1 for x in self.mesh_size)
        mesh = pde.init_mesh(*mesh_size)
        p = 1
        q = p + 2
        space = LagrangeFESpace(mesh, p)
        S = BilinearForm(space)
        S.add_integrator(ScalarDiffusionIntegrator(q=q))
        A = S.assembly()

        b = LinearForm(space)
        b.add_integrator(ScalarSourceIntegrator(pde.source, q=q))

        F = b.assembly()

        node = mesh.entity('node')
        
        if hasattr(self.pde, 'identify_boundary_edge'):
            if self.gd == 1:
                bcf = pde.is_dirichlet_boundary(node).flatten()
            else:
                bcf = pde.is_dirichlet_boundary(node)
            A, F = DirichletBC(space=space, gd=pde.dirichlet, threshold=bcf).apply(A, F)
        else:
            A, F = DirichletBC(space=space, gd=pde.dirichlet).apply(A, F)
        uh = spsolve(A, F)
        self.tmr.send(f'FEM solving time')
        next(self.tmr)
        return uh

    def show(self):
        """Visualize training results and compare with FEM solutions"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        node = self.mesh.entity('node').detach().numpy()
        net_pred = self.pred.flatten().detach().numpy()
        fem_pred = self.fem().detach().numpy()
        er = net_pred - fem_pred

        # 绘制损失曲线 
        fig = plt.figure()
        ax1 = fig.add_subplot()

        Loss = bm.tensor(self.Loss)
        ax1.plot(bm.log10(Loss).numpy(), 'r-', linewidth=2)
        ax1.set_title('Training Loss', fontsize=12)
        ax1.set_xlabel('training epochs*100', fontsize=10)
        ax1.set_ylabel('log10(Loss)', fontsize=10)
        ax1.grid(True)
        fig = plt.figure()
        if self.gd == 1:
            ax2 = fig.add_subplot()
            ax2.plot(node, net_pred, 'b-', label="PENN Prediction")
            ax2.plot(node, fem_pred, 'y--', label="FEM Prediction")
            ax2.plot(node, er, 'r--', label="Error: PENN - FEM")
            plt.xlabel('x', fontsize=12)
            plt.ylabel('u(x)', fontsize=12)
            plt.title('Comparison between PENN and FEM Solution', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle=':')

        elif self.gd == 2:
            ax2 = fig.add_subplot(131, projection="3d")
            surf1 = ax2.plot_trisurf(node[:, 0], node[:, 1], net_pred,
                                     cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.8)
            ax2.set_title('PENN Solution', fontsize=12)
            ax2.set_xlabel('x', fontsize=10)
            ax2.set_ylabel('y', fontsize=10)    
            ax2.set_zlabel('u(x,y)', fontsize=10)
            fig.colorbar(surf1, ax=ax2, shrink=0.5, label='value')

            ax3 = fig.add_subplot(132, projection="3d")
            surf2 = ax3.plot_trisurf(node[:, 0], node[:, 1],
                                     fem_pred, cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
            ax3.set_title('FEM Solution', fontsize=12)
            ax3.set_xlabel('x', fontsize=10)
            ax3.set_ylabel('y', fontsize=10)    
            ax3.set_zlabel('u(x,y)', fontsize=10)
            fig.colorbar(surf2, ax=ax3, shrink=0.5, label='value')

            ax4 = fig.add_subplot(133, projection="3d")
            surf3 = ax4.plot_trisurf(node[:, 0], node[:, 1],
                                     er, cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
            ax4.set_title('Error: PENN - FEM', fontsize=12)
            ax4.set_xlabel('x', fontsize=10)
            ax4.set_ylabel('y', fontsize=10)    
            ax4.set_zlabel('u(x,y)', fontsize=10)
            fig.colorbar(surf3, ax=ax4, shrink=0.5, label='value')

            fig.suptitle('Comparison between PENN and FEM Solution', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle=':')

        plt.tight_layout()      
        plt.show() 





