import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from typing import Union, Optional

from fealpy.backend import bm
from fealpy.utils import timer
from fealpy.typing import TensorLike
from fealpy.model import ComputationalModel, PDEModelManager
from fealpy.model.poisson import PoissonPDEDataT

from fealpy.ml import gradient, optimizers, activations

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, ISampler


class PoissonPINNModel(ComputationalModel):
    """A Physics-Informed Neural Network (PINN) model for solving Poisson equations.
    
    This class implements a PINN framework to solve Poisson PDE problems using neural networks.
    It handles PDE residual calculation, boundary condition enforcement, and training process.
    The model supports both uniform and random sampling strategies for collocation points.
    
    Parameters:
        options(dict): If None, default parameters from get_options() will be used.
            Configuration dictionary containing:
            - pde(int or HelmholtzPDEDataT): PDE definition;
            - lr(float): Learning rate;
            - epochs(int): Number of training epochs;
            - weights(tuple): Weight for the equation loss and boundary loss;
            - hidden_size(tuple): Tuple of hidden layer sizes;
            - npde(int): Number of PDE collocation points;
            - nbc(int): Number of boundary collocation points;
            - activation(str): Activation function (options: 'Tanh', 'ReLU', 'LeakyReLU', 'Sigmoid', 'LogSigmoid', 'Softmax', 'LogSoftmax');
            - optimizer(str): Optimization algorithm (options: 'Adam', 'SGD');
            - sampling_mode(str): Sampling strategy (options: 'linspace' or 'random');
            - step_size(int): Period of learning rate decay;
            - gamma(float): Multiplicative factor of learning rate decay;
            - pbar_log(bool): Whether to use progress bar for logging;
            - log_level(str): Logging level (options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
            
    Attributes:
        pde(PoissonPDEDataT): The Poisson PDE problem definition.

        gd(int): Geometric dimension of the PDE.

        domain(list): Computational domain boundaries.

        npde(int): Number of PDE collocation points.

        nbc(int): Number of boundary collocation points.

        weights(tuple): Weight for the equation loss and boundary loss.

        hidden_size(tuple): Tuple of hidden layer sizes.

        activation(nn.Module): Activation function.

        optimizer(torch.optim.Optimizer): Optimization algorithm.

        scheduler(torch.optim.lr_scheduler.StepLR): Learning rate scheduler.

        tmr(timer): Timer object for recording training and solution process times.

        mesh(TriangleMesh or UniformMesh): Discretization mesh for error estimation.

        net(nn.Module): Neural network architecture.

        Loss(list): Training loss history.

        error_fem(list): Error history compared with finite element method.

        options(dict): Configuration dictionary passed during initialization.

        solution_flag(bool): Used to mark whether the true solution exists. Default is True.
            
    Methods:
        get_options(): Get default configuration parameters for the model.

        set_pde(): Initialize the PDE problem.

        set_network(): Configure the neural network architecture.

        set_mesh(): Initialize computational mesh.

        pde_residual(): Compute PDE residual.

        bc_residual(): Compute boundary condition residual.

        predict(): Make predictions at given points.

        fem(): Solve Poisson's equation using Finite Element Method (FEM) for comparison
            if PDE don't have an analytical solution.

        run(): Execute training and prediction process.

        show(): Visualize results.
    
    Reference:
        https://wnesm678i4.feishu.cn/wiki/Me8lw5ryxigAMbkcnL8cWVQpn4g?from=from_copylink

    Examples:
        >>> from fealpy.backend import bm  
        >>> bm.set_backend('pytorch')   # Set the backend to PyTorch  
        >>> from fealpy.ml import PoissonPINNModel  
        >>> options = PoissonPINNModel.get_options()    # Get the default options of the network  
        >>> model = PoissonPINNModel(options=options)  
        >>> model.run()   # Train the network  
        >>> model.show()   # Show the results of the network training  )
    """
    def __init__(self, options: dict = {}):
        self.options = self.get_options()
        self.options.update(options)
        
        self.pbar_log = self.options['pbar_log']
        self.log_level = self.options['log_level']
        super().__init__(pbar_log=self.pbar_log, log_level=self.log_level)
  
        self.lr = self.options['lr']   # learning rate
        self.epochs = self.options['epochs']  # epochs
        self.hidden_size = self.options['hidden_size']     # hidden layer sizes
        self.activation = activations[self.options['activation']]    # activation function
        self.npde = self.options['npde']   # number of PDE collocation points
        self.nbc = self.options['nbc']   # number of boundary collocation points
        self.weights = self.options['weights']   # weights
        self.tmr = timer()   # timer

        self.set_pde(self.options['pde'])  # PDE 
        self.set_mesh(self.options['mesh_size'])  # mesh
        self.set_network()  # network

    @classmethod
    def get_options(cls):
        """Get default configuration parameters for the model.
        
        Defines and returns default configurations for the model through a command-line argument parser,
        including PDE problem number, grid size, network structure, and optimizer parameters.
        
        Returns
            options(dict): Dictionary containing all configuration parameters with parameter names as keys and default values
        """

        import argparse

        # Argument parsing
        parser = argparse.ArgumentParser(description="Poisson equation solver using PINN.")

        parser.add_argument('--pde',default=1, type=int,
                            help="Built-in PDE example ID for different Poisson problems, default is 1.")
        
        parser.add_argument('--mesh_size',
                            default=30, type=int,
                            help='Number of grid points along each dimension, default is 30.')

        parser.add_argument('--sampling_mode', 
                            default='random', type=str,
                            help="Sampling method for collocation points: 'random' or 'linspace', default is 'random'")

        parser.add_argument('--npde',
                            default=400, type=int,
                            help='Number of PDE samples, default is 400.')

        parser.add_argument('--nbc',
                            default=100, type=int,
                            help='Number of boundary condition samples, default is 100.')
    
        parser.add_argument('--weights',
                            default=(1, 30), type=tuple,
                            help='The first value is the weight for the equation loss, and the second ' \
                            'value is the weight for the boundary loss., default is (1, 30).')
        
        parser.add_argument('--hidden_size',
                            default=(32, 32, 16), type=tuple,
                            help='Default hidden sizes, default is (32, 32, 16).')

        parser.add_argument('--optimizer', 
                            default="Adam",  type=str,
                            help="Optimizer to use for training, default is Adam, options are 'Adam' , 'SGD'.")

        parser.add_argument('--activation',
                            default="Tanh", type=str,
                            help="Activation function, default is Tanh, " \
                            "options are 'Tanh', 'ReLU', 'LeakyReLU', 'Sigmoid', 'LogSigmoid', 'Softmax', 'LogSoftmax'.")

        parser.add_argument('--lr',
                            default=0.001, type=float,
                            help='Learning rate for the optimizer, default is 0.001.')
        
        parser.add_argument('--step_size',
                            default=0, type=int,
                            help='Period of learning rate decay, default is 0.')

        parser.add_argument('--gamma',
                            default=0.99, type=float,
                            help='Multiplicative factor of learning rate decay. Default: 0.99.')

        parser.add_argument('--epochs',
                            default=2000, type=int,
                            help='Number of training epochs, default is 2000.')
        
        parser.add_argument('--pbar_log',
                            default=True, type=bool,
                            help='Whether to show progress bar, default is True')

        parser.add_argument('--log_level',
                            default='INFO', type=str,
                            help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')
        options = vars(parser.parse_args())
        return options
    
    def set_pde(self, pde: Union[PoissonPDEDataT, int]=1):
        """Initialize the PDE problem definition.
        
        Parameters:
            pde(Union[PoissonPDEDataT, int]): Either a Poisson's equation problem object or the ID (integer) of a predefined example. 
                If an integer, the corresponding predefined Poisson's equation problem is retrieved from the PDE model manager.
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('poisson').get_example(pde)
        else:
            self.pde = pde 
        self.gd = self.pde.geo_dimension()
        self.domain = self.pde.domain()

    def set_network(self, net=None):
        """Configure the neural network architecture and optimizer, learning rate scheduler.
        
        Parameters:
            net(torch.nn.Module, optional): Custom network architecture. If None, creates default MLP.
                Defaults to None (auto-create network).
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

        opt = optimizers[self.options.get('optimizer', 'Adam')]
        self.optimizer = opt(params=self.net.parameters(), lr=self.lr)

        step_size = self.options.get('step_size', 0)
        gamma = self.options.get('gamma', 0.99)
        self.set_steplr(step_size, gamma)

    def set_mesh(self, mesh_size: int=30, mesh=None):
        """Create computational mesh.
        
        Creates a computational mesh over the domain defined by the PDE based on the specified mesh size.
        
        Parameters:
            mesh_size(tuple of int): Number of nodes in each dimension.

            mesh: Mesh object. If None, creates a default mesh based on the PDE domain and mesh size.
        """
        if mesh == None:
            self.mesh_size = (mesh_size, ) * self.gd
            cell_size = tuple(x - 1 for x in self.mesh_size)
            self.mesh = self.pde.init_mesh(*cell_size)
        else:
            self.mesh = mesh

    def set_steplr(self, step_size: int=0, gamma: float=0.9):
        """Create learning rate scheduler
        
        Initializes a learning rate scheduler for decaying the learning rate periodically during training.
        
        Parameters:
            step_size(int): Default is 0. Period for learning rate decay, i.e., decay every step_size epochs. No scheduler is used if step_size = 0.
            
            gamma(float): default is 0.9. Multiplicative factor for learning rate decay, new_lr = current_lr * gamma.
        """
        if step_size == 0:
            self.steplr = None
        else:
            self.steplr = StepLR(self.optimizer, step_size, gamma)

    def pde_residual(self, p: TensorLike) -> TensorLike:
        """Compute PDE residual (Laplacian(u) + f).
        
        Parameters:
            p(TensorLike): Collocation points where residual is evaluated.
            
        Returns:
            TensorLike: PDE residual values at input points.
                
        Notes:
            Uses automatic differentiation to compute second derivatives.
            The residual is calculated as Δu + f where Δ is the Laplacian operator.
        """
        u = self.net(p)
        f = self.pde.source(p)

        grad_u = gradient(u, p, create_graph=True)  ## (npde, dim)
        laplacian = bm.zeros(u.shape[0])    
        
        for i in range(p.shape[-1]):
            u_ii = gradient(grad_u[..., i], p, create_graph=True, split=True)[i]   # compute ∂²u/∂x_i²
            laplacian += u_ii.flatten()

        assert f.shape == laplacian.shape, \
            f"Shape mismatch: f.shape={f.shape}, laplacian.shape={laplacian.shape}."
        val = laplacian + f
        return val

    def bc_residual(self, p: TensorLike) -> TensorLike:
        """Compute boundary condition residual (u - g).
        
        Parameters:
            p(TensorLike): Boundary points where residual is evaluated.

        Returns:
            TensorLike: Boundary condition residual values at input points.

        Notes:
            g represents the Dirichlet boundary condition values.
            The residual is calculated as u - g where u is the network prediction.
        """
        u = self.net(p).flatten()
        bc = self.pde.dirichlet(p)
        assert u.shape == bc.shape, \
            f"Shape mismatch: u.shape={u.shape}, bc.shape={bc.shape}."
        val = u - self.pde.dirichlet(p)
        return val

    def run(self):
        """Execute the training process for the PINN model.
        
        Notes:
            Training process includes:
            1. Collocation point sampling;
            2. PDE and BC residual computation;
            3. Loss backpropagation;
            4. Periodic error evaluation;
            
            The loss function combines PDE residual and boundary condition terms.
        """
        next(self.tmr)
        # sampler
        sampler_pde = ISampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])
        sampler_bc = BoxBoundarySampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])
        mse = nn.MSELoss(reduction='mean')

        self.Loss = []
        self.error = []
        w = self.weights
        mesh = self.mesh

        for epoch in range(self.epochs+1):
            self.optimizer.zero_grad()

            # sample
            if (self.options['sampling_mode'] == 'linspace') :
                if epoch == 0:
                    ''' 'linspace' sampling mode only works one times '''
                    spde = sampler_pde.run(self.npde)
                    sbc = sampler_bc.run(self.nbc)
            else:
                spde = sampler_pde.run(self.npde)
                sbc = sampler_bc.run(self.nbc)

            # compute residuals
            pde_res = self.pde_residual(spde)
            bc_res = self.bc_residual(sbc)

            # compute loss
            mse_pde = mse(pde_res, bm.zeros_like(pde_res))
            mse_bc = mse(bc_res, bm.zeros_like(bc_res))

            loss = w[0] * mse_pde + w[1] * mse_bc
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                # L² error 
                try:
                    if hasattr(self.pde, 'solution'):
                        self.solution_flag = True
                        error = self.net.estimate_error(self.pde.solution, mesh, coordtype='c')
                        self.error.append(error.item())
                except NotImplementedError:
                    # The exact solution is unknown，so Mark exact solution as nonexistent .
                    self.solution_flag = False
                self.Loss.append(loss.item())
                self.logger.info(f"epoch: {epoch}, Loss: {loss.item():.6f}")  
               
        self.tmr.send(f'PINN training time')
        if self.solution_flag:
            next(self.tmr)

    def predict(self, p: TensorLike) -> TensorLike:
        """Make predictions using the trained network.
        
        Parameters:
            p (TensorLike): Input points where prediction is needed.

        Returns:
            TensorLike: Network predictions at input points.
        """
        return self.net(p)
    
    def fem(self):
        """Solve Poisson's equation using Finite Element Method (FEM) for comparison 
        if PDE don't have an analytical solution.
        
        Returns:
            uh(TensorLike): FEM solution results

        Notes:
            q=1, p=q+2, where p is the polynomial degree of the finite element space.
        """
        from fealpy.functionspace import LagrangeFESpace
        from fealpy.fem import BilinearForm, LinearForm
        from fealpy.fem  import ScalarDiffusionIntegrator, ScalarSourceIntegrator
        from fealpy.fem import DirichletBC
        from fealpy.solver import spsolve

        pde = self.pde
        mesh = self.mesh
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
        """Visualize training results and solution comparisons.
        
        Notes:
            Creates plots showing:
            1. Training loss history.
            2. Error compared to FEM solution.
            3. For 1D/2D problems: comparison between predicted and true solutions.
            
            Uses matplotlib for visualization with separate subplots for different metrics.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        Loss = bm.log10(bm.tensor(self.Loss)).numpy()

        # loss curve
        axes.plot(Loss, 'r-', linewidth=2)
        axes.set_title('PINN Training Loss', fontsize=12)
        axes.set_xlabel('training epochs*100', fontsize=10)
        axes.set_ylabel('log10(Loss)', fontsize=10)
        axes.grid(True)

        # PINN vs exact error 
        if self.solution_flag:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            error = bm.log10(bm.tensor(self.error)).numpy()
            axes.plot(error, 'b--', linewidth=2)
            axes.set_title('L2 Error between PINN Solution and Exact Solution', fontsize=12)
            axes.set_ylabel('log10(Error)', fontsize=10)
            axes.set_xlabel('training epochs*100', fontsize=10)
            axes.grid(True)

        if self.gd <= 2:
            mesh = self.mesh
            node = mesh.entity('node')
            u_pred = self.net(node).detach().numpy().flatten()  # PINN solution            
            if self.solution_flag:
                u_true = self.pde.solution(node).detach().numpy()   # exact solution
            else:
                u_true = self.fem().detach().numpy()  # FEM solution
            node = node.detach().numpy()
            fig = plt.figure()
            if self.gd == 1:
                # plot PINN solution and exact solution
                plt.plot(node, u_true, 'b-', linewidth=2, label='Exact Solution')
                plt.plot(node, u_pred, 'g--', linewidth=2, label='PINN Prediction')
                plt.plot(node, u_pred-u_true, 'r-', linewidth=2, label='Error: PINN-Exact')
                plt.xlabel('x', fontsize=12)
                plt.ylabel('u(x)', fontsize=12)
                plt.title('Comparison between PINN and Exact Solution', fontsize=14)
                plt.legend(fontsize=12)
                plt.grid(True, linestyle=':')
            else:
                # plot PINN solution
                ax1_3d = fig.add_subplot(131, projection='3d')
                surf1 = ax1_3d.plot_trisurf(
                    node[:, 0], node[:, 1], u_pred,
                    cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.8)
                ax1_3d.set_title('PINN Solution')
                ax1_3d.set_xlabel('X')
                ax1_3d.set_ylabel('Y')
                ax1_3d.set_zlabel('u(x,y)')
                fig.colorbar(surf1, ax=ax1_3d, shrink=0.5, label='Value')

                # plot exact solution
                ax2_3d = fig.add_subplot(132, projection='3d')
                surf2 = ax2_3d.plot_trisurf(
                    node[:, 0], node[:, 1], u_true,
                    cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
                ax2_3d.set_xlabel('X')
                ax2_3d.set_ylabel('Y')
                ax2_3d.set_zlabel('u(x,y)')
                fig.colorbar(surf2, ax=ax2_3d, shrink=0.5, label='Value')

                ax4 = fig.add_subplot(133, projection="3d")
                surf3 = ax4.plot_trisurf(node[:, 0], node[:, 1],
                                        u_pred - u_true, cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
                ax4.set_xlabel('x', fontsize=10)
                ax4.set_ylabel('y', fontsize=10)    
                ax4.set_zlabel('u(x,y)', fontsize=10)
                fig.colorbar(surf3, ax=ax4, shrink=0.5, label='value')

                if self.solution_flag:
                    ax2_3d.set_title('Exact Solution')
                    ax4.set_title('Error: PINN - Exact', fontsize=12)
                    plt.suptitle('Comparison between PINN and Exact Solution')
                else:
                    ax2_3d.set_title('FEM Solution')
                    ax4.set_title('Error: PINN - FEM', fontsize=12)
                    plt.suptitle('Comparison between PINN and FEM Solution')

        plt.tight_layout()      
        plt.show()  


