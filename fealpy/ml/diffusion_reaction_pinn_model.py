import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from typing import Union, Optional

from fealpy.backend import bm
from fealpy.utils import timer
from fealpy.typing import TensorLike
from fealpy.model import ComputationalModel, PDEModelManager
from fealpy.model.diffusion_reaction import DiffusionReactionPDEDataT

from fealpy.ml import gradient, optimizers, activations

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, ISampler

class DiffusionReactionPINNModel(ComputationalModel):
    def __init__(self, options: dict = {}):
        self.options = self.get_options()
        self.options.update(options)

        self.pbar_log = self.options.get('pbar_log', True)
        self.log_level = self.options.get('log_level', 'INFO')
        super().__init__(pbar_log=self.pbar_log, log_level=self.log_level)
       
        self.lr = self.options['lr']
        self.epochs = self.options['epochs']
        self.hidden_size = self.options['hidden_size']
        self.hidden_size = self.options['hidden_size']
        self.activation = activations[self.options['activation']]
        self.npde = self.options['npde']
        self.nbc = self.options['nbc']
        self.weights = self.options['weights']
        self.tmr = timer()  

        self.set_pde(self.options['pde'])
        self.set_mesh(self.options['mesh_size'])
        self.set_network()


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

        # Argument parsing
        parser = argparse.ArgumentParser(description=
                """
                A simple example of using PINN to solve Poisson equation.
                """)

        parser.add_argument('--pde',default=3, type=int,
                            help="Built-in PDE example ID for different Poisson problems, default is 3.")
        
        parser.add_argument('--mesh_size', default=30, type=int,
                            help='Number of grid points along each dimension, default is 30.')

        parser.add_argument('--sampling_mode', default='linspace', type=str,
                            help="Sampling method for collocation points: 'random' or 'linspace', default is 'linspace'")
       
        parser.add_argument('--npde', default=29, type=int,
                            help='Number of PDE samples, default is 29.')

        parser.add_argument('--nbc', default=30, type=int,
                            help='Number of boundary condition samples, default is 30.')
    
        parser.add_argument('--weights', default=(1, 30), type=tuple,
                            help='The first value is the weight for the equation loss, and the second ' \
                            'value is the weight for the boundary loss., default is (1, 30).')
        
        parser.add_argument('--hidden_size', default=(50, 50, 50, 50), type=tuple,
                            help='Default hidden sizes, default is (50, 50, 50, 50).')

        parser.add_argument('--optimizer', default="Adam",  type=str,
                            help="Optimizer to use for training, default is Adam, options are 'Adam' , 'SGD'.")

        parser.add_argument('--activation', default="Tanh", type=str,
                            help="Activation function, default is Tanh, " \
                            "options are 'Tanh', 'ReLU', 'LeakyReLU', 'Sigmoid', 'LogSigmoid', 'Softmax', 'LogSoftmax'.")

        parser.add_argument('--lr', default=0.001, type=float,
                            help='Learning rate for the optimizer, default is 0.001.')
        
        parser.add_argument('--step_size', default=0, type=int,
                            help='Period of learning rate decay, default is 0.')

        parser.add_argument('--gamma', default=0.99, type=float,
                            help='Multiplicative factor of learning rate decay. Default: 0.99.')

        parser.add_argument('--epochs', default=5000, type=int,
                            help='Number of training epochs, default is 5000.')
        
        parser.add_argument('--pbar_log', default=True, type=bool,
                            help='Whether to show progress bar, default is True')

        parser.add_argument('--log_level', default='INFO', type=str,
                            help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')
        options = vars(parser.parse_args())
        return options
    
    def set_pde(self, pde: Union[DiffusionReactionPDEDataT, int]=1):
        """Initialize the PDE problem definition.
        
        Parameters
            pde : Union[[DiffusionReactionPDEDataT, int]
                Either a predefined PDE object or string identifier for built-in examples.
                Defaults to 'sin' example problem.
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('diffusion_reaction').get_example(pde)
        else:
            self.pde = pde 
        self.gd = self.pde.geo_dimension()
        self.domain = self.pde.domain()

    def set_network(self, net=None):
        """Configure the neural network architecture.
        
        Parameters
            net : torch.nn.Module, optional
                Custom network architecture. If None, creates default MLP.
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

    def pde_residual(self, p: TensorLike) -> TensorLike:
        """Compute PDE residual (Laplacian(u) + f).
        
        Parameters
            p : TensorLike
                Collocation points where residual is evaluated.
            
        Returns
            TensorLike
                PDE residual values at input points.
                
        Notes
            Uses automatic differentiation to compute second derivatives.
            The residual is calculated as Δu + f where Δ is the Laplacian operator.
        """
        u = self.net(p)
        f = self.pde.source(p)
        reaction = (self.pde.reaction_coef() * u).flatten()
        
        # 一阶导数计算
        grad_u = gradient(u, p, create_graph=True)  ## (npde, dim)
        laplacian = bm.zeros(u.shape[0])    # 拉普拉斯项初始化
        
        for i in range(p.shape[-1]):
            u_ii = gradient(grad_u[..., i], p, create_graph=True, split=True)[i]   # 计算 ∂²u/∂x_i²
            laplacian += u_ii.flatten()

        assert f.shape == laplacian.shape, \
            f"Shape mismatch: f.shape={f.shape}, laplacian.shape={laplacian.shape}."
        val = laplacian + f + reaction.flatten()
        return val

    def bc_residual(self, p: TensorLike) -> TensorLike:
        """Compute boundary condition residual (u - g).
        
        Parameters
            p : TensorLike
                Boundary points where residual is evaluated.
                
        Returns
            TensorLike
                Boundary condition residual values at input points.
                
        Notes
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
        
        Notes
            Training process includes:
            1. Collocation point sampling;
            2. PDE and BC residual computation;
            3. Loss backpropagation;
            4. Periodic error evaluation;
            
            The loss function combines PDE residual and boundary condition terms.
        """
        next(self.tmr)
        self.Loss = []
        w = self.weights

        self.sampler_pde = ISampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])
        self.sampler_bc = BoxBoundarySampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])

        mse = nn.MSELoss(reduction='mean')

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

            # 计算残差
            pde_res = self.pde_residual(spde)
            bc_res = self.bc_residual(sbc)

            # 计算损失
            mse_pde = mse(pde_res, bm.zeros_like(pde_res))
            mse_bc = mse(bc_res, bm.zeros_like(bc_res))

            loss = w[0] * mse_pde + w[1] * mse_bc
            loss.backward()
            # 更新参数
            self.optimizer.step()

            if epoch % 100 == 0:
                self.Loss.append(loss.item())
                self.logger.info(f"epoch: {epoch}, Loss: {loss.item():.6f}")  

        self.tmr .send(f'PINN training time')

    def predict(self, p: TensorLike) -> TensorLike:
        """Make predictions using the trained network.
        
        Parameters
            p : TensorLike
                Input points where prediction is needed.
                
        Returns
            TensorLike
                Network predictions at input points.
        """
        return self.net(p)
    
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
        from fealpy.fem import ScalarDiffusionIntegrator, ScalarSourceIntegrator, ScalarMassIntegrator
        from fealpy.fem import DirichletBC
        from fealpy.solver import spsolve

        pde = self.pde
        mesh_size = tuple(x - 1 for x in self.mesh_size)
        mesh = pde.init_mesh(*mesh_size)
        p = 1
        q = p + 2
        r = self.pde.reaction_coef()
        space = LagrangeFESpace(mesh, p)
        S = BilinearForm(space)
        S.add_integrator(ScalarDiffusionIntegrator(coef=1, q=q))
        S.add_integrator(ScalarMassIntegrator(coef=-r, q=q))
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
        
        Notes
            Creates plots showing:
            1. Training loss history.
            2. Error compared to FEM solution.
            3. For 1D/2D problems: comparison between predicted and true solutions.
            
            Uses matplotlib for visualization with separate subplots for different metrics.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        Loss = bm.log10(bm.tensor(self.Loss)).numpy()

        # 绘制损失曲线
        axes.plot(Loss, 'r-', linewidth=2)
        axes.set_title('Training Loss', fontsize=12)
        axes.set_xlabel('training epochs*100', fontsize=10)
        axes.set_ylabel('log10(Loss)', fontsize=10)
        axes.grid(True)

        mesh = self.mesh
        node = mesh.entity('node')
  
        u_pred = self.net(node).detach().numpy().flatten()  # PINN预测解
        u_fem = self.fem().detach().numpy()   # 有限元解
        node = node.detach().numpy()
        fig = plt.figure()
 
        # 子图1：PINN预测解
        ax1_3d = fig.add_subplot(131, projection='3d')
        surf1 = ax1_3d.plot_trisurf(
            node[:, 0], node[:, 1], u_pred,
            cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.8)
        ax1_3d.set_title('PINN Solution')
        ax1_3d.set_xlabel('X')
        ax1_3d.set_ylabel('Y')
        ax1_3d.set_zlabel('u(x,y)')
        fig.colorbar(surf1, ax=ax1_3d, shrink=0.5, label='Value')

        # 子图2：有限元解
        ax2_3d = fig.add_subplot(132, projection='3d')
        surf2 = ax2_3d.plot_trisurf(
            node[:, 0], node[:, 1], u_fem,
            cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
        ax2_3d.set_title('FEM Solution')
        ax2_3d.set_xlabel('X')
        ax2_3d.set_ylabel('Y')
        ax2_3d.set_zlabel('u(x,y)')
        fig.colorbar(surf2, ax=ax2_3d, shrink=0.5, label='Value')

        ax4 = fig.add_subplot(133, projection="3d")
        surf3 = ax4.plot_trisurf(node[:, 0], node[:, 1],
                                u_pred - u_fem, cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
        ax4.set_title('Error: PINN - FEM', fontsize=12)
        ax4.set_xlabel('x', fontsize=10)
        ax4.set_ylabel('y', fontsize=10)    
        ax4.set_zlabel('u(x,y)', fontsize=10)
        fig.colorbar(surf3, ax=ax4, shrink=0.5, label='value')
        plt.suptitle('Comparison between PINN and FEM Solution')

        plt.tight_layout()      
        plt.show()  # 显示第二个Figure
