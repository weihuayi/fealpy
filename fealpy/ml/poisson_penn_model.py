import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from ..backend import  bm
from ..utils import timer
from ..typing import TensorLike

from typing import Union, Optional
from ..model import ComputationalModel, PDEDataManager
from ..model.poisson import PoissonPDEDataT
from . import gradient
from ..mesh import UniformMesh
from .modules import Solution
from . import optimizers, activations


class PoissonPennModel(ComputationalModel):
    """
    """
    def __init__(self, options=None):

        default_options = self.get_options()
        self.options = options

        self.set_pde(self.options.get('pde', default_options['pde']))
        self.gd = self.pde.geo_dimension()
        self.domain = self.pde.domain()
        self.set_mesh(self.options.get('mesh_size', default_options['mesh_size']))

        # 网络超参数、激活函数
        self.lr = self.options.get('lr', default_options['lr'])
        self.epochs = self.options.get('epochs', default_options['epochs'])
        self.hidden_sizes = self.options.get('hidden_sizes', default_options['hidden_sizes'])
        self.activation = activations[self.options.get('activation', default_options['activation'])]

        self.set_network()  # 网络
       
       
        # 优化器与学习率调度器
        opt = optimizers[self.options.get('optimizer', default_options['optimizer'])]

        self.optimizer = opt(params=self.net.parameters(), lr=self.lr)
        step_size = self.options.get('step_size', default_options['step_size'])
        gamma = self.options.get('gamma', default_options['gamma'])
        self.steplr = StepLR(self.optimizer, step_size, gamma)




    @classmethod
    def get_options(cls):

        import argparse
        parser = argparse.ArgumentParser(description=
        """
        A simple example of using PENN to solve Poisson equation.
        """)

        parser.add_argument('--backend',
                            default='pytorch', type=str,
                            help="Computational backend to use, default backend is 'pytorch'.")

        parser.add_argument('--pde',default=1, type=int,
                            help="Built-in PDE example ID (1-5 for different Poisson problems), default is 1.")
        
        parser.add_argument('--mesh_size',
                            default=(30, ), type=tuple,
                            help='网格剖分数, default is 30.')


        parser.add_argument('--hidden_sizes',
                            default=(50, 50, 50), type=tuple,
                            help='Default hidden sizes, default is (50, 50, 50).')


        parser.add_argument('--optimizer', 
                            default="Adam",  type=str,
                            help='Optimizer to use for training, default is Adam')

        parser.add_argument('--activation',
                            default="LogSigmoid", type=str,
                            help='Activation function, default is LogSigmoid')

        parser.add_argument('--lr',
                            default=0.001, type=float,
                            help='Learning rate for the optimizer, default is 0.001.')

        parser.add_argument('--step_size',
                            default=4000, type=int,
                            help='Period of learning rate decay, default is 0.')

        parser.add_argument('--gamma',
                            default=0.9, type=float,
                            help='Multiplicative factor of learning rate decay. Default: 0.9.')

        parser.add_argument('--epochs',
                            default=2000, type=int,
                            help='Number of training epochs, default is 3000.')

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
        
        Parameters
            pde : Union[PoissonPDEDataT, int]
                PDE object or built-in example ID.
        """
        if isinstance(pde, int):
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
                    layers.append(self.activation())
            net = nn.Sequential(*layers)
        self.net = Solution(net)


    def set_mesh(self, mesh_size:tuple):
        """Create computational mesh.
        
        Args:
            mesh_size: tuple of int, mesh size
            mesh: Mesh instance or mesh type name string
        """
        n = len(mesh_size)
        if n == 1:
            mesh_size = mesh_size*self.gd
        else:
            assert n == self.gd, "Mesh size should be a tuple of length 1 or geo_dimension."
        # exten = tuple( x for pair in zip((0,)*self.gd, mesh_size) for x in pair)
        # self.mesh = UniformMesh(self.domain, exten)
        self.mesh = self.pde.init_mesh(*mesh_size)

    def shape_function(self, p: TensorLike) -> TensorLike:
        """Compute shape function values at points p.
        
        Args:        
            p: (n_points, geo_dimension) tensor of points.
        
        Returns:
            (n_points, n_shape_functions) tensor of shape function values.
        """
        domain = self.domain
        is_bc = self.pde.is_dirichlet_boundary(p)
        val = bm.zeros(p.shape[0], dtype=bm.float64)
        non_bc_mask = ~is_bc
        if non_bc_mask.any():
            p_non_bc = p[non_bc_mask]
            if len(p_non_bc.shape) == 1:
                p_non_bc = p_non_bc.reshape(-1, 1)
            x0 = (domain[0] + domain[1]) / 2
            temp_val = bm.exp(-((p_non_bc[:, 0] - x0)**2) / 2)
            for i in range(1, self.gd):
                x0 = (domain[2*i] + domain[2*i+1]) / 2
                temp_val *= bm.exp(-((p_non_bc[:, i] - x0)**2) / 2)
            val[non_bc_mask] = temp_val
        return val

    def scaling_function(p: TensorLike) -> TensorLike:
        """Compute scaling function values at points p.
        
        Args:
            p: (n_points, geo_dimension) tensor of points.
        
        Returns:
            (n_points, n_shape_functions) tensor of scaling function values.
        """
        pass

    def loss(self, p: TensorLike, u: TensorLike) -> TensorLike:
        pass
        
    def run(self):
        pass

    def show(self):
        pass


