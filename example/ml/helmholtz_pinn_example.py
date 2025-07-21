import argparse
import torch.nn as nn
from torch.optim import Adam, SGD

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        A simple example of using PINN to solve Poisson equation.
        """)

parser.add_argument('--backend',
                    default='pytorch', type=str,
                    help="Computational backend to use, default backend is 'pytorch'.")

parser.add_argument('--pde',default=1, type=int,
                    help="Built-in PDE example ID (1-5 for different Helmholtz problems), default is 1.")

parser.add_argument('--complex', 
                    default=True, type=bool,
                    help="Enable complex-valued solution modeling, default is True")

parser.add_argument('--wave', 
                    default=1.0, type=float,
                    help="Wave number k for Helmholtz equation (Δu + k²u + f = 0), default is 1.0")

parser.add_argument('--meshtype', 
                    default='uniform_tri', type=str, choices=['uniform_tri', 'uniform_quad', 'uniform_tet', 'uniform_hex'],
                    help="Mesh type: 2D options: 'uniform_tri' (triangles), 'uniform_quad' (quadrangles); "
                         "3D options: 'uniform_tet' (tetrahedrons), 'uniform_hex' (hexahedrons)")

parser.add_argument('--sampling_mode', 
                    default='random', type=str,
                    help="Sampling method for collocation points: 'random' or 'linspace', default is 'random'")

parser.add_argument('--npde',
                    default=1500, type=int,
                    help='Number of PDE samples, default is 400.')

parser.add_argument('--nbc',
                    default=300, type=int,
                    help='Number of boundary condition samples, default is 100.')

parser.add_argument('--hidden_sizes',
                    default=(50, 50, 50, 50, 50, 50, 50), type=tuple,
                    help='Default hidden sizes, default is (50, 50, 50, 50, 50, 50, 50).')

parser.add_argument('--loss',
                    default=nn.MSELoss(reduction='mean'), type=callable,
                    help='Loss function to use, default is nn.MSELoss')

parser.add_argument('--optimizer', 
                    default=Adam,  type=callable,
                    help='Optimizer to use for training, default is Adam')

parser.add_argument('--activation',
                    default=nn.Tanh(), type=callable,
                    help='Activation function, default is nn.Tanh')

parser.add_argument('--lr',
                    default=0.01, type=float,
                    help='Learning rate for the optimizer, default is 0.001.')

parser.add_argument('--step_size',
                    default=1000, type=int,
                    help='Period of learning rate decay, default is 0.')

parser.add_argument('--gamma',
                    default=0.9, type=float,
                    help='Multiplicative factor of learning rate decay. Default: 0.9.')

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


from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.ml.helmholtz_pinn_model import HelmholtzPINNModel
model = HelmholtzPINNModel(options)
model.run()
model.show()

