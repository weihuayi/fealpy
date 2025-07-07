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

parser.add_argument('--pde', 
                    default='sin', type=str,
                    help="Name of the PDE model, default is 'sin'")

parser.add_argument('--meshtype', 
                    default='uni', type=str,
                    help="Type of mesh to use: 'uni' for uniform or 'tri' for triangular, default is 'uni'")

parser.add_argument('--sampling_mode', 
                    default='random', type=str,
                    help="Sampling method for collocation points: 'random' or 'linspace', default is 'random'")

parser.add_argument('--npde',
                    default=400, type=int,
                    help='Number of PDE samples, default is 400.')

parser.add_argument('--nbc',
                    default=100, type=int,
                    help='Number of boundary condition samples, default is 100.')

parser.add_argument('--hidden_sizes',
                    default=(32, 32, 16), type=tuple,
                    help='Default hidden sizes, default is (32, 32, 16).')

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
                    default=0.001, type=float,
                    help='Learning rate for the optimizer, default is 0.001.')

parser.add_argument('--epochs',
                    default=4000, type=int,
                    help='Number of training epochs, default is 4000.')



options = vars(parser.parse_args())


from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.ml.poisson_pinn_model import PoissonPINNModel
model = PoissonPINNModel(options)
model.train()
model.show()

