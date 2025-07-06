import argparse
import torch.nn as nn
from torch.optim import Adam
# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        A simple example of using PINN to solve Poisson equation.
        """)

parser.add_argument('--backend',
                    default='pytorch', type=str,
                    help='Default backend is numpy')

parser.add_argument('--pde', 
                    default='sinsinsin', type=str,
                    help='Name of the PDE model, default is ')

parser.add_argument('--meshtype', 
                    default='uni', type=str,
                    help="['uni', 'tri'], default is 'uni'")

parser.add_argument('--sampling_mode', 
                    default='random', type=str,
                    help="['random', 'linspace'], default is 'uni'")

parser.add_argument('--npde',
                    default=200, type=int,
                    help='Number of PDE samples, default is 200.')

parser.add_argument('--nbc',
                    default=200, type=int,
                    help='Number of boundary condition samples, default is .')

parser.add_argument('--hidden_sizes',
                    default=(32, 16, 8), type=tuple,
                    help='Default hidden sizes, default is')

parser.add_argument('--loss',
                    default=nn.MSELoss(reduction='mean'), type=callable,
                    help='Loss function, default is nn.MSELoss')

parser.add_argument('--optimizer', 
                    default=Adam,  type=callable,
                    help='Optimizer, default is Adam')

parser.add_argument('--activation',
                    default=nn.Tanh(), type=callable,
                    help='Activation function, default is nn.Tanh')

parser.add_argument('--lr',
                    default=0.005, type=float,
                    help='Learning rate, default is 0.01.')

parser.add_argument('--epochs',
                    default=1000, type=int,
                    help='Number of training epochs, default is .')

parser.add_argument('--step_size',
                    default=0, type=int,
                    help='Period of learning rate decay, default is 0.')

parser.add_argument('--gamma',
                    default=0.9, type=float,
                    help='Multiplicative factor of learning rate decay. Default: 0.1.')


options = vars(parser.parse_args())


from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.ml.poisson_pinn_model import PoissonPINNModel
model = PoissonPINNModel(options)
model.train()
model.show()

