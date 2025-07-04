import argparse
import torch.nn as nn
# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        A simple example of using PINN to solve Poisson equation.
        """)

parser.add_argument('--backend',
                    default='pytorch', type=str,
                    help='Default backend is numpy')

parser.add_argument('--pde', 
                    default='coscos', type=str,
                    help='Name of the PDE model, default is boxpoly3d')

parser.add_argument('--npde',
                    default=600, type=int,
                    help='Number of PDE samples, default is 200.')

parser.add_argument('--nbc',
                    default=100, type=int,
                    help='Number of boundary condition samples, default is 100.')

parser.add_argument('--hidden_sizes',
                    default=(64, 32, 32, 16), type=tuple,
                    help='Default hidden sizes, default is (30, 10)')

parser.add_argument('--loss',
                    default=nn.MSELoss(), type=callable,
                    help='Loss function, default is nn.MSELoss')

parser.add_argument('--optimizer', 
                    default='Adam', type=str,
                    help='Optimizer, default is Adam')

parser.add_argument('--activation',
                    default=nn.Tanh(), type=callable,
                    help='Activation function, default is nn.Tanh')

parser.add_argument('--lr',
                    default=0.005, type=float,
                    help='Learning rate, default is 0.01.')

parser.add_argument('--epochs',
                    default=600, type=int,
                    help='Number of training epochs, default is 200.')

parser.add_argument('--step_size',
                    default=0, type=int,
                    help='Period of learning rate decay, default is 0.')

parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Multiplicative factor of learning rate decay. Default: 0.1.')

# parser.add_argument('--log_level',
#                     default='INFO', type=str,
#                     help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())


from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.ml.poisson_pinn_model import PoissonPINNModel
model = PoissonPINNModel(options)
loss, error, e= model.train()
print(f'Loss: {loss[-1]}, Error: {error[-1]}, E: {e[-1]}')

# model.show()
import matplotlib.pyplot as plt
plt.plot(loss, label='Loss', color='blue')
plt.plot(error, label='Error', color='red') 
plt.plot(e, label='E', color='yellow')   
plt.show()
