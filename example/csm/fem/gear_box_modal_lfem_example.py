import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Arbitrary-order finite element method for solving linear elasticity eigenvalue problems
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')

parser.add_argument('--pde',
                    default=2, type=int,
                    help='index of the linear elasticity  model, default is 1')

parser.add_argument('--mesh_file',
                    default='/home/why/fealpy/data/LANXIANG_KETI_0506.inp', type=str,
                    help='mesh file name')

parser.add_argument('--shaft_system_file',
                    default='/home/why/fealpy/data/matlab-total-shaft.mat', type=str,
                    help='the stiffness and mass matrix of the shaft')

parser.add_argument('--space_degree',
        default=1, type=int,
        help='Degree of Lagrange finite element space, default is 1')

parser.add_argument('--neigen',
        default=6, type=int,
        help='Number of eigenvalues to compute, default is 6')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())

from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.csm.fem import GearBoxModalLFEMModel
model = GearBoxModalLFEMModel(options)
model.solve()
