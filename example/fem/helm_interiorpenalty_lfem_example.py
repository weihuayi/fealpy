import argparse
from fealpy.fem import HelmInteriorPenaltyLFEMModel 
from fealpy.backend import bm


# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Arbitrary-order finite element method for solving linear elasticity eigenvalue problems
        """)

parser.add_argument('--backend',
                    default='numpy', type=str,
                    help='Default backend is numpy')

parser.add_argument('--pde',
                    default='bessel2d', type=str,
                    help='Name of the PDE model, default is bessel2d')

parser.add_argument('--nx', default=20, type=int,
                    help='Number of subdivisions in x direction for mesh, default=20')

parser.add_argument('--ny', default=20, type=int,
                    help='Number of subdivisions in y direction for mesh, default=20')

parser.add_argument('--init_mesh',
                    default='uniform_tri', type=str,
                    help='Type of mesh, default is uniform_tri')

parser.add_argument('--space_degree',
                    default=1, type=int,
                    help='Degree of Lagrange finite element space, default is 1')

parser.add_argument('--wave_number',
                    default=10, type=float,
                    help='Wave number, default is 10')

parser.add_argument('--gamma',
                    default=-0.07, type=float,
                    help='Gamma parameter, default is -0.07')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

parser.add_argument('--method', type=str, default='standard',
                    choices=['standard', 'interior_penalty'],
                    help='Choose finite element method: standard or interior_penalty')


options = vars(parser.parse_args())
bm.set_backend(options['backend'])

mode = HelmInteriorPenaltyLFEMModel(options)

mode.plot_error_comparison()
mode.plot_solution()