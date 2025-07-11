import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        virtual element method for poisson equation with Dirichlet boundary
        condition 
        """)

parser.add_argument('--backend',
    default='numpy', type=str,
    help='Default backend is numpy')

parser.add_argument('--pde',
    default='SinSin_Sin_Dir_2D', type=str,
    help='Name of the PDE model')

parser.add_argument('--mesh_type',
    default='uniform_poly', type=str,
    help='Type of mesh, default is tri')

parser.add_argument('--nx',
    default=10, type=int)

parser.add_argument('--ny',
    default=10, type=int)

parser.add_argument('--space_degree',
    default=3, type=int,
    help='Polynomial degree of the nonconforming virtual element space')


parser.add_argument('--pbar_log',
    default=True, type=bool,
    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
    default='INFO', type=str,
    help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
options = vars(parser.parse_args())

from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.vem.poisson_noncvem_2d_model import PoissonNonCVEMModel
model = PoissonNonCVEMModel(options)
model.solve()
model.error()
