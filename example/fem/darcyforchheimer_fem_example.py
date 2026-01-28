import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        finite element method for solving Darcy-Forchheimer problems
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')

parser.add_argument('--pde',
                    default='3', type=int,
                    help='Name of the PDE model, default is 1')

parser.add_argument('--init_mesh',
                    default='uniform_tri', type=str,
                    help='Type of mesh, default is uniform_tri')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())

from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.fem.darcyforchheimer_fem_model import DarcyForchheimerFEMModel
model = DarcyForchheimerFEMModel(options)
model.solve['TPDv']()
