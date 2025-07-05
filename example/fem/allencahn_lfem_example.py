from fealpy.backend import bm
import argparse
from fealpy.fem.allencahn_lfem_model import AllenCahnLFEMModel

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Arbitrary-order finite element method for solving Allen-Cahn equation
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')

parser.add_argument('--pde',
                    default='sincoscos', type=str,
                    help='Name of the PDE model, default is sincoscos')

parser.add_argument('--init_mesh',
                    default='tri', type=str,
                    help='Type of mesh, default is tri, options are tri, quad, moving_tri, moving_quad')

parser.add_argument('--space_degree',
                    default=1, type=int,
                    help='Degree of Lagrange finite element space, default is 1')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

parser.add_argument('--assemble_method',
                    default= None, type=str,
                    help='Assemble method, default is None, options are None, iso , fast')

parser.add_argument('--quadrature',
                    default=4, type=int,
                    help='index of quadrature formula, default is 4')

parser.add_argument('--time_step',
                    default = 0.0001 , type=float,
                    help='Time step size, default is 0.001')

parser.add_argument('--time_strategy',
                    default='backward', type=str,
                    help='Time step strategy, default is backward, options are backward, forward, moving_mesh')

parser.add_argument('--up',
                    default=2, type=int,
                    help='Degree of velocity function space, default is 2')

parser.add_argument('--solve',
                    default='direct', type=str,
                    help='Solver method, default is direct, options are direct, cg')

parser.add_argument('--is_volume_conservation',
                    default=True, type=bool,
                    help='Whether to use Lagrange multiplier, default is True')

parser.add_argument('--mm_param',
                    default=None, type=dict,
                    help='Parameters for moving mesh, default is None')

options = vars(parser.parse_args())

from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.fem import AllenCahnLFEMModel
model = AllenCahnLFEMModel(options)

model.run(save_vtu_enabled=True, error_estimate_enabled=True)