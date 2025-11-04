import argparse


# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Finite Element Solution for Truss Tower.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')

parser.add_argument('--pde',
                    default=3, type=int,
                    help='id of the PDE model, default is 3.')

parser.add_argument('--init_mesh',
                    default='edgemesh', type=str,
                    help='Type of mesh, default is EdgeMesh.')

parser.add_argument('--space_degree',
        default=1, type=int,
        help='Degree of Finite Element Space, default is 1.')

parser.add_argument('--E',
                    default=2.1e11, type=float,
                    help='Type of mesh, default is the truss Young modulus.')

parser.add_argument('--nu',
                    default=0.3, type=float,
                    help='Type of mesh, default is the truss Poisson ratio.')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True.')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL.')

options = vars(parser.parse_args()) 

from fealpy.backend import backend_manager as bm
bm.set_backend(options['backend'])