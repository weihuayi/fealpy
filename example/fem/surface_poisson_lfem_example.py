import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Arbitrary-order Isoparametric Finite Element Method on Surfaces.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')

parser.add_argument('--pde',
                    default='surfacelevelset', type=str,
                    help='Name of the PDE model, default is surfacelevelset')

parser.add_argument('--init_mesh',
                    default='uniform_tet', type=str,
                    help='Type of mesh, default is uniform_tet')

parser.add_argument('--space_degree',
        default=1, type=int,
        help='Degree of Isoparametric Finite Element Space, default is 1')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())

from fealpy.backend import backend_manager as bm
bm.set_backend(options['backend'])

from fealpy.fem import SurfacePoissonLFEMModel
model = SurfacePoissonLFEMModel(options)
model.set_pde(options['pde'])