import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Finite Element Solution for 3D Timoshenko Beam.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')

parser.add_argument('--pde',
                    default=2, type=int,
                    help='id of the PDE model, default is 2')

parser.add_argument('--init_mesh',
                    default='edgemesh', type=str,
                    help='Type of mesh, default is EdgeMesh')

parser.add_argument('--E',
                    default=2.07e11, type=float,
                    help='Type of mesh, default is Young modulus')

parser.add_argument('--nu',
                    default=0.276, type=float,
                    help='Type of mesh, default is Poisson ratio')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args()) 

from fealpy.backend import backend_manager as bm
bm.set_backend(options['backend'])

from fealpy.csm.fem import TimoshenkoBeamModel
model = TimoshenkoBeamModel(options)
model.__str__()
#model.solve['cg']()
