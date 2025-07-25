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
                    default=1, type=int,
                    help='index of the linear elasticity  model, default is 1')

parser.add_argument('--mesh_type',
                    default='uniform_tet', type=str,
                    help='Type of mesh, default is uniform_tet')

parser.add_argument('--nx', type=int, default=10,
                    help='Number of divisions along x-direction, default is 10')

parser.add_argument('--ny', type=int, default=2,
                    help='Number of divisions along y-direction, default is 2')

parser.add_argument('--nz', type=int, default=2,
                    help='Number of divisions along z-direction, default is 2')

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

from fealpy.csm.model import CSMModelManager
from fealpy.fem import LinearElasticityEigenLFEMModel
pde = CSMModelManager("linear_elasticity").get_example(1)
options['pde'] = pde
n = 1
options['nx'] = n * 10 
options['ny'] = n * 2 
options['nz'] = n * 2
model = LinearElasticityEigenLFEMModel(options)
model.solve()
