import argparse
from fealpy.fem import CurlCurlUPMLModel
from fealpy.backend import bm


parser = argparse.ArgumentParser(description=
        """
        Arbitrary-order finite element method for curl–curl problems in the presence of UMPL
        """)

parser.add_argument('--backend',
                    default='numpy', type=str,
                    help='Default backend is numpy')

parser.add_argument('--pde',
                    default=2, type=int,
                    help='Name of the PDE model, default is 2')

parser.add_argument('--nx', default=30, type=int,
                    help='Number of subdivisions in x direction for mesh, default=50')

parser.add_argument('--ny', default=30, type=int,
                    help='Number of subdivisions in y direction for mesh, default=50')

parser.add_argument('--nz', default=30, type=int,
                    help='Number of subdivisions in z direction for mesh, default=50')

parser.add_argument('--init_mesh',
                    default='uniform_tet', type=str,
                    help='Type of mesh, default is uniform_tet')

parser.add_argument('--space_degree',
                    default=0, type=int,
                    help='Degree of Lagrange finite element space, default is 0')

parser.add_argument('--omega',
                    default=8, type=float,
                    help='Angular frequency, default is 8*bm.pi')

parser.add_argument('--mu',
                    default=1.0, type=float,
                    help='magnetic permeability, default is 1.0')

parser.add_argument('--epsilon',
                    default=1.0, type=float,
                    help='permittivity, default is 1.0')

parser.add_argument('--box',
                    default=[-1, 1,-1, 1,-1, 1],
                    help='mesh, default is [-1,1,-1,1,-1,1]')

parser.add_argument('--limits',
                    default=[(-0.75, 0.75), (-0.75, 0.75), (-0.75, 0.75)],
                    help='Computational domain limits: e.g., "[(-0.75,0.75),(-0.75,0.75),(-0.75,0.75)]"')

parser.add_argument('--delta',
                    default=0.25, type=float,
                    help='Discretization parameter, default is 0.25')

parser.add_argument('--solver',
                    default="minres", type=str,
                    help='Solver type, default is direct')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')


options = vars(parser.parse_args())
bm.set_backend(options['backend'])

model = CurlCurlUPMLModel(options)
a = model.run()
print(a)
