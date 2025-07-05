import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Arbitrary-order HuZhang finite element method for solving linear elasticity problems
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')

parser.add_argument('--pde',
                    default='boxtri2d', type=str,
                    help='Name of the PDE model, default is boxtri2d')

parser.add_argument('--init_mesh',
                    default='uniform_tri', type=str,
                    help='Type of mesh, default is uniform_tri')

parser.add_argument('--space_degree',
                    default=3, type=int,
                    help='Degree of Lagrange finite element space, default is 3')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())


from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.fem.linear_elasticity_huzhang_fem_model import LinearElasticityHuzhangFEMModel
model = LinearElasticityHuzhangFEMModel(options)
model.run['onestep']()

