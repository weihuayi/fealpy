from fealpy.backend import bm
import argparse
from fealpy.fem import ParabolicSTFEMModel

parser = argparse.ArgumentParser(description=
        """        
        Space-time finite element method for solving parabolic PDEs
        """)

parser.add_argument('--backend',
                    default='numpy', type=str,
                    help='Default backend is numpy')

parser.add_argument('--pde',
                    default=5, type=int,
                    help='PDE example number (default: 3) other options: 4')

parser.add_argument('--init_mesh',
                    default='uniform_tri', type=str,
                    help='Initial mesh type (default: uniform_tri)')

parser.add_argument('--mesh_size', default={"nx":1, "ny": 5}, type=dict,
                    help='Mesh size (default: {"nx": 10, "ny": 10})')

parser.add_argument('--space_degree',
                    default=2, type=int,
                    help='Space degree (default: 1)')

parser.add_argument('--quadrature', default=4, type=int,
                    help='Quadrature order (default: 4)')

parser.add_argument('--assemble_method', default=None, type=str,
                    help='Assembly method (default: None) , options: None ,SUPG')

parser.add_argument('--solver', default='direct', type=str,
                    help='Solver type (default: direct)')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())
bm.set_backend(options['backend'])

model = ParabolicSTFEMModel(options)
# model.run.set('refine')
model.run()
# model.show_solution()
# model.slicing_error(p = 0.3)
# model.interpolate_error()
# model.L2_project_error()
# model.to_vtk()
