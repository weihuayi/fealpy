import argparse


parser = argparse.ArgumentParser(description="""
                                 Linear FEM for 3D Truss (Edge elements).
                                 """)

parser.add_argument('--backend', 
                    default='numpy', type=str, 
                    help='Backend: numpy/cupy, default numpy')

parser.add_argument('--pde',
                    default=3, type=int,
                    help='ID of the PDE model, default 3')

parser.add_argument('--init_mesh',
                    default='edgemesh', type=str,
                    help='Type of mesh, default is EdgeMesh')

parser.add_argument('--space_degree',
        default=1, type=int,
        help='Degree of Finite Element Space, default is 1')

parser.add_argument('--E', 
                    default=1500.0, type=float, 
                    help="Young's modulus")

parser.add_argument('--nu',
                    default=0.3, type=float,
                    help='Type of mesh, default is the bar Poisson ratio')

parser.add_argument('--plot', 
                    action='store_true', 
                    help='Plot the truss structure')

parser.add_argument('--scale', default=10.0, type=float, 
                    help='Deformation scale factor for plotting')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())

import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.csm.fem.bar_model import BarModel
bm.set_backend(options['backend'])

model = BarModel(options)
K, F = model.linear_system()
K_bc, F_bc = model.apply_bc(K, F)
uh = model.solve(K_bc, F_bc)
strain, stress = model.compute_strain_and_stress(uh)
mstress = model.calculate_von_mises_stress(stress)

model.show(uh, strain, stress, mstress)