import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Finite element analysis for fluid-structure interaction (FSI) in hydraulic valve systems,
        with linear elasticity for structural deformation in steady-state conditions.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="Default backend is numpy")

parser.add_argument('--pde',
                    default=4, type=int,
                    help="index of the linear elasticity  model, default is 4")

parser.add_argument('--mesh_type',
                    default='uniform_tet', type=str,
                    help="Type of mesh, default is uniform_tet")

parser.add_argument('--space_degree',
        default=1, type=int,
        help="Degree of Lagrange finite element space, default is 1")

parser.add_argument('--E', 
                    default=2.1e11, type=float, 
                    help="Young's modulus (E) in GPa for the elastic material")

parser.add_argument('--nu',
                    default=0.3, type=float,
                    help="Poisson's ratio (nu) for the elastic material, default is 0.3")

parser.add_argument('--rho',
                    default=7800, type=float,
                    help="density for the elastic material, default is 7800")

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())


from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.csm.fem.hydraulic_pipe_lfem_model import  HydraulicPipeLFEMModel
model = HydraulicPipeLFEMModel(options)

A, F = model.linear_system()
A1, F1 = model.apply_bc(A, F)
uh = model.solve(A1, F1)
print("max displacement:", float(bm.max(bm.abs(uh))))
print(float(bm.linalg.norm(uh)))
model.show(uh)
print("-----------------------------")
