import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        C^m-conforming finite element method for solving polyharmonic problems
        of the form Δ^{m+1} u = f on 2D/3D domains.
        """)

parser.add_argument('--backend',
    default='numpy', type=str,
    help='Default backend is numpy')

parser.add_argument('--pde',
    default='sinsinbi', type=str,
    help='Name of the PDE model (e.g., sinsinbi, sinsinsinbi, sinsintri)')

parser.add_argument('--init_mesh',
    default='tri', type=str,
    help='Type of mesh, default is tri')

parser.add_argument('--mesh_size',
    default=10, type=int)

parser.add_argument('--space_degree',
    default=5, type=int,
    help='Polynomial degree of the finite element space (recommended >= 2m+1)')

parser.add_argument('--smoothness',
    default=1, type=int,
    help='Smoothness order m, where the PDE is Δ^{m+1} u = f')

parser.add_argument('--pbar_log',
    default=True, type=bool,
    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
    default='INFO', type=str,
    help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
options = vars(parser.parse_args())

from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.fem.polyharmonic_cr_fem_model import PolyharmonicCrFEMModel
model = PolyharmonicCrFEMModel(options)
model.solve()
model.error()
