import argparse
from fealpy.data import get_data_path

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Linear finite element method for solving linear elasticity eigenvalue problems
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')

parser.add_argument('--pde',
                    default=2, type=int,
                    help='index of the linear elasticity  model, default is 1')

parser.add_argument('--mesh_file',
                    default='box_case1.inp', type=str,
                    help='mesh file name')

parser.add_argument('--shaft_system_file',
                    default='shaft_case1.mat', type=str,
                    help='the stiffness and mass matrix of the shaft')

parser.add_argument('--neigen',
        default=1, type=int,
        help='Number of eigenvalues to compute, default is 1')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())

options['mesh_file'] = get_data_path('gear', options['mesh_file'])
options['shaft_system_file'] = get_data_path('gear', options['shaft_system_file'])

from fealpy.backend import bm
bm.set_backend(options['backend'])

fname = options['mesh_file'].stem + '.vtu' 

from fealpy.csm.fem import GearBoxModalLinearFEMModel
model = GearBoxModalLinearFEMModel(options)
PS, PM, N0, N1, N2, dof_nodes, dof_comps = model.construct_system['box']()
eps = model.solve(PS, PM)
model.post_process(eps, N0, N1, N2)
model.mesh.to_vtk(fname=fname)
#model.to_mtx(fname=options['mesh_file'].stem+'.mtx', S=PS, M=PM)
model.to_abaqus(S=PS, M=PM, dof_nodes=dof_nodes, dof_comps=dof_comps)
model.write_abaqus_frequency_input()

