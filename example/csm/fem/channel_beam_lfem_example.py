import argparse


# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Finite Element Solution for Channel Beam.
        """)

parser.add_argument('--backend',
                default='numpy', type=str,
                help='Default backend is numpy')

parser.add_argument('--pde',
                default=3, type=int,
                help='id of the PDE model, default is 3.')

parser.add_argument('--init_mesh',
                default='edgemesh', type=str,
                help='Type of mesh, default is EdgeMesh.')

parser.add_argument('--space_degree',
                default=1, type=int,
                help='Degree of Finite Element Space, default is 1.')

parser.add_argument('--E',
                default=2.1e11, type=float,
                help='Type of material properties, default is the beam Young modulus.')

parser.add_argument('--nu',
                default=0.25, type=float,
                help='Type of material properties, default is the beam Poisson ratio.')

parser.add_argument('--rho',
                default=7800, type=float,
                help='Type of material properties, default is the beam steel density.')

parser.add_argument('--g',
                default=9.81, type=float,
                help='Type of material properties, default is the beam gravity.')

parser.add_argument('--pbar_log',
                default=True, type=bool,
                help='Whether to show progress bar, default is True.')

parser.add_argument('--log_level',
                default='INFO', type=str,
                help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL.')

options = vars(parser.parse_args()) 

from fealpy.backend import backend_manager as bm
bm.set_backend(options['backend'])

from fealpy.csm.fem import ChannelBeamModel
model = ChannelBeamModel(options)
model.__str__()
# model.assemble_load(load_case=2)
# K1, F1 = model.channel_beam_system(load_case=1)
# model.apply_bc(K1, F1)
# K2, F2 = model.channel_beam_system(load_case=2)
# model.apply_bc(K2, F2)
uh1 = model.solve(load_case=1)
strain1, stress1 = model.compute_strain_and_stress(uh1)
model.show(uh1, strain1, stress1)

# uh2 = model.solve(load_case=2)
# strain2, stress2 = model.compute_strain_and_stress(uh2)
# model.show(uh2, strain2, stress2)