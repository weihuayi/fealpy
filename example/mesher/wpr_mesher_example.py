
import argparse

from fealpy.backend import bm
from fealpy.mesher import WPRMesher


parser = argparse.ArgumentParser(description=
    """
    Test water purifier geometry modeling.
    This script generates the geometry for a water purification device,
    according to the specified structural parameters.
    """)

parser.add_argument('--backend',
    default = 'numpy', type = str,
    help = "Default backend is numpy. You can also choose pytorch, jax, tensorflow, etc.")

parser.add_argument('--block_length',
        default= 6.0, 
        help='Main purification chamber length size.')

parser.add_argument('--block_width',
        default= 2.0, 
        help='Main purification chamber width size.')

parser.add_argument('--inlet_length',
        default= 0.5,
        help='Inlet channel length size.')

parser.add_argument('--inlet_width',
        default= 0.8, 
        help='Inlet channel width size.')


parser.add_argument('--gap',
    default = 0.1, type = float,
    help = "Gap size between purification columns / filtration units.")

parser.add_argument('--gap_len',
    default = 1, type = float,
    help = "Length of the filtration gap region.")

parser.add_argument('--lc',
    default = 0.06, type = float,
    help = "Mesh characteristic length (grid resolution).")

parser.add_argument('--return_mesh',
    default = True, type = bool,
    help = "Whether to display the generated mesh.")

parser.add_argument('--show_figure',
    default = True, type = bool,
    help = "Whether to display the generated mesh.")


options = vars(parser.parse_args())
bm.set_backend(options['backend'])
mesher = WPRMesher(options)
mesher.generate()

