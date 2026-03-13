from fealpy.backend import bm
import argparse
from fealpy.mesher import BlockWithHoleMesher

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Generate a hollow block with cylindrical holes.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')

parser.add_argument('--block',
        default= {
                'length': 10.0,
                'width': 1.0,
                'height': 10.0,
        }, 
        help='Default backend is numpy')

parser.add_argument('--cylinders',
        default=[
                ((5.0, 0, 5.0), 1.0)
        ],
        help='Cylinders as list of ((x, y, z), radius)')

parser.add_argument('--h',
        default=0.3, type=str,
        help='Maximum mesh size')

parser.add_argument('--return_mesh',
                    default='True', type=str,
                    help='Whether to generate mesh, default is True')

parser.add_argument('--show_figure',
                    default='True', type=str,
                    help='Whether to show figure in Gmsh, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())
from fealpy.backend import bm
bm.set_backend(options['backend'])
model = BlockWithHoleMesher(options)
model.get_options()
