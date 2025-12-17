import argparse

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        Solve the Stokes equations on a tensor-product mesh for simulating
        the flow behavior inside a water purifier (filtration system).
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="Computation backend. Default is numpy. Other options: pytorch, jax, tensorflow.")

parser.add_argument('--thickness',
    default = 0.4, type = float,
    help = "Thickness (third dimension) of the purification device.")

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
    help = "Gap size between adjacent filtration pillars or porous structures.")

parser.add_argument('--gap_len',
    default = 1, type = float,
    help = "Effective length of the filtration gap region.")

parser.add_argument('--return_mesh',
    default = True, type = bool,
    help = "Whether to display the generated mesh.")

parser.add_argument('--show_figure',
    default = False, type = bool,
    help = "Whether to display the generated mesh.")

parser.add_argument('--lc',
    default = 0.4, type = float,
    help = "Mesh size used for the 2D WPR mesher.")

parser.add_argument('--space_degree',
        default=2, type=int,
        help='Degree of Lagrange finite element space, default is 2.')

parser.add_argument('--level',
        default=4, type=int,
        help='Degree of Lagrange finite element space, default is 2.')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())


from fealpy.backend import bm
from fealpy.mesh import IntervalMesh, TensorPrismMesh

from fealpy.fem import WPRLFEMModel
from fealpy.mesher import WPRMesher

options = vars(parser.parse_args())

bm.set_backend('numpy'); 

mesher = WPRMesher(options)
mesher.generate()

imesh = IntervalMesh.from_interval_domain([0, 0.4], nx=8)
model = WPRLFEMModel(options=options)
model.set_init_mesher(mesher.mesh, imesh)
model.set_inlet_condition()
model.run()
