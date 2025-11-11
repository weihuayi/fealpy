import argparse 
import matplotlib.pyplot as plt

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        The example using lagrange finite element method to solve the dld
        microfluidic 3D chip problem.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="the backend of fealpy, can be 'numpy', 'torch', 'tensorflow' or 'jax'.")

parser.add_argument('--thickness',
    default = 0.1, type = float,
    help = "Thickness of the microfluidic chip.")

parser.add_argument('--init_point',
    default = (0.0, 0.0), type = tuple,
    help = "Initial point for chip positioning.")

parser.add_argument('--chip_height',
    default = 1, type = float,
    help = "Height of the microfluidic chip.")

parser.add_argument('--inlet_length',
    default = 0.1, type = float,
    help = "Length of the inlet section.")

parser.add_argument('--outlet_length',
    default = 0.1, type = float,
    help = "Length of the outlet section.")

parser.add_argument('--radius',
    default = 1 / (3 * 5), type = float,
    help = "Radius of the pillars.")

parser.add_argument('--n_rows',
    default = 3, type = int,
    help = "Number of rows of pillars in each stage.")

parser.add_argument('--n_cols',
    default = 3, type = int,
    help = "Number of columns of pillars in each stage.")

parser.add_argument('--tan_angle',
    default = 1/7, type = float,
    help = "Tangent of the deflection angle.")

parser.add_argument('--n_stages',
    default = 2, type = int,
    help = "Number of stages (or periods) in the chip.")

parser.add_argument('--stage_length',
    default = 7, type = float,
    help = "Number of stages (or periods) in the chip.")

parser.add_argument('--lc',
    default = 0.15, type = float,
    help = "Grid size for meshing.")

parser.add_argument('--show_figure',
    default = False, type = bool,
    help = "Whether to display the generated mesh.")

parser.add_argument('--space_degree',
        default=2, type=int,
        help='Degree of Lagrange finite element space, default is 2.')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())

from fealpy.backend import bm

bm.set_backend(options['backend'])

from fealpy.geometry import DLDMicrofluidicChipModeler3D
from fealpy.mesh import LagrangeTriangleMesh, TriangleMesh
from fealpy.mesher import DLDMicrofluidicChipMesher3D
from fealpy.fem import DLDMicrofluidicChipLFEMModel3D

import gmsh

options = vars(parser.parse_args())
bm.set_backend(options['backend'])
gmsh.initialize()
modeler = DLDMicrofluidicChipModeler3D(options)
modeler.build(gmsh)
mesher = DLDMicrofluidicChipMesher3D(options)
mesher.generate(modeler, gmsh)
gmsh.fltk.run()
gmsh.finalize()

model = DLDMicrofluidicChipLFEMModel3D(options)
model.set_init_mesher(mesher)
# model.mesh = mesh
model.set_space_degree(options['space_degree'])
model.set_inlet_condition()
uh, ph = model.run()