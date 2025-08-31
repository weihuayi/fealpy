import argparse 
import matplotlib.pyplot as plt


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        The example using lagrange finite element method to solve the dld
        microfluidic chip problem.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="the backend of fealpy, can be 'numpy', 'torch', 'tensorflow' or 'jax'.")

parser.add_argument('--init_point',
    default = (0.0, 0.0), type = tuple,
    help = "Initial point for chip positioning.")

parser.add_argument('--chip_height',
    default = 4.0, type = float,
    help = "Height of the microfluidic chip.")

parser.add_argument('--inlet_length',
    default = 1.0, type = float,
    help = "Length of the inlet section.")

parser.add_argument('--outlet_length',
    default = 1.0, type = float,
    help = "Length of the outlet section.")

parser.add_argument('--radius',
    default = 1 / (2.5 * 3), type = float,
    help = "Radius of the pillars.")

parser.add_argument('--n_rows',
    default = 8, type = int,
    help = "Number of rows of pillars in each stage.")

parser.add_argument('--n_cols',
    default = 4, type = int,
    help = "Number of columns of pillars in each stage.")

parser.add_argument('--tan_angle',
    default = 1/8, type = float,
    help = "Tangent of the deflection angle.")

parser.add_argument('--n_stages',
    default = 3, type = int,
    help = "Number of stages (or periods) in the chip.")

parser.add_argument('--stage_length',
    default = 1, type = float,
    help = "Number of stages (or periods) in the chip.")

parser.add_argument('--lc',
    default = 0.08, type = float,
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

from fealpy.geometry import DLDMicrofluidicChipModeler
from fealpy.mesh import LagrangeTriangleMesh, TriangleMesh
from fealpy.mesher import DLDMicrofluidicChipMesher
from fealpy.fem import DLDMicrofluidicChipLFEMModel
from fealpy.mmesh.tool import high_order_meshploter

import gmsh
box = [0.0, 1.0, 0.0, 1.0]
holes = [[0.3, 0.3, 0.1], [0.3, 0.7, 0.1], [0.7, 0.3, 0.1], [0.7, 0.7, 0.1]]
# holes = [[0.5, 0.5, 0.2]]
mesh = TriangleMesh.from_box_with_circular_holes(box=box, holes=holes, h=0.02)


options = vars(parser.parse_args())
bm.set_backend(options['backend'])
gmsh.initialize()
modeler = DLDMicrofluidicChipModeler(options)
modeler.build(gmsh)
mesher = DLDMicrofluidicChipMesher(options)
mesher.generate(modeler, gmsh)
gmsh.fltk.run()
gmsh.finalize()

model = DLDMicrofluidicChipLFEMModel(options)
model.set_init_mesher(mesher)
# model.mesh = mesh
model.set_space_degree(options['space_degree'])
model.set_inlet_condition()
uh, ph = model.run()