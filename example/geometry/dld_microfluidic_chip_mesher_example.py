
import argparse

from fealpy.backend import bm 
from fealpy.geometry import DLDMicrofluidicChipModeler
from fealpy.mesher import DLDMicrofluidicChipMesher

import gmsh

parser = argparse.ArgumentParser(description=
    """
    Test microfluidic chip geometry modeling.
    This script generates the geometry for a microfluidic chip using the specified parameters.
    """)

parser.add_argument('--backend',
    default = 'numpy', type = str,
    help = "Default backend is numpy. You can also choose pytorch, jax, tensorflow, etc.")
 
parser.add_argument('--init_point',
    default = (0.0, 0.0), type = tuple,
    help = "Initial point for chip positioning.")

parser.add_argument('--chip_height',
    default = 5.0, type = float,
    help = "Height of the microfluidic chip.")

parser.add_argument('--inlet_length',
    default = 1.0, type = float,
    help = "Length of the inlet section.")

parser.add_argument('--outlet_length',
    default = 1.0, type = float,
    help = "Length of the outlet section.")

parser.add_argument('--radius',
    default = 0.1, type = float,
    help = "Radius of the pillars.")

parser.add_argument('--n_rows',
    default = 8, type = int,
    help = "Number of rows of pillars in each stage.")

parser.add_argument('--n_cols',
    default = 6, type = int,
    help = "Number of columns of pillars in each stage.")

parser.add_argument('--tan_angle',
    default = 1/10, type = float,
    help = "Tangent of the deflection angle.")

parser.add_argument('--n_stages',
    default = 7, type = int,
    help = "Number of stages (or periods) in the chip.")

parser.add_argument('--lc',
    default = 0.1, type = float,
    help = "Grid size for meshing.")

parser.add_argument('--show_figure',
    default = True, type = bool,
    help = "Whether to display the generated mesh.")


options = vars(parser.parse_args())
bm.set_backend(options['backend'])
gmsh.initialize()
modeler = DLDMicrofluidicChipModeler(options)
modeler.build(gmsh)
mesher = DLDMicrofluidicChipMesher(options)
mesher.generate(modeler, gmsh)
gmsh.finalize()
