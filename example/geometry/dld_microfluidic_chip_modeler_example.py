
import argparse

from fealpy.backend import bm 
from fealpy.geometry import DLDMicrofluidicChipModeler

import gmsh

parser = argparse.ArgumentParser(description=
    """
    Test microfluidic chip mesh generation.
    This script generates a microfluidic chip mesh using the given parameters.
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

options = vars(parser.parse_args())
bm.set_backend(options['backend'])
gmsh.initialize()
modeler = DLDMicrofluidicChipModeler(options)
modeler.build(gmsh)
modeler.show()
gmsh.finalize()
