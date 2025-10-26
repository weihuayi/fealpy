from fealpy.backend import backend_manager as bm
from fealpy.mesher import ChipMesher

import argparse

## 参数解析
parser = argparse.ArgumentParser(description=
    """
    Generate a perforated chip mesh using Gmsh for testing.
    This script builds a rectangle with multiple circular holes arranged in a staggered pattern.
    """)

parser.add_argument('--backend',
    default='numpy', type=str,
    help="Default backend is numpy. You can also choose pytorch, jax, tensorflow, etc.")


parser.add_argument('--box',
    default = [0.0, 0.75, 0.0, 0.41], type=int,
    help="Bounding box of the rectangular domain as [x0 x1 y0 y1]. Default: [0.0, 0.75, 0.0, 0.41].")

parser.add_argument('--center',
    default = (0.1, 0.05), type=float,
    help="Center coordinates (cx, cy) of the first circular hole. Default: (0.1, 0.05).")

parser.add_argument('--radius',
    default = 0.029, type=float,
    help="Radius of each circular hole. Default: 0.029.")

parser.add_argument('--l1',
    default = 0.1, type=float,
    help="Vertical spacing between circles in a column. Default: 0.1.")

parser.add_argument('--l2',
    default = 0.1, type=float,
    help="Horizontal spacing between circle columns. Default: 0.1.")

parser.add_argument('--h',
    default = 0.04, type=float,
    help="Vertical shift (staggering) of alternating columns. Default: 0.04.")

parser.add_argument('--lc',
    default = 0.01, type=float,
    help="Target mesh element size (characteristic length). Default: 0.01.")

parser.add_argument('--return_mesh',
                    default='True', type=str,
                    help='Whether to generate mesh, default is True')

parser.add_argument('--show_figure',
                    default='False', type=str,
                    help='Whether to show figure in Gmsh, default is True')

options = vars(parser.parse_args())

bm.set_backend(options['backend'])
mesher = ChipMesher(options=options)

import matplotlib.pyplot as plt

mesh = mesher.mesh
fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes, cellcolor="#80e673")
plt.show()
