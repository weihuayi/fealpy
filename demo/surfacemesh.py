import sys

import numpy as np
import scipy.io as sio
import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh.meshio import load_mat_mesh


def u(p):
    pi = np.pi
    return np.sin(pi*p[:, 0]) * np.sin(pi*p[:, 1]) * np.sin(pi*p[:, 2])


axes = a3.Axes3D(pl.figure())

f = sys.argv[1]
smesh = load_mat_mesh(f)
bc = smesh.barycenter()
uc = u(bc)
smesh.add_plot(axes, cellcolor=uc, showcolorbar=True)
pl.show()

