#!/usr/bin/env python3
#

"""
"""

import sys
import numpy as np

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.tools import showmultirate
from fealpy.tools import MatlabShow
p = int(sys.argv[1])
with open('40vem{}.space'.format(p), 'rb') as f:
    data = pickle.load(f)

print(data)


with open('40error-{}.data'.format(p), 'rb') as f:
    error = pickle.load(f)


uh = data[-1][2*p]
mesh = data[0].mesh
NN = mesh.number_of_nodes()
print(NN)


# plot = MatlabShow()
# plot.show_solution(mesh, uh, '{}.fig'.format(p))
if True:
    fig = plt.figure()
    axes = Axes3D(fig)
    node = mesh.entity('node')
    axes.plot_trisurf(node[:, 0], node[:, 1], uh, cmap=plt.cm.jet, lw=0.0)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)

    #showmultirate(plt, 27, error[0], error[1], error[2], propsize=8)
    plt.show()
