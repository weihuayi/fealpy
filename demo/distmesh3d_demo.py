
from time import time

import numpy as np

import pylab as pl
import mpl_toolkits.mplot3d as a3

from fealpy.mesh.level_set_function import Sphere 
from fealpy.mesh.level_set_function import DistDomain3d 
from fealpy.mesh.distmesh import DistMesh3d
from fealpy.mesh.sizing_function  import huniform



fd = Sphere()
fh = huniform
bbox = [-1, 1, -1, 1, -1, 1]
h = 0.1

domain = DistDomain3d(fd, fh, bbox)
distmesh3d = DistMesh3d(domain, h)
distmesh3d.run(100)
mesh = distmesh3d.mesh
ax0 = a3.Axes3D(pl.figure())
mesh.add_plot(ax0)
pl.show()
