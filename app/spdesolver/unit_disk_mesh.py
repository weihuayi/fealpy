#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.geometry import SquareWithCircleHoleDomain

domain = SquareWithCircleHoleDomain() 
mesh = TriangleMesh.from_domain_distmesh(domain, maxit=100)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
