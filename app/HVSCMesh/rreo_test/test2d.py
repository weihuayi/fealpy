import numpy as np
import matplotlib.pyplot as plt

from mesh_quality import RadiusRatioQuality

import MeshExample2d
from optimizer import *

mesh = MeshExample2d.triangle_domain()

mesh_quality = RadiusRatioQuality(mesh)
mesh = iterate_solver(mesh,funtype=1)

mesh.to_vtk(fname='opt_mesh.vtu')
