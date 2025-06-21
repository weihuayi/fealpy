import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh
from fealpy.mesh.mesh_quality import RadiusRatioQuality

import MeshExample3d
from optimizer import *

mesh = MeshExample3d.unit_sphere()
node = mesh.entity('node')
mesh_quality = RadiusRatioQuality(mesh)
q = mesh_quality(node)
show_mesh_quality(q,ylim=3000)

mesh = iterate_solver_wolfe(mesh) 

node = mesh.entity('node')
q = mesh_quality(node)
show_mesh_quality(q,ylim=3000)
