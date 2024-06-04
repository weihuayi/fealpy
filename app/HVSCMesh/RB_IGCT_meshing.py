import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import DistMesher2d
from domain import RB_IGCT_Domain

domain = RB_IGCT_Domain()
mesher = DistMesher2d(domain,hmin=5)
mesh = mesher.meshing(maxit=50)

fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()
