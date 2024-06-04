import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import DistMesher2d
from domain import Rectangle_BJT_Domain

domain = Rectangle_BJT_Domain()
mesher = DistMesher2d(domain,hmin=10)
mesh = mesher.meshing(maxit=50)

fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()

