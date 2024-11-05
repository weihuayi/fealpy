import numpy as np
from fealpy.mesh.mesh_tools import find_node
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

p = np.random.rand(10, 2)
vor = Voronoi(p)
voronoi_plot_2d(vor)
fig = plt.gcf()
axes = fig.gca()
find_node(axes, vor.vertices, showindex=True)
find_node(axes, vor.points, showindex=True)
print(vor.vertices)
print("Region:", vor.regions)
print("Ridge_points:", vor.ridge_points)
plt.show()
