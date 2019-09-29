import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import CosCosData

pde = CosCosData()

quadtree = pde.init_mesh(n=2, meshtype='quadtree')
options = quadtree.adaptive_options(maxsize=1, HB=True)


pmesh = quadtree.to_pmesh()
axes0 = plt.subplot(1, 2, 1)
pmesh.add_plot(axes0)
pmesh.find_cell(axes0, showindex=True)

NC = pmesh.number_of_cells()
numrefine = np.zeros(NC, dtype=np.int)
numrefine[0:4] = 0 
numrefine[4:8] = 2

quadtree.adaptive(numrefine, options)
pmesh = quadtree.to_pmesh()
print(options['HB'])

axes1 = plt.subplot(1, 2, 2)
pmesh.add_plot(axes1)
pmesh.find_cell(axes1, showindex=True)
plt.show()



