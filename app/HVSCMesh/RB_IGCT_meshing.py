import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import DistMesher2d
from domain import RB_IGCT_Domain
from doping import RB_IGCT_Doping

def sizing_function(p):
    Doping = RB_IGCT_Doping()
    doping = Doping(p)
    doping = 1/doping
    doping[doping>40] = 40
    return doping

domain = RB_IGCT_Domain(fh=sizing_function)
mesher = DistMesher2d(domain,hmin=0.5)

mesh = mesher.meshing(maxit=200)
node = mesh.entity('node')

doping = RB_IGCT_Doping()
mesh.nodedata["doping"] = doping(node)
print(np.max(doping(node)))

size = 1/mesh.nodedata["doping"]
print(np.max(size))
size[size>40] = 40
mesh.nodedata["size"] = size

#print(size)
mesh.to_vtk(fname = "RB_IGCT_mesh.vtu")

fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()
