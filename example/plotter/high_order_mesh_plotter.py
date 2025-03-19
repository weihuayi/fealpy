import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.backend import backend_manager as bm
from fealpy import logger
logger.setLevel('WARNING')

from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh

from fealpy.utils import timer

t = timer()
next(t)
mesh = TriangleMesh.from_unit_sphere_surface(refine=3)
t.send("Generate mesh on unit sphere!")
next(t)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')

mesh.add_plot(axes)
plt.show()

logger.info("Generate mesh on unit sphere!")


