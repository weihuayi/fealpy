import gmsh
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TetrahedronMesh
from fealpy.mesh import QuadrangleMesh


mesh = TetrahedronMesh.from_unit_sphere_gmsh(0.1)
mesh = TetrahedronMesh.from_unit_cube()

# Create the 3D plot
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
plt.show()





