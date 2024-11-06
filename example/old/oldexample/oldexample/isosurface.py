import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.draw import ellipsoid
import load_cube

# create an object and read in data from file 
cube=load_cube.CUBE(gaussian_cube_file.cube)

# Obtain the surface mesh setting a specific isovalue (0.2)
verts, faces = measure.marching_cubes(cube.data, 0.2)

# Display resulting triangular mesh using Matplotlib. 
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(111, projection='3d')

# Generate triangles
mesh = Poly3DCollection(verts[faces])
ax.add_collection3d(mesh)

ax.set_xlabel("x-axis: a = 6")
ax.set_ylabel("y-axis: b = 10")
ax.set_zlabel("z-axis: c = 16")

ax.set_xlim(0, 100)  
ax.set_ylim(0, 100)  
ax.set_zlim(0, 100) 

plt.show()
