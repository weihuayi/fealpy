import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.level_set_function import dcircle
from fealpy.mesh import QuadrangleMesh
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh 

box = [-2, 2, -2, 2]
n = 3 
qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad') 
qmesh.print()



