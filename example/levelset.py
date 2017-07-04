import numpy as np
import matplotlib.pyplot as plt
from fealpy import dcircle
from fealpy.Mesh import QuadrangleMesh
from fealpy.Mesh import rectangledomainmesh 

box = [-2, 2, -2, 2]
n = 3 
qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad') 
qmesh.print()



