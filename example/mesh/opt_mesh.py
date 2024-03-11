import numpy as np
import meshio 
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

mesh1 = TriangleMesh.from_meshio("/home/why/data/mesh.msh")
mesh1.vtkview()

