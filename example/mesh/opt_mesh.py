import numpy as np
import meshio 
import matplotlib.pyplot as plt
import ipdb

ipdb.set_trace()
data = meshio.ansys.read('/home/why/data/mesh0.msh')



from fealpy.mesh import TriangleMesh
mesh1 = TriangleMesh.from_meshio("/home/why/data/mesh.msh")
mesh1.vtkview()

