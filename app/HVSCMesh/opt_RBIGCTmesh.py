import numpy as np
from Doping import TotalDoping
import RBIGCTModel

#mesh = RBIGCTModel.struct_mesh()
mesh = RBIGCTModel.unstruct_mesh()
node = mesh.entity("node")
Doping = TotalDoping(node)
mesh.nodedata["Doping"] = Doping
#mesh.to_vtk(fname="struct_mesh.vtu")
mesh.to_vtk(fname="unstructmesh.vtu")

