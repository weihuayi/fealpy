import numpy as np
from Doping import TotalDoping
import RBIGCTModel

'''
mesh = RBIGCTModel.struct_mesh()
mesh.to_vtk(fname="structmesh.vtu")
'''
mesh = RBIGCTModel.unstruct_mesh()
mesh.to_vtk(fname="unstructmesh.vtu")

