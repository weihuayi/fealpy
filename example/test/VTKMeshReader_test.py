

from fealpy.mesh.VTKMeshReader import VTKMeshReader


reader = VTKMeshReader('112ew.vtu')

node = reader.get_point()
cell = reader.get_cell()
print(node)
print(cell, len(cell))
