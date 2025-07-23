
from fealpy.mesh import TetrahedronMesh 
from fealpy.mesh.abaqus_inp_file_parser import AbaqusInpFileParser

class GearBoxModel:

    def __init__(self, file):
        self.file = file
        self.parser = AbaqusInpFileParser()
        self.parser.parse(file)


    def init_mesh(self):
        """
        """
        return self.parser.to_mesh(TetrahedronMesh)



# Example usage:
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    model = GearBoxModel('/home/why/fealpy/data/LANXIANG_KETI_0506.inp')

    mesh = model.init_mesh()
    mesh.to_vtk('gear_box.vtu')



