
import numpy as np
import vtk
import vtk.util.numpy_support as vnp


class VTKMeshReader():
    def __init__(self, fname):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(fname)
        reader.Update()
        self.vtkmesh = reader.GetOutput()

    def get_point(self):
        node = vnp.vtk_to_numpy(self.vtkmesh.GetPoints().GetData())
        return node

    def get_cell(self):
        cell = vnp.vtk_to_numpy(self.vtkmesh.GetCells().GetData())
        return cell

    def get_cell_type(self):
        pass

    def get_point_data(self):
        pass

    def get_cell_data(self):
        pass
        

