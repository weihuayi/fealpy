import numpy as np
import vtk
import vtk.util.numpy_support as vnp

import multiprocessing
import time

class MeshWriter:
    """

    Notes
    -----
    用于在数值模拟过程中输出网格和数据到 vtk 文件中
    """
    def __init__(self, mesh, simulation=None, args=None):

        GD = mesh.geo_dimension()
        TD = mesh.top_dimension()

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        if TD == 3:
            NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        cellType = mesh.vtk_cell_type()
        node, cell = mesh.to_vtk()

        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))

        cells = vtk.vtkCellArray()
        cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))

        self.mesh =vtk.vtkUnstructuredGrid() 
        self.mesh.SetPoints(points)
        self.mesh.SetCells(cellType, cells)
        
        pdata = self.mesh.GetPointData()
        for key, val in mesh.nodedata.items():
            d = vnp.numpy_to_vtk(val)
            d.SetName(key)
            pdata.AddArray(d)

        cdata = self.mesh.GetCellData()
        for key, val in mesh.celldata.items():
            d = vnp.numpy_to_vtk(val)
            d.SetName(key)
            cdata.AddArray(d)

        self.simulation = simulation
        if self.simulation is not None:
            self.queue = multiprocessing.Queue()
            self.process = multiprocessing.Process(None, simulation,
                    args= args + (self.queue, ))
        else:
            self.queue = None
            self.process = None

    def write(self, fname='test.vtk'):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self.mesh)
        writer.Write()

    def run(self, fname='test.vtk'):
        """

        Notes
        -----

        动态写入时间有关的数据
        """
        self.process.start()
        pdata = self.mesh.GetPointData()
        cdata = self.mesh.GetCellData()
        while True:
            if not self.queue.empty():
                data = self.queue.get()
                if data != -1:
                    for key, val in data.items():
                        datatype, data = val
                        d = vnp.numpy_to_vtk(data)
                        d.SetName(key)
                        if datatype == 'celldata':
                            cdata.AddArray(d)
                        elif datatype == 'pointdata':
                            pdata.AddArray(d)
                else:
                    print('exit program!')
                    self.process.join()
                    break
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self.mesh)
        writer.Write()
