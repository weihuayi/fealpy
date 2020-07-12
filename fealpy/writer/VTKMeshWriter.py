import numpy as np
import vtk
import vtk.util.numpy_support as vnp

import multiprocessing
import time

class VTKMeshWriter:
    """

    Notes
    -----
    用于在数值模拟过程中输出网格和数据到 vtk 文件中
    """
    def __init__(self, fname, simulation=None, args=None):

        self.vtkmesh =vtk.vtkUnstructuredGrid() 
        self.writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self.vtkmesh)

        self.simulation = simulation
        if self.simulation is not None:
            self.queue = multiprocessing.Queue()
            self.process = multiprocessing.Process(None, simulation,
                    args= args + (self.queue, ))
        else:
            self.queue = None
            self.process = None

    def set_mesh_data(self, mesh, etype='cell', index=np.s_[:]):
        node, cell, cellType, NC = mesh.to_vtk(etype=etype, index=index)
        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))
        cells = vtk.vtkCellArray()
        cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))
        self.vtkmesh.SetPoints(points)
        self.vtkmesh.SetCells(cellType, cells)

        self.set_point_data(mesh.nodedata)
        self.set_cell_data(mesh.celldata)

    def set_point_data(self, data):
        pdata = self.vtkmesh.GetPointData()
        for key, val in data.items():
            d = vnp.numpy_to_vtk(data)
            d.SetName(key)
            pdata.AddArray(d)

    def set_cell_data(self, data):
        cdata = self.vtkmesh.GetCellData()
        for key, val in data.items():
            d = vnp.numpy_to_vtk(data)
            d.SetName(key)
            cdata.AddArray(d)


    def write(self):
        self.writer.Write()

    def write_with_time(self):
        """

        Notes
        -----

        动态写入时间有关的数据
        """
        self.process.start()
        i = 0
        while True:
            if not self.queue.empty():
                data = self.queue.get()
                if isinstance(data, dict):
                    for key, val in data.items():
                        datatype, data = val
                        d = vnp.numpy_to_vtk(data)
                        d.SetName(key)
                        if datatype == 'celldata':
                            cdata.AddArray(d)
                        elif datatype == 'pointdata':
                            pdata.AddArray(d)
                    writer.WriteNextTime(i)    
                    i += 1
                elif isinstance(data, int):
                    if data > 0: # 这里是总的时间层
                        writer.SetNumberOfTimeSteps(data)
                        writer.Start()
                    elif data == -1:
                        self.process.join()
                        print('Simulation stop!')
                        writer.Stop()
                        break
