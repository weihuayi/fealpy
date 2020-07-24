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
    def __init__(self, mesh, simulation=None, args=None, etype='cell',
            index=np.s_[:]):

        GD = mesh.geo_dimension()
        TD = mesh.top_dimension()

        NN = mesh.number_of_nodes()

        node, cell, cellType, NC = mesh.to_vtk(etype=etype, index=index)

        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))

        cells = vtk.vtkCellArray()
        cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))

        self.mesh =vtk.vtkUnstructuredGrid() 
        self.mesh.SetPoints(points)
        self.mesh.SetCells(cellType, cells)

        pdata = self.mesh.GetPointData()

        for key, val in mesh.nodedata.items():
            if val is not None:
                d = vnp.numpy_to_vtk(val[:])
                d.SetName(key)
                pdata.AddArray(d)

        cdata = self.mesh.GetCellData()
        for key, val in mesh.celldata.items():
            if val is not None:
                d = vnp.numpy_to_vtk(val[:])
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

    def write(self, fname='test.vtu'):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self.mesh)
        writer.Write()

    def run(self, fname='test.vtu'):
        """

        Notes
        -----

        动态写入时间有关的数据
        """
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self.mesh)
        self.process.start()
        cdata = self.mesh.GetCellData()
        pdata = self.mesh.GetPointData()
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
