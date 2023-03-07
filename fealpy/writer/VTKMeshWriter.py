import numpy as np
import vtk
import vtk.util.numpy_support as vnp

import multiprocessing
import time

class VTKMeshWriter:
    """

    Notes
    -----
    用于在数值模拟过程中输出网格和数据到 vtk 数据文件中 
    """
    def __init__(self, simulation=None, args=tuple()):

        self.simulation = simulation
        if self.simulation is not None:
            self.queue = multiprocessing.Queue()
            self.process = multiprocessing.Process(None, simulation,
                    args= args + (self.queue, ))
        else:
            self.queue = None
            self.process = None

    def __call__(self, fname, mesh):
        return self.write_to_vtk(fname, mesh) 

    def write_to_vtk(self, fname, mesh):
        """

        Notes
        -----
        """
        node, cell, cellType, NC = mesh.to_vtk()
        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))

        cells = vtk.vtkCellArray()
        cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))

        vtkmesh =vtk.vtkUnstructuredGrid() 
        vtkmesh.SetPoints(points)
        vtkmesh.SetCells(cellType, cells)

        pdata = vtkmesh.GetPointData()
        for key, val in mesh.nodedata.items():
            if val.dtype == np.bool_:
                d = vnp.numpy_to_vtk(val.astype(np.int_))
            else:
                d = vnp.numpy_to_vtk(val)
            d.SetName(key)
            pdata.AddArray(d)

        cdata = vtkmesh.GetCellData()
        for key, val in mesh.celldata.items():
            if val.dtype == np.bool_:
                d = vnp.numpy_to_vtk(val.astype(np.int_))
            else:
                d = vnp.numpy_to_vtk(val)
            d.SetName(key)
            cdata.AddArray(d)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(vtkmesh)
        writer.Write()

    def run(self):
        """

        Notes
        -----

        执行模拟程序， 写入文件

        """
        self.process.start()
        i = 0
        while True:
            if not self.queue.empty():
                data = self.queue.get()
                if isinstance(data, dict):
                    name = data['name']
                    mesh = data['mesh']
                    self.write_to_vtk(name, mesh)
                elif data == -1:
                    print('Simulation stop!')
                    self.process.join()
                    break
                else:
                    pass #TODO: 增加更多的接口协议
