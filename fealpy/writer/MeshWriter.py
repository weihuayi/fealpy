import numpy as np
import vtk
import vtk.util.numpy_support as vnp

import multiprocessing
import time

class MeshWriter:
    def __init__(self, node, cell, celltype,  
            celllocation=None,
            nodedata=None, 
            celldata=None, 
            meshdata=None, 
            simulation=None):

        NN = node.shape[1]
        NC = cell.shape[0]
        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))
        cells = vtk.vtkCellArray()
        cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))

        self.mesh =vtk.vtkUnstructuredGrid() 
        self.mesh.SetPoints(points)
        self.mesh.SetCells(celltype, cells)

        self.queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(None, simulation,
                args=(self.queue, ))

    def run(self, fname='test.vtk'):
        self.process.start()
        pdata = self.mesh.GetPointData()
        while True:
            if not self.queue.empty():
                data = self.queue.get()
                if data is not -1:
                    for key, val in data.items():
                        d = vnp.numpy_to_vtk(val)
                        d.SetName(key)
                        pdata.AddArray(d)
                else:
                    print('exit program!')
                    self.process.join()
                    break
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self.mesh)
        writer.Write()
