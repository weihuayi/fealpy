import numpy as np
from paraview.simple import *
import pvtk
import pvtk.util.numpy_support as vnp

import multiprocessing
import time

class MeshPlotter:
    def __init__(self, node, cell, celltype, show,
            celllocation=None,
            nodedata=None,
            celldata=None,
            meshdata=None,
            simulation=None):
        self.show = show
        NN = node.shape[1]
        NC = cell.shape[0]
        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))
        cells = vtk.vtkCellArray()
        cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))

        self.mesh =vtk.vtkUnstructuredGrid() 
        self.mesh.SetPoints(points)
        self.mesh.SetCells(celltype, cells)

        if celldata is not None:
            cdata = self.mesh.GetCellData()
            for key, value in celldata.items():
                data = vnp.numpy_to_vtk(value)
                data.SetName(key)
                cdata.AddArray(data)

        if nodedata is not None:
            ndata = self.mesh.GetPointData()
            for key, value in nodedata.items():
                data = vnp.numpy_to_vtk(value)
                data.SetName(key)
                ndata.AddArray(data)

        self.source = TrivialProducer()
        self.source.GetClientSideObject().SetOutput(self.mesh) 
        self.view = GetActiveViewOrCreate('RenderView')
        self.view.ViewSize = [2418, 1297]
        self.display = Show(self.source, self.view)

        LUT = GetColorTransferFunction(self.show[1])
        self.display.RescaleTransferFunctionToDataRange(True, False)
        self.display.SetScalarBarVisibility(self.view, True)
        self.display.NonlinearSubdivisionLevel = 4
        ColorBy(self.display, self.show)
        Render()
        print('test')

        self.queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(None, simulation,
                args=(self.queue,))

    def run(self):
        self.process.start()
        while True:
            if not self.queue.empty():
                data = self.queue.get()
                if data is not -1:
                    d = vnp.numpy_to_vtk(data)
                    d.SetName(self.show[1])
                    if self.show[0] is 'POINTS':
                        self.mesh.GetPointData().AddArray(d)
                    elif self.show[0] is 'CELLS':
                        self.mesh.GetCellData().AddArray(d)

                    Delete(self.source)
                    self.source = TrivialProducer()
                    self.source.GetClientSideObject().SetOutput(self.mesh) 
                    self.display = Show(self.source, self.view)

                    self.display.RescaleTransferFunctionToDataRange(True, False)
                    self.display.SetScalarBarVisibility(self.view, True)
                    self.display.NonlinearSubdivisionLevel = 4
                    LUT = GetColorTransferFunction(self.show[1])
                    ColorBy(self.display, self.show)
                    Render()
                else:
                    print('exit program!')
                    self.process.join()
                    break
