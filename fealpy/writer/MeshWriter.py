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
    def __init__(self, mesh, simulation=None):

        GD = mesh.geo_dimension()
        TD = mesh.top_dimension()

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        if TD == 3:
            NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        node = mesh.entity('node')
        if GD == 1:
            node = np.concatenate((node, np.zeros((node.shape[0], 2), dtype=mesh.ftype)), axis=1)
        elif GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=mesh.ftype)), axis=1)
        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))

        cell = mesh.entity('cell')
        cellType = mesh.vtk_cell_type()
        if isinstance(cell, tuple): # Polygon Mesh
            mcell, mcellLocation = cell
            NV = mesh.ds.number_of_vertices_of_cells()
            cell = np.zeros(len(mcell) + NC, dtype=np.int_)
            isIdx = np.ones(len(mcell) + NC, dtype=np.bool_)
            isIdx[0] = False
            isIdx[np.add.accumulate(NV+1)[:-1]] = False
            cell[~isIdx] = NV
            cell[isIdx] = mcell
        else:
        cells = vtk.vtkCellArray()
        cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))

        self.mesh =vtk.vtkUnstructuredGrid() 
        self.mesh.SetPoints(points)
        self.mesh.SetCells(celltype, cells)

        pdata = self.mesh.GetPointData()
        for key, val in mesh.nodedata.items():
            d = vnp.numpy_to_vtk(val)
            d.SetName(key)
            pdata.AddArray(d)

        self.simulation = simulation
        if self.simulation is not None:
            self.queue = multiprocessing.Queue()
            self.process = multiprocessing.Process(None, simulation,
                    args=(self.queue, ))
        else:
            self.queue = None
            self.process = None


    def write(self, fname='test.vtk'):
        writer = vtk.vtkUnstructuredGridWriter()
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
