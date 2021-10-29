import sys
import meshio
import numpy as np
import scipy.io as sio

from fealpy.mesh import LagrangeTriangleMesh, MeshFactory, LagrangeWedgeMesh

import vtk
from vtk.numpy_interface import dataset_adapter as dsa

class vtkReader():
    def meshio_read(fname, Mesh):
        data = meshio.read(fname)
        node = data.points
        cell = data.cells[0][1]

        mesh = Mesh(node, cell)
        mesh.to_vtk(fname='write.vtu')

    def vtk_read(fname, node_keylist, cell_keylist, Mesh): 
        """
        param[in] fname 文件名
        param[in] node_keylist 节点数据的名字
        param[in] cell_keylist 单元数据的名字
        param[in] Mesh 网格类型
        """
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(fname)
        reader.Update()
        vmesh = reader.GetOutput()
        wmesh = dsa.WrapDataObject(vmesh)

        node = wmesh.GetPoints() #节点数据
        cell = wmesh.GetCells().reshape(-1, 7)[:, 1:] #单元数据
        mesh = LagrangeTriangleMesh(node, cell)
        nh = 25
        h = 0.5
        mesh = LagrangeWedgeMesh(mesh, h, nh)

        for key in node_keylist:
            arr = vmesh.GetPointData().GetArray(key)
            dl = arr.GetNumberOfTuples()
            data = np.zeros(dl, dtype=np.int)
            for i in range(dl):
                data[i] = arr.GetComponent(i, 0)
            mesh.nodedata[key] = data
        for key in cell_keylist:
            arr = vmesh.GetCellData().GetArray(key)
            dl = arr.GetNumberOfTuples()
            data = np.zeros(dl, dtype=np.int)
            for i in range(dl):
                data[i] = arr.GetComponent(i, 0)
            mesh.celldata[key] = data

fname = sys.argv[1]
#vtkReader.meshio_read(fname, TriangleMesh)
vtkReader.vtk_read(fname, ['uh'], ['uhgrad'], LagrangeWedgeMesh) 





