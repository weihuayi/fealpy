import vtk
import numpy as np
from vtk.util import numpy_support as vns


def meshactor(mesh):
    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()
    node = mesh.entity('node')
    cell = mesh.entity('cell')

    GD = mesh.geo_dimension()

    if GD == 1:
        node = np.r_['1', node.reshape(-1, 1), np.zeros((NN, 2))]
    elif GD == 2:
        node = np.r_['1', node, np.zeros((NN, 1))]

    points = vtk.vtkPoints()
    points.SetData(vns.numpy_to_vtk(node))
    cells = vtk.vtkCellArray()
    cells.SetCells(NC, vns.numpy_to_vtkIdTypeArray(cell))

    celltype = mesh.vtk_cell_type()
    vmesh = vtk.vtkUnstructuredGrid()
    vmesh.SetPoints(points)
    vmesh.SetCells(celltype, cells)

    if len(mesh.celldata) > 0:
        cdata = vmesh.GetCellData()
        for key, value in mesh.celldata.items():
            data = vns.numpy_to_vtk(value)
            data.SetName(key)
            cdata.AddArray(data)

    if len(mesh.nodedata) > 0:
        ndata = vmesh.GetPointData()
        for key, value in mesh.nodedata.items():
            data = vns.numpy_to_vtk(value)
            data.SetName(key)
            ndata.AddArray(data)

    if len(mesh.edgedata) > 0:
        ndata = vmesh.GetPointData()
        for key, value in mesh.edgedata.items():
            data = vns.numpy_to_vtk(value)
            data.SetName(key)
            ndata.AddArray(data)

    if (GD == 3) and (len(mesh.facedata) > 0):
        ndata = vmesh.GetPointData()
        for key, value in mesh.facedata.items():
            data = vns.numpy_to_vtk(value)
            data.SetName(key)
            ndata.AddArray(data)

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(vmesh)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


class Actor(vtk.vtkActor):

    def __init__(
            self,
            polydata=None,
            color='Black',
            alpha=1,
            wire=False,
            backfacecolor=None,
            legend=None,
            texture=None,
            comnormals=False):

        vtk.vtkActor.__init__(self)

        self.colors = vtk.vtkNamedColors()
        self.polydata = None

        self.mapper = vtk.vtkPolyDataMapper()
        self.SetMapper(self.mapper)

        properties = self.GetProperty()
        properties.RenderPointsAsSpheresOn()

        if polydata is not None:
            if comnormals is True:
                norms = vtk.vtkPolyDataNormals()
                norms.SetInputData(polydata)
                norms.ComputePointNormalsOn()
                norms.ComputeCellNormalsOn()
                norms.FlipNormalsOff()
                norms.ConsistencyOn()
                norms.Update()
                self.polydata = norms.GetOutput()
            else:
                self.polydata = polydata

            self.mapper.SetInputData(self.polydata)

        if alpha is not None:
            properties.SetOpacity(alpha)

        if color is None:
            self.mapper.ScalarVisibilityOn()
            properties.SetColor(self.colors.GetColor3d(color))
        else:
            self.mapper.ScalarVisibilityOff()
            c = self.colors.GetColor3d(color)
            properties.SetColor(c)
            properties.SetAmbient(0.1)
            properties.SetAmbientColor(c)
            properties.SetDiffuse(1)

        if wire is True:
            properties.SetRepresentationToWireframe()

        if texture is not None:
            properties.SetColor(1., 1., 1.)
            self.mapper.ScalarVisibilityOff()
            self.texture(texture)

        if (backfacecolor is not None) and (alpha == 1):
            bp = vtk.vtkProperty()
            c = self.colors.GetColor3d(backfacecolor)
            bp.SetDiffuseColor(c)
            bp.SetOpacity(alpha)
            self.SetBackfaceProperty(bp)

        if legend is not None:
            self._legend = legend
