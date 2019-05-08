import vtk
import fealpy.plotter.colors as colors


def meshactor(mesh):
    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()
    node = mesh.entity('node')
    edge = mesh.entity('edge')

    GD = mesh.geo_dimension()

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

    if mesh.meshtype is 'tri':


class Actor(vtk.vtkActor):

    def __init__(
            self,
            polydata=None,
            color='gold',
            alpha=1,
            wire=False,
            backfacecolor=None,
            legend=None,
            texture=None,
            comnormals=False):

        vtk.vtkActor.__init__(self)

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
            properties.SetColr(colors.getColor('gold'))
        else:
            self.mapper.ScalarVisibilityOff()
            color = colors.getColor(color)
            properties.SetColor(color)
            properties.SetAmbient(0.1)
            properties.SetAmbientColor(color)
            properties.SetDiffuse(1)

        if wire is True:
            properties.SetRepresentationToWireframe()

        if texture is not None:
            properties.SetColor(1., 1., 1.)
            self.mapper.ScalarVisibilityOff()
            self.texture(texture)

        if (backfacecolor is not None) and (alpha == 1):
            bp = vtk.vtkProperty()
            bp.SetDiffuseColor(colors.getColor(backfacecolor))
            bp.SetOpacity(alpha)
            self.SetBackfaceProperty(bp)

        if legend is not None:
            self._legend = legend
