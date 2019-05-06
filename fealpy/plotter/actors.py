import vtk
import fealpy.plotter.colors as colors


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
