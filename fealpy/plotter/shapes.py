import vtk
from fealpy.plotter.actors import Actor


def Sphere(
        position=[0, 0, 0],
        radius=1,
        color='r',
        alpha=1,
        resolution=24
        ):

    ss = vtk.vtkSphereSource()
    ss.SetRadius(radius)
    ss.SetThetaResolution(2*resolution)
    ss.SetPhiResolution(resolution)
    ss.Update()
    polydata = ss.GetOutput()
    actor = Actor(polydata, color, alpha, wire=True)

    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(position)
    return actor
