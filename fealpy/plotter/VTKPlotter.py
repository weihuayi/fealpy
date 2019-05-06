import numpy as np
import vtk


class VTKPlotter(object):

    def __init__(
            self,
            shape=(1, 1),
            pos=(0, 0)
            ):
        self.renderWin = vtk.vtkRenderWindow()
        self.renderWin.SetPosition(pos)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWin)
        vsty = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(vsty)
 
