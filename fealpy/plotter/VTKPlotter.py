import numpy as np
import vtk
import multiprocessing
from .actors import *



__doc__ = """
This the plotter class for fealpy, which is totally based on vtk.

"""


class VTKPlotter(object):

    def __init__(
            self,
            shape=(1, 1),
            bgcolor='White',
            position=None,
            title=None,
            sharecam=True,
            interactive=True,
            simulation=None,  # 模拟程序，发送需要可视化的数据给 Plotter
            ):

        self.colors = vtk.vtkNamedColors()
        self.camera = vtk.vtkCamera()
        self.sharecam = sharecam

        self.actors = []  # 可视对象列表
        self.renderers = []  # 渲染器列表
        self.legends = []  # 可视对象的图例列表 

        self.interactive = interactive

        self.sliders = []
        self.buttons = []
        self.widgets = []

        self.crenderer = None  # 当前的渲染器
        self.cwidgets = None  # 当前的 widgets

        # render window
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.PointSmoothingOn()
        self.shape = shape
        for i in reversed(range(shape[0])):
            for j in range(shape[1]):
                renderer = vtk.vtkRenderer()
                renderer.SetBackground(self.colors.GetColor3d(bgcolor))
                x0 = i/shape[0]
                y0 = j/shape[1]
                x1 = (i+1)/shape[0]
                y1 = (j+1)/shape[1]
                renderer.SetViewport(y0, x0, y1, x1)
                self.renderers.append(renderer)

        if title is None:
            title = 'FEALPy Mesh Figure'

        self.renderWindow.SetWindowName(title)

        if interactive is True:  # rdender window interactor
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.renderWindow)
        else:
            self.interactor = None

        if simulation is not None:
            self.queue = multiprocessing.Queue()
            self.process = multiprocessing.Process(None, simulation,
                    args=(self.queue,))
        else:
            self.process = None

    def show(self, actors=None, at=None):
        """
        """
        if at is None:
            at = 0

        if at < len(self.renderers):
            self.crenderer = self.renderers[at]

        self.crenderer.AddActor(actors)

        if len(self.renderers) == 1:
            self.crenderer.SetActiveCamera(self.camera)

        self.crenderer.ResetCamera()
        self.renderWindow.AddRenderer(self.crenderer)
        self.renderWindow.Render()

        if self.interactor is not None:  # begin mouse interaction
            self.interactor.Start()

    def run(self):
        self.process.start()
        while True:
            if not self.queue.empty():
                data = self.queue.get()
                if data != -1:
                    if 'mesh' in data:
                        actor = meshactor(data['mesh'])
                        self.show(actors=actor)
                else:
                    print('exit program!')
                    self.process.join()
                    break
            else:
                self.renderWindow.Render()
