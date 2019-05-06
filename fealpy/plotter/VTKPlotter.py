import numpy as np
import vtk
import multiprocessing

from .colors import getColor, printc


__doc__ = """
This the plotter class for fealpy, which is totally based on vtk.

Here I directly use many codes from
[vtkplotter](https://github.com/marcomusy/vtkplotter). And I learn many things
from vtkplotter.
"""


class VTKPlotter(object):

    def __init__(
            self,
            shape=(1, 1),
            numrenders=None,
            position=(0, 0),
            size='auto',
            screensize='auto',
            title=None,
            bgcolor='SlateGray',
            axistype=1,
            infinity=False,
            sharecam=True,
            verbose=True,
            interactive=None,
            depthpeeling=False,
            offscreen=False,
            simulation=None  # 模拟程序，发送需要可视化的数据给 Plotter
            ):

        if interactive is None:
            if (numrenders is not None) or (shape != (1, 1)):
                interactive = False
            else:
                interactive = True

        if interactive is False:
            verbose = False

        self.colors = vtk.vtkNamedColors()
        self.verbose = verbose
        self.actors = []  # 可视对象列表
        self.renderers = []  # 渲染器列表
        self._legends = []  # 可视对象的图例列表 
        self.lengendsize = 0.15  # 图例的大小
        self.legendbc = (0.96, 0.96, 0.9)  # 图例的背景颜色
        self.legendpos = 2  # 1=topright, 2=top-right, 3=bottom-left
        self.picked3d = None  # 可视对象上的点击点的三维坐标
        self.bgcolor = bgcolor  # 背景颜色
        self.offscreen = offscreen

        self.shape = shape
        self.position = position
        self.interactive = interactive
        self.axistype = axistype  # 坐标轴的类型
        self.title = title  # 窗口的标题
        self.xtitle = 'x'
        self.ytitle = 'y'
        self.ztitle = 'z'
        self.sharecam = sharecam  # 多个渲染对象时是否共用一个照相机
        self.infinity = infinity  # 平行投影
        self.camthickness = 2000

        self.flat = True  # 'flat' 插值样式
        self.phong = False  # 'phong' 插值样式
        self.gouraud = False  # 'gouraud' 插值样式

        self.bculling = False  # back face culling
        self.fculling = False  # front face culling

        self.camera = vtk.vtkCamera()
        self.sliders = []
        self.buttons = []
        self.widgets = []
        self.cwidgets = None  # 当前的 widgets
        self.axesexist = []

        # render window
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.PointSmoothingOn()
        if screensize == 'auto':
            aus = self.renderWindow.GetScreenSize()
            if aus and len(aus) == 2 and aus[0] > 100 and aus[1] > 100:  # seems ok
                if aus[0]/aus[1] > 2:  # looks like there are 2 or more screens
                    screensize = (int(aus[0]/2), aus[1])
                else:
                    screensize = aus
            else:  # it went wrong, use a default 1.5 ratio
                screensize = (2160, 1440)

        x, y = screensize
        if numrenders is not None:
            # number of renderers. Find out the best
            # arrangement based on minimum nr. of empty renderers
            if shape != (1, 1):
                printc('Warning: having set N, shape is ignored.', c=1)
            nx = int(np.sqrt(int(numrenders*y/x)+1))
            ny = int(np.sqrt(int(numrenders*x/y)+1))
            lm = [(nx, ny), (nx, ny+1), (nx-1, ny), (nx+1, ny), (nx, ny-1),
                  (nx-1, ny+1), (nx+1, ny-1), (nx+1, ny+1), (nx-1, ny-1)]
            ind, minl = 0, 1000
            for i, m in enumerate(lm):
                l = m[0]*m[1]
                if numrenders <= l < minl:
                    ind = i
                    minl = l
            shape = lm[ind]
        if size == 'auto':  # figure out a reasonable window size
            f = 1.5
            xs = y/f*shape[1]  # because y<x
            ys = y/f*shape[0]
            if xs > x/f:  # shrink
                xs = x/f
                ys = xs/shape[1]*shape[0]
            if ys > y/f:
                ys = y/f
                xs = ys/shape[0]*shape[1]
            self.size = (int(xs), int(ys))
            if shape == (1, 1):
                self.size = (int(y/f), int(y/f))  # because y<x
            if self.verbose and shape != (1, 1):
                print('Window size =', self.size, 'shape =', shape)

        self.shape = shape
        for i in reversed(range(shape[0])):
            for j in range(shape[1]):
                renderer = vtk.vtkRenderer()
                renderer.SetUseDepthPeeling(depthpeeling)
                renderer.SetBackground(self.colors.GetColor3d(bgcolor))
                x0 = i/shape[0]
                y0 = j/shape[1]
                x1 = (i+1)/shape[0]
                y1 = (j+1)/shape[1]
                renderer.SetViewport(y0, x0, y1, x1)
                self.renderers.append(renderer)
                self.axesexist.append(None)

        if 'full' in size and not offscreen:  # full screen
            self.renderWindow.SetFullScreen(True)
            self.renderWindow.BordersOn()
        else:
            self.renderWindow.SetSize(int(self.size[0]), int(self.size[1]))

        self.renderWindow.SetPosition(position)

        if title is None:
            title = 'fealpy figure'

        self.renderWindow.SetWindowName(title)

        # rdender window interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWindow)

    def show(
            self,
            actors=None,
            at=None):

        if at is None:
            at = 0

        if at < len(self.renderers):
            self.renderer = self.renderers[at]

        self.renderer.AddActor(actors)
        print(self.renderer)

        self.camera.SetParallelProjection(self.infinity)
        self.camera.SetThickness(self.camthickness)

        if self.sharecam:
            for r in self.renderers:
                r.SetActiveCamera(self.camera)

        if len(self.renderers) == 1:
            self.renderer.SetActiveCamera(self.camera)

        self.renderer.ResetCamera()
        self.renderWindow.Render()

        # begin mouse interaction
        self.interactor.Start()
