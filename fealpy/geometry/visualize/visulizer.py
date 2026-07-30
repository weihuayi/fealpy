"""
可视化渲染引擎 — VTK 窗口管理与交互.

仅负责: 渲染窗口创建、交互器风格、show() / show_file() 便捷函数.
不包含任何数据读取或格式转换逻辑 (这些属于 io 模块).
"""

from __future__ import annotations

import os
import re
import ctypes
import ctypes.util
from pathlib import Path
from typing import List, Tuple, Optional

import vtk


# ============================================================
# X11 UTF-8 窗口标题
# ============================================================

def _set_x11_utf8_title(render_window: "vtk.vtkRenderWindow", title: str) -> None:
    """
    通过 X11 _NET_WM_NAME 属性设置 UTF-8 窗口标题.

    VTK 默认的 XStoreName 仅支持 Latin-1, 此处额外设置 _NET_WM_NAME
    (UTF8_STRING) 使现代窗口管理器能正确显示中文等 Unicode 字符.
    同时仍用 VTK SetWindowName 写入 WM_NAME 作为 ASCII 回退.
    """
    if title.isascii():
        render_window.SetWindowName(title)
    else:
        safe = ''.join(c for c in title if ord(c) < 128)
        if not safe.strip():
            safe = "CAD Viewer"
        render_window.SetWindowName(safe)

    if title.isascii():
        return

    try:
        libx11 = ctypes.util.find_library("X11")
        if libx11 is None:
            return

        xlib = ctypes.cdll.LoadLibrary(libx11)

        xlib.XOpenDisplay.restype = ctypes.c_void_p
        display = xlib.XOpenDisplay(None)
        if not display:
            return

        win_id_raw = render_window.GetGenericWindowId()
        if win_id_raw is None:
            xlib.XCloseDisplay(display)
            return

        if isinstance(win_id_raw, str):
            m = re.search(r"0*([0-9a-fA-F]+)", win_id_raw)
            if not m:
                xlib.XCloseDisplay(display)
                return
            win_id = int(m.group(1), 16)
        elif isinstance(win_id_raw, int):
            win_id = win_id_raw
        else:
            try:
                win_id = int(win_id_raw) if win_id_raw else 0
            except (ValueError, TypeError):
                win_id = 0

        if not win_id:
            xlib.XCloseDisplay(display)
            return

        xlib.XInternAtom.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ]
        xlib.XInternAtom.restype = ctypes.c_ulong

        xlib.XChangeProperty.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong,
            ctypes.c_ulong, ctypes.c_ulong,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
        ]

        net_wm_name = xlib.XInternAtom(display, b"_NET_WM_NAME", 0)
        utf8_string = xlib.XInternAtom(display, b"UTF8_STRING", 0)

        title_bytes = title.encode("utf-8")
        xlib.XChangeProperty(
            display, ctypes.c_ulong(win_id),
            net_wm_name, utf8_string,
            8,  # format=8 for char data
            0,  # PropModeReplace
            title_bytes, len(title_bytes),
        )

        xlib.XFlush(display)
        xlib.XCloseDisplay(display)

    except Exception:
        pass  # X11 操作失败时保持 ASCII 后备标题


# ============================================================
# DXF 2D 交互器风格
# ============================================================

class _Dxf2DInteractorStyle(vtk.vtkInteractorStyleUser):
    """DXF 专用 2D 交互：左键平移、滚轮缩放、右键重置."""

    def __init__(self, renderer: vtk.vtkRenderer, zoom_factor: float = 1.2):
        super().__init__()
        self._renderer = renderer
        self._zoom_factor = max(float(zoom_factor), 1.01)
        self._panning = False
        self._last_position: Optional[Tuple[int, int]] = None

        self.SetDefaultRenderer(renderer)
        self.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        self.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self.AddObserver("MouseWheelForwardEvent", self._on_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self._on_wheel_backward)
        self.AddObserver("RightButtonPressEvent", self._on_right_press)

    def _render(self) -> None:
        interactor = self.GetInteractor()
        if interactor is not None and interactor.GetRenderWindow() is not None:
            interactor.GetRenderWindow().Render()

    def _on_left_press(self, _obj, _event) -> None:
        interactor = self.GetInteractor()
        if interactor is None:
            return
        self._panning = True
        self._last_position = interactor.GetEventPosition()

    def _on_left_release(self, _obj, _event) -> None:
        self._panning = False
        self._last_position = None

    def _display_to_world(
        self, x: int, y: int, display_depth: float,
    ) -> Tuple[float, float, float]:
        self._renderer.SetDisplayPoint(float(x), float(y), display_depth)
        self._renderer.DisplayToWorld()
        world = self._renderer.GetWorldPoint()
        if world[3] == 0.0:
            return (world[0], world[1], world[2])
        return (
            world[0] / world[3],
            world[1] / world[3],
            world[2] / world[3],
        )

    def _on_mouse_move(self, _obj, _event) -> None:
        if not self._panning or self._last_position is None:
            return
        interactor = self.GetInteractor()
        if interactor is None:
            return

        current_position = interactor.GetEventPosition()
        camera = self._renderer.GetActiveCamera()
        focal = camera.GetFocalPoint()
        position = camera.GetPosition()

        self._renderer.SetWorldPoint(focal[0], focal[1], focal[2], 1.0)
        self._renderer.WorldToDisplay()
        display_depth = self._renderer.GetDisplayPoint()[2]

        old_world = self._display_to_world(
            self._last_position[0], self._last_position[1], display_depth,
        )
        new_world = self._display_to_world(
            current_position[0], current_position[1], display_depth,
        )
        motion = tuple(old_world[i] - new_world[i] for i in range(3))

        camera.SetFocalPoint(*(focal[i] + motion[i] for i in range(3)))
        camera.SetPosition(*(position[i] + motion[i] for i in range(3)))
        self._last_position = current_position
        self._renderer.ResetCameraClippingRange()
        self._render()

    def _zoom(self, factor: float) -> None:
        camera = self._renderer.GetActiveCamera()
        camera.Zoom(factor)
        self._renderer.ResetCameraClippingRange()
        self._render()

    def _on_wheel_forward(self, _obj, _event) -> None:
        self._zoom(self._zoom_factor)

    def _on_wheel_backward(self, _obj, _event) -> None:
        self._zoom(1.0 / self._zoom_factor)

    def _on_right_press(self, _obj, _event) -> None:
        self._panning = False
        self._last_position = None
        self._renderer.ResetCamera()
        self._render()


# ============================================================
# 渲染入口
# ============================================================

def show(
    *actors: vtk.vtkActor,
    title: str = "CAD Viewer",
    background: Tuple[float, float, float] = (0.95, 0.95, 0.98),
    window_size: Tuple[int, int] = (1200, 800),
    show_axes: bool = True,
    block: bool = True,
    view_mode: str = "auto",
) -> vtk.vtkRenderWindowInteractor:
    """
    打开交互式 VTK 窗口显示 Actor 列表.

    Parameters
    ----------
    *actors : vtk.vtkActor
        要显示的 Actor 对象 (vtkActor 或 vtkTextActor3D 混传).
    title : str
        窗口标题.
    background : tuple
        背景色 RGB (0-1).
    window_size : tuple
        窗口尺寸 (宽, 高).
    show_axes : bool
        是否显示坐标轴.
    block : bool
        True = 阻塞直到窗口关闭; False = 非阻塞.
    view_mode : str
        "2d": 平面模式 — 左键拖拽平移, 滚轮缩放 (适合 DXF).
        "3d": 立体模式 — 左键旋转, 中键平移, 滚轮缩放 (适合 STEP).
        "auto": 根据是否有 Z 轴数据自动选择 (默认).

    Returns
    -------
    vtk.vtkRenderWindowInteractor
        渲染窗口交互器 (用于后续控制).
    """
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(background)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(window_size)
    _set_x11_utf8_title(render_window, title)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    for actor in actors:
        renderer.AddViewProp(actor)

    if show_axes:
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(10.0, 10.0, 10.0)
        axes.SetShaftTypeToCylinder()
        renderer.AddActor(axes)

    if view_mode == "2d":
        camera = renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        style = _Dxf2DInteractorStyle(renderer)
        interactor.SetInteractorStyle(style)
        print("   🖱️ 左键拖拽平移 | 滚轮缩放 | 右键重置")
    else:
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        print("   🖱️ 左键旋转 | 中键平移 | 右键缩放 | 滚轮缩放")

    renderer.ResetCamera()
    render_window.Render()

    _set_x11_utf8_title(render_window, title)

    if block:
        interactor.Initialize()
        interactor.Start()

    return interactor


def show_file(
    filepath: str,
    **kwargs,
) -> vtk.vtkRenderWindowInteractor:
    """
    自动检测文件类型并可视化.

    支持: .step / .stp (STEP) 和 .dxf (DXF).

    Parameters
    ----------
    filepath : str
        文件路径.
    **kwargs
        传递给 show() 的额外参数.

    Returns
    -------
    vtk.vtkRenderWindowInteractor
    """
    from ..io import (
        read_step, read_dxf, get_shape_info,
        step_to_actor, step_to_edges_actor,
        dxf_to_actors,
    )

    ext = Path(filepath).suffix.lower()

    if ext in (".step", ".stp"):
        print(f"📂 读取 STEP 文件: {filepath}")
        shape = read_step(filepath)
        info = get_shape_info(shape)
        print(f"   {info}")

        actors = [
            step_to_actor(shape, color=(0.65, 0.75, 0.85)),
            step_to_edges_actor(shape, color=(0.0, 0.0, 0.0)),
        ]
        title = f"STEP: {Path(filepath).name}"
        kwargs.setdefault("view_mode", "3d")
        kwargs.setdefault("show_axes", False)

    elif ext == ".dxf":
        print(f"📂 读取 DXF 文件: {filepath}")
        doc = read_dxf(filepath)
        print(f"   实体: {doc.total_entities}, 图层: {doc.layers}")
        if doc.extents:
            print(f"   包围盒: {doc.extents}")

        actors = dxf_to_actors(doc, text_visible=True)
        title = f"DXF: {Path(filepath).name}"
        kwargs.setdefault("view_mode", "2d")
        kwargs.setdefault("show_axes", False)

    else:
        raise ValueError(f"不支持的文件格式: {ext} (支持 .step/.stp/.dxf)")

    return show(*actors, title=title, **kwargs)
