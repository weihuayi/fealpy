"""
visualize — VTK 可视化渲染.

只负责渲染窗口和交互，所有数据读取和转换由 io 模块完成.

Public API:
  - show(*actors, title, ...)  → vtkRenderWindowInteractor
  - show_file(filepath, ...)   → vtkRenderWindowInteractor
"""

from .visulizer import show, show_file

__all__ = [
    "show",
    "show_file",
]
