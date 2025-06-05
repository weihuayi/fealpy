from typing import Any, Optional, Dict, Literal
from fealpy.geometry.geometry_kernel_adapter_base import GeometryKernelAdapterBase

# 定义支持的几何内核名称（示例）
AdapterName = Literal["occ"]

class GeometryKernelManager:
    # 属性
    current_adapter_name: AdapterName
    available_kernels: Dict[AdapterName, GeometryKernelAdapterBase]

    # 管理方法
    def __init__(self, default_adapter: Optional[str]=None) -> None: ...
    def load_adapter(self, name: str) -> None: ...
    def set_adapter(self, name: str) -> None: ...
    def get_current_adapter(self, logger_msg=None) -> GeometryKernelAdapterBase: ...

    # entity construct
    # 点、线、曲线
    def add_point(self, x, y, z) -> Any: ...  # 创建点
    def add_line(self, start_point, end_point) -> Any: ...  # 两点连线
    def add_arc(self, start, middle, end) -> Any: ...  # 三点圆弧
    def add_arc_center(self, center, start, end) -> Any: ...  # 圆心+起点+终点 圆弧
    def add_spline(self, points: list) -> Any: ...  # 样条曲线
    def add_curve_loop(self, edges: list) -> Any: ...  # 闭合线框
    def add_face_loop(self, faces: list) -> Any: ...  # 闭合面框
    # 面、体
    def add_surface(self, wire) -> Any: ...  # 平面面（基于闭合线框）
    def add_volume(self, shell) -> Any: ...  # 边界面包成的体
    # def add_helix(self, radius: float, pitch: float,height: float,
    #               start_point: Tuple[float, float, float] = (0, 0, 0),
    #               axis_direction: Tuple[float, float, float] = (0, 0, 1)) -> Any: ...  # 螺旋线
    # def add_extrude(self, profile, vec) -> Any: ...  # 拉伸成体/面
    # def add_revolve(self, profile, axis, angle) -> Any: ...  # 旋转成体/面
    # def add_sweep(self, profile: Any, path: Any) -> Any: ...  # 扫掠体/面



    # bool operate
    def boolean_union(self, *shapes) -> Any: ...  # 并集（已部分实现）
    def boolean_cut(self, shape, tool) -> Any: ...  # 差集（已实现为cut）
    def boolean_intersect(self, shape1, shape2) -> Any: ...  # 交集
    def boolean_fragment(self, *shapes) -> Any: ...  # 非融合切割（保留碎片）
    # def boolean_slice(self, shape, plane) -> Any: ...  # 平面切割

    # geometry operate
    def translate(self, shape, vec) -> Any: ...  # 平移（需扩展参数）
    def rotate(self, shape, rotation_point, rotation_axis, angle) -> Any: ...  # 旋转（需支持任意轴）
    # def mirror(self, shape, plane) -> Any: ...  # 镜像
    # def scale(self, shape, factor) -> Any: ...  # 均匀缩放

    # entity sample
    # 2d entity 
    # 创建矩形面（左下角坐标 + 宽高）
    def add_rectangle(self, x_min, y_min, z_min, dx, dy) -> Any: ...
    # 创建椭圆面（圆心坐标 + 半径）
    def add_disk(self, xc, yc, zc, rx, ry) -> Any: ...
    # 创建圆面（圆心坐标 + 半径）
    def add_circle(self, xc, yc, zc, r) -> Any: ...
    # 创建多边形面（顶点列表）
    def add_polygon(self, points) -> Any: ...
    # 创建圆环面（圆心坐标 + 内外半径）
    def add_ring(self, xc, yc, zc, r_inner, r_outer) -> Any: ...

    # 3d entity
    # 创建立方体（左下角坐标 + 三轴长度）
    def add_box(self, x_min, y_min, z_min, dx, dy, dz) -> Any: ...
    # 创建椭球面（圆心坐标 + 半径）
    def add_ellipsoid(self, xc, yc, zc, rx, ry, rz) -> Any: ...
    # 创建球体（中心坐标 + 半径）
    def add_sphere(self, xc, yc, zc, radius) -> Any: ...
    # 创建圆柱体（底面中心坐标 + 半径 + 高度 + 轴向）
    def add_cylinder(self, xc, yc, zc, r, height, *, axis=(0, 0, 1)) -> Any: ...
    # 创建圆锥/圆台体（底面中心坐标 + 底/顶半径 + 高度）
    # def add_cone(self, xc, yc, zc, r_bottom, r_top, height) -> Any: ...
    # 创建棱锥体（底面形状 + 顶点坐标）
    # def add_pyramid(self, base_shape, apex_point) -> Any: ...
    # 创建圆环体（中心坐标 + 主半径 + 截面半径）
    def add_torus(self, xc, yc, zc, major_r, minor_r) -> Any: ...
    # 创建空心圆柱体（底面中心坐标 + 外/内半径 + 高度 + 轴向）
    def add_hollow_cylinder(self, xc, yc, zc,outer_radius, inner_radius, height, *, axis=(0, 0, 1)) -> Any: ...

    # shape discrete
    def shape_discrete(self, shape, deflection: float = 0.1) -> Any: ...  # 离散化（网格化）

    # file io
    def import_step(self, filename) -> Any: ...  # 导入STEP
    def import_stl(self, filename) -> Any: ...  # 导入STL
    def import_brep(self, filename) -> Any: ...  # 导入BREP
    def export_step(self, *shape, filename) -> None: ...  # 导出STEP
    def export_stl(self, *shape, filename, resolution=0.1) -> None: ...  # 导出STL（网格化）
    def export_brep(self, *shape, filename) -> None: ...  # 原生BREP格式

    # display
    def display(self,
                *shapes: Any,
                background: str = "#FFFFFF",  # 背景颜色（十六进制）
                window_size: tuple = (1024, 768),  # 窗口大小 (width, height)
                show_axes: bool = True,  # 是否显示坐标系
                zoom_all: bool = True,  # 自动缩放适应所有模型
                **kwargs  # 其他显示参数（如透明度、颜色）
                ) -> None: ...