# 液压管件几何与网格接口设计文档 (fealpy_hydraulic_pipe_geometry_mesh_interface_design)

## 一、 概述

本文档旨在为液压管件流固耦合（FSI）优化场景中的**几何建模模块**与**网格生成模块**定义标准化的对外接口与数据流转规范。设计遵循高内聚、低耦合的原则，确保几何参数化设计与网格离散化过程物理分离，以支持后续无缝接入天工 CAX 工作流平台的自动化参数优化循环。

**涵盖场景：**
1. 90° 高压弯管（单弯头）
2. 90° 分流三通（单三通）



## 二、 系统数据流与耦合架构

在参数化优化计算中，几何与网格模块的数据流向呈现严格的单向依赖关系：

`Optimizer (Design Variables)` $\rightarrow$ `Geometry Module (B-Rep / STEP / CAD Object)` $\rightarrow$ `Mesh Module (Nodes, Elements, Markers)` $\rightarrow$ `Solver (CFD/FEA)`

- **数据解耦原则**：网格模块不直接读取优化参数，几何模块不感知网格尺寸限制。二者通过标准的**边界表示（B-Rep）**或**统一的拓扑数据结构**进行数据交换。



## 三、 几何建模模块接口设计

几何模块的职责是将输入的标量参数组合转化为具有严格拓扑关系的几何实体，并将其打包为跨模块的通用数据载体。

### 3.1 跨模块数据载体设计 (`UniversalGeometryData`)

作为几何与网格模块之间的标准“中间件”，承载几何拓扑信息与边界元数据：

```python
class UniversalGeometryData:
    def __init__(self):
        # 核心形状数据：以序列化的标准格式（STEP）字节流存储在内存中，切断底层 DLL 依赖
        self._shape_bytes: bytes = b""
        
        # 几何对象的附加元数据：如面标识(Inlet/Outlet/Wall)映射、物理组名称等
        self.metadata: dict = {}

    @classmethod
    def from_pythonocc(cls, shape: TopoDS_Shape, metadata: dict = None) -> 'UniversalGeometryData':
        """从 pythonocc 接收 TopoDS_Shape，极速序列化为内存字节流"""
        pass

    def build_in_gmsh(self) -> list:
        """供网格模块调用：在 gmsh 当前模型中从内存字节流反序列化实体，返回 (dim, tag) 列表"""
        pass
        
    def export_to_step(self, filepath: str) -> None:
        """（可选）将内存数据持久化为本地文件，用于调试或备份"""
        pass
```

### 3.2 基础几何接口 (`BasePipeGeometry`)

所有管件几何生成器需实现以下基础接口：

```python
class BasePipeGeometry:
    def __init__(self, parameters: dict):
        """初始化几何参数"""
        pass
        
    def update_parameters(self, new_parameters: dict) -> bool:
        """
        更新几何参数（用于优化迭代）。
        返回 bool 值指示参数是否合法且无拓扑冲突。
        """
        pass

    def build_geometry(self) -> None:
        """基于当前参数，调用底层 CAD 内核 (如 pythonocc) 构建几何实体，并识别边界元数据"""
        pass

    def export_geometry_data(self) -> UniversalGeometryData:
        """
        输出用于网格划分的通用几何数据载体。
        包含序列化的 B-Rep 字节流与识别好的面/体元数据(metadata)。
        """
        pass
```
### 3.2 90° 高压弯管参数定义 (ElbowPipeGeometry)
默认参数字典 (parameters) 结构：
```JSON
{
    "D": 25.0,                  // 内径 (mm)
    "bend_angle": 90.0,         // 弯角 (度)
    "R_bend_inner": 37.5,       // 内弯半径 (mm)，默认 1.5D
    "L_in_ratio": 5.0,          // 入口直管段比例 (5D)
    "L_out_ratio": 10.0,        // 出口直管段比例 (10D)
    "wall_thickness": 5.0       // 壁厚 (mm，用于生成外表面)
}
```

注：直管段绝对长度 $L_{in}$ 与 $L_{out}$ 在 build_geometry 内部由 $D \times ratio$ 动态计算，确保参数化联动。

### 3.3 90° 分流三通参数定义 (TJunctionPipeGeometry)

默认参数字典 (parameters) 结构：
```JSON
{
    "D_main": 32.0,             // 主管内径 (mm)
    "D_branch": 25.0,           // 支管内径 (mm)
    "intersect_angle": 90.0,    // 主支管夹角 (度)
    "R_fillet": 12.25,          // 支管过渡圆角半径 (mm)
    "L_in_ratio": 5.0,          // 入口直管段比例
    "L_out_ratio": 10.0,        // 出口直管段比例
    "wall_thickness": 5.0       // 壁厚 (mm)
}
```

## 四、 网格生成模块接口设计

网格模块的职责是接收来自几何模块的标准拓扑数据，并结合网格控制参数，生成适用于有限元（FEM）或有限体积（FVM）分析的离散网格。

### 4.1 基础网格接口 (BasePipeMeshGenerator)

```Python
class BasePipeMeshGenerator:
    def __init__(self):
        self.geometry_data: UniversalGeometryData = None
        self.mesh_config = {}
        self.mesh_data = None
        
    def set_geometry(self, geom_data: UniversalGeometryData) -> None:
        """
        接收几何模块输出的通用数据载体。
        将通过 geom_data.build_in_gmsh() 在 Gmsh 中重建几何，并解析 metadata。
        """
        pass

    def set_mesh_config(self, config: dict) -> None:
        """
        设置网格划分控制参数。
        """
        pass

    def generate_mesh(self) -> None:
        """执行网格离散化算法 (包含流体域和固体薄壁域的物理组分配与局部加密)"""
        pass

    def export_mesh_data(self) -> MeshData:
        """
        输出标准化网格数据结构。
        """
        pass
```

### 4.2 网格控制配置 (mesh_config)

针对液压管件 FSI 场景，需支持全局尺寸与局部加密（特别是弯道和三通圆角区域）：
```JSON
{
    "global_size": 2.0,              // 全局基础网格尺寸 (mm)
    "boundary_layer": {              // 流体边界层网格配置
        "enable": true,
        "first_layer_height": 0.1,
        "growth_rate": 1.2,
        "layers": 5
    },
    "local_refinement": [            // 局部特征加密
        {
            "type": "curvature",     // 针对高曲率区域（如 12.25mm 圆角）
            "min_size": 0.5,
            "max_angle": 15.0        // 每 15 度至少一个单元
        }
    ],
    "element_type": "tetrahedron"    // 单元类型：tetrahedron / hexahedron
}
```

### 4.3 核心输出数据结构 (MeshData)

网格模块需向求解器输出统一的数据对象，需包含严格的边界标识：
- nodes: 节点坐标数组，shape = (N_nodes, 3)
- elements: 单元节点连结性数组，shape = (N_elems, 4) 或 (N_elems, 8)
- markers:
  - Volume: 区分流体域(fluid)与固体域(solid_wall)。
  - Boundary: 区分入口(inlet)、出口(outlet)、流固耦合交界面(fsi_interface)、外壁面固定端(fixed_support)。

## 五、 模块接口协作伪代码 (数据流验证)

以下为工作流平台调度几何与网格模块的预期调用逻辑：
```Python
# 1. 优化器给出当前迭代的几何参数
current_design_vars = {"D_main": 32.0, "D_branch": 25.0, "R_fillet": 12.25, ...}

# 2. 几何模块工作流 (pythonocc 环境)
geom_module = TJunctionPipeGeometry(current_design_vars)
geom_module.build_geometry()
# 返回 UniversalGeometryData 对象，包含序列化内存字节流与面标识字典
univ_geom_data = geom_module.export_geometry_data() 

# 3. 网格模块工作流 (gmsh 环境)
mesh_config = {"global_size": 2.0, "local_refinement": [...]}
mesh_module = BasePipeMeshGenerator()

mesh_module.set_geometry(univ_geom_data)     # 读入通用数据载体并在 gmsh 中重建实体
mesh_module.set_mesh_config(mesh_config)     # 读入网格设置
mesh_module.generate_mesh()                  # 执行划分
ready_for_solver_mesh = mesh_module.export_mesh_data() # 输出计算网格

# 4. 传递给 Solver (FSI/CFD)
# solver.load_mesh(ready_for_solver_mesh)
```