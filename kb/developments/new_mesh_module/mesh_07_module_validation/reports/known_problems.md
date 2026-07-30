
## 已知问题

1. HalfEdgeMesh、DartMesh、UniformMesh 系列网格还未迁移；

> [!Note]
> 决策：不进入新网格体系，仅把原文件放入对应文件夹内。
> 例如各种 uniform mesh 放入 `mesh/uniform_mesh` 中。

2. 没有 BDF、INP 文件存取功能、VTK 存取功能的调用方式与之前的不兼容（部分决策，具体方案待讨论）；

> [!NOTE]
> 部分决策：文件格式读写优先考虑基于 `meshio` 统一实现，以复用其对多种网格文件格式的支持。
> 对外提供统一的 `read_mesh` 和 `write_mesh` 函数，作为新网格体系的通用文件读写入口。
> 同时在 `TriangleMesh` 等工厂类中保留或补充 `from_vtu` 等老式构造接口，兼容已有调用方式；这类接口内部复用统一读写层，而不重复实现格式解析。
> 当前尚未决定具体模块位置、格式到 schema 的映射、属性 / cell data 的转换规则、混合单元处理、错误检查和依赖配置，待后续讨论具体做法设计。


3. 没有高阶形状格式（对应 Lagrange<...>Mesh）、没有多边形和多面体形状格式；

> [!Note]
> 决策：进一步实现即可。

4. 原先网格类中可能存在特殊的算法没有迁移（特别是 NodeMesh）；


5. 缺少各种 from_<...> 的兼容性构造方法，除了 from_box。

6. Schema 测试文件较为混乱，每一种形状的测试不统一；

> [!NOTE]
> 决策：可以基于算法来组织测试，而不是形状。
> 例如，改为按 test_normal.py、test_shape_function.py 这样写测试文件。
