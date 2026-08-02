# FEALPy | AGENTS | 仓库级 AI 协作入口

本文件是 FEALPy 仓库的根级 AI 协作入口，供进入本仓库的各类 AI 协作工具加载。

各类 AI 协作工具在本仓库中应以本文件为统一协作入口；特定工具如需额外的入口文件或配置，应引用本文件，不得维护内容相同但独立演化的并列主定义。

本文件负责：

* 明确 FEALPy 的仓库定位及其在算海体系中的角色；
* 将任务按算法域路由到正确的模块和专项资产；
* 保护 FEALPy 作为智能 CAX 共性基础算法库的架构边界；
* 约束 AI 在本仓库中的修改行为、测试策略和依赖管理；
* 规定 FEALPy 仓库特有的最小核验和收口要求。

本文件不替代：

* SuanhaiOS 中适用于算海团队的治理资产；
* 个人全局默认（`~/.codex/AGENTS.md`，自动继承）；
* README、Develop_note.md、代码 docstring、测试和 Demo；
* 启元仓库中的 `qiyuan/kb/fealpy/` 中维护的稳定知识与设计原理；
* 人类承担专业判断、算法正确性确认、API 设计和发布责任。

目标目录存在更近的 `AGENTS.md` 时，应同时遵循更具体的规则。

## 一、仓库定位

FEALPy 是面向高校、科研院所、工程仿真开发者、人工智能研究者和企业研发团队的开源智能 CAX 共性基础算法库。

核心算法域：

* 网格生成与优化
* 数值离散方法（有限元、有限差分、有限体积、虚拟元、间断伽辽金、相容离散算子等）
* 仿真求解器
* 优化算法
* AI for Science

架构特征：

* 多后端张量抽象层，将上层算法与具体计算后端解耦，可连接 NumPy、PyTorch、JAX、MindSpore、PaddlePaddle、MPI 和 Numba 等计算生态
* 模块化与接口标准化：开发者可基于统一接口实现算法，在不同后端和异构硬件环境中迁移运行
* 张量化架构：以张量为基本数据结构，降低算法重复实现、AI 方法融合和硬件适配的成本

定位边界：

* 服务于新算法研究、候选方法比较、数值实验、设计验证和应用原型开发
* 不替代 ANSYS、Abaqus、COMSOL 等面向工程应用的成熟商业软件
* 软件仓库：代码、测试和 Demo 是第一事实源；文档和知识库是对代码的解释与组织，不自动构成当前实现的权威描述
* 当前分支中的代码是事实，README 和 Develop_note.md 是约定的表达，两者出现矛盾时以代码实际行为为准

## 二、任务启动前的最小检查

开始实质性工作前，应先识别：

* 当前工作目录、Git 分支及已有未提交修改；
* 当前任务属于哪个算法域（网格 / 离散 / 求解 / 优化 / AI for Science / 基础设施）；
* 当前任务是新功能、Bug 修复、重构、测试补充还是文档；
* 当前任务涉及的核心模块及其依赖关系；
* 是否需要读取启元仓库 `qiyuan/kb/fealpy/` 中的稳定知识（设计原理、接口规范、已知限制）；
* 是否涉及 `src/` 或 `exten/` 中的 C/C++ 扩展（需要编译工具链和 pybind11）。

必须遵循：

* AI 是算法研发协作者，不是算法正确性、数学严格性或发布就绪的最终责任主体；
* 确定性表达不得超过可确认的代码行为、测试结果和明确文档；
* 不得把代码存在写成能力已经验证，不得把单个示例写成稳定接口，不得把历史设计写成当前实现；
* 优先读取代码、测试和 Demo 作为事实来源，其次才是文档和注释；
* 信息不足、模块边界模糊，或涉及多模块接口变更、公开 API 破坏或后端抽象层修改时，应采用条件化结论并显式标注未决问题。

## 三、按算法域的路由

以下按算法域组织核心模块及其职责。当前目录位置为参考，目录结构可能随重构调整，以算法域描述为路由依据。

### 3.1 网格生成与优化

涉及网格数据结构、网格生成、网格优化和计算几何基础。

* 新网格系统（推荐）：`fealpy/mesh/` — Schema → Storage → Topology → View 四层架构
* 网格生成器：`fealpy/mesher/`
* 网格优化：`fealpy/meshopt/`
* 计算几何：`fealpy/geometry/`

注意事项：

* 新的网格相关代码必须基于 `fealpy/mesh/`（新网格系统），不得依赖 `fealpy/mesh_old/`（遗留代码，仅做兼容维护）
* 新网格类型按 Schema → Storage → Topology → View 四层接入，通过 Factory 元类注册

### 3.2 数值离散方法

涉及函数空间、积分公式以及各种数值离散格式。

* 函数空间：`fealpy/functionspace/` — 涵盖标量和矢量有限元空间、虚拟元空间、多项式空间等
* 积分公式：`fealpy/quadrature/` — 涵盖常见几何单形上的数值积分格式
* 有限元法：`fealpy/fem/` — Integrator → Form → Model 三层架构
* 有限差分法：`fealpy/fdm/`
* 有限体积法：`fealpy/fvm/`
* 虚拟元法：`fealpy/vem/`
* 间断伽辽金法：`fealpy/cdg/`
* 相容离散算子：`fealpy/cdo/`
* 计算电磁学：`fealpy/cem/`
* 计算流体力学：`fealpy/cfd/`
* 计算固体力学：`fealpy/csm/`
* 流固耦合：`fealpy/fsi/`
* 有限点方法：`fealpy/fpm/`

注意事项：

* 新数值方法按 Integrator → Form → Model 三层模式接入，遵循 `fealpy/typing.py` 中的类型别名体系
* 跨物理场的通用积分器和形式放在 `fealpy/fem/`，领域专有扩展放在对应子包

### 3.3 仿真求解

涉及线性/非线性求解器、稀疏线性系统、物理模型和瞬态问题。

* 求解器：`fealpy/solver/` — 涵盖 Krylov 子空间方法、预处理迭代法和直接法
* 稀疏张量系统：`fealpy/sparse/` — COO/CSR 格式，支持 batch 维度和多后端
* PDE 定义：`fealpy/pde/`
* 物理模型：`fealpy/model/` — 完整物理问题建模
* 物理场定义：`fealpy/physics/` — 场量、本构关系等
* 材料本构：`fealpy/material/` — 材料模型与参数
* 瞬态问题：`fealpy/time/` — 时间积分格式与时间步进

注意事项：

* 稀疏运算通过 `fealpy/sparse/` 进行，它提供与 scipy 兼容的 API 并支持后端切换
* 矩阵稀疏模式可能受舍入误差影响——文档中已有关于精度消除的说明
* `model/`、`physics/`、`material/` 的职责边界尚在演化中，不确定时读取各模块的实际代码和 docstring 确认

### 3.4 优化算法

* `fealpy/opt/`

### 3.5 AI for Science

* 机器学习：`fealpy/ml/`
* 统一神经网络多物理：`fealpy/unml/`

### 3.6 通用基础设施（跨域共享）

* 多后端张量抽象（⭐ 最核心的基础设施）：`fealpy/backend/`
* 分布式/并行计算：`fealpy/distributed/` — MPI 并行支持
* 公共工具：`fealpy/common/`
* 类型别名（TensorLike, Index, CoefLike, SourceLike 等）：`fealpy/typing.py`
* 装饰器：`fealpy/decorator/`
* 日志：`fealpy/logs.py`

### 3.7 其他专项模块

以下模块按需定位，不重复展开：

* `fealpy/cgp/` — 计算地球物理
* `fealpy/graph/` — 图算法
* `fealpy/mmesh/` — 移动网格方法
* `fealpy/operator/` — 算子工具
* `fealpy/pathplanning/` — 路径规划
* `fealpy/tools/`、`fealpy/utils/` — 辅助工具

### 3.8 可视化与 IO

* VTK 可视化：`fealpy/plotter/`
* matplotlib 绘图：`fealpy/plotting/`
* 文件写出：`fealpy/writer/`

### 3.9 应用项目

* `app/` — 各独立应用项目，与核心库分离

### 3.10 C/C++ 扩展

* `src/` — FEALPy 自有原生扩展，通过 CMake + pybind11 构建
* `exten/` — 第三方扩展封装
* 涉及修改时需确认 cmake + pybind11 + 编译器可用
* 新 cgal 相关代码放入 `src/cgal/`，`exten/cgal/` 为历史遗留

## 四、核心架构约束

### 4.1 多后端张量抽象

FEALPy 的核心竞争力在于将上层算法与计算后端解耦。必须遵守：

* 所有张量操作必须通过 `fealpy.backend` 进行，不得在算法代码中直接 `import numpy`、`import torch`、`import jax` 等
* `fealpy/backend/` 遵循 Python Array API Standard，`FUNCTION_MAPPING` 覆盖常见张量操作
* `TensorLike` 协议（`backend/base.py`）定义了完整的张量接口契约，张量参数和返回值应使用此类型
* `BackendManager` 使用线程局部存储管理当前后端，支持不同线程使用不同后端
* 新后端需实现 `BackendProxy` 子类并在 `ATTRIBUTE_MAPPING` 和 `FUNCTION_MAPPING` 中注册；新增后端涉及 API 设计决策和长期维护承诺，应在任务中显式说明动机和范围
* `TRANSFORMS_MAPPING` 覆盖 grad, hessian, jvp, vjp 等自动微分变换——涉及 AD 的算法应通过此映射而非直接调用特定框架的 AD API
* 当前支持的后端列表见 `fealpy/backend/` 目录，默认后端为 NumPy

### 4.2 模块化与接口标准化

* 新数值方法按 Integrator → Form → Model 三层模式接入
* 新网格类型按 Schema → Storage → Topology → View 四层接入
* 遵循 `fealpy/typing.py` 中的集中类型别名体系，不在各模块中重复定义
* `fealpy/__init__.py` 不维护扁平命名空间，所有符号按子包导入
* `__init__.py` 中通过 `*` 或显式导出的符号视为公开 API；仅在模块内部使用的符号以下划线 `_` 前缀标识

## 五、关键开发约定

以下从 Develop_note.md 中提取跨模块稳定的约定。详细的 API 命名和旧系统说明见 Develop_note.md。

* Python 版本要求见 `setup.py` 中的 `python_requires`，类型注解使用 `|` 联合语法
* 命名：`ClassName`、`function_name`、`memberdata`、`module_name_file.py`
* 网格实体：`node`（节点）、`cell`（单元）、`edge`（边）、`face`（面）
* 计数变量：`NN`=节点数、`NC`=单元数、`NE`=边数、`NF`=面数、`NQ`=积分点数、`GD`=几何维度、`TD`=拓扑维度
* 循环指标约定：`c`=单元、`f`=面、`e`=边、`v`=顶点个数、`i,j,k,d`=自由度/基函数、`q`=积分点/重心坐标、`m,n`=空间/拓扑维数
* 函数空间核心接口：`basis(bc)`、`grad_basis(bc)`、`value(u, bc)`、`grad_value(u, bc)`、`interpolation(u)`、`cell_to_dof()`

## 六、测试与修改约束

### 6.1 测试

* 新测试必须写入 `tests/`，使用 pytest 风格（fixture、parametrize、tmp_path）
* `test/` 为旧测试目录，CI 不运行，仅做参考——不得往其中添加新测试
* pytest 配置在 `pytest.ini` 或 `pyproject.toml` 中，当前排除 `pinn` 和 `efficiency` 目录
* 数值测试需显式管理 tolerance，建议使用 `pytest.approx()` 或统一的 `assert_allclose` fixture
* 运行新测试：`pytest tests/`；运行旧测试（可能失败）：`pytest test/`

### 6.2 修改

实施修改时：

* 聚焦当前任务涉及的模块、类和函数，采用满足任务要求的最小完整变更
* 修改前检查 `git status`，不覆盖已有未提交变更
* 不进行无关批量改写、目录重组或术语替换
* 不动 `fealpy/mesh_old/` 和 `fealpy/old/` 中的遗留代码（除非明确任务是清理遗留）；遗留代码中的缺陷若影响新功能，应在对应的新模块中修复，或在任务中显式说明兼容策略
* 不修改 `fealpy/fealpy.egg-info/`（构建产物）
* 不改变公开 API 签名而不检查所有调用方
* 修改核心 API 时需同步更新 `example/` 中受影响的示例，确保示例可独立运行
* 添加新依赖时必须同时约束版本号
* 涉及 `src/` 或 `exten/` 的修改需确认编译环境可用

完成前按任务范围检查：

1. 算法域归属和模块位置是否正确
2. 是否遵循后端抽象层规范
3. 是否引入直接后端依赖（裸 `import numpy/torch/jax` 等）
4. 关键数值行为是否有测试覆盖
5. 最终 diff 是否只包含任务相关修改
6. 是否意外修改了遗留代码、构建产物或第三方文件

### 6.3 提交

遵循 `~/.codex/AGENTS.md` 第五节中的 Git 与变更安全约束。此外在本仓库中：

* 不自行执行 `pip install`、`twine upload` 或 Docker 发布操作
* 不自行改变版本号、发布状态或 API 生命周期

## 七、规则分层与外部资产

本文件只维护 FEALPy 仓库级、长期稳定且高频适用的入口规则。

更具体的规则应由以下对象维护：

* 个人偏好与全局角色 → `~/.codex/AGENTS.md`
* 详细的 API 命名约定与发布步骤 → `Develop_note.md`
* 稳定知识（设计原理、接口规范、已知边界） → qiyuan `kb/fealpy/`
* 构建与 CI 细节 → `.github/workflows/`、`pyproject.toml`、`setup.py`
* 测试策略与配置 → `pytest.ini` 和 `tests/conftest.py`
* 子模块专项规则 → 未来按需建立 `fealpy/<submodule>/AGENTS.md`

模块级 `AGENTS.md` 可定义的典型内容：该模块的内部架构、不变量、测试策略、已知坑点、以及与兄弟模块的依赖契约。空白模块不需要模块级 AGENTS.md。
