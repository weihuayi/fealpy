# FEALPy 网格模块自动化测试规范指南

欢迎参与 FEALPy 网格模块的重构工作！为了保证代码的高质量、防范“改崩其他模块”的风险，我们引入了基于 `pytest` 和 GitHub Actions 的自动化测试系统。

这套系统的核心原则是：**绝对可重复、失败精确定位、拒绝“在我电脑上没问题”的玄学**。请在编写测试和提交 PR 前，仔细阅读以下规范。

---

## 1. 我的测试该放哪？ (轻量级物理目录)

请根据你测试的性质，将代码放入对应的目录，绝不允许在根目录随意堆砌：

* **`tests/mesh/unit/` (核心单元测试)**：只存放测试基础拓扑逻辑、局部几何量（如面积/体积计算）的代码。**标准：必须在秒级跑完，不依赖任何大文件**。
* **`tests/mesh/integration/` (耗时集成测试)**：专门测试文件读写、全量自适应网格加密（AMR）等。**标准：必须在测试函数上打上 `@pytest.mark.slow` 标签**，防止卡死日常提交。
* **`tests/mesh/regression/` (回归测试)**：用于存放历史 Bug 的最小复现代码。每当修复一个 Issue，请将复现代码放入此处，防止再次踩坑。
* **`tests/mesh/data/` (测试数据)**：如果必须读取基准文件，存放于此。**标准：文件节点数不得超过 50 个，严禁上传大网格文件**。

---

## 2. 测试夹具 (Fixtures) 与状态隔离

有限元计算极易受状态污染，必须严格遵守隔离法则。针对夹具的使用，我们实行**“按需作用域”**管理机制：

### 规则 A：夹具的两级管理
* **全局共享夹具**：如果是多个测试文件都要用到的标准网格（例如基础的 2D 单位正方形三角形网格），必须写入 `tests/mesh/conftest.py` 中。
* **局部专用夹具**：如果某种奇异网格（如带有特定退化节点的退化网格）仅在某个特定的测试文件（如 `test_degenerate.py`）中使用，**请直接写在该测试文件的最上方**，不要去污染全局的 `conftest.py`。

### 规则 B：绝不在工作区留下垃圾文件（强制使用 `tmp_path`）
涉及文件输出（如导出 `.vtk`）时，**绝对禁止**写到当前工程目录。必须在参数中引入 pytest 内置的 `tmp_path`，系统会为你生成一个用完即焚的沙箱。

```python
def test_export_vtk(standard_2d_triangle_mesh, tmp_path):
    output_file = tmp_path / "test_output.vtk"
    standard_2d_triangle_mesh.to_vtk(output_file) 
    assert output_file.exists()
```

### 规则 C：彻底绞杀全局随机种子
**严禁**在代码中使用 `np.random.seed()`。如果测试需要随机扰动，必须显式传递局部的生成器：`rng = np.random.default_rng(seed=42)`。

```python
def test_noisy_mesh():
    # 推荐写法：创建局部伪随机生成器
    rng = np.random.default_rng(seed=42)
    noise = rng.random(10)
    # ...
```

---

## 3. 参数化测试规范 (强制指定 `ids`)

当测试逻辑在不同维度或网格类型下通用时，强烈推荐使用 `@pytest.mark.parametrize` 压缩代码。

**强制规范**：必须显式声明 `ids` 参数。如果不写，CI 报错只会显示乱码；写了 `ids`，报错才会精准显示为 `[2D-Triangle-float64]`，一眼定位问题。

```python
import pytest

@pytest.mark.parametrize(
    "mesh_type, precision", 
    [("Triangle", "float64"), ("Tetrahedron", "float32")],
    ids=["2D-Triangle-float64", "3D-Tetra-float32"]  # <-- 必须包含此行！
)
def test_mesh_dimension(mesh_type, precision):
    # 测试逻辑...
    pass
```

---

## 4. 断言圣经：拒绝魔法数字，坚守数学不变量

鼓励使用 AI 辅助编写边界条件，但**绝不允许将断言完全交给 AI 生成的硬编码浮点数（如 `assert area == 1.2345`）**。断言必须基于**数学/几何法则**：

* **拓扑守恒定律**：无论网格如何加密，必须严格满足欧拉公式 $V - E + F = \chi$。
* **几何守恒法则**：网格细分后，所有子单元面积之和必须严格等于母单元总面积。
* **跨平台容差**：比较浮点数矩阵时，严禁使用 `==`。必须使用 `numpy.testing.assert_allclose` 吸收不同 CPU 架构下的微小精度误差。

---

## 5. CI 流水线运作机制 (你必须知道的 3 种触发方式)

为了保护团队免费计算额度并兼顾开发体验，我们的 GitHub Actions CI 采用了“分流策略”：

1.  **日常提交（极速拦截）**：当你发起普通的 `git push` 或 PR 时，CI **仅会运行**核心单元测试，确保在 3 分钟内给你反馈，不打断工作心流。
2.  **夜间巡逻（全量兜底）**：带有 `@pytest.mark.slow` 的耗时集成测试和回归测试，会在每周日的夜间定时任务中统一全量运行。
3.  **手动一键全量测试（终极武器）**：如果你这次重构了底层极其敏感的代码（如 IO 模块），心里没底不想等到周末，可以前往 GitHub 仓库的 Actions 面板，点击 `workflow_dispatch` 按钮，**手动触发一次包含集成测试在内的全量测试**。

*(注：系统配置了 `cache: 'pip'`，Numpy 等巨型依赖只会下载一次。若遇环境幽灵报错，请查看 CI 顶部的“环境自白书”排查版本差异。)*