# Coding Standards

## 一、文档目的与适用范围

本文档定义项目的代码编写规范，用于在多人协作开发过程中保持代码的一致性、可读性与可维护性。

本规范适用于：

- 项目源码
- 核心算法实现
- 数值计算模块
- 公共基础库代码

脚本类文件（例如实验脚本、一次性分析脚本）可以适当放宽部分规范，但仍建议尽量保持一致风格。

本规范基于 FEALPy 项目既有开发规范整理，并在必要处参考成熟 Python 项目实践进行补充。


## 二、命名规范

统一的命名规则可以显著提升代码可读性与维护效率。

### 2.1 类

类名采用 PascalCase（每个单词首字母大写）。

示例：

```python
class ScalarDiffusionIntegrator:
    ...
```

规则：

- 每个单词首字母大写
- 不使用下划线
- 名称应表达明确的对象概念


### 2.2 函数与方法

函数名与方法名采用 snake_case（小写加下划线）。

示例：

```python
def multi_index_matrix():
    ...
```

规则：

- 全部小写
- 单词之间使用 `_` 分隔


### 2.3 变量

变量名使用 snake_case。

示例：

```python
face_to_cell = ...
cell_to_edge = ...
```

允许使用 FEALPy 约定缩写，例如：

```python
gdof = 37680
```

表示 global degree of freedom。


### 2.4 常量

常量使用全大写加下划线。

示例：

```python
GDOF = space.number_of_global_dofs()
LDOF = space.number_of_local_dofs()
NN = mesh.number_of_nodes()
TD = 2
```

说明：

- 常量通常在创建后不再修改
- 可以使用项目约定缩写


### 2.5 内部对象

不希望用户直接访问的对象使用前导下划线。

示例：

```python
def _simplex_measure(self, index):
    ...

self._value = ...
```

说明：

- 表示内部实现细节
- 不属于公开接口


### 2.6 内部全局对象

不对用户开放的全局对象可以使用下划线前缀。

示例：

```python
_S = slice(None)
```


### 2.7 类型变量

用于类型提示的类型对象。

自定义类型命名方式与类一致：

```python
type Index = Tensor | int | slice
```

类型别名建议使用下划线前缀：

```python
_dtype = torch.dtype
_FS = FunctionSpace
```


### 2.8 几何对象命名约定

在网格与几何结构相关代码中，优先使用 `face` 作为中间维数实体的变量名或接口名。

原因：

- `face` 可以统一表示二维、三维甚至更高维结构
- 有利于代码的通用性与可扩展性

只有在实现明显依赖二维几何结构时才使用 `edge`。


## 三、源文件结构

Python 源文件建议按照以下顺序组织：

1. 导入语句
2. 全局变量
3. 类与函数定义

各部分之间使用至少一个空行分隔。

示例：

```python
from typing import Self

from ..backend import bm

_Self = TypeVar("_Self")


class TensorLike:
    ...
```


## 四、导入规范

导入顺序必须遵循以下规则：

1. Python 标准库
2. 第三方库
3. 项目内部模块

不同类别之间应使用空行分隔。


### 4.1 标准导入顺序

示例：

```python
from typing import Any, Self, overload

from scipy.sparse.linalg import eigsh

from fealpy.backend import bm
from fealpy.mesh import Mesh
from fealpy.functionspace import functionspace
from fealpy.fem import BilinearForm
from fealpy.fem import DirichletBC
```


### 4.2 FEALPy 模块导入顺序

FEALPy 内部模块建议按照有限元计算流程组织导入：

1. 工具模块

- `backend`
- `typing`
- `decorator`
- `model`

2. 网格、空间与材料

- `mesh`
- `functionspace`
- `material`

3. 算法模块

- `fem`

4. 求解器模块

- `solver`

### 4.3 绝对导入与相对导入

调用 FEALPy 基础模块时使用绝对导入：

```python
from fealpy.backend import bm
```

调用当前模块内部代码时使用相对导入：

```python
from ..model import CSMModelManager
```


### 4.4 导入语句长度

单行 import 不应明显超过 80 个字符。

必要时使用多行形式：

```python
from fealpy.fem import (
    BilinearForm,
    LinearElasticityIntegrator,
    DirichletBC
)
```


## 五、类与函数布局

### 5.1 空行规则

类与函数之间：

- 使用两个空行

类方法之间：

- 使用一个空行


示例：

```python
class TriangleMesh:

    def __init__(self):
        pass

    def construct(self):
        pass


def some_function():
    pass
```


## 六、行长度

单行代码长度建议不超过 80 个字符。

必要时可以略微超出，但不应明显超过。


## 七、代码细节规范

### 7.1 逗号后必须有空格

错误：

```python
mesh = QuadrangleMesh.from_box([-1,1,-1,1],nx=5,ny=5)
```

正确：

```python
mesh = QuadrangleMesh.from_box([-1, 1, -1, 1], nx=5, ny=5)
```


### 7.2 赋值两侧必须有空格

正确：

```python
A = D0 @ A @ D0 + D1
```

错误：

```python
uh=spsolve(A, F)
```


### 7.3 关键字参数

关键字参数通常不在等号两侧添加空格。

正确：

```python
index=flag
```

错误：

```python
index = flag
```


### 7.4 带 TypeHint 的默认值

存在类型提示时，默认值两侧需要空格：

```python
axis: int = 0
```


### 7.5 运算符两侧空格

错误：

```python
x = x+pe[:x.size(0)]
```

正确：

```python
x = x + pe[:x.size(0)]
```


### 7.6 多行表达式缩进

示例：

```python
a = bm.concat([
    cell1,
    cell2
], axis=0)
```

或：

```python
b = bm.stack(
    [x_data, y_data, z_data],
    axis=-1
)
```


## 八、Type Hint 规范

Type Hint 用于提供：

- IDE 自动提示
- 静态分析支持
- 文档生成辅助

要求：

所有面向用户的接口必须提供 Type Hint。

示例：

```python
def boundary_interpolate(
    self,
    gd: Callable | int | float | TensorLike,
    uh: TensorLike | None = None,
    *,
    threshold: Callable | TensorLike | None = None,
    method=None
) -> TensorLike:
```


## 九、文档字符串

项目使用 Python 标准 docstring。

推荐结构：

- 功能说明
- 参数说明
- 返回值说明


示例：

```python
def entity(self, etype, index=None):
    """
    Retrieve mesh entities.

    Parameters
    ----------
    etype : int or str
        Entity type.

    index : optional
        Entity index.

    Returns
    -------
    Tensor
    """
```


## 十一、对象字符串表示

重要类应实现 `__str__()` 方法，以提高调试和日志输出的可读性。

示例：

```python
def __str__(self):
    return (
        f"Material '{self.name}'\n"
        f"E: {self.E}"
    )
```


## 十二、后端统一规则

在 FEALPy 中统一使用 `bm` 作为计算后端接口。

禁止在核心代码中直接使用：

- `numpy`
- `torch`

示例：

正确：

```python
bm.zeros(...)
```

错误：

```python
np.zeros(...)
```


## 十三、代码可读性原则

当出现以下情况时应优先提升可读性：

- 表达式过长
- 数学表达式复杂
- 逻辑结构复杂

建议：

- 使用中间变量
- 将复杂逻辑拆分为函数
- 使用 `sympy` 自动生成复杂表达式


## 十四、风格优先级原则

当不同规范发生冲突时，遵循以下优先级：

1. 项目 Coding Standards
2. PEP8
3. Python 社区通用实践

最终目标是保证代码清晰可读。


## 十五、总体原则

所有代码应遵循以下核心原则：

- 简洁
- 可读
- 可维护
- 可扩展

当存在多种实现方式时，应优先选择最清晰、最易理解的写法，而不是最短或最复杂的写法。