# Documentation Style

## 一、目的与适用范围

本文档规定 FEALPy 项目中代码文档的写法与组织方式，用于统一函数、方法、类及关键对象的文档风格，提高代码可读性、接口可理解性与后续维护效率。

本文档适用于：

- `fealpy/fealpy/` 下的源码文档字符串；
- 面向用户开放的函数、方法、类与主要对象；
- 与计算模型、材料模型、网格对象、有限元组装对象等相关的主要接口说明。

本文档不规定：

- Markdown 资产正文的对话传输形式；
- 协作流程、审查流程与 Commit 规则；
- PDE 算例数据开发的专门规范；
- 一般代码格式细则与文件布局细则。

这些内容应分别遵循对应规范。本文档仅关注“代码文档如何写”。

## 二、基本原则

代码文档应遵循以下原则：

- 面向用户理解；
- 与代码实现一致；
- 信息完整但不过度堆砌；
- 术语稳定，表达清晰；
- 便于 IDE、静态分析与后续维护使用。

代码文档的首要目标不是“写得多”，而是“让用户和维护者准确理解接口的输入、输出、行为与边界”。

## 三、文档覆盖要求

### 3.1 面向用户的接口必须提供文档

所有面向用户的主要接口应提供清晰的文档字符串，尤其包括：

- 公共函数；
- 公共方法；
- 公共类；
- 主要计算模型类；
- 关键材料模型类；
- 关键网格与数据结构类。

若某接口同时面向外部使用且具有非显然行为，则不得省略文档。

### 3.2 文档与 Type Hint 配合使用

所有面向用户的接口都需要提供 Type Hint。文档字符串负责解释语义、用途与约束，Type Hint 负责表达参数与返回值的类型信息。二者应协同使用，不应互相替代。

例如：

```python
def boundary_interpolate(
    self,
    gd: Callable | int | float | TensorLike,
    uh: TensorLike | None = None,
    *,
    threshold: Callable | TensorLike | None = None,
    method=None
) -> TensorLike: ...
```

对于这类接口：

- Type Hint 用于说明参数和返回值的类型范围；
- 文档字符串用于说明这些参数在数值计算中的具体含义、使用方式与默认行为。

## 四、函数与方法文档规范

### 4.1 基本结构

函数或方法的文档字符串应优先包含以下内容：

- 功能说明；
- 参数说明；
- 返回值说明；
- 必要时补充默认行为、边界行为或特殊约束。

推荐采用如下结构：

```python
def entity(self, etype: int | str, index: Index | None = None, *, default=_default) -> Tensor:
    """
    Retrieves entities within the mesh based on the specified type (`etype`) and index location (`index`),
    returning a default value if the entity is not present.

    Parameters:
        etype (int | str): Defines the entity by its topology dimension (as an integer) or by name.
            Accepted names include 'cell', 'face', or 'edge'.
            Note that 'node' is unsupported in the current data structure.
            For polygonal meshes, 'cell_location' and 'face_location' can also be used,
            in which scenario `index` operates on the flattened entity tensor.

        index (Index | None, optional): The index specifying the location of the entity.
            Specifies the entity's position using an integer, slice, or tensor.
            Defaults to `None` if not explicitly provided.

        default (Any): The value returned when the entity is not located within the mesh.
            Defaults to the module-level variable `_default`.

    Returns:
        Tensor: The entity located at the specified `index`, or the `default` value if the entity is absent.
    """
```

### 4.2 功能说明要求

功能说明应回答以下问题：

- 该函数或方法做什么；
- 作用对象是什么；
- 在什么条件下返回什么结果；
- 是否存在默认行为或回退行为。

不应只写空泛描述，如“用于处理数据”“进行计算”等。

### 4.3 参数说明要求

参数说明应尽量说明：

- 参数类型或可接受形式；
- 参数的数学或工程含义；
- 可选值或可接受名称；
- 默认值及其行为；
- 特殊使用场景。

若参数支持多种输入形式，应明确列出，不应让用户通过源码猜测。

### 4.4 返回值说明要求

返回值说明至少应说明：

- 返回对象的类型；
- 返回对象表示什么；
- 在特殊情形下的返回行为。

若函数存在“找不到则返回默认值”“失败时返回空对象”“按索引切片后返回展平结果”等情形，应明确写出。

### 4.5 风格要求

函数与方法文档应注意：

- 先写整体作用，再写细节；
- 参数项与返回项格式统一；
- 描述句尽量完整；
- 避免只写参数名而不解释语义；
- 避免文档与代码行为不一致。

## 五、类文档规范

### 5.1 基本结构

类文档字符串通常应包含以下部分：

- 类的整体定位；
- 类的主要职责；
- 初始化参数；
- 重要属性；
- 必要时列出核心方法。

推荐采用如下结构：

```python
class QuadrangleMeshDataStructure(HomogeneousMesh):
    """
    A data structure class representing a quadrangle mesh, inheriting from the HomoMeshDataStructure.

    This class specifically handles quadrangle elements in a mesh, initializing with a given number of nodes (NN)
    and a tensor defining cells. It sets up constant tensors for local edges, faces, a counter-clockwise order,
    and cell definitions, tailored for quadrilateral geometries. Upon instantiation, it calls the `construct`
    method to further initialize or update any additional data structures necessary for the quadrangle mesh.

    Parameters:
        NN (int): The total number of nodes in the mesh.

        cell (Tensor): A tensor that defines the connectivity of cells within the mesh, specific to quadrangles.

    Attributes:
        localEdge (Tensor): A tensor defining the local edge connections within each quadrangle cell.

        localFace (Tensor): A tensor defining the local face connections within each quadrangle cell
            (though typically same as `localEdge` for quads).

        ccw (Tensor): A tensor providing a counter-clockwise ordering of indices for a standard quadrangle.

        localCell (Tensor): Defines how each local cell (quadrangle) is connected to the global node indexing.

    Methods:
        construct(): Internal method to construct additional data or perform post-initialization steps
            for the quadrangle mesh.
    """
```

### 5.2 类说明要求

类文档应优先回答以下问题：

- 该类表示什么对象；
- 继承自什么基类；
- 它负责什么数据或行为；
- 初始化时建立了哪些核心结构；
- 有哪些重要属性或内部约定。

对于数据结构类、材料类、计算模型类，这一点尤为重要。

### 5.3 参数、属性与方法说明要求

若类较复杂，应显式写出：

- `Parameters`：构造函数的关键输入；
- `Attributes`：对象建立后长期持有的核心属性；
- `Methods`：需要用户或维护者重点关注的方法。

若类较简单，可适度从简，但不能省略其核心定位。

## 六、数学与专业内容的文档表达要求

### 6.1 数学表达应清晰可读

在代码文档中描述数学对象时，应尽量保证：

- 数学意义清楚；
- 变量含义明确；
- 公式表达可读；
- 不制造歧义。

### 6.2 优先使用可直接阅读的数学字符

在文档字符串中描述数学符号时，应尽量使用可直接阅读的字符，而不是直接拷贝原始 LaTeX 形式。

例如：

- 应写 `μ`，而不是 `\mu`；
- 应写 `ν`，而不是 `\nu`；
- 应尽量写成读者一眼可理解的形式。

### 6.3 数学描述与实现必须一致

当文档中描述：

- 方程；
- 求解区域；
- 边界条件；
- 参数含义；
- 真解或源项；

这些内容必须与代码实现一致，不允许文档写的是一个区域、代码实现的是另一个区域，不允许文档与实际数学模型脱节。

## 七、PDE 算例与模型类的文档要求

### 7.1 PDE 类注释完整性要求

在 PDE 模块中添加算例时，类注释应尽量包含：

- PDE 的数学形式；
- 真解；
- 源项；
- 求解区域；
- 边界条件；
- 参考文献（如有）。

### 7.2 示例风格

例如，对于 Helmholtz 方程类，可采用如下风格：

```python
"""
2D Helmholtz problem with complex Robin (impedance-type) boundary condition:

    -Δu(x, y) - k^2·u(x, y) = 0,       (x, y) ∈ (0, 1) x (0, 1)
       ∂u/∂n + i·k·u = g(x, y),        on ∂Ω

Exact solution:

    u(x, y) = exp(i·β·k·y) * exp(-k·sqrt(β² - 1)·(x + 1))

This represents an evanescent wave propagating in the y-direction,
and exponentially decaying in the x-direction.

Parameters:
    k : wave number (float)
    beta : dimensionless propagation parameter (β > 1)

Source:
    https://www.sciencedirect.com/science/article/pii/S0045794917302602#e0010
"""
```

### 7.3 PDE 文档的特别要求

PDE 相关文档应特别注意：

- 数学形式与代码一致；
- 参数含义写清楚；
- 求解区域写清楚；
- 不写模糊的“某区域”“某方程”；
- 若引用文献，应给出明确来源。

## 八、类的字符串表示规范

### 8.1 适用对象

为了提高代码可读性和调试效率，所有主要类都应实现 `__str__` 方法，提供清晰、格式化的对象信息输出。

这类对象尤其包括：

- 材料类；
- 主要计算模型类；
- 含有关键参数状态的对象。

### 8.2 风格要求

`__str__` 的输出应满足：

- 信息清晰；
- 格式规整；
- 多行展示；
- 重点参数一目了然。

例如：

```python
class LinearElasticMaterial(ElasticMaterial):
    def __str__(self) -> str:
        """Return a nicely formatted, multi-line summary of the material."""
        return (
            f"Material '{self.name}':\n"
            f"  E (Young's modulus):      {self.E:.3g}\n"
            f"  ν (Poisson's ratio):      {self.nu:.3g}\n"
            f"  λ (Lame's first param):   {self.lam:.3g}\n"
            f"  μ (Shear modulus):        {self.mu:.3g}\n"
            f"  ρ (Density):              {self.rho:.3g}\n"
            f"  Hypothesis:               {self.hypo}\n"
            f"  Device:                   {self.device}"
        )
```

使用方式示例：

```python
material = LinearElasticMaterial()
self.logger.info(material)
```

### 8.3 目标

实现 `__str__` 的目标是：

- 让日志输出更有信息量；
- 让对象状态更易检查；
- 降低调试过程中的认知成本。

## 九、文档语言与表述风格

### 9.1 优先清楚，不追求修辞

代码文档应优先：

- 清楚；
- 直接；
- 稳定；
- 可维护。

不追求花哨表述，不使用口号式语言，不使用含糊判断。

### 9.2 英文文档字符串的基本要求

当前 FEALPy 旧规范中的示例主要采用英文文档字符串风格，因此项目中代码文档应优先保持该风格的一致性，尤其是面向公开接口的源码文档。

要求包括：

- 句子完整；
- 参数项统一；
- 返回值项统一；
- 专业术语前后一致。

### 9.3 不一致内容必须修正

以下情况必须修正：

- 文档参数名与函数签名不一致；
- 文档默认值与代码默认值不一致；
- 文档区域与代码区域不一致；
- 文档中的数学对象与实现对象不一致；
- 文档遗漏关键输入输出信息。

## 十、推荐模板

### 10.1 函数/方法模板

```python
def function_name(arg1: Type1, arg2: Type2 | None = None) -> ReturnType:
    """
    Briefly describe what the function does.

    Parameters:
        arg1 (Type1): Explain the meaning of `arg1`.

        arg2 (Type2 | None, optional): Explain the meaning, valid values,
            and default behavior of `arg2`.

    Returns:
        ReturnType: Explain what is returned and under what conditions.
    """
```

### 10.2 类模板

```python
class ClassName(BaseClass):
    """
    Briefly describe what the class represents and what its responsibility is.

    Parameters:
        arg1 (Type1): Explain the initialization parameter.

        arg2 (Type2): Explain the initialization parameter.

    Attributes:
        attr1 (TypeA): Explain the meaning of the attribute.

        attr2 (TypeB): Explain the meaning of the attribute.

    Methods:
        method_name(): Briefly explain the key method if necessary.
    """
```

### 10.3 参数较复杂的 PDE 类模板

```python
class ExamplePDE(BasePDE):
    """
    Briefly describe the PDE model.

    Governing equation:
        Write the equation in readable mathematical form.

    Exact solution:
        Write the exact solution if available.

    Domain:
        Write the computational domain explicitly.

    Boundary conditions:
        Explain the boundary conditions explicitly.

    Parameters:
        k (float): Explain the parameter.

    Source:
        Give the reference if available.
    """
```

## 十一、审阅检查清单

在提交代码前，应至少检查以下内容：

- 面向用户的主要接口是否已有文档字符串；
- Type Hint 是否已补齐；
- 文档是否解释了参数与返回值；
- 类文档是否写清楚职责、参数与关键属性；
- 数学描述是否清楚且与实现一致；
- PDE 类是否写清真解、源项、区域与边界条件；
- 主要类是否需要实现 `__str__`；
- 英文文档字符串是否通顺、统一、可读。

## 十二、边界说明

本文档只规定代码文档的写法，不替代以下规范：

- 代码格式与命名规范；
- 文件布局规范；
- PDE 模型开发规范；
- 协作、审查与提交流程规范；
- Markdown 资产写作与输出规范。

当其他规范对特定模块已有更细粒度要求时，应在不违背本文档原则的前提下遵循更具体的下位规范。