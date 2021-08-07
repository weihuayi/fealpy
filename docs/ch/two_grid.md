# 两层网格方法 

## 问题模型

考虑如下模型问题 

$$
-u'' = f, x\in(0, 1)
$$

满足边界条件

$$
u(0)=u(1)=0.
$$

```python
import numpy as np
from fealpy.mesh import IntervalMesh

node = np.array([[0.0], [1.0]], dtype=np.float64)
cell = np.array([[0, 1]], dtype=np.int_)

mesh = IntervalMesh(node, cell)
mesh.uniform_refine(4)

```


## 程序实现


## 参考文献

1. Long Chen，[Introduction to multigrid
   methods](https://www.math.uci.edu/~chenlong/226/MGintroduction.pdf).



