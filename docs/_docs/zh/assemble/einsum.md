---
title: 爱因斯坦求和
permalink: /docs/zh/assemble/einsum
key : docs-einsum-zh
author : wpx
---

# 爱因斯坦求和约定

​	    在数学中，爱因斯坦求和约定是一种标记法，也称为Einstein Summation Convention。简单来说，爱因斯坦求和就是简化掉求和式中的求和符号 $\sum$  ,这样就会使公式更加简洁，如

$$
a_ib_i := \sum^N_{i=0}a_ib_i = a_0b_0 + a_1b_1 + \cdots + a_Nb_N
$$

## 自由标

​		在爱因斯坦求和约定中，当一个下标符号仅出现一次时，则该下标为自由标，须遍历该下标所有的取值。

- 仅有一个自由标，表示矢量，例如:

$$
x_i = \boldsymbol x =(x_1,x_2,x_3)
$$

$$
\frac{\partial \phi}{\partial x_i} = \nabla \phi = (\frac{\partial \phi}{\partial x_1},\frac{\partial \phi}{\partial x_2},\frac{\partial \phi}{\partial x_3})
$$

- 有且仅有两个自由标记，表示二阶张量(矩阵)，例如

$$
\tau_{ij} = 
\begin{bmatrix}
\tau_{xx} & \tau_{xy}  &\tau_{xz} \\
\tau_{yx} & \tau_{yy}  &\tau_{yz} \\
\tau_{zx} & \tau_{zy}  &\tau_{zz} \\
\end{bmatrix}
$$

$$
u_iv_j = 
\begin{bmatrix}
u_1v_1 + u_1v_2 + u_1v_3 \\
u_2v_1 + u_2v_2 + u_2v_3 \\
u_3v_1 + u_3v_2 + u_3v_3 
\end{bmatrix}
$$

对于更高阶的张量也可依次类推

## 哑标

​        根据爱因斯坦求和约定，当下标重复出现多次时，则对该下标的索引项进行求和，该下标称为哑标

- 当仅有一个哑标，表示一个标量，例如

$$
\frac{\partial u_i}{\partial x_i} = \nabla \cdot \boldsymbol u =  \frac{\partial u_1}{\partial x_1}+\frac{\partial u_2}{\partial x_2}+\frac{\partial u_3}{\partial x_3}
$$

$$
a_ib_ic_i = \sum^N_{i=0}a_ib_ic_i = a_0b_0c_0 + a_1b_1c_1 + \cdots + a_Nb_Nc_N
$$



上式中由于 ![[公式]](https://www.zhihu.com/equation?tex=i+) 重复了两次，是一个哑标，因此对两个 ![[公式]](https://www.zhihu.com/equation?tex=i) 同时进行索引，并对其索引项进行求和。

- 当有且仅有一个自由标和一个哑标时，自由标和哑标均进行遍历，最终表示的是一个矢量，例如：

$$
\frac{\partial u_iv_j}{\partial x_j} = 
\begin{bmatrix}
\frac{\partial u_1v_1}{\partial x_1}  + \frac{\partial u_1v_2}{\partial x_2}  +\frac{\partial u_1v_3}{\partial x_3} \\ 
\frac{\partial u_2v_1}{\partial x_1}  + \frac{\partial u_2v_2}{\partial x_2}  +\frac{\partial u_2v_3}{\partial x_3} \\ 
\frac{\partial u_3v_1}{\partial x_1}  + \frac{\partial u_3v_2}{\partial x_2}  +\frac{\partial u_3v_3}{\partial x_3} \\ 
\end{bmatrix}
$$

## 补充

**事实上，自由标的个数决定了该张量的阶数**。张量的表示法中所说的下标个数的统计均是在一个独立的项之内完成的，如$a_i+b_j$，表达两个独立的项，各自按早顺序展开
$$
a_i + b_j = (a_1,a_2,a_3)+(b_1,b_2,b_3) = (a_1+b_1,a_2+b_2,a_3+b_3)
$$

# Numpy中的einsum

- 语法

<img src="./figures/einsum.png" alt="workflow" style="zoom:100%;" />

参数说明

1. **subscripts**：（**str**）指定求和的下标 ，->后面的为要输出求和形式
1. arg0,arg1,arg2，…… :与前面的下标对应的变量

- 例子

1. 转置
$$
 B_{ji} = A_{ji}
$$

```python
import numpy as np
a = np.arange(0, 9).reshape(3, 3)
print(a)
b = np.einsum('ij->ji', a)
print(b)

Output:
a: [[0 1 2]
 [3 4 5]
 [6 7 8]]
b: [[0 3 6]
 [1 4 7]
 [2 5 8]]
```

2. 全部元素求和

$$
sum = \sum_i \sum_j A_{ij}
$$

```python
import numpy as np
a = np.arange(0, 9).reshape(3, 3)
print(a)
b = np.einsum('ij->', a)
print(b)

Output:
a: [[0 1 2]
 [3 4 5]
 [6 7 8]]
b: 36
```

3. 矩阵点乘

$$
C = \sum_i\sum_j A_{ij}B_{ij}
$$

```python
import numpy as np
a = np.arange(0, 12).reshape(3, 4)
print(a)
b = np.arange(0, 12).reshape(3, 4)
print(b)
c = np.einsum('ij,ij->', a, b)
print(c)

Output:
a: [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
b: [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
c: 506
```
4.单元质量矩阵组装

​		设单元基函数为

$$
\begin{aligned}
\boldsymbol\phi := [\phi_0(\boldsymbol x), \phi_1(\boldsymbol x), \cdots, \phi_{l-1}(\boldsymbol x)],
\forall \boldsymbol x \in \tau
\end{aligned}
$$

​		则其在单元 $ \tau $ 上的质量矩阵为

$$
\begin{aligned}
    \boldsymbol H=\int_{\tau} \boldsymbol \phi^{T} \boldsymbol \phi d \boldsymbol x &=
    \begin{bmatrix}
        (\phi_{0}, \phi_{0})_{\tau} & (\phi_{0}, \phi_{1})_{\tau} & \cdots & (\phi_{0}, \phi_{l-1})_{\tau} \\
        (\phi_{1}, \phi_{0})_{\tau} & (\phi_{1}, \phi_{1})_{\tau} & \cdots & (\phi_{1}, \phi_{l-1})_{\tau} \\
        \vdots & \vdots & \ddots & \vdots \\
        (\phi_{l-1}, \phi_{0})_{\tau} & (\phi_{l-1}, \phi_{1})_{\tau} & \cdots & (\phi_{l-1}, \phi_{l-1})_{\tau}
    \end{bmatrix} \\
    &=|\tau| \sum_{i=0}^{l-1} w_i
    \begin{bmatrix}
    \phi_0(\boldsymbol x_i) \\
    \phi_1(\boldsymbol x_i) \\
    \vdots \\
    \phi_{l-1}(\boldsymbol x_i) \\
    \end{bmatrix} 
        \begin{bmatrix}
    \phi_0(\boldsymbol x_i) &
    \phi_1(\boldsymbol x_i) &
    \cdots &
    \phi_{l-1}(\boldsymbol x_i) 
    \end{bmatrix} 
\end{aligned}
$$

上式中 $w_i$ 为积分权重，$\boldsymbol x_i$ 为积分点。代码如下

```python
import numpy as np
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
#建立初始网格和空间
mesh = MF.boxmesh2d([0, 1, 0, 1], nx=10, ny=10, meshtype='tri')
space = LagrangeFiniteElementSpace(mesh, p=1)
#获取数值积分的积分点和积分权重
qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')
#获取空间基基函数
phi = space.basis(bcs)
#刚度矩阵组装
H = np.einsum('i,ijk,ijm,j -> jkm',ws,phi,phi,cellmeasure)
```

- 加速

可以通过过安装opt_einsum来加速einsum的速度

```bash
pip3 install opt-einusm
```

用法如下

```python
import numpy as np
from opt_einsum import contract

N = 10
C = np.random.rand(N, N)
I = np.random.rand(N, N, N, N)

%timeit np.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
1 loops, best of 3: 934 ms per loop

%timeit contract('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
1000 loops, best of 3: 324 us per loop
```



# 附录

1. [opt-einsum](https://pypi.org/project/opt-einsum/)
1. [numpy中的einsum](https://zhuanlan.zhihu.com/p/74462893)
1. [哑标与自由标](https://zhuanlan.zhihu.com/p/136836158)
