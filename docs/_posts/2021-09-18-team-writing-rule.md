---
title: 编程文档写作的基本原则和格式要求
tags: FEALPy team
---

写编程文档是团队科研中的一项基本活动. 为了保证产生文档的质量, 使其更易阅读分享,
这里把相应的写作原则和格式标准总结出来, 以供随时查阅.

# 基本原则 

1. 准确
1. 简洁
1. 要提供丰富的超链接

# 格式

## 文本与标点符号

- 长文本应按照语义单位进行换行, 在团队的 vim 配置文件中，默认设置了文本的宽度为 
  79 个字符.

```bash
set textwidth=79
```

- 使用英文环境下的标点符号.
- 标点符号与后面的文字要手动空一格.
- 每个行间公式后面都要有标点符号, 按照语义来选择相应的标点符号.
- 矩阵元素之间的下标要用逗号隔开, 例: $a_{i,j}$.
- 行内公式后有标点符号需要紧挨着, 例：

```latex
设 $x \in \mmathbb R$, 则有 $\cdots$.
```

## 中英文混排

- 英文单词与两边的中文要空一格, 例：

```markdown
引入 Clement-Scott-Zhang 插值算子.
```

## 数学符号意义约定

- 小写的英文或希腊字母表示标量或标量函数, 例:

```latex
f(x) = 2x + 3.
```

- 大写的英文或希腊字母表示集合或空间, 例:

```latex
$$
 H^1(\Omega)=\{\phi \in L_2(\Omega):\,
             \nabla \phi \in L_2(\Omega;\mathbb R^d)\}.
$$
```

$$
H^1(\Omega)=\{\phi \in L_2(\Omega):\,\nabla \phi \in L_2(\Omega;\mathbb R^d)\}.
$$

- 粗体的小写字母表示向量或向量函数, 例:

```latex
\boldsymbol a = [0,1,\cdots,n].
```

$$
\boldsymbol a = [0,1,\cdots,n].
$$

- 粗体的大写字母代表矩阵或矩阵函数, 例:

```latex
\boldsymbol A = \boldsymbol L \boldsymbol L^T.
```

$$
\boldsymbol A = \boldsymbol L \boldsymbol L^T.
$$

- 空心大写字母代表特殊的空间, 例:

$$
\text{实数空间:}\,\mathbb R,\quad
    \text{复数空间:}\,\mathbb C,\quad
    \text{多项式空间:}\,\mathbb P.
$$

- 花体的大写字母代表算子, 例:

```latex
\mcA : H^1_0(\Omega) \to H^{-1}(\Omega).
```

$$
\mathcal A : H^1_0(\Omega) \to H^{-1}(\Omega).
$$

## 数学公式

- 长公式应按照数学公式语义, 便于和生成的网页文件或 pdf 对照修改，例:

```markdown
$$
\begin{aligned}
    \begin{bmatrix}
        \boldsymbol A & \boldsymbol 0 &\boldsymbol B_{1} \\
        \boldsymbol 0 &  \boldsymbol A &\boldsymbol B_{2}\\
        \boldsymbol B_{1}^{T} & \boldsymbol B_{2}^{T} & \boldsymbol 0\\
    \end{bmatrix}
    \begin{bmatrix}
        \boldsymbol U_x \\
        \boldsymbol U_y \\
        \boldsymbol P 
    \end{bmatrix}
\end{aligned}
$$
```

$$
\begin{aligned}
 	\begin{bmatrix}
 		 \boldsymbol A & \boldsymbol 0 &\boldsymbol B_{1} \\
 		 \boldsymbol 0 &  \boldsymbol A &\boldsymbol B_{2}\\
    	 \boldsymbol B_{1}^{T} & \boldsymbol B_{2}^{T} & \boldsymbol 0\\
 	\end{bmatrix}
 	\begin{bmatrix}
        \boldsymbol U_x \\
        \boldsymbol U_y \\
        \boldsymbol P 
 	\end{bmatrix}
\end{aligned}
$$

- 行间公式上下段落之间要空一行, 例:

```markdown
定义有限元空间:

$$
\begin{aligned}
 \mathbbV: = \{
     v \in H_0^1(\Omega):v|_K , \forall K \in \mathcal T
 \}.
\end{aligned}
$$

其中...
```

- 行内公式与中文字体之间要有空格, 例:

```tex
设 $\Omega \subset \mathbbR^d$ 是有界的多边形区域.
```

- 行内公式和两边的美元符号`$`不能有空格,
    这样可能导致生成的网页文件不能正常显示公式.

- 公式中的 Latex 命令之间要空一格, 例:

```tex
$\Omega \subset \mathbbR^d$ % nice 

$\Omega\subset\mathbbR^d$ % bad
```
## 程序代码


# 附录

1. [Latex 学习--基础知识](https://www.cnblogs.com/cmi-sh-love/p/latex-xue-xiji-chu-zhi-shi.html)
1. [BERAMER 文档类用户手册](http://static.latexstudio.net/wp-content/uploads/2017/02/BeamerUserGuide_V3.24_zh-cn.pdf)
1. [Latex 学习笔记](https://www.jianshu.com/p/55dde2c64667)

