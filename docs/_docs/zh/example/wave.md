# 基于Fealpy的电磁反向波有限元仿真

Author:  王唯

E-mail: <abelwangwei@163.com>  

Address: 湘潭大学 $\cdot$​ 数学与计算科学学院



## 1.致谢与声明

$\qquad$​本实验是在我的导师杨伟教授的指导下完成的, 感谢杨老师的付出! 本实验还得到了魏华祎副教授的帮助以及Fealpy科学计算开源社区平台的支持, 感谢他们的工作! 本实验所有代码和相关文档全部提交至Fealpy社区, 由我和Fealpy社区共同维护!



## 2.背景介绍

### 2.1Maxwell方程简介

$\qquad$现实世界中的所有电磁现象都可以由Maxwell方程描述
$$
\left\{\begin{array}{l}
\nabla \times \boldsymbol{E}+\frac{\partial \boldsymbol{B}}{\partial t}=\mathbf{0} \\
\nabla \times \boldsymbol{H}-\frac{\partial \boldsymbol{D}}{\partial t}=\boldsymbol{J} \\
\nabla \cdot \boldsymbol{B}=0 \\
\nabla \cdot \boldsymbol{D}=\rho
\end{array}\right.
$$
$\qquad$其中, $\boldsymbol{E}$是电场强度,单位为伏特/米$(\mathrm{V}/\mathrm{m)}$; $\boldsymbol{D}$是电通量密度,单位为库仑/米$^2$$(\mathrm{C}/\mathrm{m}^2)$; $\boldsymbol{H}$磁场强度, 单位为安培/米$(\mathrm{A}/\mathrm{m)}$; $\boldsymbol{B}$磁通量密度,单位为韦伯/米$^2$$(\mathrm{Wb}/\mathrm{m}^2)$; $\boldsymbol{J}$电流密度,单位为安培/米$^2$ $(\mathrm{A}/\mathrm{m}^2)$

其材料参数由本构关系确定, 具体来说
$$
\left\{\begin{array}{l}
\boldsymbol{D}=\varepsilon \boldsymbol{E}=\varepsilon_{0} \varepsilon, \boldsymbol{E} \\
\boldsymbol{B}=\mu \boldsymbol{H}=\mu_{0} \mu_{r} \boldsymbol{H} \\
\boldsymbol{J}=\sigma \boldsymbol{E}
\end{array}\right.
$$
$\qquad$其中, $\varepsilon $是介质介电系数, 单位为法拉第/米$(\mathrm{F}/\mathrm{m)}$;$\mu $为磁导系数,单位为亨利/米$(\mathrm{H}/\mathrm{m)}$;$\sigma $为电导率,单位为西门子/米$(\mathrm{S}/\mathrm{m)}$.

$\qquad$​考虑Maxwell方程有时频转换关系
$$
\frac{\partial}{\partial t}\rightarrow \mathrm{j}\omega 
$$

### 2.2超材料简介

> $\qquad$​新型人工电磁媒质(也称超材料、新型人工电磁材料、特异介质、异向介质等)这个词首先由美国得克萨斯州大学奥斯汀分校的 Rodger. Walser教授提出,并定义为"Macroscopic composites having synthetic, three-dimensional-periodic cellular architecture designed to produce an optimized combination, no available in nature, of two or more responses to specific excitation根据维基百科,新型人工电磁媒质可以定义为" material which gains its properties from it structure rather than directly from its composition"根据等效媒质理论,新型人工电磁媒质的单元结构大小必须小于或等于亚波长尺寸上述定义或范从不同方面对新型人工电磁媒质进行了描述,可以看出,它有如下几个特点:同寻常的物理性质,而且其性质主要取决于单元结构;周期或非周期的人工单元结构排列;其单元结构具有亚波长尺寸本书中,新型人工电磁媒质有时也简为人工电磁媒质.左手媒质、负折射率媒质等都属于新型人工电磁媒质的范畴新型人工电磁媒质包含的范围很广,凡是介电常数和磁导率与普通介质有明区别的媒质都属于新型人工电磁媒质范畴.一些奇特的各向异性、双各向异媒质及手征媒质也属于新型人工电磁媒质的定义范畴.
>
> $\qquad$​新型人工电磁媒质是一种人造媒质,其电磁性质不仅取决于单元的组成成分,而且还决定于其结构.作为明确概念,新型人工电磁媒质最早可追溯到20世纪60年代.1968年,苏联科学家 Veselago在理论上提出了左手媒质的概念,并系统地分析了其物理性质.左手媒质是指介电常数和磁导率同时为负的媒质,拥有很多新奇特性,如负折射、<u>==反向波传播==</u>、逆多普 Cerenkov勒效应、反向辐,,射等.因为缺少实验验证,左手媒质的概念提出之后的很长一段时间并没有引起人们的重视.直到1996年,英国帝国理工学院理论物理学家 Pendry教授利用周期排布的金属线在较低频率上实现了类似于金属在光波段所具有的等离子特征,从而人工实现了负介电常数.1999年, Pendry教授进一步利用非磁性媒质构造出能产生负磁导率的开口谐振环(Split- ring Resonantor,SRR)结构2000年Pendry教授在 Physics Review Letters上发表了题为"Negative Refraction Makes Perfect Lens"的文章,从理论上证明利用左手媒质可实现"完美成像",从而突破成像分辨率的衍射极限,并在其之前工作的基础上提出了可能实现左手媒质的实际单元结构.几乎同时,美国科学家Smth教授也找到了可能实现左手媒质的结构10-12),随后在2001年通过实验对负折射现象进行了验证.这一成果被 Science杂志评为2003年的十大科技突破之一.从此左手媒质(或称为负折射率媒质)引起了广大科研工作者的极大兴趣,其理论和实验工作都取得了很大进展,如开口谐振环单元之间的耦合效应新的磁谐振结构、电谐振结构或同时具有负介电常数和负磁导率的新结构等.
>
> ​                                                                                                   ——蒋卫祥, 崔铁军《变换光学理论及其应用》

$\qquad$​本实验主要是仿真电磁波在左手材料中的反向传播现象, 这是一个界面问题.

### 2.3PML

$\qquad$PML(perfectly match layer, 完全匹配层)属于典型ABC 边界条件(Absorbing Boundary Condition, 吸收边界条件), 因为电磁波在边界上会发生发射, 从计算上来看, 会破坏计算域中的解, 因此在开域问题的截断边界应用完全匹配层(PML)来吸收外向行波会有更好的效果. Berenger(1994)首先提出场分量分裂PML理论, 几乎在同时, Sacks(1995)和Gedney(1996)提出各向异性介质PML, Chew和 Weedon(1994)提出基于坐标伸缩的PML, 本实验使用UPML, 详细理论可参考Monk(2003).

## 3.数学模型

令$ \Omega:=\left[a_{1}, b_{1}\right] \times\left[a_{2}, b_{2}\right] $​, 考虑二维时谐磁通密度的含有PML的Maxwell双旋度方程
$$
\nabla \times \alpha \varepsilon _{r}^{-1}\nabla \times \mu _{r}^{-1}\mathbf{B}-k^2\beta \mathbf{B}=\beta \mathbf{F}
$$
边界上采用PMC边界条件
$$
\mu _{r}^{-1}\boldsymbol{B}\times \boldsymbol{n}=0
$$
其中$\alpha, \beta$为匹配矩阵, 具体来说
$$
\alpha=\frac{1}{d_1d_2},\quad \beta =\left[ \begin{matrix}\frac{d_1}{d_2}&0\\0&\frac{d_2}{d_1}\\\end{matrix} \right]
$$
其中
$$
d_j=1+i\frac{\sigma _j\left( x_j \right)}{2\pi k},\qquad \sigma _j\left( x_j \right) =\left\{ \begin{matrix}C\left( \frac{x_j-a_{j}^{'}}{\delta _j} \right) ^2,&x_j\in \left( a_j,a_{j}^{'} \right) ,\\0,&x_j\in \left( a_{j}^{'},b_{j}^{'} \right) ,\\C\left( \frac{x_j-b_{j}^{'}}{\delta _j} \right) ^2,&x_j\in \left( b_{j}^{'},b_j \right) ,\\\end{matrix} \right.
$$
 取$ a_{1}=a_{2}=-2, b_{1}=-5, b_{2}=2, C=100, \delta_{1}=\delta_{2}=0.5$, 材料参数为
$$
\varepsilon _r=\left\{ \begin{matrix}-1.1,&\Omega _1\\1,&\Omega \backslash \Omega _1\\\end{matrix} \right. \qquad \mu _{r}^{-1}=\left\{ \begin{matrix}-1.1I,&\Omega _1\\I,&\Omega \backslash \Omega _1\\\end{matrix} \right.
$$
 原问题的变分格式为:给定$ \boldsymbol{F} \in\left[L_{2}(\Omega)\right]^{2}$,找到$ \boldsymbol{B} \in H\left(\mathrm{div}^{0} ; \Omega\right) \cap H_{0}\left(\operatorname{curl} ; \Omega ; \mu_{r}^{-1}\right)$,使得
$$
\left( \alpha^{-1} \epsilon _{r}^{-1}\nabla \times \mu _{r}^{-1}\boldsymbol{B},\nabla \times \beta ^{-1}\mu _{r}^{-1}\boldsymbol{v} \right) -k^2\left( \mu _{r}^{-1}\boldsymbol{B},\boldsymbol{v} \right) =\left( \mu _{r}^{-1}\boldsymbol{F},\boldsymbol{v} \right)
$$
 其中$\forall \boldsymbol{v} \in H\left(\operatorname{div}^{0} ; \Omega\right) \cap H_{0}\left(\operatorname{curl} ; \Omega ; \mu_{r}^{-1}\right)$​​.

此外波源项$\boldsymbol{F}=[f_1,0]^{\top}$, 其中
$$
f_1=k^2\left( 1-4\left( x^2+(y-1.45)^2 \right) \right) \exp (iky)
$$
定义在以$(0,1.45)$为圆心半径为$1/2$区域内,在其它区域$f_1=0$.超材料区域固定在区域$\Omega _1=[-2,2]\times [-2.4,-0.6]$.



## 4.算法设计

原问题考虑用有限元离散, 并且为了解的物理性质, 考虑是用Nedelec元离散. 为了提高代码的复用性, 我们采用面向对象的程序设计模式.

在实现时, 我们考虑有两个类, 其中

```python
class PML_model:
```

是描述问题和设计参数的.

```python
class PML_solution:
```

是解法器.

这里需要注意的是, 由于最后组装矩阵和向量的时候, 我们无法直接操作单刚和单载, 因此, 我们要运用张量计算技巧, 相关参数要在此之前形成, 张量内部元素要记录相关点和相关数据, 例如, 对于介电常数

```python
@cartesian
    def epsilon(self, p):
        x = p[..., 0]
        y = p[..., 1]
        idx = (x > -2) & (x < 2) & (y > -2.4) & (y < -0.6)
        w = np.size(p, 0)
        t = np.size(p, 1)
        val = np.ones([w, t, 1])
        val[idx, 0] = 1.0 / - 1.1
        return val
```

最后, 我们利用爱因斯坦求和形成相应的矩阵和向量即可. 其main函数为

```python
pml = PML_solution()
pml.get_imag()
```

## 5.数值结果

![image-20211102095248805](C:\Users\Genius\Desktop\wave\image-20211102095248805.png)

![image-20211101214151168](C:\Users\Genius\Desktop\wave\image-20211101214151168.png)



从图中可以看到, 当电磁波到达超材料区域(黑框), 波开始和之前的波以镜面形式向前传播, 有限元仿真与实际结果相符, 并且波在边界上没有发生反射, 说明电磁波在PML区域快速衰减, PML区域工作正常.

## 6.后续工作

- 反向波问题属于界面问题, 在界面处有奇性, 因此可以考虑使用自适应有限元方法.
- Fealpy目前对电磁波绘图采用的是渲染网格的方法, Fealpy需要研究其他方法, 使得数值可视化更加美观.
- 电磁场有限元计算依赖于Nedelec元, Fealpy需要继续完成此类工作.