# FEALPy：以开源社群的力量，打造中国自己的工业仿真共性基础算法库

* 核心开发者：魏华祎
* 开发单位：湘潭大学
* 编程语言：Python
* 开源类型：GNU General Public License
* 邮箱：weihuayi@xtu.edu.cn
* 托管网址：
    - https://github.com/weihuayi/fealpy
    - https://gitlab.com/weihuayi/fealpy
    - https://github.com/deepmodeling/fealpy
    - https://gitee.com/whymath/fealpy

* 帮助文档：https://www.weihuayi.cn/fealpy/docs/zh/quick-start


$\quad$ FEALPy 定位于一款开源的偏微分方程数值解算法库, 
主要用于解决工业仿真基础算法科研和人才培养中缺少自主可控算法软件平台的问题.
FEALPy 设计的首要目标是追求**简单易用, 并且不受操作系统或软件平台太多限制**. 
所以 FEALPy 完全采用 Python 语言开发, 并充分利用 Python 对象化和 Numpy 
数组化编程技术, 重构偏微分方程数值解核心的数据结构和算法，
以达到模块化和接口标准化的目的. 在实现 FEALPy 高可维护和高可扩展的同时，
让用户能够以“乐高积木”的方式快速搭建自己的数值实验程序, 
并可以根据需要灵活替换其中的模块.
FEALPy 即可用于支撑相关领域的基础科研和人才培养工作, 
又可用于 CAE 应用原型软件的快速开发验证工作.
FEALPy 的长远目标是希望借助开源社区的力量, 为广大算法科研工作者及学生提供一个简单易用、
开放共享的科学计算实践学习平台.

$\quad$ FEALPy 中的单词 feal 是忠实、可靠的意思, 是我的师兄易年余帮忙起的,
寓意很好, 因为我内心就是希望 FEALPy 能成为我们学习、
研究和应用偏微分方程数值解最忠实可靠的帮手.

## 设计特点

1. 脚本化实现, 无需编译, 安装简单.
1. 完全基于标准的 Python3 科学计算库开发, 如 Numpy、 Scipy、 Matplotlib 等.
1. 适用于 Linux、 MacOS、Windows 三大主流操作系统.
1. 基于 Numpy 的网格数组化表示和核心算法数组化实现.
    * 简短的代码实现复杂的运算
    * 算法更接近高等代数的思维习惯
    * 自动支持多线程运算
1. 高度模块化组织, 且模块内同一类对象拥有同样的变量和接口命名规则.
    * 易阅读
    * 易维护
    * 易扩展
    * 易泛化

## 已有功能

![](./fealpy.png)

$\quad$ FEALPy 目前已经集成了非常丰富的算法功能, 如

* 丰富的网格数据结构
    - `IntervalMesh` 一维区间网格
    - `TriangleMesh` 三角形网格
    - `QuadrangleMesh` 四边形网格
    - `PolygonMesh` 多边形网格
    - `HalfEdgeMesh2d` 半边网格
    - `TetrahedronMesh` 四面体网格
    - `HexahedronMesh` 六面体网格
    - `LagrangeTriangleMesh` 高阶三角形网格
    - `LagrangeQuadrangleMesh` 高阶四边形网格
    - ...... 
* 各种常用的网格自适应算法, 如二分、四叉树、八叉树等.
* 常用的基本有限元空间, 如
    - `LagrangeFiniteElementSpace` 任意维任意次的拉格朗日有限元空间,
        包括连续和间断
    - `ParametricLagrangeFiniteElementSpace` 任意次的参数有限元空间,
        包括连续和间断
    - `ScaledMonomialSpace{2d, 3d}` 任意次的缩放单项式空间
    - `HuZhangFiniteElementSapce{2d, 3d}` 任意次的 Hu-Zhang 元空间
    - `RaviartThomasFiniteElementSpace{2d, 3d}` 任意次的 RT 元空间
    - `FirstKindNedelecFiniteElementSpace2d` 二维任意次的棱元空间
    - `ConformingVirtualElementSpace2d` 二维协调 VEM 空间
    - `NonConformingVirtualElementSpace2d` 二维非协调 VEM 空间
    - `WeakGalerkinSpace2d`  二维 WG 空间
    - .....
* 功能强大的数值积分算法模块
    - `FEMeshIntegralAlg`  有限元网格上的积分算法, 同时适用于区间、三角形、 
      四边形、四面体、六面体网格.
    - `PolygonMeshIntegralAlg` 多边形网格上的积分算法. 
    - `UniformTimeLine` 均匀剖分时间离散
    - ......
* 丰富的算例, 见 https://gitlab.com/weihuayi/fealpy/example 


## 安装

```
git clone https://github.com/weihuayi/fealpy
cd fealpy
pip3 install -e .
```


## 基于 FEALPy 完成的成果

目前在湘潭大学和国内的一些高校，已经有不少的老师和学生在基于 
FEALPy 完成自己的学习和科研任务. 下面列出一些已经完成的工作

### 毕业论文
[1] 文利清. 基 于 python 语 言 虚 单 元 法 的 实 现 与 超 收 敛 研 究. 硕士论文, 2017.

[2] 樊旺旺. 基于自适应界面拟合网格求解椭圆界面问题的虚单元法. 硕士论文, 2018.

[3] 王龙娟. 虚单元法的重构型后验误差估计与自适应算法. 硕士论文, 2019.

[4] 龚欣. 一般曲面上晶体相场模型的高阶有限元数值模拟研究. 硕士论文, 2020.

[5] 扈瀚丹. 梯度恢复技术在求解线弹性问题中的应用研究. 硕士论文, 2020.

### 发表论文

[1] K. Jiang, X. Wang, J. Liu, H. Wei. An adaptive high-order surface finite 
element method for the self-consistent field theory on general curved surfaces, 
arXiv preprint arXiv:2106.07405.

[2] H. Wei, X. Wang, C. Chen, K. Jiang. High order numerical simulations for 
the polymer self-consistent field theory using the adaptive virtual element 
and spectral deferred correction methods. arXiv preprint arXiv:2002.08187.

[3] H. Cao, Y. Huang, N. Yi. Adaptive direct discontinuous Galerkin method 
for elliptic equations. Computers and Mathematics with Applications, 
97: 394-415, 2021.

[4] H. Wei, X. Huang*, A. Li. Piecewise Divergence-Free Nonconforming Virtual 
Elements for Stokes Problem in Any Dimensions. 
SIAM Journal on Numerical Analysis.59(3):1835-1856, 2021.

[5] Y. Huang, H. Wei, W. Yang, and N. Yi*. Recovery based finite element method 
for biharmonic equation in 2d. Journal of Computational Mathematics, 
38(1): 84, 2020.

[6] Y. Deng, F. Wang*, H. Wei. A posteriori error estimates of virtual element 
method for a simplified friction problem. Journal of Scientific Computing, 
83:1-20, 2020

[7] F Wang, H Wei. Virtual element methods for the obstacle problem. 
IMA Journal of  Numerical Analysis, 40(1):708-728, 2020.

[8] F. Wang, H. Wei. Virtual element method for simplified friction problem. Applied Mathematics Letters, 2018.

## 致谢 

感谢在湘潭大学求学时的导师陈艳萍教授和黄云清教授, 
感谢他们一直为我的学习成长提供的引导、支持和帮助. 

感谢我在美国加州大学欧文分校的导师陈龙教授, 感谢他在科研道路上对我一直悉心指导, 
而他开发的 MATLAB 有限元软件包 [iFEM](https://github.com/lyc102/ifem) 
是我开发 FEALPy 的灵感源泉.

感谢北京大学的李若教授, 是他用 [AFEPack](https://github.com/wangheyu/AFEPack) 
引导我在硕士阶段就走进了有限元编程的多彩世界. 

感谢鄂维南院士、深势科技的张林峰博士、以及整个 [DeepModeling](https://github.com/deepmodeling) 
社区的支持, 让自己心中的那个目标更加清晰明确, 也更有力量和信心继续走下去,
并深度参与"共同定义科学计算未来". 

感谢为 FEALPy 做出贡献的老师和学生, 黄学海(上海财经大学)、蒋凯(湘潭大学)、
王飞(西安交通大学)、曹书豪(Washington University)、
周炜恩(军科院国防科技创新研究院)、张永超(西北大学)、吴超(湖南科技大学)、
陈伟(北京大学)、彭辉(吉林大学)、杨迪(西安交通大学)、曹慧慧(湘潭大学)、
杨俊(湘潭大学)、王唯(湘潭大学)、谢玮(湘潭大学)等等.

感谢我的学生许明、 王龙娟、 刘江刚、 李奥、 王鑫、陈春雨、田甜、
王鹏祥、 王栋、梁一茹和高婷艺.


## FEALPy 的未来期待更多优秀人的参与

如果你对 FEALPy 感兴趣, 欢迎加入进来, 让我们一起共同推动 FEALPy 的发展, 让它变的更加强大,
从而可以帮助到更多的人.

FEALPy 的发展需要各种形式的外部驱动:

* 也许你想完成一篇毕业论文
* 也许你想为一篇待发表的论文增加一个数值实验, 
* 也许你想为验证一个项目的想法可行性, 
* 也许你想和我建立深度合作的关系,
* 也许你想报考我的研究生,
* 也许你想学好有限元编程,
* ......

都欢迎随时联系我, 期待优秀的你通过 FEALPy 和我产生链接!








