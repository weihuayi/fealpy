# WHYSC: WeiHuaYi's Scientific Computing Package in C++

I have developed a Python finite element package FEALPy, but I found that I also need
an additional package dedicated to my C++ scientific computing program.


# 开发的基本原则

* 易扩展
* 易组合
* 模块之间要尽量独立
* 分层

# 采用的技术：

* 面向对象
* 模板
* 尽量只采用标准的 CPP 库
* 避免过度设计 !!

# 命名规则

* NameSpaceExample
* Class_name_example
* class_member_function
* varNameExample
* Class_name_example.h
* package_interface.h


* is_empty

**模板参数**

* I    : Int type
* F    : Float type
* DIM  : dimentsion
* TDIM : toplogy dimension
* GDIM : geometry dimention
* Container : 

**成员变量**

* gdim : the geometry dimension
* tdim : the toplogy dimension

**常用变量

* cell2node or  c2n
* cell2edge or  c2e
* cell2face or  c2f
* face2node or  f2n
* i, j, k, m, n : ofen define in for loop
* NN : the total number of nodes on a mesh
* NE : the total number of edges on a mesh
* NF : the total number of faces on a mesh
* NC : the total number of cells on a mesh
* nn : the local number of nodes on a cell
* ne : the local number of edges on a cell
* nf : the local number of faces on a cell
