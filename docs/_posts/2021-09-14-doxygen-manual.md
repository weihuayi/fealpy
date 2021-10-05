---
title: Doxygen 使用
tags: doxygen
author: ccy
---

本文简单介绍 linux 下 doxygen 的安装和注释规则.

## 简介

doxygen 是一个程序化生成代码文档的工具, 适用于 `C++`, `Python` 等语言.
按照 doxygen 的格式给程序写注释, 就可以通过 doxygen 处理为详细的代码文档.

## 安装

linux

```bash
sudo apt install graphviz
sudo apt install doxygen
```

## doxygen 注释规则

Doxygen 可以用于 C++ 和 Python 等语言的程序, 只会处理符合规则的注释.

### C++ 注释规则

- 多行注释
    ```c++
    /** 注释内容
     *
     */
    ```
- 单行注释
    ```c++
    /** 注释内容 */
    ```
**注意:** Doxygen 默认注释是说明下面的代码, 如:
    ```c++
    /**  两个整数相加
     *
     */
    int add(int a, int b)
    {
        return a+b;
    }
    ```
    若要注释上面的代码用

    ```c++
    /**< 注释内容 */
    ```
    如:

    ```c++
    int a = 3; /**< 将 a 设为 3 */
    ```
### Python 注释规则

- 简单注释
    ```python
    '''! 注释内容'''
    ```

### 常见注释标记

在注释内容中通过加入一些标记来规范的写注释.

1. 文件信息
    - `\file` 文件名
    - `\date` 日期
    - `\author` 作者 
    - `\brief` 文件或函数的摘要
    - `\todo` TODO
2. 函数信息
    - `\param[int]` 输入参数的说明
    - `\param[out]` 输出参数的说明
    - `\param[in, out]` 即输入又输出参数的说明
    - `\return` 返回值说明
    - `\note` 注解, 用来描述函数的流程或注疑事项
    - `\code` 示例代码开始
    - `\endcode` 示例代码结束

3. 特殊标记
    - `\f[` 输入 latex 公式
    - `\f]` 结束 latex 公式

以上是 C++ 风格注释, Python 程序中将 `\` 改为 `@` 即可.

## 例子

### C++ 程序例子

```c++
/**
 * \file test_TetrahedronQuadrature.cpp
 * \author Chunyu Chen
 * \date 2021/09/08
 * \brief TetrahedronQuadrature.h 测试文件
 */
#include <array>
#include <vector>
#include <algorithm>
#include <math.h>

#include "TestMacro.h"
#include "geometry/Point_3.h"
#include "geometry/Vector_3.h"
#include "quadrature/TetrahedronQuadrature.h"

typedef WHYSC::GeometryObject::Point_3<double> Point;
typedef WHYSC::Quadrature::TetrahedronQuadrature TetrahedronQuadrature;

/** 
 * \brief 被积函数
 * \param p 函数的参数
 */
double f(const Point & p)
{
  return p[0];//p[0]*p[1];
}

/**
 * \brief 四面体类, 即积分区域
 */
struct Tetrahedron
{
  Point p0, p1, p2, p3; /**< 四面体顶点 */

  /**
   * \brief 四面体面积
   */
  double area()
  {
    auto v1 = p1 - p0;
    auto v2 = p2 - p0;
    auto v3 = p3 - p0;
    return std::abs(dot(cross(v1, v2), v3)/6);
  }
};

/**
 * \brief 积分测试函数
 */
void test_integral(int p = 3, double h = 0.5)
{
  Tetrahedron tet;
  tet.p0 = Point({0, 0, 0});
  tet.p1 = Point({h, 0, 0});
  tet.p2 = Point({0, h, 0});
  tet.p3 = Point({0, 0, h});
  
  TetrahedronQuadrature tetQ(p);
  int NQP = tetQ.number_of_quadrature_points();

  double val = 0.0;
  for(int i = 0; i < NQP; i++)
  {
    auto w = tetQ.quadrature_weight(i);
    auto & qpts = tetQ.quadrature_point(i);
    auto P = qpts[0]*tet.p0 + qpts[1]*tet.p1 + qpts[2]*tet.p2 + qpts[3]*tet.p3; 
    val += f(P)*w;
  }
  val *= tet.area();
  ASSERT_THROW(std::abs(val-std::pow(h, 5)/12.0)<1e-10);/**< 判断积分是否正确 */
}

int main(int args, char *argv[])
{
  int q = 4;
  double h = 0.5;
  if(args>1 && args<3)
     q = std::stoi(argv[1]);
  else if(args>=3)
  {
     q = std::stoi(argv[1]);
     h = std::stoi(argv[2]);
  }
  test_integral(q, h);
}

```

### Python 程序例子

```python

'''!
@file test.py
@author chenchunyu
@date 10/03/2021
@brief 一个测试文件
'''

import numpy as np

class Test:
    '''!
    @brief 测试类
    @todo 完善
    '''
    def __init__(self, a, b):
        '''!
        @brief 初始化函数, 构造对象时调用的函数.
        @param a 第一个参数
        @param b 第二个参数
        '''
        self.a = a
        self.b = b

    def add(self, k, b):
        '''!
        @brief 一个测试函数, 计算 @f[kx+by@f]
        @param k 第一个参数
        @param b 第一个参数
        @return 返回 k*self.a + b*self.b
        '''
        return k*self.a + b*self.b
```

## 文件生成
1. 生成 Doxyfile
```bash
doxygen -g
```

2. 修改设置, 打开 Doxyfile, 设置 
    - `USE_MATHJAX = YES`
    - `GENERATE_LATEX = NO`
    - 文件位置

3. 生成 HTML 文件
```bash
doxygen
```

4. 查看 HTML 文件
```bash
jekyll serve
```







