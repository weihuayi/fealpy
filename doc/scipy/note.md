# Why Python

## 科学家的需求

* 获得数据
* 操作和处理数据
* 可视化数据
* 交流结果: 生成图片- produce figures for reports or publications, write
    presentations

## 要求

* 充分利用已有代码
* 容易学习
* 容易与合作者, 学生, 客户交流
    + 易读
* 高效的编写代码和执行代码, 花的时间尽量少
* 单一的语言环境

## 已有的解决方案

* C, C++, Fortran
    - 很快, 优化的编译器, 大量的计算很难被其它语言超过
    - 用起来很痛苦, 没有互动,编译的过程, 复杂的语法, 人工的内存管理
* Matlab
    - 丰富的算法, 快速执行, 友好的开发环境, 商业支持
    - Base language is quite poor and can become restrictive for advanced
        users, no free
* Other scripting languages: SciLab, Octave, Igor, R, IDL, etc.
    - open-source, free, at least cheaper than matlab
    - some features can be very advanced 
    - Fewer algorithm, language is note more advanced
    - Restrited to a single type of usage

What about Python?

* Advantages:
    - Very rich scientific computing libraries
    - Well thought ourt language
    - Many libraries for other taskd than scientific computing
    - Free an open-source, widely spread, with a vibrant community
* Drawbacks:
    - less pleasant development environment thant matlab
    - Not all th algorithms that can found in more specialized software or
        toolboxes

From a script to functions
* A script is not reusable, functions are.
* Thinking in terms of functions helps breaking the problem in small bocks


* cpaste
* debug
* alias
* tab


# The Python Language

* interpreted language, interactively using
* a free software
* multi-platform
* a very readable language with clear non-verbose snytax
* a lage variety of high-quality packages are available for various
    applications
* a language very easy to interface with other languages, in particular C and
    C++
* Objected-oriented
* dynamic typing


```
>>> 'An integer: %i ; a float: %f ; another string: %s ' % (1, 0.1, 'string')
'An integer: 1; a float: 0.100000; another string: string'
>>> i = 102 >>> filename = 'processing_of_dataset_%d .txt' % i >>> filename
'processing_of_dataset_102.txt'
```

## List Comprehensions
