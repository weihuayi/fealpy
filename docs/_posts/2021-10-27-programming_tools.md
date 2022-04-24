---
title: C++ 程序调试工具
tags: gdb valgrind
author: ccy
---
# 

## 一、简介
本文简单介绍 C++ 程序运行原理及 C++ 程序调试工具.

## 二、Debug vs Release
Debug 和 Release 是 cmake 中两种不同的程序构建方式: 
- Debug 模式下没有代码优化, 会添加调试信息, 以便于程序调试(如 gdb 调试),
    等价于编译时添加选项 `-O0, -g`.
- Release 模式下启用高等级代码优化, 但是不添加调试信息, 等价于编译时添加选
    项 `-O3, -DNDEBUG`.

[(参考)](https://stackoverflow.com/questions/48754619/what-are-cmake-build-type-debug-release-relwithdebinfo-and-minsizerel/48755129).
## 三、编译优化选项 `-O*` 的区别
编译时开启优化可以提高程序执行效率和程序大小, 但是会增加编译时间. 常用编译优化
选项有 `-O0, -O1, -O2, -O3, -Og` 等, 其中不同选项是开启了不同的优化类型, `-O0`
是没有优化, `-O1` 到 `-O3` 是逐步开启了更多的优化类型, 如 `-O3` 相比与 `-O2`
多开启了
```
-fgcse-after-reload
-fipa-cp-clone
-floop-interchange
-floop-unroll-and-jam
-fpeel-loops
-fpredictive-commoning
-fsplit-loops
-fsplit-paths
-ftree-loop-distribution
-ftree-partial-pre
-funswitch-loops
-fvect-cost-model=dynamic
-fversion-loops-for-strides
```
上面的这些选项也可以单独使用.

在[这里](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)可以找到更多相关知识.

## 四、gdb 调试程序
程序调试是编程中的重要一环, 使用 gdb
可以简单的让程序在自己想要的位置停止并查看当前所有变量的值是否符合预期,
快速找到程序的出错的地方.

### 1. 安装
Ubuntu 中自带.

### 2. 简单使用
在 Debug 模式下编译 C++ 程序, 在终端命令框使用

```bash
gdb 调试文件名
```

即可对程序进行调试. 在调试过程中, 使用 
- `run argv1 argv2` 可以运行程序.
- `b k` 可以在当前文件中的第 $k$ 行设置断点, 当程序运行到这个位置时会停止.
- `c` 可以让程序继续运行到下一个断点.
- `print 变量名` 可以打印出变量的信息.
- `set var 变量名 = XXX` 可以修改程序中的变量的值.
- `n` 可以运行当前行程序.
- `s` 可以进入当前行程序调用的函数中.

### 3. 程序调用方式
C++ 程序运行时, 执行函数操作是在 **栈** 每个函数会在栈中保存自己的局部变量,
一个函数的执行环境(个人理解就是给每个函数开辟的那一块内存)就是一个 **栈帧**, 
在程序运行时会调用多个函数, 在 gdb 中可以使用
```
backtrace
```
查看代码当前位置所在的栈帧. 使用
```
frame num
```
查看 `num` 号栈帧的信息.

[参考](http://c.biancheng.net/view/8282.html)

### 4. mpi 程序的 gdb 调试
gdb 也可以调试并行程序, 需要用到一种特殊的方法: 在要调试的程序的开头添加:
```c++
  volatile int i = 0;
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("PID %d on %s ready for attach\n", getpid(), hostname);
  fflush(stdout);
  while (0 == i)
      sleep(5);
```
编译运行程序, 程序会打印出自己的进程号, 在终端打开 gdb

```bash
gdb
```
在 gdb 中使用 
```
attach pid 
```
即可进入运行的程序中, 使用 `n` 操作使程序进入到 
```c++
fflush(stdout)
```
行, 在 gdb 中使用
```
set var i = 10
```
即可正常调试程序.

[gdb 并行程序调试官方文档 #6 ](https://www.open-mpi.org/faq/?category=debugging#parallel-debuggers)

## 五、valgrind 测试程序内存使用错误
### 1. 简介
valgrind 是一个非常好用的内存使用错误检测软件.
### 2. 安装
```bash
sudo snap install valgrind
```
### 3. 使用
将自己写的程序编译以后在终端使用
```bash
valgrind <程序名> <参数1> <参数2> ...
```
即可对程序进行检测, 会返回程序中的内存使用错误, 如内存泄漏, 内存未初始化等.

[valgrind 官方使用文档](https://valgrind.org/docs/manual/mc-manual.html#mc-manual.mpiwrap)
### 4. 并行程序检测

## 六、总结
每次完成程序以后必须经过 valgrind 检测. 若出现错误, 灵活使用 gdb 与 valgrind
查出问题所在, 两个软件都不能找到错误时使用原始的打印信息的方法找错误.




