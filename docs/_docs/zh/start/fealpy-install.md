---
title: FEALPy 的安装 
permalink: /docs/zh/start/fealpy-install
key: docs-fealpy-install-zh
---


# 源文件安装

目前，首推的安装方式, 是直接从 [GitLab](https://gitlab.com/weihuayi/fealpy.git) 
或者 [GitHub](https://github.com/weihuayi/fealpy.git) 克隆安装

```bash
$ git clone https://github.com/weihuayi/fealpy.git
$ cd fealpy
$ pip install . # 复制安装
# or
$ pip install -e . # 符号链接安装，即开发者模式
```

**注意：**

1. `-e` 参数指定安装方式为**开发者模式**, 本质上是直接在系统的默认搜索路径里建
立一个符号链接到 FEALPy 源文件目录下。这样只要每次在 FEALPy 源文件目录下用 `git pull` 
更新，用户在其它地方就可以使用最新的 FEALPy.
1. 安装命令中的 `.` 表示当前目录.
1. Ubuntu 系统下， 安装命令中的 `pip`， 可能要换成 `pip3`.


# Pypi 安装

最直接的安装方式是从 [PyPi](https://pypi.org/project/fealpy/) 安装, 

```bash
pip install -U fealpy
```

但要注意，这样安装的 FEALPy 很可能不是最新版本.

