---
title: FEALPy 安装 
permalink: /docs/zh/start/fealpy-install
key: docs-fealpy-install-zh
---


# 源文件安装

目前，首推的安装方式, 是直接从 [GitHub](https://github.com/weihuayi/fealpy.git) 克隆安装

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



# Windows安装

Windows 下 matplotlib 画图如果要支持 tex 公式的显示，需要系统提前安装好 LaTex 软件，并确保 LaTex， dvipng 和 ghostscript 三个可执行程序在系统默认路径下，可执行程序的默认搜索路径由环境变量 PATH 决定，如果上面三个程序所在的路径没有加入 PATH，需要手动增加， 

1. 下载最新的[Anaconda for Windows](https://www.anaconda.com/distribution/).

2. 下载最新的 [Git for Windows](https://gitforwindows.org/).

3. 开始菜单中在 Anaconda 菜单目录下打开 Anaconda PowerShell Prompt 命令行终端, 注 意它会默认进入你的 Desktop 的上一级目录. 然后配置你自己的 Git 信息.

   ```bash
   > git config --global user.name "<Your Name>"
   > git config --global user.email "<Your Email>"
   ```
   在有些 Windows 系统中，可能默认进入的是管理员的目录，就没有 Desktop 目录。这样的 情况下，你可以找一下自己的 Desktop 目录， 然后再执行下面的安装步骤。当然， fealpy 最终 clone 到系统的任何地方都是可以的，你觉得方便就行。

4. 进入 Desktop 目录, 然后从github上克隆最新的 FEALPy

  ```bash
  > cd Desktop # 进入主目录
  > git clone https://github.com/weihuayi/fealpy.git
  ```

5. 进入 FEALPy, 运行下面的命令安装:

   ```bash
   > cd fealpy
   > pip install -e .
   ```

   这样的安装是以符号链接的方式安装到系统当中。这样的好处是， 你进入 fealpy 目录 后，用 git pull 更新 FEALPy，不用重新安装，即可以在系统的任意位置调用最新的版本.

6. 最后, 你在开始目录中可以打开 Python 集成开发环境 Spyder, 在 Ipython 中 导入测试一下， 如果没有报错就说明安装成功了

   ```bash
   In [1]: from fealpy.mesh import TriangleMesh
   In [2]:
   ```
