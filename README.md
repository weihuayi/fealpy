# FEALPy: Finite Element Analysis Library in Python

[![Join the chat at https://gitter.im/weihuayi/fealpy](https://badges.gitter.im/weihuayi/fealpy.svg)](https://gitter.im/weihuayi/fealpy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
![Python package](https://github.com/weihuayi/fealpy/workflows/Python%20package/badge.svg)
![Upload Python Package](https://github.com/weihuayi/fealpy/workflows/Upload%20Python%20Package/badge.svg)

We want to develop an efficient and easy to use finite element software
package to support our teach and research work. 

We still have lot work to do. 

关于 FEALPy 的中文帮助与安装信息请查看：
[FEALPy 帮助与安装](https://www.weihuayi.cn/fealpy/fealpy.html)

# Installation

## Common


Use `pip install -U fealpy` to install the latest release from PyPi.

Users in China can install FEALPy from mirrors such as:
- [Aliyun](https://developer.aliyun.com/mirror/pypi)
- [Tsinghua](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- [USTC](https://lug.ustc.edu.cn/wiki/mirrors/help/pypi)

## From Source

```
git clone https://github.com/weihuayi/fealpy.git
cd fealpy
pip install .
```

For developers, please use `pip install -e .[dev]` to install it in develop mode.

On Ubuntu system, maybe you should use `sudo -H pip3 install -e .[dev]` to install it in
develop mode.

## Uninstallation

`pip uninstall fealpy`

## Docker

To be added.

## Reference

* http://www.math.uci.edu/~chenlong/programming.html


## Please cite fealpy if you use it in you paper

H. Wei and Y. Huang, FEALPy: Finite Element Analysis Library in Python, https://github.com/weihuayi/fealpy, Xiangtan University, 2017-2021.
