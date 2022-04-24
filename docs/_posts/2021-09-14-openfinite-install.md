---
title: OpenFinite 安装
tags: OpenFinite
author: ccy
---

## OpenFinite 简介
[OpenFinite](https://gitlab.com/weihuayi/openfinite) 是一个 C++ 科学计算软件库,
致力于将计算数学中的高效算法应用到实际工业应用中, 现在已包含各种常见网格,
并可以对网格优化.

## 安装
**Openfinite 仓库**
```bash
git clone https://gitlab.com/weihuayi/openfinite.git
```
**安装依赖**
```bash
sudo apt install libgmp-dev libmpfr-dev libboost-dev 
sudo apt install zlib1g-dev libblas-dev liblapack-dev
sudo apt install libglu1-mesa-dev freeglut3-dev mesa-common-dev
sudo apt install freeglut3-dev
```

**安装 ParMetis** 
```bash
sudo apt install libparmetis-dev libmetis-dev
```

**安装 MPI**
```bash
sudo apt install libopenmpi-dev
```

**安装 VTK**

[下载vtk](https://vtk.org/download/)
```bash
mkdir ~/opt
tar -xvf VTK-9.0.1.tar.gz ~/opt/vtk
cd ~/opt/vtk
mkdir -p build/release
cd build/release
cmake -DCMAKE_BUILD_TYPE=Release -DVTK_WRAP_PYTHON=1 -DVTK_PYTHON_VERSION=3 -DBUILD_TESTING=0 -DCMAKE_INSTALL_PREFIX=~/.local/vtk ../..
make -j8
make install
```

**安装 cgal**

[下载cgal](https://github.com/CGAL/cgal/archive/refs/tags/v5.3.tar.gz)
```bash
tar -xvf CGAL-5.2.tar.gz ~/opt/cgal
cd ~/opt/cgal
mkdir -p build/release
cd build/release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/.local/cgal ../..
make -j8
make install
```

**安装 gmsh**
```bash
pip3 install --upgrade gmsh
```

**安装 OpenBLAS**
```bash
cd ~/opt
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS/
make install PREFIX=~/.local/openblas
```
