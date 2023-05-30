# MGIS安装指南


MGIS
---
1. 克隆 MGIS
git clone git@github.com:suanhaitech/MFrontGenericInterfaceSupport.git
cd MFrontGenericInterfaceSupport
2. 构建安装
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -Denable-c-bindings=ON -Denable-fortran-bindings=ON  -Denable-python-bindings=ON
make -j8
sudo make install
---
例子
---
1. 编译 .mfront 文件
mfront --obuild --interface=generic saint_venant_kirchhoff.mfront
注意：在编译过程中，可能出现无法找到共享库的问题，即
mfront: error while loading shared libraries: libTFELMFront.so.4.2.0-dev: cannot open shared object file: No such file or directory
解决方法如下：
  1. 检查系统默认的库路径中是否有 libTFELMFront.so.4.2.0-dev 文件
ls /usr/local/lib | grep libTFELMFront.so.4.2.0-dev
若有，显示结果如下
libTFELMFront.so.4.2.0-dev

  2. 更新动态链接器的缓存，使得动态链接器能够知道新的库文件的存在或者已删除的库文件的缺失
sudo ldconfig
---
2. Python 导入 mgis 库
from mgis import behaviour as mg
注意：在调用过程中，可能出现无法导入 mgis 模块的问题，解决方法如下：
  1. 查找 MGIS Python 绑定的安装位置
find / -name 'mgis' 2>/dev/null
上面的命令会显示：
/usr/local/lib/python3.10/site-packages/mgis
/usr/local/share/mgis
/home/heliang/MFront/MFrontGenericInterfaceSupport/build/src/CMakeFiles/Export/share/mgis
/home/heliang/MFront/MFrontGenericInterfaceSupport/build/bindings/python/mgis
/home/heliang/MFront/MFrontGenericInterfaceSupport/build/bindings/c/src/CMakeFiles/Export/share/mgis
/home/heliang/MFront/MFrontGenericInterfaceSupport/build/bindings/fortran/src/CMakeFiles/Export/share/mgis
/home/heliang/MFront/MFrontGenericInterfaceSupport/bindings/python/mgis
  2. 将此路径添加到 PYTHONPATH 环境变量中
如果你使用的是 bash，将export 命令添加到你的 ~/.bashrc 文件中
vim ~/.bashrc
在将下面的代码添加到文件的最后一行
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/site-packages/
