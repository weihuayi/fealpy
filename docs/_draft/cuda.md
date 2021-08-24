# Ubuntu 系统中 CUDA 的安装

安装 Nvidia 显卡驱动
```bash
$ sudo ubuntu-drivers autoinstall
```


安装 cupy 
```
pip3 install cupy-cuda114
python3 -m cupyx.tools.install_library --cuda 11.4 --library cutensor
```
