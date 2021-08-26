# Ubuntu 系统中 CUDA 的安装


## Ubuntu 中 Nvidia 显卡驱动的安装方法

安装 Nvidia 显卡驱动

```bash
$ sudo ubuntu-drivers autoinstall
```

查看驱动的安装情况：

```bash
$ nvidia-smi
Thu Aug 26 17:20:29 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 32%   43C    P8    11W / 180W |    674MiB /  8117MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1293      G   /usr/lib/xorg/Xorg                 96MiB |
|    0   N/A  N/A      1985      G   /usr/lib/xorg/Xorg                384MiB |
|    0   N/A  N/A      2132      G   /usr/bin/gnome-shell               45MiB |
|    0   N/A  N/A      2502      G   ...AAAAAAAAA= --shared-files       95MiB |
|    0   N/A  N/A      3281      G   ...AAAAAAAA== --shared-files       39MiB |
+-----------------------------------------------------------------------------+
```

## CUDA ToolKit 的安装方法 

从[这里](https://developer.nvidia.com/cuda-downloads)下载 CUDA ToolKit,
页面上需要根据你自己的系统情况选择合适的版本，最好选择 run file
的版本，会下载所有的文件，包括驱动。

```
$ wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda_11.4.1_470.57.02_linux.run
$ sudo sh cuda_11.4.1_470.57.02_linux.run
```

如果你已经安装了驱动，注意不要再选择安装 cuda 的驱动。

安装完成后, 在家目录的 .bashrc 中加入

```
PATH=$PATH:/usr/local/cuda-11.4/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/lib64
```

安装 cupy 
```
pip3 install cupy-cuda114
python3 -m cupyx.tools.install_library --cuda 11.4 --library cutensor
```

```
python3 -m cupyx.tools.install_library --cuda 11.4 --library cutensor
```
