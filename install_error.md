## 环境 ubuntu 16.04 安装 meshpy pip3 出错的解决办法.


解决第一个问题，AttributeError: '_NamespacePath' object has no attribute 'sort', 首先确定这个问题是 pip 出错. 解决办法

第一步, 首先卸载之前安装的 pip/pip3
```
1. sudo apt-get remove python3-pip
2. sudo apt-get remove python3-pip3
3. sudo apt autoremove
4. sudo apt autoclean
5. sudo apt autoremove

```
第二步

```
sudo apt-get install curl
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo -H python3
 
解决第二个问题, pip 安装好了, 重新安装 meshpy 会遇到第二个问题

sudo -H pip3 install --upgrade pip
sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install meshpy
```


## 环境 ubuntu 18.04 安装 meshpy 出错的问题, 需要安装　libboost-python-dev 的库

sudo apt-get install libboost-python-dev
sudo -H pip3 install meshpy
