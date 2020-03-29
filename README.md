# FEALPy: Finite Element Analysis Library in Python

[![Join the chat at https://gitter.im/weihuayi/fealpy](https://badges.gitter.im/weihuayi/fealpy.svg)](https://gitter.im/weihuayi/fealpy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

We want to develop an efficient and easy to use finite element software
package to support our teach and research work. 

We still have lot work to do. 

# Install

## Ubuntu

1. please install the python envieronment. 
```
$ sudo apt install git            # The version control tool
$ sudo apt install python3        # The python3 interpretor 
$ sudo apt install python3-pip    # The PyPA recommended tool for installing Python packages.
$ sudo apt install python3-tk     # Python interface to Tcl/Tk used by matplotlib 
```
2. clone the latest fealpy from github or gitlab:
```
$ git clone https://github.com/weihuayi/fealpy.git
```
or
```
$ git clone https://gitlab.com/weihuayi/fealpy.git
```
3. In `fealpy/`, run the following command: 
```
$ python3 setup_linux.py install --prefix=~/.local
```
which will copy the fealpy into `~/.local/lib/python3.<x>/site-packages/`.  Or run the following command:
```
$ python3 setup_linux.py develop --prefix=~/.local
```
which will create a soft link in `~/.local/lib/python3.<x>/site-packages/`.
4. 
```
$ pip3 install -U vtk
$ pip3 install -U meshio
```



## Windows: Anaconda

1. Download and install latest Anaconda for Windows. https://www.anaconda.com/distribution/
2. Download and insall latest Git for Windows, https://gitforwindows.org/
3. Open Anaconda Powershell Prompt (Anaconda3) from start menu, and config your name and email information for git. 
```
> git config --global user.name "Your Name"
> git config --global user.email "Your Email"
```
4. Then clone the latest fealpy
```
> cd Desktop
> mkdir git # create a directory named git. Of course, you can name it by another name you like.
> cd git # enter git directory
> git clone https://github.com/weihuayi/fealpy.git # clone the fealpy repo
```
or
```
> git clone https://gitlab.com/weihuayi/fealpy.git # clone the fealpy repo
```
5. enter fealpy directory, run the following command:
```
> python setup_win.py develop 
> pip install https://download.lfd.uci.edu/pythonlibs/s2jqpv5t/MeshPy-2018.2.1-cp37-cp37m-win_amd64.whl
> pip install https://download.lfd.uci.edu/pythonlibs/s2jqpv5t/pyamg-4.0.0-cp37-cp37m-win_amd64.whl
> conda install -c anaconda vtk 
```
Notice that, the above `whl` files are build for python3.7, you can find suitable
version for other python version.

## Mac
1. Download and install latest Anaconda for macOS. https://www.anaconda.com/distribution/
2. Download and insall latest Git for Mac, https://git-scm.com/download/mac.
3. Open command terminal and config your name and email information for git. 
```
$ git config --global user.name "Your Name"
$ git config --global user.email "Your Email"
```
4. Then clone the latest fealpy
```
$ cd ~ # enter your home directory 
$ mkdir git # create a directory named git. Of course, you can name it by another name you like.
$ cd git # enter git directory
$ git clone https://github.com/weihuayi/fealpy.git # clone the fealpy repo
```
or
```
$ git clone https://gitlab.com/weihuayi/fealpy.git # clone the fealpy repo
```

5. enter fealpy directory, run the following command:
```
$ python setup_mac.py develop 
$ conda install -c anaconda vtk
```

# Debug in python 

```
sudo apt-get install python3-dbg
```

Debug python program:

```
$ gdb python3
...
(gdb) run <programname>.py <arguments>
```

## Reference

* https://plot.ly/python/
* http://www.math.uci.edu/~chenlong/programming.html


## Please cite fealpy if you use it in you paper

H. Wei, FEALPy: Finite Element Analysis Library in Python, https://github.com/weihuayi/fealpy, 2017-2020.
