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
2. clone the latest fealpy from github:
```
$ git clone https://github.com/weihuayi/fealpy.git
```
3. In `fealpy/`, run the following command: 
```
$ python3 setup_linux.py install --prefix=~/.local/
```
which will copy the fealpy into `~/.local/lib/python3.6/dist-packages/`.  Or run the following command:
```
$ python3 setup_linux.py develop --prefix=~/.local/
```
which will create a soft link in `~/.local/lib/python3.6/dist-packages/`.



## Windows: Anaconda

1. Download and install latest Anaconda for Windows. https://www.anaconda.com/distribution/
2. Download and insall latest Git for Windows, https://gitforwindows.org/
3. Add the Anaconda directory into $PATH env variable, maybe reboot the windows.
4. Open the git bash, clone the latest fealpy
```
$ echo ". /c/Users/<uasername>/Anaconda3/etc/profile.d/conda.sh" >> .bashrc
$ cd Desktop
$ mkdir git # create a directory named git
$ cd git # enter git directory
$ git clone https://github.com/weihuayi/fealpy.git # clone the fealpy repo
```
5. enter fealpy directory, run the following command:
```
> python setup_win.py develop 
> pip　install https://download.lfd.uci.edu/pythonlibs/s2jqpv5t/MeshPy-2018.2.1-cp37-cp37m-win_amd64.whl
> pip install https://download.lfd.uci.edu/pythonlibs/s2jqpv5t/pyamg-4.0.0-cp37-cp37m-win_amd64.whl
```

## Mac
1. Download and install latest anaconda for macOS
2. Enter fealpy directory, run the following command:
```
$ python3 setup_mac.py develop
$ pip　install https://download.lfd.uci.edu/pythonlibs/s2jqpv5t/MeshPy-2018.2.1-cp37-cp37m-win_amd64.whl
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
