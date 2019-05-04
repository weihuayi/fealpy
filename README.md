# FEALPy: Finite Element Analysis Library in Python

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
$ sudo apt install ipython3       # An enhanced Interactive Python  
```
2. clone the latest fealpy from github:
```
$ git clone https://github.com/weihuayi/fealpy.git
```
3. In `fealpy/`, run the following command: 
```
$ sudo -H pip3 install ./
```
which will copy the fealpy into `/usr/local/lib/python3.6/dist-packages/`.  Or run the following command:
```
$ sudo -H pip3 install -e ./
```
which will create a soft link in `/usr/local/lib/python3.6/dist-packages/`.

4. You can open `ipython3` from bash shell to test fealpy.

5. If you want to upgrade numpy 

```
sudo -H pip3 --upgrade numpy
```


## Windows: Anaconda

1. Dowload visual studio IDE for windows (community 2017) from https://visualstudio.microsoft.com/ and 
   install it on you win system. Notice that you just need to install the c++ component. 

2. Download anaconda for windows from https://repo.continuum.io/archive/Anaconda3-2018.12-Windows-x86_64.exe and install it
   on your system.

3. Download the correct version `meshpy` from https://www.lfd.uci.edu/~gohlke/pythonlibs/#meshpy. For example:

```
pip install MeshPy‑2018.2.1‑cp37‑cp37m‑win_amd64.whl # For 64 bit win system with python 3.7
```

4. Download the correct version `pyamg` from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyamg. For example:

```
pip install pyamg‑4.0.0‑cp37‑cp37m‑win_amd64.whl # For 64 bit win system with python 3.7
```

5. enter fealpy directory, run the following command:

```
pip install -e ./
```

6. You can open IPython from the start menu of win system to test fealpy.

## Mac


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
