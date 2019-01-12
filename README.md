# FEALPy: Finite Element Analysis Library in Python

We want to develop an efficient and easy to use finite element software
package to support our teach and research work. 

We still have lot work to do. 

# Install

## Ubuntu

### Install by pip 

In `fealpy/`, run the following command 
```
sudo pip3 install ./
```
whill copy the fealpy into `/usr/local/lib/python3.x/dist-packages/`



In the `fealpy/`, run the following command:

```
sudo pip3 install -e ./
```
which will create a link in `/usr/local/lib/python3.x/dist-packages/`.  


### Dependency

```
sudo apt install python3 python3-pip ipython3 
sudo apt install python3-numpy
sudo apt install python3-scipy
sudo -H pip3 install matplotlib
```

```
sudo apt install petsc-dev
sudo -H pip3 install petsc4py 
```

### Option Dependency 

Install mesh generation package `meshpy`:

ububtu 18.04 install meshpy

```
sudo apt install libboost-python-dev
sudo -H pip3 install meshpy
```

Install "metis"
```
sudo apt install metis
```

```
sudo apt install python3-numexpr 
```

The following can install by pip3

```
sudo -H pip3 install pyamg
sudo -H pip3 install pycuda
sudo -H pip3 install mpi4py
sudo -H pip3 install pymetis
sudo -H pip3 install PySPH
sudo -H pip3 install nipy
sudo -H pip3 install pygmsh
```


## Windows: Anaconda

Dowload anaconda for windows  from https://www.anaconda.com/. 

Run `Anaconda3-5.3.1-Windows-x86_64.exe` and install Anaconda on you system.

Dowload the correct version `meshpy` from https://www.lfd.uci.edu/~gohlke/pythonlibs/#meshpy. For example:

```
MeshPy‑2018.1‑cp36‑cp37m‑win_amd64.whl # for python 3.7 and 64 bit win system.
```

```
pip install MeshPy‑2018.1‑cp36‑cp36m‑win_amd64.whl
```

```
conda install pyamg # amg fast solver
conda install -c conda-forge pytools
```

```
pip install -e ./
```

## Mac


## Install Boost with python3 


Dowload the latest boost from http://www.boost.org/. 

```
./bootstrap.sh --with-python=python3
./b2
```

Here we install boost into `/usr/local` directory. 

## Install VTK Wrapped by Python3 and Mayavi

We also use VTK Python 3 wrapper, one can clone VTK from `github` first:

```
git clone https://github.com/Kitware/VTK.git
cd VTK
```

Then check out the latest stable version:

```
git archive v7.1.0 --prefix=vtk7.1.0/ --format=zip  > ../vtk7.1.0.zip
```

Unzip the `vtk7.1.0.zip`, then cd the source code file:
```
cd ..
unzip vtk7.1.0.zip
cd vtk7.1.0/
```

Create a `build` directory, then cd into it and run `cmake-gui ..`
```
mkdir build
cd build/
cmake-gui ..
```
You can see the GUI of cmake, then:

1. check the `Advanced` option
2. Press `Configure` button, and a dialog box will pop up
3. Press the `Finish` button on the dialog box 
4. When configure completed, input `test` in `search` input box, uncheck
   `BUILD_TESTING`. If you check this argument, vtk will dowload many data
   which is very slow. 
4. Then, replace keyword `test` by `python` in `search` input box
    * Set `PYTHON_EXECUTABLE` as `/usr/bin/python3.5`
    * Set `VTK_PYTHON_VERSION` as `3.5`
    * Check `VTK_WRAP_PYTHON` 
5. Press `Configure` button again
6. When configure completed, press `Generate` button
7. When generate process completed, just close the GUI of cmake

Finally, we can compile VTK by `make` command

```
make -j8 # parallel compile with 8 thread (My laptop have 8 core)
sudo make install # it will intsall vtk into /usr/local/
```

Then one should append  the following lines into your bashrc file:
```
# if you install vtk into /usr/local
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.5/site-packages
```

```
# if you install vtk into /home/why/software/vtk/7.1.0/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/why/software/vtk/7.1.0/lib
export PYTHONPATH=$PYTHONPATH:/home/why/software/vtk/7.1.0/lib/python3.5/site-packages
```

After you update your bashrc file, please source it:

```
source ~/.bashrc
```
Then you can use vtk in your python 3 program or `ipython3`,  just:

```
import vtk
```

At last, one can intall `mayavi`, but one should change into root user first: 
```
$ sudo -s
# export PYTHONPATH=$PYTHONPATH:/home/why/software/vtk/7.1.0/lib/python3.5/site-packages
# pip3 install mayavi
```


## Debug python 

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
