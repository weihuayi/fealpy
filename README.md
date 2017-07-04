# FEALPy: Finite Element Analysis Library in Python

## Goal

github.com/weihuayi/fealpy

We want to develop an efficient and easy to use finite element software
package to support our teach and research work. 

We still have lot work to do. 


## Dependency

```
sudo apt-get install python3 python3-pip ipython3 
sudo apt-get install python3-numpy
sudo apt-get install python3-scipy
sudo apt-get install python3-matplotlib
sudo -H pip3 install spyder
```

```
sudo apt-get install python3-skimage
```

## Option Dependency 

Install mesh generation package `meshpy`:

```
sudo -H pip3 install meshpy
```

Install "metis"
```
sudo apt-get install metis
```

```
sudo apt-get install python3-numexpr 
```

The following can install by pip3

```
sudo -H pip3 install pyamg
sudo -H pip3 install pycuda
sudo -H pip3 install mpi4py
sudo -H pip3 install pymetis
sudo -H pip3 install PySPH
sudo -H pip3 install nipy
```

## Install VTK Wrapped by Python3

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
4. When configure completed, input `test` in `serach` input box, uncheck
   `BUILD_TESTING`. If you check this argument, vtk will dowload many data
   which is very slow. 
4. Then, replace keyword `test` by `python` in `search` input box
    * Set `PYTHON_EXECUTABLE` as `/usr/bin/python3.5`
    * Set `VTK_PYTHON_VERSION` as `3.5`
    * Check `VTK_WRAP_PYTHON` 
5. Press `Configure` button again
6. When configure completed, press `Generate` button
7. When generate process completed, just close the GUI of cmake

Finally, we can compile VTK by make command

```
make -j8 # parallel compile with 8 thread (My laptop have 8 core)
sudo make install # it will intsall vtk into /usr/local/
```

If you want to install vtk in other path, for example `~/software/vtk/7.1.0`,
you need to set `CMAKE_INSTALL_PREFIX` as `~/software/vtk/7.1.0` in the cmake
configure step. But you need do more setting in `~/.bashrc` file. For my case, I
put the following lines into my bashrc file:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/why/software/vtk/7.1.0/lib
export PYTHONPATH=$PYTHONPATH:/home/why/software/vtk/7.1.0/lib/python3.5/site-packages
```
After you update your bashrc file, please source it:

```
source ~/.bashrc
```
Then you can use vtk in your python 3 program or `ipython3`, just:

```
import vtk
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

## Anaconda

Download the latest Anaconda 

```
conda create -n py35_qt4 python=3.5 pyqt=4
conda 
conda install -c menpo mayavi=4.5.0
conda env remove --name py35_qt4
```

## Reference

* https://plot.ly/python/
* http://www.math.uci.edu/~chenlong/programming.html
