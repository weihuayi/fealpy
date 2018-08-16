import matplotlib.pyplot as plt
import numpy as np
import sys

from fealpy.pde.BiharmonicModel2d import SinSinData
from fealpy.fem.BiharmonicFEMModel import BiharmonicFEMModel
from fealpy.tools.show import show_error_table 

from fealpy.tools.show import showmultirate

m = int(sys.argv[1]) 
meshtype = int(sys.argv[2])
rtype = int(sys.argv[3])
if rtype == 1:
    rtype='simple'
elif rtype == 2:
    rtype='inv_area'

print('rtype:', rtype)
sigma = 1

pde = SinSinData()
box = [0, 1, 0, 1]
maxit = 4

Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$\| u - u_h\|$',
         '$\|\\nabla u - \\nabla u_h\|$',
         '$\|\\nabla u_h - G(\\nabla u_h) \|$',
         '$\|\\nabla u - G(\\nabla u_h)\|$',
         '$\|\Delta u - \\nabla\cdot G(\\nabla u_h)\|$',
         '$\|\Delta u -  G(\\nabla\cdot G(\\nabla u_h))\|$',
         '$\|G(\\nabla\cdot G(\\nabla u_h)) - \\nabla\cdot G(\\nabla u_h)\|$'
         ]
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    fem = BiharmonicFEMModel(

show_error_table(Ndof, errorType, errorMatrix, end='\\\\\\hline\n')

optionlist = ['k-*', 'b-o', 'r--^', 'g->', 'm-8', 'c-D','y-x']
fig = plt.figure()
axes = fig.gca()
showmultirate(axes, 1, Ndof, errorMatrix, optionlist, errorType)
axes.legend(loc=3)
axes.axis('tight')
plt.show()
