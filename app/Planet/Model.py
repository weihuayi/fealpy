
import sys
import numpy as np
import matplotlib.pyplot as plt

from PlanetHeatConductionSimulator import PlanetHeatConductionSimulator
from TPMModel import TPMModel 

from fealpy.tools.show import showmultirate, show_error_table
from scipy.sparse.linalg import spsolve

p = int(sys.argv[1])
n = int(sys.argv[2])
NT = int(sys.argv[3])
maxit = int(sys.argv[4])
#    h = float(sys.argv[5])
#    nh = int(sys.argv[6])
h = 0.005
nh = 100

pde = TPMModel()
mesh = pde.init_mesh(n=n, h=h, nh=nh, p=p)

simulator = PlanetHeatConductionSimulator(pde, mesh, p=p)

timeline = simulator.time_mesh(NT=NT)

Ndof = np.zeros(maxit, dtype=np.float)

uh0 = simulator.init_solution()  # 当前时间层的温度分布
uh1 = simulator.space.function() # 下一时间层的温度分布 
uh1 = uh0.copy()

for i in range(maxit):
    print(i)
    simulator = PlanetHeatConductionSimulator(pde, mesh, p=p)
    
    Ndof[i] = simulator.space.number_of_global_dofs()

    timeline.time_integration(uh1, simulator, spsolve)

    if i < maxit-1:
        timeline.uniform_refine()
        
        n = n+1
        h = h/2
        nh = nh*2
        mesh = pde.init_mesh(n=n, h=h, nh=nh, p=p)
    
    Tss = pde.options['Tss']
    uh = uh1*Tss
    print('uh:', uh1)

np.savetxt('01solution', uh)
mesh.nodedata['uh'] = uh

mesh.to_vtk(fname='test.vtu') 

