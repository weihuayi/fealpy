import numpy as np

import matplotlib.pyplot as plt
from fealpy.model.parabolic_model_2d import  SinCosExpData

from fealpy.functionspace.tools import function_space 
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh 
from fealpy.timeintegratoralg.TimeIntegratorAlgorithmData import TimeIntegratorAlgorithmData



box = [0, 1, 0, 1]
n = 10
mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='tri')

interval = [0.0,1.0]
model = SinCosExpData()
N = 400
s = TimeIntegratorAlgorithmData(interval,mesh,model,N)

a = 1/16
e = s.solve(a,n)
print(e)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
