import numpy as np  
import matplotlib.pyplot as plt
from fealpy.mesh.adaptive_tools import mark
from fealpy.tools2.show import showmultirate
from IntervalQuadrature import IntervalQuadrature
from DDGModel import PoissonDDGModel
from fealpy.pde.poisson_2d import KelloggData as PDE 
from DDGResidualAlg_DC import DDGResidualEstimators   
beta1 =150  # The positive parameters beta1 are chosen large enough 
beta2 = 1/12
theta = 0.65
C = 0.01
eta = 1
maxit =38 # Iterations
n =1
p=2        # Degree of polynomial approximation, p = 2,3,4
index = 2   # To choose weighted average of the diffusion coefficients                           # Example 4
errorType = ['$\eta_1$','$rel-err$']            # Case 2
Cdtype = 'D'
    
pde = PDE()
mesh = pde.init_mesh(n, meshtype='tri')           # Coarse mesh
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
qf = mesh.integrator(p+2)
IntervalQuadrature = IntervalQuadrature(p+2)
for i in range(maxit):
    print('step:', i)
    fem = PoissonDDGModel(pde, mesh, qf, IntervalQuadrature,p) # Create DDG model
    fem.solve(beta1,beta2,index,Cdtype)
    uh = fem.uh     # Numerical solution
    #print(uh)                                            
    ralg = DDGResidualEstimators(uh,mesh,pde,p)
    Ndof[i] = fem.space.number_of_global_dofs()     # Degrees of freedom
    # A posteriori estimator
    eta2 = ralg.Rf_estimate(cfun = pde.diffusion_coefficient)
    eta3 = ralg.Ju_estimate(index,cfun = pde.diffusion_coefficient)
    eta4 = ralg.Ji_estimate(index,cfun = pde.diffusion_coefficient)
    eta5 = ralg.RD_estimate(index,cfun = pde.diffusion_coefficient)
    eta1 = (eta2 + eta3 + eta4+ eta5)
    errorMatrix[0, i] = np.sqrt(np.sum(eta1))
    errorMatrix[1, i] = fem.get_rel_error(uh,index,cfun = pde.diffusion_coefficient)
    print('number of nodes:',fem.mesh.number_of_nodes())
    if i < maxit - 1:
        markedCell = mark(np.sqrt(eta4), theta=theta, method='MAX') # To choose estimator
        mesh.bisect(markedCell)

# Plot numerical results
mesh.add_plot(plt, cellcolor='w')  # Adaptive meshe
showmultirate(plt, 0, Ndof, errorMatrix,  errorType, propsize=20)

# Numerical solution 
# fig = plt.figure(figsize=(8,7))
# axes = fig.gca(projection = '3d')
# uh.add_plot(axes,cmap = 'rainbow')
plt.show()
