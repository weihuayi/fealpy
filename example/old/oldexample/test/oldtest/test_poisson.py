import sys
import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.poisson_model_2d import LShapeRSinData
from fealpy.fem.PoissonFEMModel import PoissonFEMModel

from fealpy.mesh.Tritree import Tritree

from mpl_toolkits.mplot3d import Axes3D
from fealpy.tools.show import showmultirate

p = int(sys.argv[1])
maxit = int(sys.argv[2])

class Estimator:
    def __init__(self, rho, mesh, theta, beta):
        self.mesh = mesh
        self.rho = rho
        self.theta = theta
        self.beta = beta
        self.area = mesh.entity_measure('cell')
        self.compute_eta()
        self.maxeta = np.max(self.eta)

    def compute_eta(self):
        mesh = self.mesh
        cell = mesh.entity('cell')
        Dlambda = mesh.grad_lambda()
        grad = np.einsum('ij, ijm->im', self.rho[cell], Dlambda)
        self.eta = np.sqrt(np.sum(grad**2, axis=1)*self.area)
        return self.eta

    def update(self, rho, mesh, smooth=True):
        self.rho = rho
        self.mesh = mesh
        self.area = mesh.entity_measure('cell')
        self.smooth_rho()
        self.compute_eta()

    def smooth_rho(self):
        '''
        smooth the rho
        '''
        mesh = self.mesh
        cell = mesh.entity('cell')
        node2cell = mesh.ds.node_to_cell()
        inva = 1/self.area
        s = node2cell@inva
        for i in range(3):
            crho = (self.rho[cell[:, 0]] + self.rho[cell[:, 1]] + self.rho[cell[:, 2]])/3.0
            self.rho = np.asarray(node2cell@(crho*inva))/s

    def is_uniform(self):
        stde = np.std(self.eta)/self.maxeta
        print('The current relative std of eta is ', stde)
        if stde < 0.05:
            return True
        else:
            return False

pde = LShapeRSinData()
mesh = pde.init_mesh(n=4, meshtype='tri')
integrator = mesh.integrator(3)
node = mesh.entity('node')
cell = mesh.entity('cell')
tmesh = Tritree(node, cell)
pmesh = tmesh.to_conformmesh()

tol = 1.0e-4

for i in range(maxit):
    fem = PoissonFEMModel(pde, pmesh, p, integrator)
    fem.solve()
    res = fem.get_H1_error()
    #res = fem.get_L2_error()

    estimator = Estimator(fem.uh[:], mesh, 0.3, 0.5)

    fig = plt.figure()
    axes = fig.gca() 
    mesh.add_plot(axes, cellcolor=estimator.eta, showcolorbar=True)
    
    if  res < tol:
        break
    tmesh.adaptive_refine(estimator)
    pmesh = tmesh.to_conformmesh()
    print("Steps:", i)
    mesh = estimator.mesh
fig = plt.figure()
axes = fig.gca() 
mesh.add_plot(axes, cellcolor=estimator.eta, showcolorbar=True)
   
plt.show()

