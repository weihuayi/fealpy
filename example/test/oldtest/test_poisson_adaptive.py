import sys
import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.poisson_model_2d import LShapeRSinData
from fealpy.fem.PoissonFEMModel import PoissonFEMModel

from fealpy.recovery import FEMFunctionRecoveryAlg
from fealpy.mesh import Tritree

from mpl_toolkits.mplot3d import Axes3D
from fealpy.tools.show import showmultirate

class Estimator:
    def __init__(self, rho, mesh, theta, beta):
        self.mesh = mesh
        self.rho = rho
        self.theta = theta
        self.beta = beta
        self.area = mesh.entity_measure('cell')
        self.eta = self.compute_eta()

    def compute_eta(self):
        mesh = self.mesh
        cell = mesh.entity('cell')
        Dlambda = mesh.grad_lambda()
        guh = np.einsum('ij, ijm->im', self.rho[cell], Dlambda)

        node2cell = mesh.ds.node_to_cell()
        inva = 1/mesh.area()
        asum = node2cell@inva
        rguh = np.asarray(node2cell@(guh*inva.reshape(-1, 1)))/asum.reshape(-1, 1)

        bc = np.array([1/3, 1/3, 1/3])
        err = np.einsum('ijk, j->ik', rguh[cell], bc) - guh
        eta0 = np.sqrt(np.sum(err**2, axis=1)*self.area)
        eta1 = np.sqrt(np.sum(guh**2, axis=1)*self.area)
        
        return eta1 + eta0

    def update(self, rho, mesh, smooth=True):
        self.rho = rho
        self.mesh = mesh
        self.area = mesh.entity_measure('cell')
        if smooth is True:
            self.smooth_rho()
            #self.loop_smooth_rho()
        self.eta = self.compute_eta()

    def is_extreme_node(self):
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        edge = mesh.entity('edge')
        node2node = mesh.ds.node_to_node()
        V = np.asarray(np.sum(node2node, axis=1)).reshape(-1)

        isSmall = self.rho[edge[:, 0]] < self.rho[edge[:, 1]]
        v = np.zeros(NN, dtype=np.int)
        np.add.at(v, edge[isSmall, 0], 1)
        np.add.at(v, edge[~isSmall, 1], 1)

        isExtremeNode = (v == V) | (V == (V - v))

        return isExtremeNode

    def smooth_rho(self):
        '''
        smooth the rho
        '''
        mesh = self.mesh
        cell = mesh.entity('cell')
        isExtremeNode = self.is_extreme_node()
        node2cell = mesh.ds.node_to_cell()
        inva = 1/self.area
        s = node2cell@inva
        for i in range(2):
            crho = (self.rho[cell[:, 0]] + self.rho[cell[:, 1]] + self.rho[cell[:, 2]])/3.0
            rho = np.asarray(node2cell@(crho*inva))/s
            self.rho[~isExtremeNode] = rho[~isExtremeNode]

    def loop_smooth_rho(self):
        mesh = self.mesh
        cell = mesh.entity('cell')
        node2node = mesh.ds.node_to_node()
        NN = mesh.number_of_nodes()
        v = np.bincount(cell.flat, minlength=NN)
        isExtremeNode = self.is_extreme_node()
        a = (5/8 - (3/8 + 0.25*np.cos(2*np.pi/v)**2))/v
        for i in range(2):
            self.rho = (1 - v*a)*self.rho + a*(node2node@self.rho)
            #self.rho[~isExtremeNode] = rho[~isExtremeNode]

    def is_uniform(self):
        stde = np.std(self.eta)/np.max(self.eta)
        print('The current relative std of eta is ', stde)
        if stde < 0.01:
            return True
        else:
            return False


# get the pde model
pde = LShapeRSinData()

mesh = pde.init_mesh(n=4, meshtype='tri')
node = mesh.entity('node')
cell = mesh.entity('cell')
tmesh = Tritree(node, cell)

errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$', 
             '$||\\nabla u - G(\\nabla u_h)||_{0}sim$',]

maxit = 1
ralg = FEMFunctionRecoveryAlg()
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
integrator = mesh.integrator(3)


for i in range(maxit):
    print('step:', i)
    fem = PoissonFEMModel(pde, mesh, 1, integrator)
    fem.solve()
    uh = fem.uh
    Ndof[i] = fem.mesh.number_of_nodes()
    errorMatrix[0, i] = fem.get_l2_error()
    errorMatrix[1, i] = fem.get_L2_error()
    errorMatrix[2, i] = fem.get_H1_error()
    rguh = ralg.simple_average(uh)
    eta = fem.recover_estimate(rguh)
    errorMatrix[3, i] = fem.get_recover_error(rguh)

    estimator = Estimator(uh[:], mesh, 0.3, 0.5)
    tmesh.adaptive_refine(estimator)
    if i < maxit - 1:
        mesh = estimator.mesh



mesh.add_plot(plt, cellcolor='w')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.node[:, 0]
y = mesh.node[:, 1]
cell = mesh.ds.cell
axes.plot_trisurf(x, y, cell, fem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()

