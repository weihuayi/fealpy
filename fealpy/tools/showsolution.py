import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def showsolution(plt,mesh,pde,uh,ph):
    fig = plt.figure()
    ax = Axes3D(fig)


    bc = mesh.entity_barycenter('edge')
    pc = mesh.entity_barycenter('cell')

    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()

    isYDEdge = mesh.ds.y_direction_edge_flag()
    isXDEdge = mesh.ds.x_direction_edge_flag()

    uI = np.zeros(NE, dtype = mesh.ftype)
    pI = np.zeros(NC, dtype=mesh.ftype)

    uI[isYDEdge] = pde.velocity_x(bc[isYDEdge])
    uI[isXDEdge] = pde.velocity_y(bc[isXDEdge])
    pI = pde.pressure(pc)

    hx = mesh.hx
    hy = mesh.hy
    Nx = int(1/hx)
    Ny = int(1/hy)

    ux1 = bc[:sum(isYDEdge),0]
    ux2 = bc[sum(isYDEdge):,0]
    uy1 = bc[:sum(isYDEdge),1]
    uy2 = bc[sum(isYDEdge):,1]

    uh1 = uh[:sum(isYDEdge)]
    uh2 = uh[sum(isYDEdge):]
    uI1 = uI[:sum(isYDEdge)]
    uI2 = uI[sum(isYDEdge):]

    plt.title("U solution")
    surf = ax.plot_surface(ux1.reshape((Ny,Nx+1)),uy1.reshape((Ny,Nx+1)),uh1.reshape((Ny,Nx+1)), rstride = 2,cstride = 2,cmap=plt.cm.hot)
    surf = ax.plot_surface(ux2.reshape((Ny,Nx+1)),uy2.reshape((Ny,Nx+1)),uI1.reshape((Ny,Nx+1)), rstride = 2,cstride = 2,cmap=plt.cm.hot)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
