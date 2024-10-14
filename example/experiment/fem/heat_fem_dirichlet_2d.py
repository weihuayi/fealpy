from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ScalarMassIntegrator,
    DirichletBC
)
from fealpy.sparse.linalg import sparse_cg
from fealpy.pde.heatequation_model_2d import SinSinExpData
import matplotlib.pyplot as plt
# bm.set_backend('pytorch')

def one_step(bm, pde, t, ut, dt, method):

    lform = LinearForm(space)

    if method == 'C-N':
        lform.add_integrator(ScalarSourceIntegrator(lambda p: pde.source(p,t+0.5*dt)))
        F = lform.assembly()
        LHSmat = M + 0.5 * dt * A
        RHS = (M - 0.5*dt*A) @ ut[:] + dt*F
        LHSmat, RHS = DirichletBC(space, lambda p: pde.dirichlet(p,t+0.5*dt)).apply(LHSmat,RHS,check=False)

    elif method == 'Backward Euler':
        lform.add_integrator(ScalarSourceIntegrator(lambda p: pde.source(p,t+dt)))
        F = lform.assembly()
        LHSmat = M + dt * A
        RHS = M@ut[:] + dt*F
        LHSmat, RHS = DirichletBC(space, lambda p: pde.dirichlet(p,t+dt)).apply(LHSmat,RHS,check=False)

    elif method == 'Forward Euler':
        lform.add_integrator(ScalarSourceIntegrator(lambda p: pde.source(p,t)))
        F = lform.assembly()
        LHSmat = M
        RHS = (M - dt*A) @ ut[:] + dt*F
        LHSmat, RHS = DirichletBC(space, lambda p: pde.dirichlet(p,t)).apply(LHSmat,RHS,check=False)

    uh = space.function()
    uh[:] = sparse_cg(LHSmat, RHS)
    return uh    

def solve(bm, pde, dt, step, method):

    ut = space.interpolate(pde.init_value)
    res = bm.zeros((step+2,ut.shape[0]))
    exact = bm.zeros((step+2,ut.shape[0]))
    res[0,:] = ut[:]
    exact[0,:] = ut[:]
    
    for i in range(step):

        t = i * dt
        ut = one_step(bm, pde, t, ut, dt, method)
        res[i+1,:] = ut[:]
        uexact = space.interpolate(lambda p: pde.solution(p,t+dt))
        exact[i+1,:] = uexact

    return res, exact

def plot_linear_function(uh, u):
    fig = plt.figure()
    fig.set_facecolor('white')
    axes = plt.axes(projection='3d')
    NC = mesh.number_of_cells()
    mid = mesh.entity_barycenter("cell")
    node = mesh.entity("node")
    cell = mesh.entity("cell")
    coor = node[cell]
    val = u(node).reshape(-1) 
    for ii in range(NC):
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], uh[cell[ii]], color = 'r', lw=0.0)
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], val[cell[ii]], color = 'b', lw=0.0)

if __name__ == '__main__':

    pde = SinSinExpData(k=1)
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=40, ny=40)
    space = LagrangeFESpace(mesh, p=1)

    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator())
    A = bform.assembly()

    mform = BilinearForm(space)
    mform.add_integrator(ScalarMassIntegrator())
    M = mform.assembly()

    res, exact = solve(bm, pde, dt=0.05, step=100, method='C-N')
    print(bm.max(bm.abs(res - exact)))
    plot_linear_function(res[99], lambda p: pde.solution(p,0.05*99))
