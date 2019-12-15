import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import PolygonMesh
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.pde.sobolev_equation_2d import SinSinExpData
from fealpy.functionspace import WeakGalerkinSpace2d
from fealpy.quadrature import GaussLegendreQuadrature


nu = 1
epsilon = 0.1
pde = SinSinExpData(nu, epsilon)
domain = pde.domain()
tmesh = pde.init_mesh(2, meshtype='tri')


p = 1
h = 0.1
maxit = 4
error = np.zeros((6, maxit), dtype=np.float)
for i in range(4):
    mesh = triangle(domain, h, meshtype='polygon')
    #mesh = PolygonMesh.from_mesh(tmesh)
    space = WeakGalerkinSpace2d(mesh, p=p)

    uh = space.projection(pde.init_value)
    ph = space.weak_grad(uh)

    gh = space.projection(lambda x:pde.gradient(x, 0.0), dim=2)
    dh = space.weak_div(gh)


    NE = mesh.number_of_edges()
    node = mesh.entity('node')
    edge = mesh.entity('edge')
    l = mesh.entity_measure('edge')
    qf = GaussLegendreQuadrature(p + 3)
    bcs, ws = qf.quadpts, qf.weights
    ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
    
    g = pde.solution(ps, 0.0)
    val = space.edge_value(uh, bcs)
    error[3, i] = np.sqrt(np.einsum('i, ij, j->', ws, (g - val)**2, l)/NE)

    g = pde.gradient(ps, 0.0)
    val = space.edge_value(gh, bcs)
    error[4, i] = np.sqrt(np.einsum('i, ijk, j->', ws, (g - val)**2, l)/NE)

    error[0, i] = space.integralalg.L2_error(pde.init_value, uh)
    error[1, i] = space.integralalg.L2_error(lambda x: pde.gradient(x, 0.0), ph)
    error[2, i] = space.integralalg.L2_error(lambda x: pde.gradient(x, 0.0), gh)
    error[5, i] = space.integralalg.L2_error(lambda x: pde.laplace(x, 0.0), dh)
    
    h /= 2
    #tmesh.uniform_refine()

print(error)
print(error[:, 0:-1]/error[:, 1:])

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
