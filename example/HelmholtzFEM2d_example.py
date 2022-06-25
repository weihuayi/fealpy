import numpy as np
import matplotlib.pyplot as plt

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.pde.helmholtz_3d import HelmholtzData3d
from fealpy.pde.helmholtz_2d import HelmholtzData2d


def penalty_matrix(space, q):
    """
    @brief 组装罚项矩阵
    """
    ldof = space.number_of_local_dofs()

    mesh = space.mesh
    GD = mesh.geo_dimension()
    TD = mesh.top_dimension()
    NC = mesh.number_of_cells()

    qf = mesh.integrator(q, 'face')
    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(ws)

    gphi = np.zeros((TD+1, NQ, NC, ldof, GD), dtype=np.float64)
    for i in range(TD+1):
        b = np.insert(bcs, i, 0, axis=1)
        gphi[i] = space.grad_basis(b)

    face2cell = mesh.face_to_cell()
    isBdFace = face2cell[:, 0] != face2cell[:, 1]
    n = mesh.face_unit_normal()
    measure = mesh.entity_measure('face')

    return None




pde = HelmholtzData2d() 
domain = pde.domain()
mesh = MF.boxmesh2d(domain, nx=100, ny=100, meshtype='tri')

space = LagrangeFiniteElementSpace(mesh, p=1)

penalty_matrix(space, 3)



uI = space.interpolation(pde.solution)

bc = np.array([1/3, 1/3, 1/3])
u = uI(bc)

fig, axes = plt.subplots(1, 2)
mesh.add_plot(axes[0], cellcolor=np.real(u))
mesh.add_plot(axes[1], cellcolor=np.imag(u)) 

fig, axes = plt.subplots(1, 2, subplot_kw={'projection':'3d'})
plt.show()
