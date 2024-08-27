import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh 
from fealpy.mesh.halfedge_mesh import HalfEdgeMesh2d
from fealpy.geometry import SquareWithCircleHoleDomain

from fealpy.csm import AFEMPhaseFieldCrackHybridMixModel
from fealpy.ml import timer

class Brittle_Facture_model():
    def __init__(self):
        self.E = 200 # 杨氏模量
        self.nu = 0.2 # 泊松比
        self.Gc = 1 # 材料的临界能量释放率
        self.l0 = 0.02 # 尺度参数，断裂裂纹的宽度

        self.mu = self.E / (1 + self.nu) / 2.0 # 剪切模量
        self.lam = self.E * self.nu / (1 + self.nu) / (1- 2*self.nu) # 拉梅常数
        self.kappa = self.lam + 2 * self.mu /3 # 压缩模量

    def is_boundary_disp(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return np.concatenate((np.linspace(0, 70e-3, 6), np.linspace(70e-3,
            125e-3, 26)[1:]))

    def is_disp_boundary(self, p):
        """
        @brief 标记位移增量的边界点
        """
        isDNode = np.abs(p[..., 1] - 1) < 1e-12 
        isDDof = np.c_[np.zeros(p.shape[0], dtype=np.bool_), isDNode]
        return isDDof

    def is_boundary_phase(self, p):
        """
        @brief 标记内部边界, 内部圆的点
        Notes
        -----
        内部圆周的点为 DirichletBC，相场值和位移均为 0
        """
        return np.abs((p[..., 0]-0.5)**2 + np.abs(p[..., 1]-0.5)**2 - 0.04) < 0.001

    def is_dirchlet_boundary(self, p):
        """
        @brief 标记内部边界, 内部圆的点
        Notes
        -----
        内部圆周的点为 DirichletBC，相场值和位移均为 0
        """
        return np.abs((p[..., 0]-0.5)**2 + np.abs(p[..., 1]-0.5)**2 - 0.04) < 0.001
#        return np.abs(p[..., 1]) < 1e-12

tmr = timer()
next(tmr)
model = Brittle_Facture_model()

domain = SquareWithCircleHoleDomain(hmin=0.05) 
mesh = TriangleMesh.from_domain_distmesh(domain, maxit=100)
#mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3) # 使用半边网格

simulation = AFEMPhaseFieldCrackHybridMixModel(model, mesh)

disp = model.is_boundary_disp()
stored_energy = np.zeros_like(disp)
dissipated_energy = np.zeros_like(disp)
force = np.zeros_like(disp)

for i in range(len(disp)-1):
    print('i:', i)
    simulation.newton_raphson(disp[i+1], dirichlet_phase=True, refine='rg',
                              maxit=50)

    force[i+1] = simulation.force
    stored_energy[i+1] = simulation.stored_energy
    dissipated_energy[i+1] = simulation.dissipated_energy
    mesh = simulation.mesh

    mesh.nodedata['damage'] = simulation.d
    mesh.nodedata['uh'] = simulation.uh
    mesh.celldata['H'] = simulation.H
    fname = 'test' + str(i).zfill(10)  + '.vtu'
    mesh.to_vtk(fname=fname)

tmr.send('stop')
fig = plt.figure()
axes = fig.add_subplot(111)
NN = mesh.number_of_nodes()
mesh.node += simulation.uh[:, :NN]
mesh.add_plot(axes)
#plt.show()

plt.figure()
plt.plot(disp, force, label='Force')
plt.xlabel('disp')
plt.ylabel('Force')
plt.grid(True)
plt.legend()
plt.savefig('force.png', dpi=300)
#plt.show()

plt.figure()
plt.plot(disp, stored_energy, label='stored_energy')
plt.plot(disp, dissipated_energy, label='dissipated_energy')
plt.plot(disp, dissipated_energy+stored_energy, label='total_energy')
plt.xlabel('disp')
plt.ylabel('energy')
plt.grid(True)
plt.legend()
plt.savefig('energy.png', dpi=300)
plt.show()


