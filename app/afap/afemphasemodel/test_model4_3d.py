import numpy as np
import matplotlib.pyplot as plt
import time

from fealpy.mesh import TetrahedronMesh 
from fealpy.geometry import SquareWithCircleHoleDomain

from fealpy.csm import AFEMPhaseFieldCrackHybridMixModel

class Brittle_Facture_model():
    def __init__(self):
        self.E = 220.8 # 杨氏模量
        self.nu = 0.3 # 泊松比
        self.Gc = 5e-4 # 材料的临界能量释放率
        self.l0 = 0.02 # 尺度参数，断裂裂纹的宽度

        self.mu = self.E / (1 + self.nu) / 2.0 # 剪切模量
        self.lam = self.E * self.nu / (1 + self.nu) / (1- 2*self.nu) # 拉梅常数
        self.kappa = self.lam + 2 * self.mu /3 # 压缩模量

    def init_mesh(self, n=2):
        """
        @brief 生成实始网格
        """
        mesh = TetrahedronMesh.from_crack_box()
        mesh.uniform_refine(n=n)
        return mesh

    def is_boundary_disp(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return np.linspace(0, 4.5e-2, 4501)

    def is_disp_boundary(self, p):
        """
        @brief 标记位移增量的边界点
        """
        isDNode = np.abs(p[..., 2] - 10) < 1e-12
        isDDof = np.c_[np.zeros(p.shape[0], dtype=np.bool_),
                np.zeros(p.shape[0], dtype=np.bool_), isDNode]
        return isDDof

    def is_dirchlet_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是下边界
        """
        return np.abs(p[..., 2]) < 1e-12

start = time.time()

model = Brittle_Facture_model()
mesh = model.init_mesh(n=2)

simulation = AFEMPhaseFieldCrackHybridMixModel(model, mesh)
mesh.nodedata['damage'] = simulation.d
mesh.to_vtk(fname='mesh_3d.vtu')

disp = model.is_boundary_disp()
stored_energy = np.zeros_like(disp)
dissipated_energy = np.zeros_like(disp)
force = np.zeros_like(disp)

for i in range(len(disp)-1):
    print('i:', i)
    simulation.newton_raphson(disp[i+1], maxit=50, solve='gpu')
#    simulation.newton_raphson(disp[i+1], maxit=50, atype=None)

    force[i+1] = simulation.force
    stored_energy[i+1] = simulation.stored_energy
    dissipated_energy[i+1] = simulation.dissipated_energy

    mesh.nodedata['damage'] = simulation.d
    mesh.nodedata['uh'] = simulation.uh
    mesh.celldata['H'] = simulation.H
    fname = 'test' + str(i).zfill(10)  + '.vtu'
    mesh.to_vtk(fname=fname)

end = time.time()
print('time:', end-start)

plt.figure()
plt.plot(disp, force, label='Force')
plt.xlabel('disp')
plt.ylabel('Force')
plt.grid(True)
plt.legend()
plt.savefig('model4_force.png', dpi=300)
#plt.show()

plt.figure()
plt.plot(disp, stored_energy, label='stored_energy')
plt.plot(disp, dissipated_energy, label='dissipated_energy')
plt.plot(disp, dissipated_energy+stored_energy, label='total_energy')
plt.xlabel('disp')
plt.ylabel('energy')
plt.grid(True)
plt.legend()
plt.savefig('model4_energy.png', dpi=300)
plt.show()

