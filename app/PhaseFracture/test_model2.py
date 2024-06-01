import numpy as np
import matplotlib.pyplot as plt
import time

from fealpy.mesh import TriangleMesh 
from fealpy.geometry import SquareWithCircleHoleDomain

from fealpy.csm import AFEMPhaseFieldCrackPropagationProblem2d
from fealpy.csm import IPFEMPhaseFieldCrackHybridMixModel

class Brittle_Facture_model():
    def __init__(self):
        self.E = 210 # 杨氏模量
        self.nu = 0.3 # 泊松比
        self.Gc = 2.7e-3 # 材料的临界能量释放率
        self.l0 = 0.004 # 尺度参数，断裂裂纹的宽度

        self.mu = self.E / (1 + self.nu) / 2.0 # 剪切模量
        self.lam = self.E * self.nu / (1 + self.nu) / (1- 2*self.nu) # 拉梅常数
        self.kappa = self.lam + 2 * self.mu /3 # 压缩模量

    def init_mesh(self, n=3):
        """
        @brief 生成实始网格
        """
        node = np.array([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]], dtype=np.float64)

        cell = np.array([
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5]], dtype=np.int_)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n=n)
        mesh.ds.NV = 3
        return mesh

    def is_boundary_disp(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return np.linspace(0, 2e-2, 2001)
#        return np.concatenate((np.linspace(0, 5e-3, 501), np.linspace(5e-3,
#            6.1e-3, 1101)[1:]))

    def is_disp_boundary(self, p):
        """
        @brief 标记位移增量的边界点
        """
        isDNode = np.abs(p[..., 1] - 1) < 1e-12 
        isDDof = np.c_[isDNode, np.zeros(p.shape[0], dtype=np.bool_)]
        return isDDof

    def is_inter_boundary(self, p):
        """
        @brief 标记内部边界, 内部圆的点
        Notes
        -----
        内部圆周的点为 DirichletBC，相场值和位移均为 0
        """
        return np.abs((p[..., 0]-0.5)**2 + np.abs(p[..., 1]-0.5)**2 - 0.04) < 0.001
    
    def is_dirchlet_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是下边界
        """
        return (np.abs(p[..., 1]) < 1e-12) | (np.abs(p[..., 1] - 1) < 1e-12) 

def adaptive_mesh(mesh, d0=0.49, d1=1.01, h=0.005):
    cell = mesh.entity("cell")
    node = mesh.entity("node")
    isMarkedCell = mesh.cell_area() > 0.00001
    isMarkedCell = isMarkedCell & (np.min(np.abs(node[cell, 1] - 0.5),
                                          axis=-1) < h)
    isMarkedCell = isMarkedCell & (np.min(node[cell, 0], axis=-1) > d0) & (
            np.min(node[cell, 0], axis=-1) < d1)
    return isMarkedCell

start = time.time()

model = Brittle_Facture_model()
mesh = model.init_mesh(n=4)

NC = mesh.number_of_cells()

simulation = IPFEMPhaseFieldCrackHybridMixModel(model, mesh)

disp = model.is_boundary_disp()
stored_energy = np.zeros_like(disp)
dissipated_energy = np.zeros_like(disp)
force = np.zeros_like(disp)

for i in range(len(disp)-1):
    print('i:', i)
    simulation.newton_raphson(disp[i+1], maxit=30)

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
fig = plt.figure()
axes = fig.add_subplot(111)
NN = mesh.number_of_nodes()
mesh.node += simulation.uh[:, :NN]
mesh.add_plot(axes)
plt.show()

plt.figure()
plt.plot(disp, force, label='Force')
plt.xlabel('disp')
plt.ylabel('Force')
plt.grid(True)
plt.legend()
plt.savefig('model2_force.png', dpi=300)
#plt.show()

plt.figure()
plt.plot(disp, stored_energy, label='stored_energy')
plt.plot(disp, dissipated_energy, label='dissipated_energy')
plt.plot(disp, dissipated_energy+stored_energy, label='total_energy')
plt.xlabel('disp')
plt.ylabel('energy')
plt.grid(True)
plt.legend()
plt.savefig('model2_energy.png', dpi=300)
plt.show()

