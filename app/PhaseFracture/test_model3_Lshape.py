import numpy as np
import matplotlib.pyplot as plt
import time

from fealpy.mesh import TriangleMesh 

from fealpy.csm import IPFEMPhaseFieldCrackHybridMixModel

class Brittle_Facture_model():
    def __init__(self):
        self.Gc = 8.9e-5 # 材料的临界能量释放率
        self.l0 = 1.18 # 尺度参数，断裂裂纹的宽度

        self.mu = 10.95 # 10.95GPa
        self.lam = 6.16 # 6.16GPa


    def is_boundary_disp(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return np.concatenate((np.linspace(0, 0.3, 301), np.linspace(0.3,
            -0.2, 501)[1:], np.linspace(-0.2, 1, 1201)[1:]))

    def is_disp_boundary(self, p):
        """
        @brief 标记施加力的节点
        """
        isDNode = (np.abs(p[..., 1] - 250) < 1e-12)&(np.abs(p[..., 0]-470)<1e-5)
        isDDof = np.c_[np.zeros(p.shape[0], dtype=np.bool_), isDNode]
        return isDDof

    
    def is_dirchlet_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是下边界
        """
        return np.abs(p[..., 1]) < 1e-12

def no_mesh(p):
    return (p[...,0] > 250)&(p[...,1]<250)

model = Brittle_Facture_model()

mesh = TriangleMesh.from_box(box=[0, 500, 0, 500], nx=50, ny=50,
        threshold=no_mesh)

simulation = IPFEMPhaseFieldCrackHybridMixModel(model, mesh)

mesh.nodedata['damage'] = simulation.d
fname = 'test_model3_Lshape.vtu'
mesh.to_vtk(fname=fname)

disp = model.is_boundary_disp()
stored_energy = np.zeros_like(disp)
dissipated_energy = np.zeros_like(disp)
force = np.zeros_like(disp)

for i in range(len(disp)-1):
    print('i:', i)
    simulation.newton_raphson(disp[i+1], theta=0.02)

    force[i+1] = simulation.force
    stored_energy[i+1] = simulation.stored_energy
    dissipated_energy[i+1] = simulation.dissipated_energy

    mesh.nodedata['damage'] = simulation.d
    mesh.nodedata['uh'] = simulation.uh
    mesh.celldata['H'] = simulation.H
    fname = 'test' + str(i).zfill(10)  + '.vtu'
    mesh.to_vtk(fname=fname)

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


