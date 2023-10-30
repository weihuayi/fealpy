import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import gmsh

from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags

from fealpy.mesh import TriangleMesh 
from fealpy.mesh.halfedge_mesh import HalfEdgeMesh2d

from fealpy.functionspace import LagrangeFESpace
from fealpy.csm import SpectralDecomposition
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm

from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarMassIntegrator
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import ProvidesSymmetricTangentOperatorIntegrator

from fealpy.fem import DirichletBC
from fealpy.fem import LinearRecoveryAlg
from fealpy.mesh.adaptive_tools import mark


class Brittle_Facture_model():
    def __init__(self):
        self.E = 210 # 杨氏模量
        self.nu = 0.3 # 泊松比
        self.Gc = 5e-4 # 材料的临界能量释放率
        self.l0 = 0.06 # 尺度参数，断裂裂纹的宽度

        self.mu = 8 # 剪切模量
        self.lam = 12 # 拉梅常数
        self.kappa = self.lam + 2 * self.mu /3 # 压缩模量

    def init_mesh(self):
        """
        @brief 生成实始网格
        """
        gmsh.initialize()

        gmsh.model.geo.addPoint(0,0,0,tag = 1)
        gmsh.model.geo.addPoint(8,0,0,tag = 2)
        gmsh.model.geo.addPoint(8,2,0,tag = 3)
        gmsh.model.geo.addPoint(0,2,0,tag = 4)
        gmsh.model.geo.addPoint(3.9,0,0,tag = 5)
        gmsh.model.geo.addPoint(4.1,0,0,tag = 6)
        gmsh.model.geo.addPoint(4.0,0.4,0,tag = 7)

        gmsh.model.geo.addLine(1,5,1)
        gmsh.model.geo.addLine(5,7,2)
        gmsh.model.geo.addLine(7,6,3)
        gmsh.model.geo.addLine(6,2,4)
        gmsh.model.geo.addLine(2,3,5)
        gmsh.model.geo.addLine(3,4,6)
        gmsh.model.geo.addLine(4,1,7)

        gmsh.model.geo.addCurveLoop([1,2,3,4,5,6,7],1)
        gmsh.model.geo.addPlaneSurface([1], 1)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = node_coords.reshape((-1,3))[:,:2]

        # 节点编号映射
        nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

        # 获取单元信息
        cell_type = 2 # 三角形单元的类型编号为 2
        cell_tags,cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)

        gmsh.finalize()
        # 节点编号映射到单元
        evid = np.array([nodetags_map[j] for j in cell_connectivity])
        cell = evid.reshape((cell_tags.shape[-1],-1))
        mesh = TriangleMesh(node,cell)

        return mesh

    def top_boundary_disp(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return -np.concatenate((np.linspace(0, 3.6e-2, 361), np.linspace(3.6e-2,
            1e-1, 64000)[1:]))

    def top_disp_direction(self):
        """
        @brief 上边界位移的方向
        Notes
        -----
        位移方向沿 (0, 1) 方向，即仅在 y 方向的位移变化
        """
        return np.array([0, 1], np.float_)

    def is_top_boundary(self, p):
        """
        @brief 标记上边界, y = 1 时的边界点
        """
        return (np.abs(p[..., 1] - 2) < 1e-12 ) & (np.abs(p[..., 0]-4)<1e-5)

    def is_inter_boundary(self, p):
        """
        @brief 标记内部边界, 内部圆的点
        Notes
        -----
        内部圆周的点为 DirichletBC，相场值和位移均为 0
        """
        return np.abs((p[..., 0]-0.5)**2 + np.abs(p[..., 1]-0.5)**2 - 0.04) < 0.001
    
    def is_below_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是下边界
        """
        x_condition = (p[:, 0] == 0)
        y_condition = (p[:, 1] == 0) | (p[:, 1] == 8)
        result = np.logical_and(x_condition, y_condition)
        return result


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
mesh = model.init_mesh()
#mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3) # 使用半边网格

GD = mesh.geo_dimension()
NC = mesh.number_of_cells()

simulation = SpectralDecomposition (mesh, lam=model.lam, mu=model.mu,
        Gc=model.Gc, l0=model.l0)
space = LagrangeFESpace(mesh, p=1, doforder='vdims')

recovery = LinearRecoveryAlg()

d = space.function()
H = np.zeros(NC, dtype=np.float64)  # 分片常数
uh = space.function(dim=GD)


disp = model.top_boundary_disp()
stored_energy = np.zeros_like(disp)
dissipated_energy = np.zeros_like(disp)
force = np.zeros_like(disp)

for i in range(len(disp)-1):

    k = 0
    while k < 20:
        print('i:', i)
        print('k:', k)
        NN = mesh.number_of_nodes()
        print('NN:', NN)
        node  = mesh.entity('node') 
        isTNode = model.is_top_boundary(node)
        if space.doforder == 'vdims':
            uh[isTNode, 1] = disp[i+1]
            isDof = np.c_[np.zeros(NN, dtype=np.bool_), isTNode]
            isTDof = isDof.flat[:]
        else:
            uh[1, isTNode] = disp[i+1]
            isTDof = np.r_['0', np.zeros(NN, dtype=np.bool_), isTNode]
        du = space.function(dim=GD)
        
        # 求解位移
        vspace = (GD*(space, ))
        ubform = BilinearForm(GD*(space, ))

        D = simulation.dsigma_depsilon(d, uh)
        integrator = ProvidesSymmetricTangentOperatorIntegrator(D, q=4)
        ubform.add_domain_integrator(integrator)
        ubform.assembly()
        A0 = ubform.get_matrix()
        R0 = -A0@uh.flat[:]
        
        force[i+1] = np.sum(-R0[isTDof])
        
        ubc = DirichletBC(vspace, 0, threshold=model.is_below_boundary)
        A0, R0 = ubc.apply(A0, R0)
        
        # 位移边界条件处理
        bdIdx = np.zeros(A0.shape[0], dtype=np.int_)
        bdIdx[isTDof] =1
        Tbd =spdiags(bdIdx, 0, A0.shape[0], A0.shape[0])
        T = spdiags(1-bdIdx, 0, A0.shape[0], A0.shape[0])
        A0 = T@A0@T + Tbd
        R0[isTDof] = du.flat[isTDof]
        
        du.flat[:] = spsolve(A0, R0)
        uh[:] += du
        
        # 更新参数
        strain = simulation.strain(uh)
        phip, _ = simulation.strain_energy_density_decomposition(strain)
        H[:] = np.fmax(H, phip)

        # 计算相场模型
        dbform = BilinearForm(space)
        dbform.add_domain_integrator(ScalarDiffusionIntegrator(c=model.Gc*model.l0,
            q=4))
        dbform.add_domain_integrator(ScalarMassIntegrator(c=2*H+model.Gc/model.l0, q=4))
        dbform.assembly()
        A1 = dbform.get_matrix()

        lform = LinearForm(space)
        lform.add_domain_integrator(ScalarSourceIntegrator(2*H, q=4))
        lform.assembly()
        R1 = lform.get_vector()
        R1 -= A1@d[:]

        d[:] += spsolve(A1, R1)
    
        stored_energy[i+1] = simulation.get_stored_energy(phip, d)
        dissipated_energy[i+1] = simulation.get_dissipated_energy(d)

        # 恢复型后验误差估计子
        eta = recovery.recovery_estimate(d)
            
        isMarkedCell = mark(eta, theta = 0.2)

        cm = mesh.cell_area() 
        isMarkedCell = np.logical_and(isMarkedCell, np.sqrt(cm) > model.l0/8)
        
        if np.any(isMarkedCell):
            data = {'uh0':uh[:, 0], 'uh1':uh[:, 1], 'd':d, 'H':H}
            option = mesh.bisect_options(data=data, disp=False)
            mesh.bisect(isMarkedCell, options=option)
            print('mesh refine')      
           
            # 更新加密后的空间
            space = LagrangeFESpace(mesh, p=1, doforder='vdims')
            NC = mesh.number_of_cells()
            uh = space.function(dim=GD)
            d = space.function()
            H = np.zeros(NC, dtype=np.float64)  # 分片常数

            uh[:, 0] = option['data']['uh0']
            uh[:, 1] = option['data']['uh1']
            d[:] = option['data']['d']
            H = option['data']['H']

        # 计算残量误差
        if k == 0:
            er0 = np.linalg.norm(R0)
            er1 = np.linalg.norm(R1)
        error0 = np.linalg.norm(R0)/er0
        print("error0:", error0)

        error1 = np.linalg.norm(R1)/er1
        print("error1:", error1)
        error = max(error0, error1)
        print("error:", error)
        if error < 1e-5:
            break
        k += 1
    mesh.nodedata['damage'] = d
    mesh.nodedata['uh'] = uh
    mesh.celldata['H'] = H
    fname = 'test' + str(i).zfill(10)  + '.vtu'
    mesh.to_vtk(fname=fname)

end = time.time()
print('time:', end-start)
fig = plt.figure()
axes = fig.add_subplot(111)
NN = mesh.number_of_nodes()
mesh.node += uh[:, :NN]
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

