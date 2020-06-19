
"""

Notes
-----

    1. 混合物的摩尔浓度 c 的计算公式为
        c = p/(ZRT), 
        Z^3 - (1 - B)Z^2 + (A - 3B^2 -2B)Z - (AB - B^2 -B^3) = 0
        其中:
        p 是压力 
        Z 是压缩系数 
        R 是气体常数
        T 是温度
        M_i 是组分 i 的摩尔质量
        
        混合物的密度计算公式为:

        rho = M_0 c_0 + M_1 c_1 + M_2 c_2 ...
    2. c_i : 组分 i 的摩尔浓度
       z_i : 组分 i 的摩尔分数

       c_i = z_i c

    3. 
    甲烷: methane,  CH_4,   16.04 g/mol,  0.42262 g/cm^3
    乙烷: Ethane, C_2H_6,   30.07 g/mol, 1.212 kg/m^3
    丙烷: Propane,  C_3H_8,  44.096 g/mol,  1.83 kg/m^3 
    (25度 100 kPa

    4. 气体常数  R = 8.31446261815324 	J/K/mol


References
----------
"""
import numpy as np
from scipy.sparse import coo_matrix
from fealpy.mesh import MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.timeintegratoralg.timeline import UniformTimeLine

class Model1():
    def __init__(self):
        self.domain = [0, 50, 0, 50]
        self.mesh = MeshFactory().regular(self.domain, n=50)
        self.timeline = UniformTimeLine(0, 1, 100) 
        self.space0 = RaviartThomasFiniteElementSpace2d(self.mesh, p=0)
        self.space1 = ScaledMonomialSpace2d(self.mesh, p=1) # 线性间断有限元空间

        self.vh = self.space0.function() # 速度
        self.ph = self.space0.smspace.function() # 压力
        self.ch = self.space1.function(dim=3) # 三个组分的摩尔密度
        self.options = {
                'viscosity': 1.0, 
                'permeability': 1.0,
                'temperature': 397,
                'pressure': 50,
                'porosity': 0.2,
                'injecttion_rate': 0.1,
                'compressibility': (0.001, 0.001, 0.001),
                'pmv': (1.0, 1.0, 1.0)}

        c = self.options['viscosity']/self.options['permeability']
        self.A = c*self.space0.mass_matrix()
        self.B = self.space0.div_matrix()
        self.M = self.space0.smspace.mass_matrix() #  

    def get_current_pv_matrix(self, q=None):
        """
        (c_h[i] v_h\cdot n, w_h)_{\partial K}
        """

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = edge2cell[:, 0] == edge2cell[:, 1]
        edge2dof = self.dof.edge_to_dof() 

        qf = self.integralalg.edgeintegrator if q is None else mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = mesh.bc_to_point(bcs, etype='edge')
        en = mesh.edge_unit_normal() # (NE, 2)

        # 组分浓度在积分点上的值, 这里有 3 个组分
        lval = self.ch(ps, index=edge2cell[:, 0]) # (NQ, NE, 3) 
        rval = self.ch(ps, index=edge2cell[:, 1]) # (NQ, NE, 3)
        rval[:, isBdEdge] = 0.0 # 边界值设为 0, 这样后面不产生贡献.

        # 速度在积分点上的基函数值
        gdof1 = self.space0.number_of_global_dofs()
        phi = self.space0.edge_basis(ps) # (NQ, NE, 1, 2)

        # 压力测试函数空间基函数的值
        lphi = self.space0.smspace.basis(ps, index=edge2cell[:, 0]) # (NQ, NE, 3)
        rphi = self.space0.smspace.basis(ps, index=edge2cell[:, 1]) # (NQ, NE, 3)
        gdof0 = self.space0.smspace.number_of_global_dofs()
        measure = self.integralalg.edgemeasure

        LM = np.einsum('i, ijk, ijl, ijmn, jn, j->jlm', ws, lval, lphi, phi, en, measure)
        RM = np.einsum('i, ijk, ijl, ijmn, jn, j->jlm', ws, rval, rphi, phi, en, measure)

        cell2dof = cell.space1.cell_to_dof()
        I = np.broadcast_to(edge2dof[:, :, None], LM.shape)

        J = np.broadcast_to(cell2dof[edge2cell[:, 0]][:, None, :], LM.shape)
        P = coo_matrix((LM.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))

        J = np.broadcast_to(cell2dof[edge2cell[:, 1]][:, None, :], RM.shape)
        P += coo_matrix((RM.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))

        return P.tocsr()

    def get_current_left_matrix(self, data, timeline):
        pass

    def get_current_right_vector(self, data, timeline):
        pass

    def solve(self, data, A, b, solver, timeline):
        pass




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = Model1()

    NN = model.mesh.number_of_nodes()
    print(NN)
    fig = plt.figure()
    axes = fig.gca()
    model.mesh.add_plot(axes)
    plt.show()
