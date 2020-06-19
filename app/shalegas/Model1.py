
import numpy as np
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

        lval = self.ch(ps, index=edge2cell[:, 0]) # (NQ, NE, 3) 
        rval = self.ch(ps, index=edge2cell[:, 1]) # (NQ, NE, 3)
        rval[:, isBdEdge] = 0.0 # 边界值设为 0
        phi0 = self.space0.edge_basis(ps) # (NQ, NE, 1, 2)

        lphi = self.space1.basis(ps, index=edge2cell[:, 0]) # (NQ, NE, 3)
        rphi = self.space1.basis(ps, index=edge2cell[:, 1]) # (NQ, NE, 3)
        measure = self.integralalg.edgemeasure

        LM = np.einsum('i, ijk, ijmn, ijl, jn, j->jml', ws, lval, phi0, lphi, en, measure)
        RM = np.einsum('i, ijk, ijmn, ijl, jn, j->jml', ws, rval, phi0, rphi, en, measure)

        cell2dof = cell.space1.cell_to_dof()

        ldof0 = edge2dof.shape[1]
        ldof1 = cell2dof.shape[1]
        I = np.einsum('k, ij->ijk', np.ones(ldof1), edge2dof)
        J = np.einsum('k, ij->ikj', np.ones(ldof0), cell2dof)



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
