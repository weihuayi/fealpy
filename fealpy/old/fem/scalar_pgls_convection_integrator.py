import numpy as np

class ScalarPGLSConvectionIntegrator:
    """
    @note ( c \\cdot \\nabla u, v)
    """    

    def __init__(self, A, b, q=3):
        self.A = A
        self.b = b 
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None,
            out=None):

        q = self.q
        mesh = space.mesh

        A = self.A # 假设是一个标量, 扩散系数
        b = self.b # 假设是一个二维向量, 对流系数
        
        GD = mesh.geo_dimension()
        TD = mesh.top_dimension()
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        h = cellmeasure**(1/TD) # 单元尺寸
        v = np.sqrt(b[0]**2 + b[1]**2) # 对流系数模
        pe = 0.5*v*h/A # peclet 数
        tau = 0.5*h*(1/np.tanh(pe) - 1/pe)/v

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs() 
        
        if out is None:
            C = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        else:
            C = out
        
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)
        
        gphi = space.grad_basis(bcs, index=index) 
        phi = space.basis(bcs, index=index) 

        val = np.einsum('c, qcmn, n->qcm', tau, gphi, b)
        val += phi
        
        C += np.einsum('q, qck, n, qcmn, c->ckm', ws, val, b, gphi, cellmeasure) 

        if out is None:
            return C

    def assembly_cell_matrix_fast(self, space, index=np.s_[:], cellmeasure=None):
        """
        """
        mesh = space.mesh 
        assert mesh.meshtype in ['tri', 'tet']

    def assembly_cell_matrix_ref(self, space, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元的矩阵组装
        """
        pass
