import numpy as np

from .ScalarDiffusionIntegrator import ScalarDiffusionIntegrator 

class VectorDiffusionIntegrator:
    """
    @note (c \\grad u, \\grad v)
    """    
    def __init__(self, c=None, q=None):
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @note 没有参考单元的组装方式
        """
        if isinstance(space, tuple) and not isinstance(space[0], tuple): # 由标量空间组合而成的空间
            return self.assembly_cell_matrix_for_vspace_with_sacalar_basis(
                    space, index=index, cellmeasure=cellmeasure, out=out)
        else: # 空间基函数是向量函数
            return self.assembly_cell_matrix_for_vspace_with_vector_basis(
                    space, index=index, cellmeasure=cellmeasure, out=out)

    def assembly_cell_matrix_for_vspace_with_vector_basis(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 空间基函数是向量型
        """
        p = space.p
        q = self.q if self.q is not None else p+1 

        coef = self.coef
        mesh = space.mesh
        GD = mesh.geo_dimension()

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs() 
        if out is None:
            D = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        else:
            D = out

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        gphi0 = space.grad_basis(bcs, index=index) # (NQ, NC, ldof, GD, GD)

        if coef is None:
            D += np.einsum('q, qcimn, qcjmn, c->cij', ws, gphi0, gphi0, cellmeasure, optimize=True)
        else:
            pass #todo

        if out is None:
            return D

    def assembly_cell_matrix_for_vspace_with_sacalar_basi(
            self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 标量空间拼成的向量空间 
        """

        GD = space[0].geo_dimension()
        ldof = space[0].number_of_local_dofs()

        integrator = ScalarDiffusionIntegrator(self.coef, self.q)
        # 组装标量的单元扩散矩阵
        # D.shape == (NC, ldof, ldof)
        D = inegrator.assembly_cell_matrix(space, index=index, cellmeasure=cellmeasure)

        if out is None:
            VD = n.zeros((NC, GD*ldof, GD*ldof), dtype=space[0].ftype)
        else:
            assert out.shape == (NC, GD*ldof, GD*ldof)
            VD = out
        
        if space[0].doforder == 'sdofs':
            for i in range(GD):
                VD[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] += D
        elif space0.doforder == 'vdims':
            for i in range(GD):
                VD[:, i::GD, i::GD] += D 

        if out is None:
            return VD
