import numpy as np

from .scalar_diffusion_integrator import ScalarDiffusionIntegrator 

class VectorDiffusionIntegrator:
    """
    @note (c \\grad u, \\grad v)
    """    
    def __init__(self, c=None, q=None):
        self.coef = c
        self.q = q
        self.type = 'BL10'

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @note 没有参考单元的组装方式
        """
        if isinstance(space, tuple) and not isinstance(space[0], tuple): # 由标量空间组合而成的空间
            return self.assembly_cell_matrix_for_vspace_with_sacalar_basis(space, index=index, cellmeasure=cellmeasure, out=out)
        else: # 空间基函数是向量函数
            return self.assembly_cell_matrix_for_vspace_with_vector_basis(space, index=index, cellmeasure=cellmeasure, out=out)

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

    def assembly_cell_matrix_for_vspace_with_sacalar_basis(
            self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 标量空间拼成的向量空间 
        """

        mesh = space[0].mesh
        GD = space[0].geo_dimension()
        assert len(space) == GD

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)
        ldof = space[0].number_of_local_dofs()
        integrator = ScalarDiffusionIntegrator(self.coef, self.q)
        # 组装标量的单元扩散矩阵
        # D.shape == (NC, ldof, ldof)
        D = integrator.assembly_cell_matrix(space[0], index=index, cellmeasure=cellmeasure)
        NC = len(cellmeasure)

        if out is None:
            VD = np.zeros((NC, GD*ldof, GD*ldof), dtype=space[0].ftype)
        else:
            assert out.shape == (NC, GD*ldof, GD*ldof)
            VD = out
        
        if space[0].doforder == 'sdofs':
            for i in range(GD):
                VD[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] += D
        elif space[0].doforder == 'vdims':
            for i in range(GD):
                VD[:, i::GD, i::GD] += D 

        if out is None:
            return VD

    def assembly_cell_matrix_fast(self, space, trialspace=None, testspace=None, coefspace=None, index=np.s_[:], cellmeasure=None, out=None):
        """
        @note 基于无数值积分的组装方式
        """
        self.space = space
        if isinstance(space, tuple) and not isinstance(space[0], tuple): # 由标量空间组合而成的空间
            return self.assembly_cell_matrix_for_vspace_with_sacalar_basis_fast(space,
                trialspace, testspace, coefspace, index=index, cellmeasure=cellmeasure, out=out)
        else: # 空间基函数是向量函数
            return self.assembly_cell_matrix_for_vspace_with_vector_basis_fast(space,
                trialspace, testspace, coefspace, index=index, cellmeasure=cellmeasure, out=out)

    def assembly_cell_matrix_for_vspace_with_vector_basis_fast(self, space,
        trialspace, testspace, coefspace, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 空间基函数是向量型
        """
        pass

    def assembly_cell_matrix_for_vspace_with_sacalar_basis_fast(self, space,
        trialspace, testspace, coefspace, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 标量空间拼成的向量空间 
        """

        mesh = space[0].mesh
        GD = space[0].geo_dimension()
        assert len(space) == GD

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)
        ldof = space[0].number_of_local_dofs()
        integrator = ScalarDiffusionIntegrator(self.coef, self.q)
        # 组装标量的单元扩散矩阵
        # D.shape == (NC, ldof, ldof)
        D = integrator.assembly_cell_matrix_fast(space[0], trialspace, testspace, coefspace, index=index, cellmeasure=cellmeasure)
        NC = len(cellmeasure)

        if out is None:
            VD = np.zeros((NC, GD*ldof, GD*ldof), dtype=space[0].ftype)
        else:
            assert out.shape == (NC, GD*ldof, GD*ldof)
            VD = out
        
        if space[0].doforder == 'sdofs':
            for i in range(GD):
                VD[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] += D
        elif space[0].doforder == 'vdims':
            for i in range(GD):
                VD[:, i::GD, i::GD] += D 

        if out is None:
            return VD

