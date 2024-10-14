import numpy as np

from .scalar_convection_integrator import ScalarConvectionIntegrator 

class VectorConvectionIntegrator:
    """
    @note (a * c \\cdot \\nabla u, v)
    """    

    def __init__(self, c=None, a=1 ,q=3):
        self.coef = c
        self.a = a
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
        if out is None:
            return D

    def assembly_cell_matrix_for_vspace_with_sacalar_basis(
            self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 标量空间拼成的向量空间 
        """

        GD = space[0].geo_dimension()
        ldof = space[0].number_of_local_dofs()

        integrator = ScalarConvectionIntegrator(self.coef, self.q)
        # 组装标量的单元扩散矩阵
        # D.shape == (NC, ldof, ldof)
        D = integrator.assembly_cell_matrix(space[0], index=index, cellmeasure=cellmeasure)
        NC = len(cellmeasure)

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

    def fast_assembly_cell_matrix(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        """
        mesh = space.mesh 
        assert mesh.meshtype in ['tri', 'tet']

    def assembly_cell_matrix_ref(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元的矩阵组装
        """
        pass
