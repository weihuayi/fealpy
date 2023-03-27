import numpy as np


class ConvectionIntegrator:
    """
    @note (c \\cdot \\nabla u, v)
    """    

    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None):
        """
        @note 没有参考单元的组装方式
        """
        q = self.q
        coef = self.coef
        mesh = space.mesh

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()


        gphi = space.grad_basis(bcs, index=index) 
        phi = space.basis(bcs, index=index) 

        if callable(coef):
            if hasattr(coef, 'coordtype'):
                if coef.coordtype == 'barycentric':
                    coef = coef(bcs, index)
                elif coef.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bcs, index=index)
                    coef = coef(ps)
            else:
                raise ValueError('''
                You should add decorator "cartesian" or "barycentric" on
                function `basis0`

                from fealpy.decorator import cartesian, barycentric

                @cartesian
                def basis0(p):
                    ...

                @barycentric
                def basis0(p):
                    ...

                ''')
            if coef.ndim == 3: #(NQ, NC, GD)
                shape = (3-len(coef.shape))*(1, ) + coef.shape
                C = np.einsum('q, qcn, qck, qcmn, c->ckm',ws,coef.reshape(shape),phi,gphi,cellmeasure) 
            else:
                raise ValueError("coef 的维度超出了支持范围")

        return C

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
