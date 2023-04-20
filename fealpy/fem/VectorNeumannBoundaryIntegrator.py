import numpy as np

class VectorNeumannBoundaryIntegrator:
    def __init__(self, space, gN, threshold=None, q=None):
        self.space = space
        self.gN = gN #TODO：考虑 gN 可以由 Mesh 提供
        self.q = q
        self.threshold = threshold

    def assembly_face_vector(self, space, index=np.s_[:], facemeasure=None, out=None):
        """
        @brief 组装单元向量
        """

        if isinstance(space, tuple) and ~isinstance(space[0], tuple):
            return self.assembly_face_vector_for_vspace_with_scalar_basis(space, 
                    index=index, facemeasure=facemeasure, out=out)
        else:
            return self.assembly_face_vector_for_vspace_with_vector_basis(space, 
                    index=index, facemeasure=facemeasure, out=out)
        

    def assembly_face_vector_for_vspace_with_scalar_basis(
            self, space, index=np.s_[:], facemeasure=None, out=None):
        """
        @brief 由标量空间张成的向量空间 

        @param[in] space 
        """
        # 假设向量空间是由标量空间组合而成
        space = self.space
        assert isinstance(space, tuple) and ~isinstance(space[0], tuple) 
        
        gN = self.gN
        mesh = space[0].mesh # 获取网格对像
        GD = mesh.geo_dimension()

        if facemeasure is None:
            facemeasure = mesh.entity_measure('face', index=index)

        NF = len(facemeasure)
        ldof = space[0].number_of_face_dofs() 
        if out is None:
            if space[0].doforder == 'sdofs': # 标量基函数自由度排序优先
                bb = np.zeros((NC, GD, ldof), dtype=space.ftype)
            elif space[0].doforder == 'vdims': # 向量分量自由度排序优先
                bb = np.zeros((NC, ldof, GD), dtype=space.ftype)
        else:
            bb = out

        q = self.q if self.q is not None else space[0].p + 1 
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space[0].face_basis(bcs, index=index)

        if callable(gN):
            if hasattr(gN, 'coordtype'):
                if f.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bcs, index=index)
                    val = gN(ps)
                elif f.coordtype == 'barycentric':
                    val = gN(bcs, index=index)
            else: # 默认是笛卡尔
                ps = mesh.bc_to_point(bcs, index=index)
                val = gN(ps)
        else:
            val = gN 

        if isinstance(val, (int, float)):
            if space[0].doforder == 'sdofs':
                bb += val*np.einsum('q, qci, c->ci', ws, phi, cellmeasure, optimize=True)[:, None, :]
            elif space[0].doforder == 'vdims':
                bb += val*np.einsum('q, qci, c->ci', ws, val, phi, cellmeasure, optimize=True)[:, :, None]
        elif isinstance(val, np.ndarray):
            if val.shape == (GD, ): # GD << NC
                if space[0].doforder == 'sdofs':
                    bb += np.einsum('q, d, qci, c->cdi', ws, val, phi, cellmeasure, optimize=True)
                elif space[0].doforder == 'vdims':
                    bb += np.einsum('q, d, qci, c->cid', ws, val, phi, cellmeasure, optimize=True)
            elif val.shape == (NC, GD): 
                if space[0].doforder == 'sdofs':
                    bb += np.einsum('q, cd, qci, c->cdi', ws, val, phi, cellmeasure, optimize=True)
                elif space[0].doforder == 'vdims':
                    bb += np.einsum('q, cd, qci, c->cid', ws, val, phi, cellmeasure, optimize=True)
            elif val.shape == (NQ, NC, GD):
                if space[0].doforder == 'sdofs':
                    bb += np.einsum('q, qcd, qci, c->cdi', ws, val, phi, cellmeasure, optimize=True)
                elif space[0].doforder == 'vdims':
                    bb += np.einsum('q, qcd, qci, c->cid', ws, val, phi, cellmeasure, optimize=True)

        if out is None:
            return bb 
    
    def assembly_face_vector_for_vspace_with_vector_basis(
            self, index=np.s_[:], facemeasure=None, out=None):
        """
        """
        pass


