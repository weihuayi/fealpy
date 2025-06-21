import numpy as np

class VectorBoundarySourceIntegrator:
    """
    @brief 组装向量型的边界源项，主要用于 Neuann 和 Robin 边界函数的积分
    """
    def __init__(self, source, threshold=None, q=3):
        self.source = source 
        self.q = q 
        self.threshold = threshold

    def assembly_face_vector(self, space, out=None):
        """
        @brief 组装面元向量
        """

        if isinstance(space, tuple) and ~isinstance(space[0], tuple):
            return self.assembly_face_vector_for_vspace_with_scalar_basis(
                    space, out=out)
        else:
            return self.assembly_face_vector_for_vspace_with_vector_basis(
                    space, out=out)
        

    def assembly_face_vector_for_vspace_with_scalar_basis(
            self, space, out=None):
        """
        @brief 由标量空间张成的向量空间 

        @param[in] space 
        """
        assert isinstance(space, tuple) and ~isinstance(space[0], tuple) 
        
        source = self.source
        threshold = self.threshold
        mesh = space[0].mesh # 获取网格对像
        GD = mesh.geo_dimension()

        if isinstance(threshold, np.ndarray) or isinstance(threshold, slice):
            # 在实际应用中，一般直接给出相应网格实体编号
            # threshold 有以下几种情况：
            # 1. 编号数组
            # 2. slice
            # 3. 逻辑数组
            index = threshold 
        else:
            index = mesh.ds.boundary_face_index()
            if callable(threshold): 
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)] # 通过边界的重心来判断
  
        facemeasure = mesh.entity_measure('face', index=index)
        NF = len(facemeasure)
        ldof = space[0].number_of_local_dofs(doftype='face')

        bb = np.zeros((NF, ldof, GD), dtype=space[0].ftype)

        qf = mesh.integrator(self.q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        phi = space[0].face_basis(bcs, index=index) # (NQ, NF, ldof)
        if callable(source):
            if source.coordtype == 'cartesian':
                n = mesh.face_unit_normal(index=index)
                ps = mesh.bc_to_point(bcs, index=index)
                # 在实际问题当中，法向 n  这个参数一般不需要
                # 传入 n， 用户可根据需要来计算边界面的法向梯度
                val = source(ps, n)
            elif source.coordtype == 'barycentric':
                val = source(bcs, index=index)
        else:
            val = source
        if isinstance(val, np.ndarray):
            if val.shape == (GD, ): # 假设 GD << NC
                    bb += np.einsum('q, d, qfi, f->fid', ws, val, phi, facemeasure, optimize=True)
            elif val.shape == (NF, GD): 
                    bb += np.einsum('q, fd, qfi, f->fid', ws, val, phi, facemeasure, optimize=True)
            elif val.shape == (NQ, NF, GD):
                    bb += np.einsum('q, qfd, qfi, f->fid', ws, val, phi, facemeasure, optimize=True)
            else:
                raise ValueError(f"val with shape {val.shape}, I can't deal with it!")
        else:
            raise ValueError(f"val is not a np.ndarray object! It is {type(val)}")

        gdof = space[0].number_of_global_dofs()
        face2dof = space[0].face_to_dof(index=index) # 标量的面元自由度数组
        if out is None:
            F = np.zeros((GD*gdof, ), dtype=mesh.ftype)
        else:
            assert out.shape == (GD*gdof,)
            F = out

        if space[0].doforder == 'sdofs': # 标量空间自由度优先排序
            V = F.reshape(GD, gdof)
            for i in range(GD):
                np.add.at(V[i, :], face2dof, bb[:, :, i])
        elif space[0].doforder == 'vdims': # 向量分量自由度优先排序
            V = F.reshape(gdof, GD) 
            for i in range(GD):
                np.add.at(V[:, i], face2dof, bb[:, :, i])

        if out is None:
            return F
    
    def assembly_face_vector_for_vspace_with_vector_basis(
            self, space, out=None):
        """
        """
        assert ~isinstance(self.space, tuple) 

        source = self.source
        mesh = space.mesh # 获取网格对像
        GD = mesh.geo_dimension()
  
        if isinstance(threshold, np.ndarray) or isinstance(threshold, slice):
            # 在实际应用中，一般直接给出相应网格实体编号
            # threshold 有以下几种情况：
            # 1. 编号数组
            # 2. slice
            # 3. 逻辑数组
            index = threshold 
        else:
            index = mesh.ds.boundary_face_index()
            if callable(threshold): 
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)] # 通过边界的重心来判断

        facemeasure = mesh.entity_measure('face', index=index)
        NF = len(facemeasure)
        ldof = space.number_of_face_dofs() 
        if out is None:
            bb = np.zeros((NC, ldof), dtype=space.ftype)
        else:
            bb = out

        qf = mesh.integrator(self.q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if callable(source):
            if ~hasattr(source, 'coordtype') or source.coordtype == 'cartesian':
                ps = mesh.bc_to_point(bcs, index=index)
                n = mesh.face_unit_normal(index=index)
                # 在实际问题当中，法向 n  这个参数一般不需要
                # 传入 n， 用户可根据需要来计算 Neumann 边界的法向梯度
                val = source(ps, n) 
            elif source.coordtype == 'barycentric':
                # 这个时候 source 是一个有限元函数，一定不需要算面法向
                val = source(bcs, index=index) 
        else:
            val = source 

        phi = space.face_basis(bcs, index=index) # (NQ, NF, ldof, GD)
        if isinstance(val, np.ndarray):
            if val.shape == (GD, ): # GD << NC
                    bb += np.einsum('q, d, qfid, f->fi', ws, val, phi, facemeasure, optimize=True)
            elif val.shape == (NF, GD): 
                    bb += np.einsum('q, fd, qfid, f->fi', ws, val, phi, facemeasure, optimize=True)
            elif val.shape == (NQ, NF, GD):
                    bb += np.einsum('q, qfd, qfid, f->fi', ws, val, phi, facemeasure, optimize=True)
            else:
                raise ValueError(f"val with shape {val.shape}, I can't deal with it!")
        else:
            raise ValueError(f"val is not a np.ndarray object! It is {type(val)}")

        if out is None:
            return bb 



