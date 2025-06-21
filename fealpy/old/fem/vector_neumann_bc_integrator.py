
from typing import Optional, Union, Callable
import numpy as np

class VectorNeumannBCIntegrator:
    def __init__(self, gN, threshold=None, q=None):
        self.gN = gN #TODO：考虑 gN 可以由 Mesh 提供
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
        
        gN = self.gN
        mesh = space[0].mesh # 获取网格对像
        GD = mesh.geo_dimension()
  
        if isinstance(self.threshold, np.ndarray):
            index = self.threshold
        else:
            index = mesh.ds.boundary_face_index()
            if callable(self.threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[self.threshold(bc)]

        facemeasure = mesh.entity_measure('face', index=index)
        NF = len(facemeasure)
        ldof = space[0].number_of_local_dofs(doftype='face')

        bb = np.zeros((NF, ldof, GD), dtype=space[0].ftype)

        q = self.q if self.q is not None else space[0].p + 1 
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        phi = space[0].face_basis(bcs, index=index) # (NQ, NF, ldof)
        n = mesh.face_unit_normal(index=index)

        if callable(gN):
            if ~hasattr(gN, 'coordtype') or gN.coordtype == 'cartesian':
                ps = mesh.bc_to_point(bcs, index=index)
                # 在实际问题当中，法向 n  这个参数一般不需要
                # 传入 n， 用户可根据需要来计算 Neumann 边界的法向梯度
                val = gN(ps, n)
            elif gN.coordtype == 'barycentric':
                val = gN(bcs, index=index)
        else:
            val = gN 

        if isinstance(val, np.ndarray):
            if val.shape == (GD, ): # GD << NC
                    bb += np.einsum('q, d, qfi, f->fid', ws, val, phi, facemeasure, optimize=True)
            elif val.shape == (NF, GD): 
                    bb += np.einsum('q, fd, qfi, f->fid', ws, val, phi, facemeasure, optimize=True)
            elif val.shape == (NQ, NF, GD):
                    bb += np.einsum('q, qfd, qfi, f->fid', ws, val, phi, facemeasure, optimize=True)

        gdof = space[0].number_of_global_dofs()
        face2dof = space[0].face_to_dof(index=index) # 标量的面元自由度数组
        if out is None:
            F = np.zeros((GD*gdof, ), dtype=self.ftype)
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

        gN = self.gN
        mesh = space.mesh # 获取网格对像
        GD = mesh.geo_dimension()
  
        if isinstance(self.threshold, np.ndarray):
            index = self.threshold
        else:
            index = mesh.ds.boundary_face_index()
            if callable(self.threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[self.threshold(bc)]

        facemeasure = mesh.entity_measure('face', index=index)
        NF = len(facemeasure)
        ldof = space.number_of_face_dofs() 
        if out is None:
            bb = np.zeros((NC, ldof), dtype=space.ftype)
        else:
            bb = out

        q = self.q if self.q is not None else space.p + 1 
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space.face_basis(bcs, index=index) # (NQ, NF, ldof, GD)

        if callable(gN):
            if ~hasattr(gN, 'coordtype') or gN.coordtype == 'cartesian':
                ps = mesh.bc_to_point(bcs, index=index)
                n = mesh.face_unit_normal(index=index)
                # 在实际问题当中，法向 n  这个参数一般不需要
                # 传入 n， 用户可根据需要来计算 Neumann 边界的法向梯度
                val = gN(ps, n) 
            elif gN.coordtype == 'barycentric':
                # 这个时候 gN 是一个有限元函数，一定不需要算面法向
                val = gN(bcs, index=index) 
        else:
            val = gN 

        if isinstance(val, np.ndarray):
            if val.shape == (GD, ): # GD << NC
                    bb += np.einsum('q, d, qfid, f->fi', ws, val, phi, facemeasure, optimize=True)
            elif val.shape == (NF, GD): 
                    bb += np.einsum('q, fd, qfid, f->fi', ws, val, phi, facemeasure, optimize=True)
            elif val.shape == (NQ, NF, GD):
                    bb += np.einsum('q, qfd, qfid, f->fi', ws, val, phi, facemeasure, optimize=True)
        if out is None:
            return bb 


