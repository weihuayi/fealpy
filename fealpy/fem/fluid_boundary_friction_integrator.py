from typing import Optional, Union, Callable
import numpy as np


class FluidBoundaryFrictionIntegrator:

    def __init__(self, mu, threshold=None, q=None):
        """
        @brief 

        @param[in] mu 
        """
        self.mu = mu 
        self.q = q
        self.threshold = threshold

    def assembly_face_matrix(self, space, out=None):
        """
        @brief 组装面元向量
        """
        if isinstance(space, tuple) and ~isinstance(space[0], tuple):
            return self.assembly_face_matrix_for_vspace_with_scalar_basis(
                    space, out=out)
        else:
            return self.assembly_face_matrix_for_vspace_with_vector_basis(
                    space, out=out)

    def assembly_face_matrix_for_vspace_with_scalar_basis(space, out=None):
        """
        """
        assert isinstance(space, tuple) and ~isinstance(space[0], tuple) 
        
        mu = self.mu
        mesh = space[0].mesh # 获取网格对像
        GD = mesh.geo_dimension()
  
        if isinstance(self.threshold, np.ndarray):
            index = self.threshold
        else:
            index = mesh.ds.boundary_face_index()
            if callable(self.threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[self.threshold(bc)]

        cindex = mesh.ds.face_to_cell()[index, 0] # 边界面的左边的单元

        facemeasure = mesh.entity_measure('face', index=index)
        NF = len(facemeasure)
        ldof = space[0].number_of_local_dofs(doftype='face')

        q = self.q if self.q is not None else space[0].p + 1 #TODO： 是否合合适？
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        if space[0].doforder == 'sdofs': # 标量自由度优先排序 
            pass
        elif space[0].doforder == 'vdims':
            pass
        return K

