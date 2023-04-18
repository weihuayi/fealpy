import numpy as np

class VectorNeumannBoundaryIntegrator:
    def __init__(self, space, gN, threshold=None, q=None):
        self.space = space
        self.gN = gN
        self.q = q
        self.threshold = threshold

    def assembly_face_vector(self, space, index=np.s_[:], facemeasure=None, out=None):
        """
        @brief 组装单元向量

        @param[in] space 

        @note f 是一个向量函数，返回形状可以为
            * (GD, ) 常向量情形
            * (NC, GD) 分片常向量情形
            * (NQ, NC, GD) 变向量情形
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
        """
        pass
    
    def assembly_face_vector_for_vspace_with_vector_basis(
            self, index=np.s_[:], facemeasure=None, out=None):
        """
        """
        pass


